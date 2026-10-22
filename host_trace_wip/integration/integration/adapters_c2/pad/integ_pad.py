# INTEG_PAD=1: SGLang's CUDA-graph padding, emulated in our arms for an A/B against running unpadded. The default,
# INTEG_PAD=0, is the source adapters' code path: no padding.
# - Decode: bs goes up to the decode graph bucket. The list is generate_decode_cuda_graph_batch_sizes at the serving
#   max_bs (DECODE_MAX_BS below), which DecodeCudaGraphRunner captures; _pad_to_bucket picks the bucket. A bs above the list's max runs unpadded, as the engine runs it eagerly.
#   Padded rows are filled the way build_decode_registry's slots fill them: seq_lens / seq_lens_cpu get the backend's
#   get_cuda_graph_seq_len_fill_value(); req_pool_indices, positions and out_cache_loc get 0. input_ids are not read,
#   and get 0 here.
# - Extend / mixed: the token count goes up to the prefill bucket (cuda_graph_config.prefill.bs, PrefillCudaGraphRunner's
#   load_batch). It is skipped beyond the list's max or a 2x padding factor (can_run_graph). Padded tokens get 0 in
#   input_ids, positions and out_cache_loc; the request fields are unchanged.
# Padding runs eagerly before the forward, as the runners' fill_from does: one F.pad launch per padded tensor.
import bisect
import collections
import copy
import os

import torch
import torch.nn.functional as F

PAD = os.environ.get("INTEG_PAD", "0") == "1"
_MAX_PREFILL_PADDING_FACTOR = 2  # prefill_cuda_graph_runner._MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR
_DECODE_BS = ("req_pool_indices", "seq_lens", "orig_seq_lens")  # bs axis; seq lengths get the fill value
_TOKENS = ("input_ids", "positions", "out_cache_loc")  # token axis, 0


# Decode buckets: the serving default's list. one_batch's drivers set the decode max_bs to their largest --batch-size, so the
# resolved cuda_graph_config.decode.bs would end with an exact bucket at it. The serving max_bs on a GPU over 160 GB is
# 512 (memory_hook.py, the B200 tier), which gives 52 buckets.
DECODE_MAX_BS = int(os.environ.get("INTEG_PAD_DECODE_MAX_BS", "512"))


class Pad:
    def __init__(self, backend):
        from sglang.srt.arg_groups.cuda_graph_hook import generate_decode_cuda_graph_batch_sizes
        from sglang.srt.runtime_context import get_exec, get_server_args

        cfg = get_exec().graph.cuda_graph_config
        self.decode_bs = sorted(generate_decode_cuda_graph_batch_sizes(get_server_args(), DECODE_MAX_BS))
        self.prefill_tokens = sorted(cfg.prefill.bs)
        self.fill = backend.get_cuda_graph_seq_len_fill_value()
        # per kind: calls, padded calls, rows computed, rows asked for, pad launches; unpadded: fields left at raw size
        self.stats = collections.Counter()
        self.unpadded = collections.Counter()
        print(f"integ_pad: decode buckets {self.decode_bs}, prefill buckets up to {self.prefill_tokens[-1]}, fill {self.fill}", flush=True)

    def _bucket(self, n, buckets):
        i = bisect.bisect_left(buckets, n)
        return buckets[i] if i < len(buckets) else None

    def decode_size(self, bs, count=False):
        """The padded bs of a decode of bs requests, or None (no padding). count: tally it as a prep-path call
        (PrepF, whose padding is in its trace: pad_launches counts the traced F.pad calls per call, 6)."""
        size = self._bucket(bs, self.decode_bs)
        size = size if size is not None and size != bs else None
        if count:
            self.stats["prep_calls"] += 1
            self.stats["prep_rows_asked"] += bs
            self.stats["prep_rows_computed"] += size or bs
            if size:
                self.stats["prep_padded_calls"] += 1
                self.stats["prep_pad_launches"] += 6
        return size

    def batch(self, fb, input_ids, positions):
        """(padded ForwardBatch, input_ids, positions, kind, raw rows, padded rows), or None to run unpadded."""
        mode = fb.forward_mode
        if mode.is_decode():
            kind, raw = "decode", input_ids.shape[0]
            size = self._bucket(raw, self.decode_bs)
        elif mode.is_extend() and not mode.is_target_verify():
            kind, raw = "extend", input_ids.shape[0]
            size = self._bucket(raw, self.prefill_tokens)
            if size is not None and size > raw * _MAX_PREFILL_PADDING_FACTOR:
                size = None
        else:
            return None
        self.stats[f"{kind}_calls"] += 1
        self.stats[f"{kind}_rows_asked"] += raw
        if size is None or size == raw:
            self.stats[f"{kind}_rows_computed"] += raw
            return None
        extra = size - raw
        self.stats[f"{kind}_padded_calls"] += 1
        self.stats[f"{kind}_rows_computed"] += size
        p = copy.copy(fb)
        p_ids, p_pos = F.pad(input_ids, (0, extra)), F.pad(positions, (0, extra))
        p.input_ids, p.positions = p_ids, p_pos
        self.stats[f"{kind}_pad_launches"] += 2
        names = _TOKENS + (_DECODE_BS if kind == "decode" else ())
        for n in names[2:]:
            t = getattr(fb, n, None)
            if isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 1 and t.shape[0] == raw:
                setattr(p, n, F.pad(t, (0, extra), value=self.fill if n in ("seq_lens", "orig_seq_lens") else 0))
                self.stats[f"{kind}_pad_launches"] += 1
        if kind == "decode":
            p.batch_size = size
            if fb.seq_lens_cpu is not None:
                p.seq_lens_cpu = F.pad(fb.seq_lens_cpu, (0, extra), value=self.fill)
            if fb.seq_lens_sum is not None:
                p.seq_lens_sum = fb.seq_lens_sum + extra * self.fill
        for n, t in vars(fb).items():  # CUDA fields of the raw size left as they are (none expected)
            if n not in names and isinstance(t, torch.Tensor) and t.is_cuda and t.dim() >= 1 and t.shape[0] == raw:
                self.unpadded[f"{kind}:{n}"] += 1
        return p, p_ids, p_pos, kind, raw, size

    @staticmethod
    def slice_out(out, kind, raw, size, n_reqs):
        """The output's tensor fields cut to the raw rows: decode rows are requests, extend's per-token fields (hidden
        states with capture_hidden_mode FULL) tokens; extend's logits are already one row per request."""
        res = copy.copy(out)
        for n, t in vars(out).items():
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == size and (kind == "decode" or size != n_reqs):
                setattr(res, n, t[:raw])
        return res

    def summary(self):
        return {"buckets": {"decode": self.decode_bs, "prefill_max": self.prefill_tokens[-1]}, "fill": self.fill,
                "stats": dict(self.stats), "unpadded_fields": dict(self.unpadded)}


def check(arm, kind, got, raw_args, padded_args):
    """A padded traced call's raw rows (got) against the model's own eager forward on the same batch unpadded
    ("bitwise", the A/B's correctness) and padded ("bitwise_padded_eager": the same rows and GEMM shapes)."""
    got = got.clone()
    torch.cuda.synchronize()
    init = arm.orig_init_metadata or arm.backend.init_forward_metadata
    wants = {}
    for tag, (ids, pos, fb) in (("padded", padded_args), ("unpadded", raw_args)):
        init(fb)
        want = arm.orig_forward(ids, pos, fb).next_token_logits
        wants[tag] = want[: got.shape[0]] if tag == "padded" else want
    torch.cuda.synchronize()
    rec = {"kind": kind, "shape": list(got.shape), "rows_computed": padded_args[0].shape[0], "max_ref": float(wants["unpadded"].float().abs().max())}
    for tag, key in (("unpadded", ""), ("padded", "_padded_eager")):
        w = wants[tag]
        same = got.shape == w.shape
        rec["bitwise" + key] = bool(same and torch.equal(got, w))
        rec["max_abs" + key] = float((got.float() - w.float()).abs().max()) if same else None
        rec["argmax_equal" + key] = bool(same and torch.equal(got.argmax(-1), w.argmax(-1)))
    return rec
