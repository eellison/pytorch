# Arm F: SGLang's model forward under host-trace replay on the eager line (no torch.compile).
#
# Runs with the arm-D flags (EagerRunner for decode and extend). EagerRunner keeps load_batch eager (without its copy,
# NO_COPY); install() swaps model.forward for a proxy, so only the model call is traced, and for plain decode and
# extend swaps init_forward_metadata for views of buffers kept here, filled inside the traced step (METADATA).
# Decode and extend are separate HostTraceReplay entries. Every CUDA tensor the forward touches is a top-level
# argument: params/buffers (one per tensor, swapped in by _reparametrize_module), plain tensor attributes of modules,
# the per-layer KV buffers, the ForwardBatch's CUDA tensor fields and the metadata's tensor fields. max_extend_len
# (the extend kernel's grid) is a top-level int. Sizes (bs, T, kv lengths) are the tensors' own sizes: no bucketing,
# no padding.
#
# torch.cuda._host_trace_replay must be imported before the first CuTe compile (model warm-up): import this module
# before ob.load_model.
import collections
import contextlib
import copy
import dataclasses
import os
import traceback

import torch
import torch.cuda._host_trace as ht
import triton
import triton.language as tl
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.nn.utils.stateless import _reparametrize_module

import integ_pad  # INTEG_PAD (adapters_c2/pad)

FAMILIES = tuple(os.environ.get("ARMF_FAMILIES", "blas,attention,rng,reduce").split(","))
# sglang ops harvested by the extern family (src_sgl1 only; ARMF_EXTERN= on int5b)
EXTERN = [n for n in os.environ.get("ARMF_EXTERN", "fused_inplace_qknorm,apply_rope_inplace,_run_activation_inplace,store_cache").split(",") if n]
ht.raise_unexpected = bool(os.environ.get("ARMF_HT_RAISE"))
if "ARMF_EXTERN_CAP" in os.environ:  # diagnostic: which keys a smaller cap refuses (adapter summary "refused")
    import torch.cuda._host_trace_harvest as _harvest

    _harvest._EXTERN_CAP = int(os.environ["ARMF_EXTERN_CAP"])
# "0": launch the Triton decode stage-2 kernel without PDL (same kernel), standing in for patches/02_triton_pdl.diff
TRITON_PDL = os.environ.get("ARMF_TRITON_PDL", "1") == "1"
# "run_buffer": one buffer per run for temporaries, stable addresses; "eager" repatches moved nodes per call (stage 4)
MEMORY = os.environ.get("ARMF_MEMORY", "run_buffer")
# "trace": plain decode and extend attention metadata in buffers kept by ArmF, filled inside the traced
# step (ArmF._metadata, ArmF._fill); "eager": the backend's own init_forward_metadata per call
METADATA = os.environ.get("ARMF_METADATA", "trace")
# copy positions and out_cache_loc into buffers kept by ArmF inside the traced step: every layer's rope and KV store
# then read a fixed address, so a replay patches 2 nodes for fresh batch tensors, not 2 per layer
STAGE = os.environ.get("ARMF_STAGE", "1") == "1"
# set SGLang's SGLANG_EAGER_INPUT_NO_COPY unless it is set: EagerRunner.load_batch then hands the forward the batch's own
# tensors instead of copying them into its static buffers (0.3-0.5 ms per step); they are arguments of the trace either way
NO_COPY = os.environ.get("ARMF_NO_COPY", "1") == "1"
# the prefix tensors (parameters, attrs, KV pools, stage buffers) have fixed layouts: static_shapes makes their sizes and
# strides constants of the trace (as Dynamo's force_parameter_static_shapes), so a replay evaluates no per-weight shape
# arithmetic
STATIC = os.environ.get("ARMF_STATIC", "0") == "1"

DECLINE_SITES = collections.Counter()
_orig_declined_init = ht.Declined.__init__


def _declined_init(exc, *args, **kwargs):
    _orig_declined_init(exc, *args, **kwargs)
    frames = [f for f in traceback.extract_stack()[:-1] if "/torch/" not in f.filename and not f.filename.endswith("adapter.py")]
    where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].lineno} {frames[-1].line}" if frames else "?"
    DECLINE_SITES[(str(exc)[:200], where[:200])] += 1


ht.Declined.__init__ = _declined_init


def _cuda_tensor(x):
    return isinstance(x, torch.Tensor) and x.is_cuda


def _cuda_fields(names, d, cache):
    """The names, other than input_ids and positions, whose values in d are CUDA tensors. cache maps the values' types
    to the tensor-valued names, so a call tests only those."""
    sig = tuple(map(type, map(d.get, names)))
    cand = cache.get(sig)
    if cand is None:
        cand = cache[sig] = tuple(n for n, ty in zip(names, sig) if issubclass(ty, torch.Tensor) and n not in ("input_ids", "positions"))
    return tuple(n for n in cand if d[n].is_cuda)


class _ReadLog:
    """Non-tensor ForwardBatch / ForwardMetadata fields the traced forward reads: their values are baked."""

    reads = collections.Counter()


def _shell(obj, traced, field_names):
    """A shallow copy of a ForwardBatch-like dataclass with its CUDA tensor fields replaced by the traced ones;
    reads of its other dataclass fields are logged."""
    base = type(obj)
    cls = _SHELLS.get(base)
    if cls is None:
        names = frozenset(field_names)

        def __getattribute__(self, name, _names=names, _base=base):
            v = object.__getattribute__(self, name)
            if name in _names and not _cuda_tensor(v) and v is not None:
                _ReadLog.reads[f"{_base.__name__}.{name}"] += 1
            return v

        cls = _SHELLS[base] = type(f"Shell{base.__name__}", (base,), {"__getattribute__": __getattribute__})
    s = copy.copy(obj)
    for k, v in traced.items():
        object.__setattr__(s, k, v)
    object.__setattr__(s, "__class__", cls)
    return s


_SHELLS = {}


@triton.jit
def _inclusive_scan(lens, out, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + 1 + i, tl.cumsum(tl.load(lens + i, mask=i < n, other=0), 0), mask=i < n)


def _indptr(lens, out):
    """out[1:] = cumsum(lens), one Triton launch: torch.cumsum's two CUB kernels are an eager boundary of the trace."""
    n = lens.shape[0]
    _inclusive_scan[(1,)](lens, out, n, BLOCK=triton.next_power_of_2(n))


def _int_field(md):
    """The host int that feeds the extend attention grid: Triton max_extend_len, trtllm max_seq_len_q."""
    return "max_extend_len" if hasattr(md, "max_extend_len") else "max_seq_len_q"


_PATH_FIELDS = ("return_logprob", "capture_hidden_mode", "is_prefill_only", "attn_tp_sequence_sharded")

# the traced extend step's ForwardBatch.extend_start_loc, for _pruned_states; None outside it
_EXTEND_START_LOC = [None]


def _pruned_states(lp, hidden_states, before_norm, aux, md):
    """LogitsProcessor._get_pruned_states, with prefill-without-logprobs' last-token gather spelled
    h.index_select(0, extend_start_loc + extend_seq_lens - 1) in the traced extend step: the same rows as
    h[cumsum(extend_seq_lens) - 1], which breaks the graph twice (cumsum, index.Tensor)."""
    from sglang.srt.layers.logits_processor import LogitsProcessor

    start, mode = _EXTEND_START_LOC[0], md.forward_mode
    plain = mode.is_extend() and not (mode.is_target_verify() or mode.is_draft_extend_v2() or md.extend_return_logprob)
    if start is None or not plain:
        return LogitsProcessor._armf_get_pruned_states(lp, hidden_states, before_norm, aux, md)
    last = start + md.extend_seq_lens - 1
    if aux is not None:
        aux = aux.index_select(0, last) if isinstance(aux, torch.Tensor) else [h.index_select(0, last) for h in aux]
    before_norm = None if before_norm is None else before_norm.index_select(0, last)
    return hidden_states.index_select(0, last), before_norm, aux, None, None, []


def _plain_backend(b):
    """A Triton or trtllm-MHA backend without sliding windows, DCP or a translating KV pool: _metadata's cases."""
    if b.use_sliding_window_kv_pool or b.kv_index_translator.is_translating:
        return False
    if type(b).__name__ == "TritonAttnBackend":
        return b.dcp_size == 1 and not b.sliding_window_size and b.swa_v_head_dim is None
    return type(b).__name__ == "TRTLLMHAAttnBackend" and b._swa_kv_pool is None


class ArmF:
    def __init__(self, model_runner):
        self.mr = model_runner
        self.model = model_runner.model
        self.orig_forward = self.model.forward  # bound method of the class
        self.pool = model_runner.token_to_kv_pool
        self.backend = model_runner.attn_backend
        index, self.params, self.names, self.slots = {}, [], [], []
        for mn, m in self.model.named_modules():
            for k, t in [*m._parameters.items(), *m._buffers.items()]:
                if t is None:
                    continue
                if id(t) not in index:
                    index[id(t)] = len(self.params)
                    self.params.append(t.detach())
                self.names.append(f"{mn}.{k}" if mn else k)
                self.slots.append(index[id(t)])
        pb = {id(t) for t in self.params}
        # plain CUDA tensor attributes (not parameters or buffers), e.g. a rotary cache held as an attribute
        self.attrs = [(m, k, v) for m in self.model.modules() for k, v in vars(m).items() if _cuda_tensor(v) and id(v) not in pb]
        self.trtllm = type(self.backend).__name__ == "TRTLLMHAAttnBackend"
        if self.trtllm:
            # the FlashInfer workspace and multi-CTA counters are backend attributes the attention ops write
            self.attrs += [(self.backend, k, getattr(self.backend, k)) for k in ("workspace_buffer", "_multi_ctas_kv_counter_buffer")]
        self.md_trace = METADATA == "trace" and _plain_backend(self.backend)
        if self.md_trace:  # the KV index table _fill's kernels read
            owner = self.backend if self.trtllm else self.backend.kv_index_translator
            self.attrs.append((owner, "req_to_token", owner.req_to_token))
        self.kbuf = list(self.pool.k_buffer)
        self.vbuf = list(self.pool.v_buffer)
        # the arguments every call starts with: the tensors above, fixed here
        self.stage = tuple(torch.empty(1 << 15, dtype=torch.int64, device=self.backend.device) for _ in range(2)) if STAGE else ()
        self.prefix = (*self.params, *(v for _, _, v in self.attrs), *self.kbuf, *self.vbuf, *self.stage)
        self.fields = {}  # (ForwardBatch class, metadata class) -> their field names, and _cuda_fields' caches
        extern = tuple(getattr(torch.ops.sglang, n).default for n in EXTERN)
        if self.trtllm and extern:
            import trtllm_shim  # noqa: F401

            extern += (torch.ops.sglang_ht.trtllm_decode.default, torch.ops.sglang_ht.trtllm_context.default)
        # store_cache writes KV pool rows at indices: verified on small caches
        indexed = {torch.ops.sglang.store_cache.default: ("indices", "k_cache", "v_cache")}
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed) if extern else HarvestProvider(FAMILIES)
        self.entries = {}  # key -> HostTraceReplay
        self.outs = {}  # key -> (output template, tensor field names)
        self.calls = collections.Counter()
        self.check = False  # compare each call against the model's own eager forward
        self.checks = []
        layer = next(m for m in self.model.modules() if type(m).__name__ == "RadixAttention")
        self.layer_heads = (layer.tp_q_head_num, layer.tp_q_head_num // layer.tp_k_head_num, layer.qk_head_dim != layer.v_head_dim)
        self.orig_init_metadata = None
        self.md_bufs = None  # (batch capacity, kv_indices capacity, buffers): _metadata's
        self.md_cache = {}  # (kind, batch size) -> _metadata's metadata, views of md_bufs
        self.fill = False  # the metadata _metadata set last is _metadata's, to fill
        self.pad = integ_pad.Pad(self.backend) if integ_pad.PAD else None
        if self.pad is None:  # INTEG_PAD=0: model.forward is the source adapter's forward itself
            self.forward = self._forward_unpadded

    def install(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.model.forward = self.forward
        if NO_COPY:
            from sglang.srt.environ import envs

            if not envs.SGLANG_EAGER_INPUT_NO_COPY.is_set():
                envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)
        if self.trtllm and EXTERN:
            import trtllm_shim

            trtllm_shim.install(self.backend)
        if not hasattr(LogitsProcessor, "_armf_get_pruned_states"):
            LogitsProcessor._armf_get_pruned_states = LogitsProcessor._get_pruned_states
            LogitsProcessor._get_pruned_states = _pruned_states
        if not TRITON_PDL:
            self.backend.use_pdl = False
        if self.md_trace:
            self.orig_init_metadata = self.backend.init_forward_metadata
            self.backend.init_forward_metadata = self._metadata

    def _metadata(self, fb):
        """The backend's init_forward_metadata for a plain decode or extend, with no kernel here: its tensors
        are views of buffers kept here, at fixed addresses, and the traced step fills them (_fill), so the graph runs
        those kernels and a replay patches no node for them. Anything else takes the backend's own."""
        b, mode, bs = self.backend, fb.forward_mode, fb.batch_size
        self.fill = False
        if self.trtllm:
            from sglang.srt.layers.cp.utils import is_cp_active

            kind = "decode" if mode.is_decode() else "extend" if mode.is_extend() and not (mode.is_target_verify() or mode.is_draft_extend_v2()) else None
            n = 0
            if kind is None or fb.spec_info is not None or is_cp_active(fb):
                return self.orig_init_metadata(fb)
        else:
            if mode.is_decode():
                kind, n = "decode", fb.seq_lens_sum
            elif not mode.is_decode_or_idle() and not mode.is_target_verify() and fb.extend_prefix_lens_cpu is not None:
                kind, n = "extend", sum(fb.extend_prefix_lens_cpu)
            else:
                kind = n = None
            if n is None or fb.spec_info is not None or kind == "decode" and self._lean_gate(fb):
                return self.orig_init_metadata(fb)
            b._dense_one_shot_kv_indptr = None
        if self.md_bufs is None or self.md_bufs[0] < bs or self.md_bufs[1] < n:
            self._alloc_metadata(max(256, 1 << (bs - 1).bit_length()), 1 << max(16, (n - 1).bit_length()))
        md = self.md_cache.get((kind, bs))
        if md is None:
            md = self.md_cache[(kind, bs)] = self._new_metadata(kind, bs)
        if kind == "extend":
            setattr(md, _int_field(md), int(max(fb.extend_seq_lens_cpu)))
        b.forward_metadata = md
        self.fill = True

    def _alloc_metadata(self, cap, kv_cap):
        b, i32, f32, dev = self.backend, torch.int32, torch.float32, self.backend.device
        z = lambda *shape, dtype=i32: torch.zeros(shape, dtype=dtype, device=dev)
        if self.trtllm:  # cache_seqlens, cu_seqlens_q (decode: arange; extend: filled), cu_seqlens_k, page table
            bufs = (z(cap), torch.arange(cap + 1, dtype=i32, device=dev), z(cap + 1), z(cap + 1), z(cap, b.max_num_pages))
        else:  # kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, Lean's Mp, Lp, Op, locks (unused, Lean off), qo_indptr
            h, s, d, p, m = b.num_head, b.max_kv_splits, b.v_head_dim, b.lean_total_programs, b.lean_block_m
            bufs = (z(cap + 1), z(kv_cap, dtype=torch.int64), z(cap, h, s, d, dtype=f32), z(cap, h, s, dtype=f32), z(cap),
                    z(p, m, dtype=f32), z(p, m, dtype=f32), z(p, m, d, dtype=f32), z(p), z(cap + 1))
        self.md_bufs, self.md_cache = (cap, kv_cap, bufs), {}

    def _new_metadata(self, kind, bs):
        b, bufs = self.backend, self.md_bufs[2]
        if self.trtllm:
            from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMMHAMetadata

            cs, arange, cu_q, cu_k, pt = bufs
            q = cu_q[: bs + 1] if kind == "extend" else arange[: bs + 1]
            return TRTLLMMHAMetadata(cache_seqlens_int32=cs[:bs], cu_seqlens_q=q, cu_seqlens_k=cu_k[: bs + 1], page_table=pt[:bs])
        from sglang.srt.layers.attention.triton_backend import ForwardMetadata

        indptr, kv, logits, lse, splits, mp, lp, op, locks, qo = bufs
        if kind == "extend":
            return ForwardMetadata(None, None, None, None, indptr[: bs + 1], kv, qo[: bs + 1], None, None, b.window_kv_indptr, None, None, None)
        return ForwardMetadata(logits[:bs], lse[:bs], None, splits[:bs], indptr[: bs + 1], kv, None, None, None,
                               b.window_kv_indptr, None, None, None, lean_Mp=mp, lean_Lp=lp, lean_Op=op, lean_locks=locks)

    def _fill(self, kind, fb, md):
        """_metadata's tensors, filled by the kernels the backend's eager metadata runs (Triton: the same calls;
        trtllm: the fused update_trtllm_mha_graph_metadata of its graph replay). The same values: kv_indices and the
        page table are wider than eager's, and the kernels bound their reads by the lengths. The extend cu_seqlens_q is
        cumsum(extend_seq_lens), eager's cu_seqlens_k without a prefix, in its own buffer: eager's alias of the two
        would be a written argument another step reads."""
        b = self.backend
        if self.trtllm:
            from sglang.kernels.ops.kvcache.trtllm_mha_graph_metadata import Q_MODE_CUMSUM, Q_MODE_NONE, update_trtllm_mha_graph_metadata

            ext = kind == "extend"
            update_trtllm_mha_graph_metadata(req_pool_indices=fb.req_pool_indices, seq_lens=fb.seq_lens, req_to_token=b.req_to_token,
                                             cache_seqlens=md.cache_seqlens_int32, cu_seqlens_k=md.cu_seqlens_k, page_table=md.page_table,
                                             bs=fb.seq_lens.shape[0], seqlen_offset=0, max_seq_pages=b.max_num_pages, page_size=b.page_size,
                                             cu_seqlens_q=md.cu_seqlens_q if ext else None, qlens=fb.extend_seq_lens if ext else None,
                                             q_mode=Q_MODE_CUMSUM if ext else Q_MODE_NONE)
            return
        lens = fb.extend_prefix_lens if kind == "extend" else fb.seq_lens
        _indptr(lens, md.kv_indptr)
        b.kv_index_translator.fill_packed_read_stream(req_pool_indices=fb.req_pool_indices, seq_lens=lens, indptr=md.kv_indptr,
                                                      total_tokens=md.kv_indices.numel(), out=md.kv_indices)
        if kind == "extend":
            _indptr(fb.extend_seq_lens, md.qo_indptr)
        else:
            b.get_num_kv_splits(md.num_kv_splits, fb.seq_lens)

    def _lean_gate(self, fb):
        """Decode's Lean decision is host-side (seq_lens_sum): taken here, outside the trace; a call it would turn
        on runs eagerly. In the trace the backend's flag is fixed to the decision taken."""
        from sglang.srt.environ import envs

        b = self.backend
        if not hasattr(b, "_lean_decode_seqlen_gate"):  # not the Triton backend
            return False
        if b.enable_deterministic or envs.SGLANG_DISABLE_LEAN_ATTENTION.get() or b.enable_lean_attention is not None:
            return b.enable_lean_attention
        h, g, mla = self.layer_heads
        return b._lean_decode_seqlen_gate(h, g, fb.batch_size, fb.seq_lens_sum, mla)

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        """INTEG_PAD=1: the engine's padded batch (integ_pad), its forward, the raw rows; else _forward_unpadded."""
        p = None
        if self.pad is not None and not (kwargs and any(v is not None for v in kwargs.values())):
            p = self.pad.batch(forward_batch, input_ids, positions)
        if p is None:
            return self._forward_unpadded(input_ids, positions, forward_batch, **kwargs)
        fb, ids, pos, kind, raw, size = p
        if kind == "decode":  # the padded bs's metadata, as the runner's replay init: _metadata's buffers or the backend's own
            self.backend.init_forward_metadata(fb)
        check, self.check = self.check, False
        try:
            out = self._forward_unpadded(ids, pos, fb)
        finally:
            self.check = check
        out = integ_pad.Pad.slice_out(out, kind, raw, size, forward_batch.batch_size)
        if check:
            self.checks.append(integ_pad.check(self, kind, out.next_token_logits, (input_ids, positions, forward_batch), (ids, pos, fb)))
        return out

    def _forward_unpadded(self, input_ids, positions, forward_batch, **kwargs):
        mode = forward_batch.forward_mode
        fill, self.fill = self.fill, False
        if kwargs and any(v is not None for v in kwargs.values()):
            self.calls["eager_kwargs"] += 1
            if fill:
                self._fill("decode" if mode.is_decode() else "extend", forward_batch, self.backend.forward_metadata)
            return self.orig_forward(input_ids, positions, forward_batch, **kwargs)
        if mode.is_decode():
            if not fill and self._lean_gate(forward_batch):  # _metadata filled only with Lean off
                self.calls["eager_lean"] += 1
                return self.orig_forward(input_ids, positions, forward_batch)
            kind = "decode"
        elif mode.is_extend() and not mode.is_target_verify():
            kind = "extend"
        else:
            self.calls["eager_mode"] += 1
            return self.orig_forward(input_ids, positions, forward_batch)
        md = self.backend.forward_metadata
        classes = (type(forward_batch), type(md))
        if classes not in self.fields:
            self.fields[classes] = ([f.name for f in dataclasses.fields(forward_batch)], [f.name for f in dataclasses.fields(md)], {}, {})
        fb_fields, md_fields, fb_cache, md_cache = self.fields[classes]
        fbd, mdd = vars(forward_batch), vars(md)
        fb_t = _cuda_fields(fb_fields, fbd, fb_cache)
        md_t = _cuda_fields(md_fields, mdd, md_cache)
        ints = (getattr(md, _int_field(md)),) if kind == "extend" else ()
        # host fields that pick code paths are baked into the trace, so each value gets its own entry
        host = tuple(repr(fbd.get(n)) for n in _PATH_FIELDS)
        loc = fbd.get("out_cache_loc")
        stage = bool(self.stage) and _cuda_tensor(loc) and max(positions.shape[0], loc.shape[0]) <= self.stage[0].shape[0] and positions.dtype == loc.dtype == torch.int64
        # decode: one entry per batch size, so the native path evaluates one family of variants, not every batch size's
        bs = positions.shape[0] if kind == "decode" else None
        key = (kind, mode, fb_t, md_t, len(ints), host, fill, stage, bs)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step(key, fb_fields, md_fields), opaque=(self.provider,), memory=MEMORY,
                                                             static_shapes=range(len(self.prefix)) if STATIC else ())
        self.calls[kind] += 1
        args = self.prefix + (input_ids, positions, *[fbd[n] for n in fb_t], *[mdd[n] for n in md_t], *ints)
        self._ctx = (forward_batch, md)
        try:
            flat = entry(*args)
        finally:
            self._ctx = None
        template, names = self.outs[key]
        out = object.__new__(type(template))  # copy.copy(template), without copy's dispatch
        vars(out).update(vars(template))
        vars(out).update(zip(names, flat))
        if self.check:
            self._check(kind, out, input_ids, positions, forward_batch)
        return out

    def _check(self, kind, out, input_ids, positions, forward_batch):
        got = out.next_token_logits.clone()
        torch.cuda.synchronize()
        if self.orig_init_metadata is not None:  # the reference runs on the backend's own metadata
            self.orig_init_metadata(forward_batch)
        want = self.orig_forward(input_ids, positions, forward_batch).next_token_logits
        torch.cuda.synchronize()
        same = got.shape == want.shape
        self.checks.append({"kind": kind, "shape": list(want.shape), "bitwise": bool(same and torch.equal(got, want)),
                            "max_abs": float((got.float() - want.float()).abs().max()) if same else None,
                            "max_ref": float(want.float().abs().max()),
                            "argmax_equal": bool(same and torch.equal(got.argmax(-1), want.argmax(-1)))})

    def _step(self, key, fb_fields, md_fields):
        kind, _, fb_t, md_t, n_ints, _, fill, stage, _ = key
        n_p, n_a, n_l = len(self.params), len(self.attrs), len(self.kbuf)
        proxy = self

        def step(*flat):
            fb, md = proxy._ctx
            i = 0
            params = flat[i:i + n_p]; i += n_p
            attrs = flat[i:i + n_a]; i += n_a
            kb = list(flat[i:i + n_l]); i += n_l
            vb = list(flat[i:i + n_l]); i += n_l
            st = flat[i:i + len(proxy.stage)]; i += len(proxy.stage)
            input_ids, positions = flat[i], flat[i + 1]; i += 2
            fbt = dict(zip(fb_t, flat[i:i + len(fb_t)])); i += len(fb_t)
            mdt = dict(zip(md_t, flat[i:i + len(md_t)])); i += len(md_t)
            ints = flat[i:]
            if stage:
                positions = st[0][:positions.shape[0]].copy_(positions)
                fbt["out_cache_loc"] = st[1][:fbt["out_cache_loc"].shape[0]].copy_(fbt["out_cache_loc"])
            fbt.update(input_ids=input_ids, positions=positions, next_token_logits_buffer=None)
            if "batch_size" in fb_fields:
                fbt["batch_size"] = fbt["seq_lens"].shape[0]
            shell_fb = _shell(fb, fbt, fb_fields)
            if n_ints:
                mdt[_int_field(md)] = ints[0]
            shell_md = _shell(md, mdt, md_fields)
            b, pool = proxy.backend, proxy.pool
            saved_md, saved_k, saved_v, saved_lean = b.forward_metadata, pool.k_buffer, pool.v_buffer, getattr(b, "enable_lean_attention", False)
            saved_attrs = [(m, k, getattr(m, k)) for m, k, _ in proxy.attrs]
            try:
                b.forward_metadata = shell_md
                pool.k_buffer, pool.v_buffer = kb, vb
                if kind == "decode" and saved_lean is None:
                    b.enable_lean_attention = False  # the decision taken in _lean_gate
                for (m, k, _), t in zip(proxy.attrs, attrs):
                    setattr(m, k, t)
                if kind == "extend":
                    _EXTEND_START_LOC[0] = fbt.get("extend_start_loc")
                if fill:
                    proxy._fill(kind, shell_fb, shell_md)
                with _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions, shell_fb)
            finally:
                _EXTEND_START_LOC[0] = None
                b.forward_metadata, pool.k_buffer, pool.v_buffer = saved_md, saved_k, saved_v
                if hasattr(b, "enable_lean_attention"):
                    b.enable_lean_attention = saved_lean
                for m, k, v in saved_attrs:
                    setattr(m, k, v)
            names = [f.name for f in dataclasses.fields(out) if isinstance(getattr(out, f.name), torch.Tensor)]
            template = copy.copy(out)
            for n in names:
                setattr(template, n, None)
            proxy.outs[key] = (template, names)
            return tuple(getattr(out, n) for n in names)

        return step

    def counters(self):
        es = list(self.entries.values())
        out = {k: sum(getattr(e, k, 0) for e in es) for k in ("traces", "replays", "eager", "uncaptured", "structural", "guards", "folds")}
        out["variants"] = sum(len(e.variants) for e in es)
        out["calls"] = dict(self.calls)
        return out

    def summary(self):
        import torch.cuda._host_trace_tape as T
        from torch.cuda._host_trace_launch import KernelLaunch

        res = {"families": list(FAMILIES), "extern": EXTERN, "args": {"params": len(self.params), "attrs": [f"{type(m).__name__}.{k}" for m, k, _ in self.attrs], "kv_layers": len(self.kbuf)}}
        res.update(self.counters())
        res["harvests"] = self.provider.harvests
        res["bindings"] = len(self.provider.bindings)
        res["entries"] = {}
        for key, e in self.entries.items():
            ops, whys, host_ops, per_variant = collections.Counter(), collections.Counter(), collections.Counter(), []
            for v in e.variants:
                k = n_e = o = n_h = 0
                vops = collections.Counter()
                for _, rec in v.tape.launches:
                    if isinstance(rec, KernelLaunch):
                        k += 1
                    elif isinstance(rec, T.OpaqueCall):
                        o += 1
                    elif isinstance(rec, T.EagerCall) and rec.host:
                        n_h += 1
                        host_ops[rec.name] += 1
                    elif isinstance(rec, T.EagerCall):
                        n_e += 1
                        ops[rec.name] += 1
                        vops[rec.name] += 1
                        whys[(rec.name, (rec.reason or "no traced host")[:200])] += 1
                steps = v.captured.lowered.steps
                per_variant.append({"kernels": k, "eager_calls": n_e, "host": n_h, "opaque": o,
                                    "boundaries": sum(not isinstance(s, range) and not s.call.host for s in steps),
                                    "segments": len(v.captured.segments), "learns": getattr(v, "learns", None), "eager_ops": dict(vops)})
            res["entries"][f"{key[0]}/{key[1].name}/{key[-1]}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants),
                "eager_ops": dict(ops.most_common(30)), "host_ops": dict(host_ops.most_common(30)),
                "eager_reasons": [f"{n}x {op}: {why}" for (op, why), n in whys.most_common(20)],
                "per_variant": per_variant, "declines": sorted({str(d)[:400] for d in e.declines})[:30],
                "reasons": list(getattr(e, "_reasons", {}))[:20], "triton_fallbacks": list(e.triton_fallbacks)[:20]}
        res["decline_sites"] = [f"{n}x {m} @ {w}" for (m, w), n in DECLINE_SITES.most_common(20)]
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["baked_host_reads"] = dict(_ReadLog.reads.most_common(40))
        res["checks"] = self.checks
        res["pad"] = self.pad.summary() if self.pad is not None else None
        return res
