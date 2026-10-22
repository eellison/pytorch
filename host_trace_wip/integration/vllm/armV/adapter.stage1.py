# Arm V: vLLM's eager model forward (enforce_eager, Model Runner V2) under host-trace replay, after sglang/armF/adapter.py.
#
# install() swaps the model's forward for ArmV.forward, so only the model call is traced: input prep and the FlashInfer
# metadata build stay eager (stage 1). Every Attention layer runs direct-call (the Python bodies of
# unified_kv_cache_update / unified_attention_with_output, so the trace sees their ops), and FlashInfer's trtllm-gen
# calls go through trtllm_shim's dispatcher ops. Every CUDA tensor the forward touches is a top-level argument:
# params/buffers (swapped in by _reparametrize_module), plain tensor attributes of modules (each layer's kv_cache), the
# trtllm workspace and counters, input_ids, positions, the slot mapping and the metadata's tensors. Each (decode |
# prefill | mixed, token count, request counts, metadata ints) is its own HostTraceReplay entry: exact sizes, no
# padding. The forward context the traced step sees is a copy with the traced metadata.
#
# Decode max_seq_len (the trtllm-gen kernel's split-KV choice and a kernel parameter) is the actual maximum per step
# in vLLM's eager metadata, so as an extern scalar it would be a new key, and a new harvest, every step. ARMV_DECODE_MAX_SEQ
# "model" (default) sets it to max_model_len in the live metadata, as vLLM's FULL cudagraph capture does
# (model_states/default.py prepare_attn, for_capture): the trace, its eager fallbacks and the check's reference all
# run with it. "actual" leaves vLLM's value. ARMV_PREFILL_MAX_KV does the same for the prefill max_kv_len, which in a
# mixed batch is the maximum over all requests, decodes included, so it also moves every step.
import collections
import copy
import os
import time
import traceback

import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.nn.utils.stateless import _reparametrize_module

FAMILIES = tuple(os.environ.get("ARMV_FAMILIES", "blas,attention,rng,reduce").split(","))
MEMORY = os.environ.get("ARMV_MEMORY", "run_buffer")
DECODE_MAX_SEQ = os.environ.get("ARMV_DECODE_MAX_SEQ", "model")
PREFILL_MAX_KV = os.environ.get("ARMV_PREFILL_MAX_KV", "model")
ht.raise_unexpected = bool(os.environ.get("ARMV_HT_RAISE"))

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


def _same_view(a, b):
    return a is b or (a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype)


class ArmV:
    def __init__(self, runner):
        import vllm.v1.attention.backends.flashinfer as fi
        from vllm.model_executor.layers.attention.attention import Attention

        import trtllm_shim

        self.runner = runner
        self.model = runner.model
        self.orig_forward = self.model.forward
        self.max_model_len = runner.max_model_len
        self.fi = fi
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
        self.layers = [m for m in self.model.modules() if isinstance(m, Attention)]
        trtllm_shim.install()
        fi._get_trtllm_workspace_buffer()
        # plain CUDA tensor attributes (each layer's kv_cache), the trtllm workspace (a module global) and counters
        self.attrs = [(m, k, v) for m in self.model.modules() for k, v in vars(m).items() if _cuda_tensor(v) and id(v) not in pb]
        self.attrs += [(fi, "trtllm_workspace_buffer", fi.trtllm_workspace_buffer), (trtllm_shim, "COUNTERS", trtllm_shim.COUNTERS)]
        self.prefix = (*self.params, *(v for _, _, v in self.attrs))
        c, cache = torch.ops._C, torch.ops._C_cache_ops
        extern = (c.rms_norm.default, c.fused_add_rms_norm.default, c.rotary_embedding.default, c.silu_and_mul.default,
                  cache.reshape_and_cache_flash.default, torch.ops.vllm_ht.trtllm_decode.default, torch.ops.vllm_ht.trtllm_context.default)
        # reshape_and_cache_flash writes rows of the interleaved K/V views of each layer's cache at slot = block * block_size
        # + row: verified on small caches (src_vllm1 CHANGES item 14)
        indexed = {cache.reshape_and_cache_flash.default: ("slot_mapping", "key_cache", "value_cache", 2)}
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed)
        self.entries = {}  # key -> HostTraceReplay
        self.calls = collections.Counter()
        self.host_s = collections.defaultdict(list)  # kind -> host seconds of the replayed call (entry to return, no sync)
        self.check = False  # compare each call against the model's own eager forward
        self.checks = []
        self._ctx = None

    def install(self):
        for layer in self.layers:
            layer.use_direct_call = True
        self.model.forward = self.forward

    def _eager(self, why, *args, **kwargs):
        self.calls[why] += 1
        return self.orig_forward(*args, **kwargs)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.flashinfer import FlashInferDecodeKernel, FlashInferMetadata, FlashInferTrtllmAPIDecode, TRTLLMPrefill

        t0 = time.perf_counter()
        if intermediate_tensors is not None or inputs_embeds is not None or input_ids is None:
            return self._eager("eager_inputs", input_ids, positions, intermediate_tensors, inputs_embeds)
        fc = get_forward_context()
        amd, sm = fc.attn_metadata, fc.slot_mapping
        if not isinstance(amd, dict) or not amd or not isinstance(sm, dict) or not sm:
            return self._eager("eager_no_metadata", input_ids, positions)
        md = next(iter(amd.values()))
        slot = next(iter(sm.values()))
        dec, pre = md.decode, md.prefill
        if (type(md) is not FlashInferMetadata or md.use_cascade or any(v is not md for v in amd.values())
                or any(v is not slot for v in sm.values())
                or not (dec is None or type(dec) is FlashInferTrtllmAPIDecode and dec.kernel is FlashInferDecodeKernel.TRTLLM_GEN
                        and dec.q_len_per_req == 1 and dec.q_cu_seq_lens is None and dec.mask is None)
                or not (pre is None or type(pre) is TRTLLMPrefill)):
            return self._eager("eager_metadata_form", input_ids, positions)
        stock = []  # (metadata part, vLLM's max_seq_len) replaced by max_model_len
        if dec is not None and DECODE_MAX_SEQ == "model" and dec.max_seq_len != self.max_model_len:
            stock.append((dec, dec.max_seq_len))
            dec.max_seq_len = self.max_model_len
        if pre is not None and PREFILL_MAX_KV == "model" and pre.max_seq_len != self.max_model_len:
            stock.append((pre, pre.max_seq_len))
            pre.max_seq_len = self.max_model_len
        kind = "mixed" if dec is not None and pre is not None else "decode" if dec is not None else "prefill"
        md_slot = None if _same_view(md.slot_mapping, slot) else md.slot_mapping
        ints = (md.num_actual_tokens, md.num_decodes, md.num_decode_tokens, md.num_prefills, md.num_prefill_tokens,
                dec.max_seq_len if dec is not None else None, (pre.max_q_len, pre.max_seq_len) if pre is not None else None)
        key = (kind, input_ids.shape[0], ints, md_slot is None)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step(key), opaque=(self.provider,), memory=MEMORY)
        self.calls[kind] += 1
        args = [*self.prefix, input_ids, positions, slot]
        if md_slot is not None:
            args.append(md_slot)
        if dec is not None:
            args += [dec.block_tables, dec.seq_lens]
        if pre is not None:
            args += [pre.block_tables, pre.seq_lens, pre.cum_seq_lens_q, pre.cum_seq_lens_kv]
        self._ctx = (fc, md)
        replays = entry.replays
        try:
            (out,) = entry(*args)
        finally:
            self._ctx = None
        if entry.replays > replays:
            self.host_s[kind].append(time.perf_counter() - t0)
        if self.check:
            self._check(kind, out, input_ids, positions, stock)
        return out

    def _check(self, kind, out, input_ids, positions, stock):
        got = out.clone()
        torch.cuda.synchronize()
        want = self.orig_forward(input_ids, positions)
        torch.cuda.synchronize()
        row = {"kind": kind, "shape": list(want.shape), "bitwise": bool(got.shape == want.shape and torch.equal(got, want)),
               "max_abs": float((got.float() - want.float()).abs().max()) if got.shape == want.shape else None}
        if stock:  # vLLM's stock eager, with its own max_seq_len values
            for part, n in stock:
                part.max_seq_len = n
            ref = self.orig_forward(input_ids, positions)
            torch.cuda.synchronize()
            for part, _ in stock:
                part.max_seq_len = self.max_model_len
            row["stock_parts"] = [type(part).__name__ for part, _ in stock]
            row["bitwise_vs_stock"] = bool(torch.equal(got, ref))
            row["max_abs_vs_stock"] = float((got.float() - ref.float()).abs().max())
        self.checks.append(row)

    def _step(self, key):
        from vllm.forward_context import override_forward_context

        kind, _, _, md_slot_shared = key
        n_p, n_a = len(self.params), len(self.attrs)
        proxy = self

        def step(*flat):
            fc, md = proxy._ctx
            i = 0
            params = flat[i:i + n_p]; i += n_p
            attrs = flat[i:i + n_a]; i += n_a
            input_ids, positions, slot = flat[i:i + 3]; i += 3
            md_slot = slot if md_slot_shared else flat[i]
            i += not md_slot_shared
            shell = copy.copy(md)
            shell.slot_mapping = md_slot
            if md.decode is not None:
                shell.decode = copy.copy(md.decode)
                shell.decode.block_tables, shell.decode.seq_lens = flat[i:i + 2]; i += 2
            if md.prefill is not None:
                shell.prefill = copy.copy(md.prefill)
                p = shell.prefill
                p.block_tables, p.seq_lens, p.cum_seq_lens_q, p.cum_seq_lens_kv = flat[i:i + 4]; i += 4
            sfc = copy.copy(fc)
            sfc.attn_metadata = dict.fromkeys(fc.attn_metadata, shell)
            sfc.slot_mapping = dict.fromkeys(fc.slot_mapping, slot)
            saved = [(m, k, getattr(m, k)) for m, k, _ in proxy.attrs]
            try:
                for (m, k, _), t in zip(proxy.attrs, attrs):
                    setattr(m, k, t)
                with override_forward_context(sfc), _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions)
            finally:
                for m, k, v in saved:
                    setattr(m, k, v)
            return (out,)

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

        res = {"families": list(FAMILIES), "decode_max_seq": DECODE_MAX_SEQ, "prefill_max_kv": PREFILL_MAX_KV,
               "args": {"params": len(self.params), "attrs": len(self.attrs), "attr_names": sorted({f"{type(m).__name__}.{k}" for m, k, _ in self.attrs})}}
        res.update(self.counters())
        res["harvests"] = self.provider.harvests
        res["bindings"] = len(self.provider.bindings)
        # replayed calls only: median and count
        res["host_ms"] = {k: (sorted(v)[len(v) // 2] * 1e3, len(v)) for k, v in self.host_s.items() if v}
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
            res["entries"][f"{key[0]}/T{key[1]}/{key[2]}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants),
                "eager_ops": dict(ops.most_common(30)), "host_ops": dict(host_ops.most_common(30)),
                "eager_reasons": [f"{n}x {op}: {why}" for (op, why), n in whys.most_common(20)],
                "per_variant": per_variant, "declines": sorted({str(d)[:400] for d in e.declines})[:30],
                "reasons": list(getattr(e, "_reasons", {}))[:20], "triton_fallbacks": list(e.triton_fallbacks)[:20]}
        res["decline_sites"] = [f"{n}x {m} @ {w}" for (m, w), n in DECLINE_SITES.most_common(20)]
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["checks"] = self.checks
        return res
