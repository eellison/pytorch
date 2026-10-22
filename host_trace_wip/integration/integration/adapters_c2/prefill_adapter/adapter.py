# Arm F: SGLang's model forward under host-trace replay on the eager line (no torch.compile).
#
# Runs with the arm-D flags (EagerRunner for decode and extend). EagerRunner keeps load_batch and
# attn_backend.init_forward_metadata eager; install() swaps model.forward for a proxy, so only the model call is
# traced. Decode and extend are separate HostTraceReplay entries. Every CUDA tensor the forward touches is a
# top-level argument: params/buffers (one per tensor, swapped in by _reparametrize_module), plain tensor attributes
# of modules, the per-layer KV buffers, the ForwardBatch's CUDA tensor fields and the triton ForwardMetadata's
# tensor fields. max_extend_len (the extend kernel's grid) is a top-level int. Sizes (bs, T, kv lengths) are the
# tensors' own sizes: no bucketing, no padding.
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
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.nn.utils.stateless import _reparametrize_module

FAMILIES = tuple(os.environ.get("ARMF_FAMILIES", "blas,attention,rng,reduce").split(","))
# sglang ops harvested by the extern family (src_sgl1 only; ARMF_EXTERN= on int5b)
EXTERN = [n for n in os.environ.get("ARMF_EXTERN", "fused_inplace_qknorm,apply_rope_inplace,_run_activation_inplace,store_cache").split(",") if n]
ht.raise_unexpected = bool(os.environ.get("ARMF_HT_RAISE"))
# extern operand cap in bytes; smaller is a diagnostic: which keys it refuses (adapter summary "refused")
EXTERN_CAP = int(os.environ.get("ARMF_EXTERN_CAP", 1 << 30))
# "1": the keys learned at a miss allocate from a pool dropped after them (best-memory mode)
LEARN_POOL = os.environ.get("ARMF_LEARN_POOL") == "1"
# "1": a call whose key would be learned runs eagerly while a worker thread learns the key (later calls replay)
BACKGROUND = os.environ.get("ARMF_BACKGROUND") == "1"
# the provider's bindings: loaded at start, saved after a warm-up (learns persisted across restarts)
BINDINGS = os.environ.get("ARMF_BINDINGS")
# "0": launch the Triton decode stage-2 kernel without PDL (same kernel), standing in for patches/02_triton_pdl.diff
TRITON_PDL = os.environ.get("ARMF_TRITON_PDL", "1") == "1"
# "run_buffer": one buffer per run for temporaries, stable addresses; "eager" repatches moved nodes per call (stage 4)
MEMORY = os.environ.get("ARMF_MEMORY", "run_buffer")
# the extend entries' memory: a run buffer grows with each new max T and the caching allocator keeps every older
# one reserved; "eager" makes eager's requests (about 1 us each, small next to a prefill). Best-memory mode
# (ARMF_LEARN_POOL=1) defaults to "eager".
EXTEND_MEMORY = os.environ.get("ARMF_EXTEND_MEMORY", "eager" if LEARN_POOL else MEMORY)

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
        self.kbuf = list(self.pool.k_buffer)
        self.vbuf = list(self.pool.v_buffer)
        # the arguments every call starts with: the tensors above, fixed here
        self.prefix = (*self.params, *(v for _, _, v in self.attrs), *self.kbuf, *self.vbuf)
        self.fields = {}  # (ForwardBatch class, metadata class) -> their field names
        extern = tuple(getattr(torch.ops.sglang, n).default for n in EXTERN)
        if self.trtllm and extern:
            import trtllm_shim  # noqa: F401

            extern += (torch.ops.sglang_ht.trtllm_decode.default, torch.ops.sglang_ht.trtllm_context.default)
        # store_cache writes KV pool rows at indices: verified on small caches
        indexed = {torch.ops.sglang.store_cache.default: ("indices", "k_cache", "v_cache")}
        # src_sglpre: extern keys at a new size learn without a model run, reading views of the KV pool and workspace
        lendable = (*self.kbuf, *self.vbuf, *(v for _, k, v in self.attrs if k == "workspace_buffer"))
        self.provider = HarvestProvider(
            FAMILIES + ("extern",), extern_ops=extern, indexed=indexed, lendable=lendable, extern_cap=EXTERN_CAP, extern_digest="rows"
        ) if extern else HarvestProvider(FAMILIES)
        self.loaded = self.provider.load(BINDINGS) if BINDINGS else 0
        self.warm_stats = None
        self.entries = {}  # key -> HostTraceReplay
        self.outs = {}  # key -> (output template, tensor field names)
        self.calls = collections.Counter()
        self.check = False  # compare each call against the model's own eager forward
        self.checks = []
        layer = next(m for m in self.model.modules() if type(m).__name__ == "RadixAttention")
        self.layer_heads = (layer.tp_q_head_num, layer.tp_q_head_num // layer.tp_k_head_num, layer.qk_head_dim != layer.v_head_dim)

    def install(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.model.forward = self.forward
        if self.trtllm and EXTERN:
            import trtllm_shim

            trtllm_shim.install(self.backend)
        if not hasattr(LogitsProcessor, "_armf_get_pruned_states"):
            LogitsProcessor._armf_get_pruned_states = LogitsProcessor._get_pruned_states
            LogitsProcessor._get_pruned_states = _pruned_states
        if not TRITON_PDL:
            self.backend.use_pdl = False

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
        mode = forward_batch.forward_mode
        if kwargs and any(v is not None for v in kwargs.values()):
            self.calls["eager_kwargs"] += 1
            return self.orig_forward(input_ids, positions, forward_batch, **kwargs)
        if mode.is_decode():
            if self._lean_gate(forward_batch):
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
            self.fields[classes] = ([f.name for f in dataclasses.fields(forward_batch)], [f.name for f in dataclasses.fields(md)])
        fb_fields, md_fields = self.fields[classes]
        fbd, mdd = vars(forward_batch), vars(md)
        fb_t = tuple(n for n in fb_fields if n not in ("input_ids", "positions") and _cuda_tensor(fbd.get(n)))
        md_t = tuple(n for n in md_fields if _cuda_tensor(mdd.get(n)))
        ints = (getattr(md, _int_field(md)),) if kind == "extend" else ()
        # host fields that pick code paths are baked into the trace, so each value gets its own entry
        host = tuple(repr(fbd.get(n)) for n in _PATH_FIELDS)
        key = (kind, mode, fb_t, md_t, len(ints), host)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step(key, fb_fields, md_fields), opaque=(self.provider,), memory=EXTEND_MEMORY if kind == "extend" else MEMORY, learn_pool=LEARN_POOL, background=BACKGROUND)
        self.calls[kind] += 1
        args = self.prefix + (input_ids, positions, *[fbd[n] for n in fb_t], *[mdd[n] for n in md_t], *ints)
        self._ctx = (forward_batch, md)
        try:
            flat = entry(*args)
        finally:
            self._ctx = None
        template, names = self.outs[key]
        out = copy.copy(template)
        for n, t in zip(names, flat):
            setattr(out, n, t)
        if self.check:
            self._check(kind, out, input_ids, positions, forward_batch)
        return out

    def _check(self, kind, out, input_ids, positions, forward_batch):
        got = out.next_token_logits.clone()
        torch.cuda.synchronize()
        want = self.orig_forward(input_ids, positions, forward_batch).next_token_logits
        torch.cuda.synchronize()
        same = got.shape == want.shape
        self.checks.append({"kind": kind, "shape": list(want.shape), "bitwise": bool(same and torch.equal(got, want)),
                            "max_abs": float((got.float() - want.float()).abs().max()) if same else None,
                            "max_ref": float(want.float().abs().max()),
                            "argmax_equal": bool(same and torch.equal(got.argmax(-1), want.argmax(-1)))})

    def _step(self, key, fb_fields, md_fields):
        kind, _, fb_t, md_t, n_ints, _ = key
        n_p, n_a, n_l = len(self.params), len(self.attrs), len(self.kbuf)
        proxy = self

        def step(*flat):
            fb, md = proxy._ctx
            i = 0
            params = flat[i:i + n_p]; i += n_p
            attrs = flat[i:i + n_a]; i += n_a
            kb = list(flat[i:i + n_l]); i += n_l
            vb = list(flat[i:i + n_l]); i += n_l
            input_ids, positions = flat[i], flat[i + 1]; i += 2
            fbt = dict(zip(fb_t, flat[i:i + len(fb_t)])); i += len(fb_t)
            mdt = dict(zip(md_t, flat[i:i + len(md_t)])); i += len(md_t)
            ints = flat[i:]
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

    def warm_up(self, prefill, lo, hi, step=1, mode=None, log=print):
        """Range warm-up of the extend entry (warm.py, mode ARMF_WARM_MODE "sweep" or "cover"); prefill(lens) runs
        one real prefill of those requests."""
        import warm

        self.warm_stats = warm.warm_up(self, prefill, lo, hi, step, mode or os.environ.get("ARMF_WARM_MODE", "sweep"), log)
        if BINDINGS:
            self.warm_stats["saved"] = self.provider.save(BINDINGS)
        return self.warm_stats

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
        res["persisted"] = {"loaded": self.loaded, "restored": self.provider.restored, "stale": self.provider.stale[:10]}
        res["deferred"] = sum(getattr(e, "deferred", 0) for e in self.entries.values())
        res["background_errors"] = [x for e in self.entries.values() for x in getattr(e, "background_errors", [])][:10]
        res["warm"] = self.warm_stats
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
            res["entries"][f"{key[0]}/{key[1].name}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants),
                "eager_ops": dict(ops.most_common(30)), "host_ops": dict(host_ops.most_common(30)),
                "eager_reasons": [f"{n}x {op}: {why}" for (op, why), n in whys.most_common(20)],
                "per_variant": per_variant, "declines": sorted({str(d)[:400] for d in e.declines})[:30],
                "reasons": list(getattr(e, "_reasons", {}))[:20], "triton_fallbacks": list(e.triton_fallbacks)[:20]}
        res["decline_sites"] = [f"{n}x {m} @ {w}" for (m, w), n in DECLINE_SITES.most_common(20)]
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["baked_host_reads"] = dict(_ReadLog.reads.most_common(40))
        res["checks"] = self.checks
        return res
