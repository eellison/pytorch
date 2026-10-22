# Arm F: SGLang's model forward under host-trace replay on the eager line (no torch.compile).
# adapter_cpp.py: adapter.py (md5 66e97237, then stage 4 merged from md5 5a058047) for install_sgl_cpp; each entry gets
# static_prefix=len(self.prefix) (ARMF_STATIC_PREFIX=0: off) and memory=ARMF_MEMORY when set. ARMF_ONECALL and
# ARMF_TRUST_STATICS (both default "1" on a build with them): the bound call is model.forward (below).
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

FAMILIES = tuple(os.environ.get("ARMF_FAMILIES", "blas,attention,rng,reduce").split(","))
# sglang ops harvested by the extern family (src_sgl1 only; ARMF_EXTERN= on int5b)
EXTERN = [n for n in os.environ.get("ARMF_EXTERN", "fused_inplace_qknorm,apply_rope_inplace,_run_activation_inplace,store_cache").split(",") if n]
ht.raise_unexpected = bool(os.environ.get("ARMF_HT_RAISE"))
if "ARMF_EXTERN_CAP" in os.environ:  # diagnostic: which keys a smaller cap refuses (adapter summary "refused")
    import torch.cuda._host_trace_harvest as _harvest

    _harvest._EXTERN_CAP = int(os.environ["ARMF_EXTERN_CAP"])
# "0": launch the Triton decode stage-2 kernel without PDL (same kernel), standing in for patches/02_triton_pdl.diff
TRITON_PDL = os.environ.get("ARMF_TRITON_PDL", "1") == "1"
# ARMF_MEMORY -> HostTraceReplay memory. "run_buffer" (default): one buffer per run for temporaries, stable addresses;
# "eager" repatches moved nodes per call (stage 4). "planned" is the per-run arena (user 2026-10-04; "arena" is an alias)
# and "held" is C4's buffer held across calls, which overlays before the src_sglplan merge call "planned".
# planned, packed, planned_no_reuse, planned_no_split: src_sglplan's modes, on a merged overlay only.
# arena_scratch: the arena with keyed sites' scratch as arena members ("planned_scratch").
_MERGED = hasattr(R, "_PLANNED")
MEMORY_ENV = os.environ.get("ARMF_MEMORY", "run_buffer")
MEMORY = {"held": "held" if _MERGED else "planned", "arena": "planned", "arena_scratch": "planned_scratch"}.get(MEMORY_ENV, MEMORY_ENV)
if MEMORY_ENV in ("planned", "arena", "packed", "planned_no_reuse", "planned_no_split", "arena_scratch") and (not _MERGED or MEMORY not in ("held", *R._PLANNED)):
    raise ValueError(f"ARMF_MEMORY={MEMORY_ENV} needs an overlay with src_sglplan merged (SGLCPP_SRC=src_sglcpp)")
print(f"adapter_cpp: ARMF_MEMORY={MEMORY_ENV} runs HostTraceReplay memory={MEMORY!r}", flush=True)
# C++ entry switches (default "1", on as the build has them): "0" takes the path before C2's variant order, C6's
# run_buffer placement reuse or C9's delta evaluate
# native switches (name, env, default); form_held is off by default (a form switch forgets the exec's held params)
_SWITCHES = (("variant_order", "ARMF_VARIANT_ORDER", "1"), ("placement_reuse", "ARMF_PLACEMENT_REUSE", "1"), ("delta", "ARMF_DELTA", "1"), ("entry_descriptors", "ARMF_ENTRY_DESCRIPTORS", "1"), ("form_held", "ARMF_FORM_HELD", "0"))
SWITCHES = {name: os.environ.get(env, default) == "1" for name, env, default in _SWITCHES}
for _name, _env, _default in _SWITCHES:
    if hasattr(torch._C, f"_host_trace_{_name}"):
        getattr(torch._C, f"_host_trace_{_name}")(SWITCHES[_name])
    elif SWITCHES[_name] != (_default == "1"):
        raise ValueError(f"ARMF switch {_env}={int(SWITCHES[_name])} needs a build with torch._C._host_trace_{_name}")
# "trace": plain decode and extend attention metadata in buffers kept by ArmF, filled inside the traced
# step (ArmF._metadata, ArmF._fill); "eager": the backend's own init_forward_metadata per call
# integration sglang_stock (INTEG_STOCK=1, default here): stock SGLang + the hook. The region starts at model.forward;
# SGLang's prep and init_forward_metadata run eagerly as shipped (their tensors are arguments, max_seq_len_q a symbolic
# int), no staging copies, SGLang's own input copy (no SGLANG_EAGER_INPUT_NO_COPY), SGLang's own LogitsProcessor, and
# attention through FlashInfer's own API with the trtllm-gen fork on the path (INTEG_FORK=1: no trtllm_shim ops).
STOCK = os.environ.get("INTEG_STOCK", "1") == "1"
FORK = os.environ.get("INTEG_FORK", "1") == "1"
# INTEG_ATTN: "fork" (default with INTEG_FORK=1: flashinfer.trtllm_trace, the attn lane's Python launcher port) or "cpp"
# (flashinfer.trtllm_cpp: FlashInfer's own trtllm-gen C++ launcher traced, land/core/trtllm_cpp; SGL_LINE=rd2cpp sets it)
ATTN = os.environ.get("INTEG_ATTN", "fork" if FORK else "shim")
FORK = ATTN in ("fork", "cpp")
METADATA = os.environ.get("ARMF_METADATA", "eager" if STOCK else "trace")
# copy positions and out_cache_loc into buffers kept by ArmF inside the traced step: every layer's rope and KV store
# then read a fixed address, so a replay patches 2 nodes for fresh batch tensors, not 2 per layer
STAGE = os.environ.get("ARMF_STAGE", "0" if STOCK else "1") == "1"
# "kernel" (default): the staging copies are an elementwise kernel (add.out of 0), not copy_'s cudaMemcpyAsync: a
# replay then sets a kernel node for a fresh batch tensor (about 0.5 us) where a memcpy node's setter takes about 5 us
STAGE_COPY = os.environ.get("ARMF_STAGE_COPY", "kernel")  # "memcpy": copy_
# set SGLang's SGLANG_EAGER_INPUT_NO_COPY unless it is set: EagerRunner.load_batch then hands the forward the batch's own
# tensors instead of copying them into its static buffers (0.3-0.5 ms per step); they are arguments of the trace either way
NO_COPY = os.environ.get("ARMF_NO_COPY", "0" if STOCK else "1") == "1"
STATIC_PREFIX = os.environ.get("ARMF_STATIC_PREFIX", "1") == "1"
# "1": after an entry's first call, its calls whose key is the same take one C++ call (torch._C._HostTraceBound)
BOUND = os.environ.get("ARMF_BOUND", "1") == "1" and hasattr(torch._C, "_HostTraceBound")
WHY = bool(os.environ.get("ARMF_BOUND_WHY"))
# "1": the bound is model.forward itself (the one-call entry): it reads forward_metadata off the backend, takes
# pp_proxy_tensors=None, resets self.fill after a hit, and hands any call no plan takes to ArmF._forward
ONECALL = BOUND and os.environ.get("ARMF_ONECALL", "1") == "1" and hasattr(torch._C._HostTraceBound, "statics_changed")
# "1": the bound's calls skip the check of the leading arguments (params, attrs, KV buffers) once they passed it: a
# change to one in place (set_, resize_, .data =) must call self.bound.statics_changed(). Their identity is fixed at
# __init__ either way (self.prefix)
TRUST_STATICS = ONECALL and os.environ.get("ARMF_TRUST_STATICS", "1") == "1"
WHY_SEEN = collections.Counter()

DECLINE_SITES = collections.Counter()
_orig_declined_init = ht.Declined.__init__


def _declined_init(exc, *args, **kwargs):
    _orig_declined_init(exc, *args, **kwargs)
    frames = [f for f in traceback.extract_stack()[:-1] if "/torch/" not in f.filename and not f.filename.endswith(("adapter.py", "adapter_cpp.py"))]
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
        if self.trtllm and FORK:
            if ATTN == "cpp":
                import flashinfer.trtllm_cpp  # noqa: F401  (the trtllm_cpp overlay's)
            else:
                import flashinfer.trtllm_trace  # noqa: F401  (the fork's; stock FlashInfer has none)
        elif self.trtllm and extern:
            import trtllm_shim  # noqa: F401

            extern += (torch.ops.sglang_ht.trtllm_decode.default, torch.ops.sglang_ht.trtllm_context.default)
        # store_cache writes KV pool rows at indices: verified on small caches
        indexed = {torch.ops.sglang.store_cache.default: ("indices", "k_cache", "v_cache")}
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed) if extern else HarvestProvider(FAMILIES)
        self.entries = {}  # key -> HostTraceReplay
        self.outs = {}  # key -> (output template, tensor field names)
        self.calls = collections.Counter()
        self.installed = False
        self._check_on = False  # compare each call against the model's own eager forward (check)
        self.checks = []
        layer = next(m for m in self.model.modules() if type(m).__name__ == "RadixAttention")
        self.layer_heads = (layer.tp_q_head_num, layer.tp_q_head_num // layer.tp_k_head_num, layer.qk_head_dim != layer.v_head_dim)
        self.orig_init_metadata = None
        self.md_bufs = None  # (batch capacity, kv_indices capacity, buffers): _metadata's
        self.md_cache = {}  # (kind, batch size) -> _metadata's metadata, views of md_bufs
        self.fill = False  # the metadata _metadata set last is _metadata's, to fill
        # call objects (forward_batch, forward_metadata), then the roots: the backend and this adapter (fill)
        if ONECALL:
            self.bound = torch._C._HostTraceBound(self.prefix, 2, 1, (self.backend, self), derived=((0, "forward_metadata"),),
                                                  none_kwargs=("pp_proxy_tensors",), resets=((3, "fill", False),),
                                                  fallback=self._forward, trust_statics=TRUST_STATICS)
        else:
            self.bound = torch._C._HostTraceBound(self.prefix, 2, 2, (self.backend, self)) if BOUND else None
        self.planned = set()  # the keys with a plan in self.bound
        self.plan_specs = []  # ARMF_BOUND_WHY: the plans' guards, to name the one a NotImplemented call fails
        self.lean_limits = None

    def install(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.installed = True
        self.model.forward = self.bound if ONECALL and not self.check else self.forward
        if NO_COPY:
            from sglang.srt.environ import envs

            if not envs.SGLANG_EAGER_INPUT_NO_COPY.is_set():
                envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)
        if self.trtllm and EXTERN and not FORK:
            import trtllm_shim

            trtllm_shim.install(self.backend)
        if not STOCK and not hasattr(LogitsProcessor, "_armf_get_pruned_states"):
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

    @property
    def check(self):
        return self._check_on

    @check.setter
    def check(self, on):
        # ONECALL: model.forward is the bound, which skips forward's check of a hit
        self._check_on = on
        if self.installed and ONECALL:
            self.model.forward = self.forward if on else self.bound

    def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None, **kwargs):
        # ModelRunner passes pp_proxy_tensors=None on every call when the model's forward takes it
        if ONECALL:
            hits = sum(self.bound.hits())
            out = self.bound(input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors, **kwargs)
            if self.check and sum(self.bound.hits()) > hits:
                kind = "decode" if forward_batch.forward_mode.is_decode() else "extend"
                self._check(kind, out, input_ids, positions, forward_batch, bound=True)
            return out
        if self.bound is not None and pp_proxy_tensors is None and not kwargs:
            out = self.bound(input_ids, positions, forward_batch, self.backend.forward_metadata)
            if out is NotImplemented and WHY and self.plan_specs:
                self._why(forward_batch, self.backend.forward_metadata)
            if out is not NotImplemented:
                self.fill = False
                if self.check:
                    kind = "decode" if forward_batch.forward_mode.is_decode() else "extend"
                    self._check(kind, out, input_ids, positions, forward_batch, bound=True)
                return out
        return self._forward(input_ids, positions, forward_batch, pp_proxy_tensors, **kwargs)

    def _forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None, **kwargs):
        if ONECALL and WHY and self.plan_specs:
            self._why(forward_batch, self.backend.forward_metadata)
        mode = forward_batch.forward_mode
        fill, self.fill = self.fill, False
        if pp_proxy_tensors is not None or any(v is not None for v in kwargs.values()):
            self.calls["eager_kwargs"] += 1
            if fill:
                self._fill("decode" if mode.is_decode() else "extend", forward_batch, self.backend.forward_metadata)
            return self.orig_forward(input_ids, positions, forward_batch, pp_proxy_tensors=pp_proxy_tensors, **kwargs)
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
        key = (kind, mode, fb_t, md_t, len(ints), host, fill, stage)
        entry = self.entries.get(key)
        if entry is None:
            kw = {"static_prefix": len(self.prefix)} if STATIC_PREFIX else {}
            entry = self.entries[key] = R.HostTraceReplay(self._step(key, fb_fields, md_fields), opaque=(self.provider,), memory=MEMORY, **kw)
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
        if self.bound is not None and key not in self.planned:
            self.planned.add(key)
            self._plan(key, entry, forward_batch, md, fb_fields, md_fields)
        if self.check:
            self._check(kind, out, input_ids, positions, forward_batch)
        return out

    def _lean_limits(self):
        """Per batch size b < 4097, the least seq_lens_sum the Lean gate turns on at (2**62: none), so that the gate
        is seq_lens_sum >= limits[b]; checked against the gate on either side of each limit."""
        if self.lean_limits is None:
            h, g, mla = self.layer_heads
            gate = self.backend._lean_decode_seqlen_gate
            limits = []
            for b in range(4097):
                lo, hi = 0, 1 << 62
                if gate(h, g, b, hi, mla):
                    while lo < hi:
                        mid = (lo + hi) // 2
                        lo, hi = (lo, mid) if gate(h, g, b, mid, mla) else (mid + 1, hi)
                    assert lo == 0 or not gate(h, g, b, lo - 1, mla)
                    assert not gate(h, g, b, lo // 2, mla) and gate(h, g, b, 2 * lo + 1, mla), "the Lean gate is not a threshold"
                limits.append(hi)
            self.lean_limits = tuple(limits)
        return self.lean_limits

    def _plan(self, key, entry, fb, md, fb_fields, md_fields):
        """The guards under which a call's key is `key`: as forward computes it, from the same fields."""
        from sglang.srt.environ import envs

        kind, mode, fb_t, md_t, n_ints, _, fill, _ = key
        FB, MD, BACKEND, ARMF = 0, 1, 2, 3
        fbd = vars(fb)
        same = [(FB, "forward_mode", mode), (ARMF, "fill", fill), *((FB, n, fbd.get(n)) for n in _PATH_FIELDS)]
        below = []
        b = self.backend
        if kind == "decode" and hasattr(b, "_lean_decode_seqlen_gate"):
            if envs.SGLANG_DISABLE_LEAN_ATTENTION.get():  # read once: the environment is the process's
                same.append((BACKEND, "enable_lean_attention", b.enable_lean_attention))
            else:
                same += [(BACKEND, "enable_deterministic", b.enable_deterministic), (BACKEND, "enable_lean_attention", b.enable_lean_attention)]
                if not b.enable_deterministic and b.enable_lean_attention is None:
                    below.append((FB, "seq_lens_sum", FB, "batch_size", self._lean_limits()))
        other = [(FB, n) for n in fb_fields if n not in fb_t and n not in ("input_ids", "positions")]
        other += [(MD, n) for n in md_fields if n not in md_t]
        tensors = [(FB, n) for n in fb_t] + [(MD, n) for n in md_t]
        ints = [(MD, _int_field(md))] if n_ints else []
        template, names = self.outs[key]
        self.bound.add(entry, (type(fb), type(md)), tuple(same), tuple(below), tuple(other), tuple(tensors), tuple(ints), template, tuple(names))
        if WHY:
            self.plan_specs.append(((type(fb), type(md)), same, below, other, tensors, ints))

    def _why(self, fb, md):
        """The first guard of each plan a call fails, as the C++ checks them (diagnostics)."""
        objs = (fb, md, self.backend, self)
        ds = [vars(o) for o in objs]
        reasons = []
        for types, same, below, other, tensors, ints in self.plan_specs:
            r = None
            if (type(fb), type(md)) != types:
                r = ("types", type(fb).__name__, type(md).__name__)
            for s_, n, want in same:
                if r is None and ds[s_].get(n) is not want:
                    r = ("same", s_, n, repr(ds[s_].get(n))[:60], repr(want)[:60])
            for (vs, vn, is_, in_, limits) in below:
                v, i = ds[vs].get(vn), ds[is_].get(in_)
                if r is None and not (type(v) is int and type(i) is int and 0 <= i < len(limits) and v < limits[i]):
                    r = ("below", vn, type(v).__name__, v, type(i).__name__, i)
            for s_, n in other:
                if r is None and _cuda_tensor(ds[s_].get(n)):
                    r = ("other", s_, n)
            for s_, n in tensors:
                if r is None and not _cuda_tensor(ds[s_].get(n)):
                    r = ("tensor", s_, n, type(ds[s_].get(n)).__name__)
            for s_, n in ints:
                if r is None and type(ds[s_].get(n)) is not int:
                    r = ("int", s_, n, type(ds[s_].get(n)).__name__)
            reasons.append(r or "guards pass: the entry call missed")
        WHY_SEEN[str(reasons)] += 1
        if WHY_SEEN[str(reasons)] <= 2:
            print("ARMF_BOUND_WHY", reasons, flush=True)

    def _check(self, kind, out, input_ids, positions, forward_batch, bound=False):
        got = out.next_token_logits.clone()
        torch.cuda.synchronize()
        if self.orig_init_metadata is not None:  # the reference runs on the backend's own metadata
            self.orig_init_metadata(forward_batch)
        want = self.orig_forward(input_ids, positions, forward_batch).next_token_logits
        torch.cuda.synchronize()
        same = got.shape == want.shape
        self.checks.append({"kind": kind, "bound": bound, "shape": list(want.shape), "bitwise": bool(same and torch.equal(got, want)),
                            "max_abs": float((got.float() - want.float()).abs().max()) if same else None,
                            "max_ref": float(want.float().abs().max()),
                            "argmax_equal": bool(same and torch.equal(got.argmax(-1), want.argmax(-1)))})

    def _step(self, key, fb_fields, md_fields):
        kind, _, fb_t, md_t, n_ints, _, fill, stage = key
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
                loc = fbt["out_cache_loc"]
                if STAGE_COPY == "kernel":
                    positions = torch.add(positions, 0, out=st[0][:positions.shape[0]])
                    fbt["out_cache_loc"] = torch.add(loc, 0, out=st[1][:loc.shape[0]])
                else:
                    positions = st[0][:positions.shape[0]].copy_(positions)
                    fbt["out_cache_loc"] = st[1][:loc.shape[0]].copy_(loc)
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
        out["bound"] = sum(self.bound.hits()) if self.bound is not None else 0
        out["calls"] = dict(self.calls)
        return out

    def summary(self):
        import torch.cuda._host_trace_tape as T
        from torch.cuda._host_trace_launch import KernelLaunch

        res = {"memory": {"env": MEMORY_ENV, "mode": MEMORY}, "switches": SWITCHES, "families": list(FAMILIES), "extern": EXTERN, "args": {"params": len(self.params), "attrs": [f"{type(m).__name__}.{k}" for m, k, _ in self.attrs], "kv_layers": len(self.kbuf)}}
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
            res["entries"][f"{key[0]}/{key[1].name}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants),
                "eager_ops": dict(ops.most_common(30)), "host_ops": dict(host_ops.most_common(30)),
                "eager_reasons": [f"{n}x {op}: {why}" for (op, why), n in whys.most_common(20)],
                "per_variant": per_variant, "declines": sorted({str(d)[:400] for d in e.declines})[:30],
                "reasons": list(getattr(e, "_reasons", {}))[:20], "triton_fallbacks": list(e.triton_fallbacks)[:20],
                "retrace_causes": {str(c): n for c, n in getattr(e, "retrace_causes", {}).items()},
                "redispatches": getattr(e, "redispatches", None), "respecs": getattr(e, "respecs", None), "learned": getattr(e, "learned", None)}
        res["decline_sites"] = [f"{n}x {m} @ {w}" for (m, w), n in DECLINE_SITES.most_common(20)]
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["baked_host_reads"] = dict(_ReadLog.reads.most_common(40))
        res["checks"] = self.checks
        return res
