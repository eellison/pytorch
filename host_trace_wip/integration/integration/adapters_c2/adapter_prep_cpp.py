# Arm F: SGLang's model forward under host-trace replay on the eager line (no torch.compile).
# adapter_prep_cpp.py: adapter_prep.py for install_sgl_cpp (python_sglprepcpp.sh): PrepF's entries take
# static_prefix, and their replays go through a torch._C._HostTraceBound (PREP_BOUND), one entry for all bs.
# PREP_TGV=1 (arm M, python_sglm.sh): install() also points SGLang's CuTe DSL bf16 GEMM (TGV) at its run functions
# instead of its Library-registered custom op (armF/tgv/adapter.py), so the CuTe route records its launches.
# ARMF_BS_KEY (default 1): 0 = one decode entry for bs > 1 (armF/bs); arm M uses 0.
# PREP_BS_META (default 0, armF/bs/adapter's ARMF_BS_META): the trtllm metadata kernel and _prep_decode_kernel launch
# with a fixed block (req_to_token's rows) and an unspecialized bs, so no guard pins the batch size; TGV's choice of
# kernel is tested on runs of bs (_by_runs).
# PREP_M=1: arm M, the defaults of PREP_TGV, PREP_BS_META become 1, ARMF_BS_KEY 0, ARMF_STAGE_COPY kernel.
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
import inspect
import os
import time
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
M = os.environ.get("PREP_M") == "1"
TGV = os.environ.get("PREP_TGV", "1" if M else "0") == "1"
BS_META = os.environ.get("PREP_BS_META", "1" if M else "0") == "1"
_META_KERNEL, _PREP_KERNEL = [], []


def _by_runs(choose, cap=1 << 16):
    """A bool choose(m, n, k) for a SymInt m: at each static (n, k), the runs of m in [0, cap] where it is True,
    tested in order with plain comparisons (no And/Or guard nodes), so the guards hold wherever the choice is the
    same; beyond cap, choose itself."""
    runs = {}

    def run(m, n, k):
        if not isinstance(m, torch.SymInt) or not m <= cap:
            return choose(m, n, k)
        r = runs.get((n, k))
        if r is None:
            r = runs[(n, k)] = []
            for i in range(cap + 1):
                if choose(i, n, k):
                    if r and r[-1][1] == i - 1:
                        r[-1][1] = i
                    else:
                        r.append([i, i])
        for a, e in r:
            if m < a:
                return False
            if m <= e:
                return True
        return False
    return run

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
        # req_to_token has a row per request, so its row count bounds every batch size (the fixed cumsum width)
        self.meta_block = triton.next_power_of_2(self.backend.req_to_token.shape[0]) if self.trtllm else None
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
        # with src_sglm_abc (prefill agent's A): extern keys at a new size learn without a model run, on views of the KV pool and workspace
        lend = {"lendable": (*self.kbuf, *self.vbuf, *(v for _, k, v in self.attrs if k == "workspace_buffer"))} if "lendable" in inspect.signature(HarvestProvider).parameters else {}
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed, **lend) if extern else HarvestProvider(FAMILIES)
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
        self._grab = None

    def install(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.model.forward = self.forward
        if TGV:
            from sglang.srt.layers.quantization import unquant
            import sglang.kernels.ops.gemm.cutedsl_bf16_gemm as tgv

            def gemm(x, w, bias=None):
                return tgv._tgv_bf16_gemm_run(x, w, bias)

            def gemm_out(x, w, out, bias=None):
                tgv._tgv_bf16_gemm_out_run(x, w, out, bias)
                return out

            # unquant imports cutedsl_bf16_gemm_out from the module at each call
            tgv.cutedsl_bf16_gemm, tgv.cutedsl_bf16_gemm_out = gemm, gemm_out
            # served, the module (and its from_dlpack) is imported with the model, ahead of the CuTe route's hooks
            from cutlass.cute import runtime as cute_runtime
            from torch.cuda import _host_trace_cute

            _host_trace_cute.install()
            tgv.from_dlpack = cute_runtime.from_dlpack
            if unquant._cutedsl_bf16_gemm is not None:
                unquant._cutedsl_bf16_gemm = gemm
            if BS_META:
                # SGLang's GEMM choices test m before the static n, k (use_bf16_splitk_gemm hashes m into its table:
                # Eq(bs, value); use_cutedsl_bf16_gemm's m <= 8 and m <= 72 gate (n, k) sets): guards where the choice
                # does not change. Same choices, one guard: m in the union of the runs where the choice is True.
                unquant.use_bf16_splitk_gemm = _by_runs(unquant.use_bf16_splitk_gemm)
                if unquant._use_cutedsl_bf16_gemm is not None:
                    unquant._use_cutedsl_bf16_gemm = _by_runs(unquant._use_cutedsl_bf16_gemm)
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
            if BS_META:
                _trtllm_metadata(fb.req_pool_indices, fb.seq_lens, b.req_to_token, md.cache_seqlens_int32, md.cu_seqlens_k, md.page_table,
                                 b.max_num_pages, b.page_size, self.meta_block, md.cu_seqlens_q if ext else None, fb.extend_seq_lens if ext else None)
                return
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
        if self._grab is not None:  # PrepF's template capture
            self._grab.append((forward_batch, md, fill, fb_fields, md_fields, fb_t, md_t))
        ints = (getattr(md, _int_field(md)),) if kind == "extend" else ()
        # host fields that pick code paths are baked into the trace, so each value gets its own entry
        host = tuple(repr(fbd.get(n)) for n in _PATH_FIELDS)
        loc = fbd.get("out_cache_loc")
        stage = bool(self.stage) and _cuda_tensor(loc) and max(positions.shape[0], loc.shape[0]) <= self.stage[0].shape[0] and positions.dtype == loc.dtype == torch.int64
        # decode: one entry per batch size, so the native path evaluates one family of variants, not every batch size's
        n = positions.shape[0]
        bs = None if kind != "decode" or not BS_KEY else n if BS_KEY == 1 else n == 1
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

    def _body(self, kind, fill, stage, n_ints, params, attrs, kb, vb, st, input_ids, positions, fbt, fb, fb_fields, mdt, md, md_fields, ints):
        """The traced step's model call on traced tensors: the shells, the swapped backend state, _fill, the forward."""
        proxy = self
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
        return out, shell_fb

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
            out, _ = proxy._body(kind, fill, stage, n_ints, params, attrs, kb, vb, st, input_ids, positions, fbt, fb, fb_fields, mdt, md, md_fields, ints)
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
        return res


# Decode prep inside the trace (armF/prep). PrepF replaces one_batch.decode: for a plain trtllm decode batch the
# device work of prepare_for_decode and ForwardBatch.init_new (input_ids cast, last_loc gather, alloc_decode_kernel,
# req_to_token write, seq_lens / orig_seq_lens + 1, positions), the model and the greedy sampler are one traced entry.
# What stays in Python per step is host bookkeeping with no CUDA launch: the allocator's free-page slicing (the page
# count comes from seq_lens_cpu), request counters, seq_lens_cpu / seq_lens_sum. The ForwardBatch the model sees is
# the one the first step built through SGLang's own path (template), with the step's tensors swapped in.
# "0": current F (ArmF only)
PREP = os.environ.get("ARMF_PREP", "1") == "1"
# "0": the sampler runs after the entry, eagerly, on a ForwardBatch copied from the template
PREP_SAMPLE = os.environ.get("ARMF_PREP_SAMPLE", "1") == "1"
# "1": one bound per prep key (prefix held in C++, trust_statics), the HostTraceReplay call only on a miss
PREP_BOUND = os.environ.get("PREP_BOUND", "1") == "1"
# "1": the batch size in the prep key (one entry per bs, as adapter_prep.py)
PREP_BS_KEY = os.environ.get("PREP_BS_KEY", "0") == "1"
# decode entry key: 1 = one entry per bs (change 13), 0 = bs 1 and one family for bs > 1 (armF/bs ARMF_BS_KEY)
BS_KEY = int(os.environ.get("ARMF_BS_KEY", "0" if M else "1"))
# armF/adapter_cpp.py: "kernel" stages positions / out_cache_loc with add.out (a replay sets kernel nodes, not memcpy nodes)
STAGE_COPY = os.environ.get("ARMF_STAGE_COPY", "kernel" if M else "memcpy")
BRK = [] if os.environ.get("PREP_BREAK") else None  # per prep step (us): host fields, seq_lens_cpu+1, tolist, evict check, page count+free_pages view, req counters+key, replay call, outputs
_PREP_FB = frozenset(("seq_lens", "orig_seq_lens", "out_cache_loc", "req_pool_indices"))


@triton.jit
def _prep_decode_kernel(tok, ids, seq, new_seq, orig, new_orig, pos, rpi, r2t, r2t_stride, free, loc,
                        BLOCK: tl.constexpr, PAGE: tl.constexpr, PAGED: tl.constexpr):
    """Row pid of prepare_for_decode + init_new: input_ids, seq_lens + 1, orig_seq_lens + 1, positions
    (clamp(seq_lens + 1 - 1, 0)) and the KV slot, written to req_to_token[rpi, seq_lens]. Paged: alloc_decode_kernel's
    arithmetic on seq_lens + 1, last_loc = req_to_token[rpi, seq_lens - 1]; page 1: the allocator's slice."""
    pid = tl.program_id(0)
    tl.store(ids + pid, tl.load(tok + pid))
    s = tl.load(seq + pid)
    tl.store(new_seq + pid, s + 1)
    tl.store(new_orig + pid, tl.load(orig + pid) + 1)
    tl.store(pos + pid, tl.maximum(s, 0))
    row = r2t + tl.load(rpi + pid).to(tl.int64) * r2t_stride
    if PAGED:
        i = tl.arange(0, BLOCK)
        m = i <= pid
        sj = tl.load(seq + i, mask=m, other=0)
        new_pages = tl.where(m, (sj + PAGE) // PAGE - (sj + PAGE - 1) // PAGE, 0)
        own = (s + PAGE) // PAGE - (s + PAGE - 1) // PAGE
        if own == 0:
            slot = tl.load(row + s - 1).to(tl.int64) + 1
        else:
            slot = tl.load(free + tl.sum(new_pages) - own).to(tl.int64) * PAGE
        tl.store(loc + pid, slot)
    else:
        slot = tl.load(free + pid)
    tl.store(row + s, slot.to(tl.int32))


def _prep_kernel():
    """_prep_decode_kernel for PREP_BS_META: the batch's tensors are views at moving offsets (free slots), so no 16-byte
    alignment guards."""
    if not _PREP_KERNEL:
        _PREP_KERNEL.append(triton.jit(do_not_specialize=["tok", "seq", "orig", "rpi", "free", "loc"])(_prep_decode_kernel.fn))
    return _PREP_KERNEL[0]


def _trtllm_metadata(req_pool_indices, seq_lens, req_to_token, cache_seqlens, cu_seqlens_k, page_table, max_num_pages, page_size,
                     block, cu_seqlens_q=None, qlens=None):
    """update_trtllm_mha_graph_metadata (no SWA, seqlen_offset 0) for PREP_BS_META: the wrapper's
    BS_BLOCK=next_power_of_2(bs) guards bs (its bit ops) and Triton specializes the bs argument (== 1, % 16); the block
    is only the cumsum's width, masked by bs, so any block >= bs gives the same values."""
    from sglang.kernels.ops.kvcache.trtllm_mha_graph_metadata import Q_MODE_CUMSUM, Q_MODE_NONE, update_trtllm_mha_graph_metadata_kernel as kernel

    if not _META_KERNEL:
        _META_KERNEL.append(triton.jit(do_not_specialize=["bs"])(kernel.fn))
    bs = seq_lens.shape[0]
    _META_KERNEL[0][(bs + 1,)](req_pool_indices, seq_lens, req_to_token, None, None, qlens, cache_seqlens, cu_seqlens_k, cu_seqlens_q,
                               page_table, None, None, bs, 0, max_num_pages, 0, 0, 0, req_to_token.stride(0), page_table.stride(0), 0,
                               PAGE_SIZE=page_size, HAS_SWA=False, HAS_SWA_OUT=False, Q_MODE=Q_MODE_NONE if qlens is None else Q_MODE_CUMSUM,
                               PAGE_BLOCK=512, BS_BLOCK=block, SKIP_PAGE_TABLE=0)


class PrepF(ArmF):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        # (batch, reqs, len, sampling_info ids) -> [template tuple, "grab" or None (not eligible), batch, reqs, sampling_info]:
        # the entry holds what its key's ids name, so no id in a key is reused while the key is here
        self.prep = {}
        self.orig_decode = None
        self.r2t_i = None  # index in attrs of req_to_token
        self._pctx = None
        self._free_view = (None, None)  # (allocator free_pages object, its [:bs] view)
        self.bounds = {}  # prep key -> its _HostTraceBound (PREP_BOUND)

    def install(self):
        global _evict_from_tree_cache, _alloc_token_slots
        super().install()
        if not (PREP and self.trtllm and self.md_trace):
            return
        import sglang.benchmark.one_batch as ob
        from sglang.srt.mem_cache.allocation import alloc_token_slots as _alloc_token_slots
        from sglang.srt.mem_cache.common import evict_from_tree_cache as _evict_from_tree_cache

        r2t = self.mr.req_to_token_pool.req_to_token
        idx = [i for i, (_, k, v) in enumerate(self.attrs) if k == "req_to_token" and v.data_ptr() == r2t.data_ptr() and v.shape == r2t.shape]
        if not idx:
            return
        self.r2t_i = idx[-1]
        self.orig_decode = ob.decode
        ob.decode = self.decode
        self._install_serving()

    def _install_serving(self):
        """SGLang's serving loop (Scheduler, TpModelWorker): update_running_batch's prepare_for_decode is deferred
        (its forward_mode set) to the worker's forward_batch_generation of that batch, which runs it with the step as
        decode does, at the scheduler's sampling_info (_forward_isolation swaps in a copy for the forward). Under the
        overlap scheduler the deferred prepare runs on the forward stream, so the schedule stream then waits for the
        step: its later reads of seq_lens, req_to_token and the free pages follow the step's writes."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers import tp_worker
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        import sglang.benchmark.one_batch as ob

        W = tp_worker.TpModelWorker
        if getattr(W.forward_batch_generation, "_prepf", None) is not None:
            return
        orig_update, orig_prepare, orig_fbg = Scheduler.update_running_batch, ScheduleBatch.prepare_for_decode, W.forward_batch_generation
        updating, pending = [None], {}  # the scheduler in update_running_batch; id(batch) -> (batch, sampling_info, scheduler)

        def update_running_batch(sched, batch):
            updating[0] = sched
            try:
                return orig_update(sched, batch)
            finally:
                updating[0] = None

        def prepare_for_decode(batch):
            sched, updating[0] = updating[0], None
            if sched is None or not batch.spec_algorithm.is_none() or ob.require_mlp_sync():
                return orig_prepare(batch)
            batch.forward_mode = ForwardMode.DECODE
            pending[id(batch)] = (batch, batch.sampling_info, sched)

        def forward_batch_generation(worker, batch=None, *args, **kwargs):
            p = pending.pop(id(batch), None)
            if p is None:
                return orig_fbg(worker, batch, *args, **kwargs)
            _, si, sched = p
            ran = []

            def run(*_):
                fsi, batch.sampling_info = batch.sampling_info, si
                try:
                    orig_prepare(batch)
                finally:
                    batch.sampling_info = fsi
                ran.append(orig_fbg(worker, batch, *args, **kwargs))
                return ran[0].next_token_ids, None

            # the non-overlap loop passes pp_proxy_tensors=None
            if (args or any(v is not None for v in kwargs.values()) or worker.model_runner is not self.mr or not worker.pp_group.is_last_rank or worker.is_dllm()
                    or tp_worker.get_exec().features.enable_encoder_swa_bounded_replay):
                run()
            else:
                worker.set_hicache_consumer(batch.hicache_consumer_index)
                next_ids, logits = self.decode(batch.input_ids, batch, self.mr, orig=run, si=si)
                if not ran:
                    self.calls["serve_prep"] += 1
                    worker._maybe_finalize_elastic_cuda_graph_scale()
                    ran.append(GenerationBatchResult(logits_output=LogitsProcessorOutput(next_token_logits=logits), next_token_ids=next_ids))
            if sched.enable_overlap:
                done = torch.cuda.Event()
                done.record()
                sched.schedule_stream.wait_event(done)
            return ran[0]

        forward_batch_generation._prepf = self
        Scheduler.update_running_batch, ScheduleBatch.prepare_for_decode, W.forward_batch_generation = update_running_batch, prepare_for_decode, forward_batch_generation

    def _eligible(self, batch, mr, si):
        from sglang.srt.managers import schedule_batch as sbm
        from sglang.srt.mem_cache.allocation import _alloc_page_size
        import sglang.benchmark.one_batch as ob

        tc = batch.tree_cache
        alloc, page = tc.token_to_kv_pool_allocator, _alloc_page_size(batch)
        paged_ok = type(alloc).__name__ == "PagedTokenToKVPoolAllocator" and not alloc.debug_mode and not hasattr(alloc, "c128_attn_allocator")
        checks = {
            "spec": batch.spec_algorithm.is_none(),
            "beam": getattr(batch, "beam_tail", None) is None and all(getattr(r, "beam_group", None) is None for r in batch.reqs),
            "penalizer": si is not None and not si.penalizer_orchestrator.is_required,
            "encdec": not batch.model_config.is_encoder_decoder,
            "hisparse": batch.hisparse_coordinator is None and getattr(mr, "hisparse_coordinator", None) is None,
            "mamba": not sbm.get_exec().mamba.enable_mamba_extra_buffer,
            "mlp_sync": not ob.require_mlp_sync(),
            "logprob": not batch.return_logprob,
            "hidden": not batch.return_hidden_states,
            "grammar": not batch.has_grammar,
            "lora": mr.lora_manager is None,
            "allocator": paged_ok and alloc.page_size == page if page > 1 else type(alloc).__name__ == "TokenToKVPoolAllocator",
            "runner": not any(getattr(mr, n, None) for n in ("msprobe_debugger", "canary_manager", "eplb_manager", "enable_elastic_ep")),
            "swa": not tc.supports_swa(),
            "translator": not mr.kv_index_translator.is_translating,
            "mrope": not mr.model_config.model_is_mrope,
            "ngram": not mr.ngram_embedding_manager.enabled,
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            self.calls["prep_ineligible:" + ",".join(bad)] += 1
        return not bad

    def _make_template(self, grabbed, batch, mr):
        """The template from the step SGLang ran itself: its ForwardBatch at model.forward, trace-mode metadata."""
        from sglang.srt.mem_cache.allocation import _alloc_page_size

        if len(grabbed) != 1:
            return None
        fb, md, fill, fb_fields, md_fields, fb_t, md_t = grabbed[0]
        bs = len(batch.reqs)
        why = [n for n, ok in (("fill", fill), ("md", self.md_cache.get(("decode", bs)) is md),
                                ("fields", set(fb_t) == _PREP_FB), ("mlp", fb.global_num_tokens_cpu is None),
                                ("spec", fb.spec_info is None), ("ngram", getattr(fb, "ngram_embedding_info", None) is None)) if not ok]
        if why:
            self.calls["prep_template_refused:" + ",".join(why)] += 1
            return None
        si = fb.sampling_info
        sample = (PREP_SAMPLE and si.is_all_greedy and getattr(mr, "sampling_observer", None) is None and not si.has_custom_logit_processor
                  and si.logit_bias is None and si.acc_additive_penalties is None and si.acc_scaling_penalties is None
                  and getattr(si, "sampling_mask_batch_indices", None) is None and not si.grammars)
        page = _alloc_page_size(batch)
        host = tuple(repr(vars(fb).get(n)) for n in _PATH_FIELDS)
        return (fb, md, fb_fields, md_fields, md_t, host, bool(sample), page > 1, page)

    def decode(self, tok, batch, mr, orig=None, si=None):
        """one_batch.decode(tok, batch, mr); orig runs the step SGLang's own way (default one_batch.decode), si is the
        batch's sampling_info outside the forward (default batch.sampling_info)."""
        orig = orig or self.orig_decode
        si = batch.sampling_info if si is None else si
        k = (id(batch), id(batch.reqs), len(batch.reqs), id(si))
        e = self.prep.get(k)
        if e is None:
            self.prep = {o: v for o, v in list(self.prep.items())[-7:] if v[1] is not batch}
            e = self.prep[k] = ["grab" if self._eligible(batch, mr, si) else None, batch, batch.reqs, si]
        t = e[0]
        bs = len(batch.reqs)
        if isinstance(t, tuple) and self.md_cache.get(("decode", bs)) is not t[1]:  # metadata buffers regrown
            t = e[0] = "grab"
        if t is None:
            return orig(tok, batch, mr)
        if t == "grab":
            self._grab = []
            try:
                out = orig(tok, batch, mr)
            finally:
                grabbed, self._grab = self._grab, None
            e[0] = self._make_template(grabbed, batch, mr)
            return out
        fb, md, fb_fields, md_fields, md_t, host, sample, paged, page = t
        T = [time.perf_counter()] if BRK is not None else None
        # prepare_for_decode's host part (the plain case _eligible admits)
        batch.forward_mode = fb.forward_mode
        batch.mamba_track_seqlens_cpu = batch.mamba_prefill_track_mask_cpu = batch.input_embeds = None
        batch.mamba_cow_src_indices = batch.mamba_cow_dst_indices = batch.mamba_clear_indices = None
        if getattr(batch, "attn_cp_metadata", None) is not None:
            batch.attn_cp_metadata = None
        tc = batch.tree_cache
        if T: T.append(time.perf_counter())
        nxt = batch.seq_lens_cpu + 1
        if T: T.append(time.perf_counter())
        nl = nxt.tolist()
        if T: T.append(time.perf_counter())
        if paged:
            alloc = tc.token_to_kv_pool_allocator
            # evict_from_tree_cache's own early-out, without its per-call imports
            if tc is not None and not tc.is_chunk_cache() and bs * page > alloc.available_size():
                _evict_from_tree_cache(tc, bs * page)
            if T: T.append(time.perf_counter())
            free = alloc.free_pages
            if bs > len(free):
                alloc.merge_and_sort_free()
                free = alloc.free_pages
            new_pages = sum(s % page == 1 for s in nl)
            if new_pages > len(free):
                raise RuntimeError(f"Decode out of memory. Try to lower your batch size. Try to allocate {bs} tokens.")
            # free_pages is only ever reassigned, never written in place: a view of the same object stays valid
            if self._free_view[0] is not free or self._free_view[1].shape[0] != bs:
                self._free_view = (free, free[:bs])
            view = self._free_view[1]
            if new_pages:
                alloc.free_pages = free[new_pages:]
        else:
            view = _alloc_token_slots(tc, bs)
            if T: T.append(time.perf_counter())
        if T: T.append(time.perf_counter())
        batch.req_to_token_pool.alloc_aux_to_lengths(req_pool_indices_cpu=batch.req_pool_indices_cpu, target_seq_lens_cpu=nxt)
        for req in batch.reqs:
            req.decode_batch_idx += 1
            kv = req.kv
            kv.kv_allocated_len += 1
            kv.kv_committed_len += 1
        batch.seq_lens_cpu = nxt
        batch.seq_lens_sum = sum(nl)
        mr.forward_pass_id += 1
        key = ("prep", fb.forward_mode, md_t, host, paged, sample, bool(self.stage) and bs <= self.stage[0].shape[0]) + ((bs,) if PREP_BS_KEY else ())
        self.calls["prep"] += 1
        bound = self.bounds.get(key)
        if T: T.append(time.perf_counter())
        flat = bound(tok, view, batch, md) if bound is not None else NotImplemented
        if flat is NotImplemented:  # not a replay hit: nothing ran
            self.calls["prep_miss"] += 1
            entry = self.entries.get(key)
            if entry is None:
                entry = self.entries[key] = R.HostTraceReplay(self._prep_step(key, md_t, paged, page, sample, key[6]), opaque=(self.provider,), memory=MEMORY, static_prefix=len(self.prefix),
                                                                 static_shapes=range(len(self.prefix)) if STATIC else ())
            mdd = vars(md)
            self._pctx = t
            try:
                flat = entry(*self.prefix, tok, view, batch.seq_lens, batch.orig_seq_lens, batch.req_pool_indices, *[mdd[n] for n in md_t])
            finally:
                self._pctx = None
            if PREP_BOUND and bound is None:
                bound = self.bounds[key] = torch._C._HostTraceBound(self.prefix, 2, 2, (), trust_statics=True)
                tensors = ((0, "seq_lens"), (0, "orig_seq_lens"), (0, "req_pool_indices")) + tuple((1, n) for n in md_t)
                bound.add(entry, (type(batch), type(md)), (), (), (), tensors, (), None, ())
        if T: T.append(time.perf_counter())
        template, names = self.outs[key]
        n = len(names)
        batch.input_ids, batch.seq_lens, batch.orig_seq_lens = flat[n], flat[n + 1], flat[n + 2]
        batch.out_cache_loc = flat[n + 4] if paged else view
        if sample:
            out = flat[names.index("next_token_logits")]
            next_ids = flat[-1]
        else:
            out = object.__new__(type(template))
            vars(out).update(vars(template))
            vars(out).update(zip(names, flat))
            f = object.__new__(type(fb))
            vars(f).update(vars(fb))
            vars(f).update(input_ids=flat[n], positions=flat[n + 3], seq_lens=flat[n + 1], orig_seq_lens=flat[n + 2],
                           out_cache_loc=batch.out_cache_loc, seq_lens_cpu=nxt, seq_lens_sum=batch.seq_lens_sum, sampling_info=batch.sampling_info)
            next_ids = mr.sample(out, f)
            out = out.next_token_logits
        if T:
            T.append(time.perf_counter())
            BRK.append([round((b - a) * 1e6, 1) for a, b in zip(T, T[1:])])
        return next_ids, out

    def _prep_step(self, key, md_t, paged, page, sample, stage):
        n_p, n_a, n_l, n_s = len(self.params), len(self.attrs), len(self.kbuf), len(self.stage)
        proxy, r2t_i = self, self.r2t_i

        def step(*flat):
            fb, md, fb_fields, md_fields = proxy._pctx[:4]
            i = 0
            params = flat[i:i + n_p]; i += n_p
            attrs = flat[i:i + n_a]; i += n_a
            kb = list(flat[i:i + n_l]); i += n_l
            vb = list(flat[i:i + n_l]); i += n_l
            st = flat[i:i + n_s]; i += n_s
            tok, free, seq, orig, rpi = flat[i:i + 5]; i += 5
            mdt = dict(zip(md_t, flat[i:]))
            n = seq.shape[0]
            ids = torch.empty(n, dtype=torch.int64, device=seq.device)
            new_seq, new_orig, pos = torch.empty_like(seq), torch.empty_like(orig), torch.empty_like(seq)
            loc = torch.empty(n, dtype=torch.int64, device=seq.device) if paged else free
            r2t = attrs[r2t_i]
            (_prep_kernel() if BS_META else _prep_decode_kernel)[(n,)](tok, ids, seq, new_seq, orig, new_orig, pos, rpi, r2t, r2t.stride(0), free, loc,
                                      BLOCK=proxy.meta_block if BS_META else triton.next_power_of_2(n), PAGE=page, PAGED=paged)
            fbt = {"seq_lens": new_seq, "orig_seq_lens": new_orig, "req_pool_indices": rpi, "out_cache_loc": loc}
            from sglang.srt.model_executor.forward_context import ForwardContext, forward_context, has_forward_context

            # ModelRunner._forward_raw's context (the attention layers read the backend from it)
            ctx = contextlib.nullcontext() if has_forward_context() else forward_context(ForwardContext(attn_backend=proxy.backend))
            with ctx:
                out, shell_fb = proxy._body("decode", True, stage, 0, params, attrs, kb, vb, st, ids, pos, fbt, fb, fb_fields, mdt, md, md_fields, ())
            extra = (ids, new_seq, new_orig, pos) + ((loc,) if paged else ())
            if sample:
                extra += (proxy.mr.sample(out, shell_fb),)
            names = [f.name for f in dataclasses.fields(out) if isinstance(getattr(out, f.name), torch.Tensor)]
            template = copy.copy(out)
            for nm in names:
                setattr(template, nm, None)
            proxy.outs[key] = (template, names)
            return tuple(getattr(out, nm) for nm in names) + extra

        return step
