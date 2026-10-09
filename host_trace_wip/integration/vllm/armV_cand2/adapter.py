# Arm V: vLLM's eager model forward (enforce_eager, Model Runner V2) under host-trace replay, after sglang/armF/adapter.py.
#
# install() swaps the model's forward for ArmV.forward, so only the model call is traced; input prep stays eager. With
# ARMV_METADATA "trace" (default) execute_model's attention prep (prepare_attn's block-table gather and slot mappings,
# build_slot_mappings_by_layer, and model_state.prepare_attn's FlashInfer metadata build) is deferred: they return a
# _Pending; ArmV.forward runs prepare_attn inside the traced step and rebuilds the metadata there (_step_md); "eager" builds
# the metadata before the forward as vLLM does (stage 1). Every Attention layer runs direct-call (the Python bodies of
# unified_kv_cache_update / unified_attention_with_output, so the trace sees their ops), and FlashInfer's trtllm-gen
# calls go through trtllm_shim's dispatcher ops. Every CUDA tensor the forward touches is a top-level argument:
# params/buffers (swapped in by _reparametrize_module), plain tensor attributes of modules (each layer's kv_cache), the
# trtllm workspace and counters, input_ids, positions, and the metadata's tensors (eager metadata) or the prep's inputs and
# the runner buffers its kernels write (traced metadata). Each (decode |
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

import numpy as np

import torch
import torch.cuda._host_trace as ht
import triton
import triton.language as tl
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.nn.utils.stateless import _reparametrize_module

FAMILIES = tuple(os.environ.get("ARMV_FAMILIES", "blas,attention,rng,reduce").split(","))
MEMORY = os.environ.get("ARMV_MEMORY", "run_buffer")
DECODE_MAX_SEQ = os.environ.get("ARMV_DECODE_MAX_SEQ", "model")
PREFILL_MAX_KV = os.environ.get("ARMV_PREFILL_MAX_KV", "model")
# "sym" (integration, for the stock-semantics A/B): vLLM's own max_seq_len (the batch's max sequence length, used by
# both the decode and the prefill part) enters the family step as a symbolic int, so the trtllm split choice is
# guarded (a variant per split region) instead of pinned. Open path (the fork) only; the closed path's extern key holds it.
SYM_MAX_SEQ = DECODE_MAX_SEQ == "sym"
METADATA = os.environ.get("ARMV_METADATA", "trace")
INPUTS = os.environ.get("ARMV_INPUTS", "trace")
# prepare_inputs' device launches (vllm/v1/worker/gpu/input_batch.py), deferred into the traced step with ARMV_INPUTS "trace"
INPUT_FNS = ("prepare_prefill_inputs", "prepare_pos_seq_lens", "combine_sampled_and_draft_tokens")
HOSTCUTS = os.environ.get("ARMV_HOSTCUTS", "1") != "0"  # armV/hostcuts.py: STAGE, sampler dirty flag, fused request-state writes
# install_sgl_cpp (python_vllm_cpp.sh, arm Vcpp): HostTraceReplay(static_prefix=len(prefix)) skips the bind and evaluate
# of the leading params/attrs/KV arguments after their first check. ARMV_STATIC_PREFIX=0: off. Absent on int5b (arm V).
STATIC_PREFIX = os.environ.get("ARMV_STATIC_PREFIX", "1") == "1" and "static_prefix" in R.HostTraceReplay.__init__.__code__.co_varnames
# ARMV_FAMILY=1 (default): one entry per (kind, token-count bucket, launch form); the step takes the token and request
# counts from its tensors' sizes and the split from top-level ints, so a new size is a new row of the family's
# variant, not a new trace (after SGLang's armF/bs BS_KEY=3 and BS_META). 0: an entry per exact sizes (_split).
FAMILY = os.environ.get("ARMV_FAMILY", "1") == "1"
# ARMV_NO_REJIT=1 (integration A/B): keep vLLM's _gather_block_tables_kernel and the indptr kernel with Triton's stock
# specialization (num_reqs == 1 / % 16, pointer alignment are guards in core) instead of the do_not_specialize re-jits
NO_REJIT = os.environ.get("ARMV_NO_REJIT", "1") == "1"  # default 1 since 10-08 (the user: same kernels as stock vLLM)
MD_STOCK_DIAG = os.environ.get("ARMV_MD_STOCK_DIAG", "0") == "1"
MD_STOCK = os.environ.get("ARMV_MD_STOCK", "0") == "1"
# ARMV_STOCK=1 (arm V-stock, the "core done" configuration): vLLM builds its metadata eagerly outside the traced region,
# as it does stock (use with ARMV_METADATA=eager ARMV_HOSTCUTS=0 and the max_seq policies "actual"); the region starts at
# model.forward; the metadata's device tensors are arguments and its Python ints (token/request counts, max_seq_len,
# max_q_len) symbolic int arguments, so guards and dispatch see the stock values. One entry per kind and form.
STOCK = os.environ.get("ARMV_STOCK", "0") == "1"
# ARMV_DIRECT_CALL=0: keep vLLM's attention custom-op call path (unified_attention op) instead of the direct call
DIRECT_CALL = os.environ.get("ARMV_DIRECT_CALL", "1") == "1"
_STOCK_INTS = ("num_actual_tokens", "num_decodes", "num_decode_tokens", "num_prefills", "num_prefill_tokens")  # diagnostics: vLLM's metadata build inside the trace (soundness check)  # diagnostics: try vLLM's metadata builder inside the trace
STATIC_SHAPES = os.environ.get("ARMV_STATIC_SHAPES", "0") == "1"
# ARMV_WIDE_SLOTS=1: block_tables.slot_mappings one column wider than max_num_batched_tokens, so the step's
# slot_mappings[:, :T] is a real slice at T == max_num_batched_tokens too (a full-width slice returns self: a guard)
WIDE_SLOTS = os.environ.get("ARMV_WIDE_SLOTS", "0") == "1"
# ARMV_PAD=1 (integration A/B; default 0 = V runs exact token and request counts, which vLLM's enforce_eager dispatch
# already gives): pad each batch as vLLM's default FULL_AND_PIECEWISE dispatch would. Tokens go up to the next
# cudagraph capture size (<= ARMV_PAD_MAX_TOKENS); a uniform decode batch also pads its requests (<= ARMV_PAD_MAX_REQS,
# FULL decode graphs). The padded batch is the step's T.
PAD = os.environ.get("ARMV_PAD", "0") == "1"
# ARMV_VP=1 (arm VP, integration): a decode step's entry also computes the logits, the greedy/temperature sampler
# (gumbel_sample), num_sampled/num_rejected and post_update (the request-state writes) after the model, so one replay
# covers forward + sampling; runner.sample / postprocess_sampled then hand back its outputs. A step whose sampling
# needs anything else (logits processing, logprobs, top-k/p, grammar, drafts) runs eagerly ("eager_vp_form").
VP = os.environ.get("ARMV_VP", "0") == "1"
# ARMV_VP_UVA=1 (default with VP): no copies outside the step either. The input staging slots are read by the step's
# kernels through their UVA views (hostcuts uva mode), and the step writes the sampled tokens and num_sampled into
# pinned UVA output slots, which AsyncOutput reads instead of its two device-to-host copies.
VP_UVA = VP and os.environ.get("ARMV_VP_UVA", "1") == "1"
# ARMV_VP_PINNED=1 (pinned lane, with ARMV_VP_UVA=0): the same single launch through plain pinned host <-> device copies,
# which host tracing records as memcpy nodes. The step takes the pinned staging views (idx_mapping, query_start_loc)
# and copies them into the device buffers first, and copies the sampled tokens and num_sampled into pinned output
# slots last, both with copy_(non_blocking=True) as vLLM's own staging and AsyncOutput do. No UVA view anywhere.
VP_PINNED = VP and not VP_UVA and os.environ.get("ARMV_VP_PINNED", "0") == "1"
VP_SLOTS = 4  # output slots in rotation: more than the steps async scheduling keeps in flight
PAD_MAX_TOKENS = int(os.environ.get("ARMV_PAD_MAX_TOKENS", "512"))
PAD_MAX_REQS = int(os.environ.get("ARMV_PAD_MAX_REQS", "256"))
PAD_SIZES = [1, 2, 4, *range(8, 257, 8), *range(272, 1025, 16)]  # vLLM v0.29's default cudagraph_capture_sizes
# src_vllmcpp item 16: the K/V views of a KV layer lent as one group for an out-of-band learn
LEND_GROUPS = os.environ.get("ARMV_LEND_GROUPS", "1") == "1"
BS_SPLIT = tuple(int(x) for x in os.environ.get("ARMV_BS_SPLIT", "2,32").split(","))
FAM_INTS = ("num_actual_tokens", "num_decodes", "num_decode_tokens", "num_prefills", "num_prefill_tokens")
ENTRY_BENCH = bool(os.environ.get("ARMV_ENTRY_BENCH"))  # time 50 extra (idempotent) calls of each key's entry at its 12th replay
ht.raise_unexpected = bool(os.environ.get("ARMV_HT_RAISE"))
# diagnostics: ARMV_HT_SET="torch.cuda._host_trace.trace_pdl=0,torch.cuda._host_trace_capture.keep_programmatic_edges=0"
for _kv in filter(None, os.environ.get("ARMV_HT_SET", "").split(",")):
    import importlib

    _path, _val = _kv.split("=")
    _mod, _attr = _path.rsplit(".", 1)
    setattr(importlib.import_module(_mod), _attr, bool(int(_val)))
# ARMV_TRACED_IMPLS=1: the five _C ops (rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul,
# reshape_and_cache_flash) trace through armV/vllm_ops_trace.py's launchers (src_vllmcpp CHANGES item 21) and leave the
# extern list; a decline still takes the extern harvest. Default 0: extern harvest, as before.
TRACED_IMPLS = os.environ.get("ARMV_TRACED_IMPLS", "0") == "1"
# ARMV_TRTLLM_FORK=1 (launcher python_vllm_cpp_fork.sh): FlashInfer's trtllm-gen calls trace through the fork's
# flashinfer.trtllm_trace (sglang/fork/VLLM_NOTES.md): trtllm_shim is not installed, its ops leave the extern list and
# its COUNTERS the attrs. Default 0: trtllm_shim's extern ops, as before.
TRTLLM_FORK = os.environ.get("ARMV_TRTLLM_FORK", "0") == "1"

DECLINE_SITES = collections.Counter()
DUMP_LAUNCH = os.environ.get("ARMV_DUMP_LAUNCH", "")  # diagnostics: summary()["launch_dump"] of launches whose name has it
DUMPS = {}
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


def _set(owner, k, v):
    if isinstance(owner, list):
        owner[k] = v
    else:
        setattr(owner, k, v)


def _get(owner, k):
    return owner[k] if isinstance(owner, list) else getattr(owner, k)


@triton.jit
def _kv_indptr_kernel(seq_lens, out, n, page, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    blocks = (tl.load(seq_lens + i, mask=i < n, other=0) + page - 1) // page
    tl.store(out + 1 + i, tl.cumsum(blocks, 0), mask=i < n)
    tl.store(out + i, 0, mask=i == 0)


# FAMILY: a fixed BLOCK (>= every request count) and no specialization of n or of the pointers' alignment (seq_lens and
# out are slices at the decode count): next_power_of_2(n) on a SymInt and Triton's == 1 / % 16 rules would guard n
_kv_indptr_fixed = triton.jit(do_not_specialize=["n"], do_not_specialize_on_alignment=["seq_lens", "out"])(_kv_indptr_kernel.fn)
KV_BLOCK = [None]  # FAMILY: next_power_of_2(max_num_reqs + 1), set by ArmV


def _kv_indptr(seq_lens, page, out):
    """out[0] = 0, out[1:] = cumsum(ceil(seq_lens / page)) in one Triton launch (sglang/armF change 9): torch.cumsum's
    two CUB kernels are an eager boundary of the trace."""
    n = seq_lens.shape[0]
    if KV_BLOCK[0] is not None:
        _kv_indptr_fixed[(1,)](seq_lens, out, n, page, BLOCK=KV_BLOCK[0])
    else:
        _kv_indptr_kernel[(1,)](seq_lens, out, n, page, BLOCK=triton.next_power_of_2(n))


class _Pending:
    """One step's deferred attention prep: what execute_model passed to prepare_attn and model_state.prepare_attn.
    It stands in for the block tables, the slot mappings (both levels) and the metadata dict; live() builds the real
    ones eagerly, once (eager fallbacks and the check)."""

    def __init__(self, input_batch):
        self.input_batch = input_batch
        self.md_args = None  # (cudagraph_mode, attn_groups, kv_cache_config, for_capture)
        self.built = None  # (attn_metadata, slot_mappings_by_layer) once live() ran, max_seq_len policies applied
        self.stock = []  # live()'s _policy result: [(part, vLLM's max_seq_len)]
        self.inputs = None  # prepare_inputs' recorded launches: [(name, args, kwargs, out)], out the logits_indices buffer
        self.inputs_done = False
        self.key = None  # adapter_cpp.py: the decode key its bound plan guards (None: no fast path)


class _FormMismatch(Exception):
    pass


def _vp_gumbel(logits, expanded_idx_mapping, temperature, seed, pos, use_fp64):
    """vLLM's gumbel_sample (sample/gumbel.py) for the VP step, apply_temperature=False and no drafting/logits cache,
    with its final local_argmax.gather(-1, max_block_idx) as a one-hot float32 multiply + sum: the trace runs aten.gather,
    aten.index and an int64 sum eagerly (a boundary each). Same selected values, so the same tokens."""
    from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel

    expanded_idx_mapping, pos = expanded_idx_mapping.contiguous(), pos.contiguous()
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64 if use_fp64 else torch.float32)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](local_argmax, local_argmax.stride(0), local_max, local_max.stride(0), None, 0, 0, None, logits,
                                                    logits.stride(0), expanded_idx_mapping, seed, pos, temperature, vocab_size, BLOCK_SIZE=BLOCK_SIZE,
                                                    IS_DRAFTING=False, APPLY_TEMPERATURE=False, USE_FP64=use_fp64, PER_TOKEN_COL=False)
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    # one-hot select in float32 (the trace's sum host takes floating dtypes): one nonzero term per row and token ids
    # < 2**24, so the sum is the selected id exactly
    sel = torch.arange(num_blocks, device=logits.device) == max_block_idx
    return (local_argmax.float() * sel).sum(dim=-1).long()


class ArmV:
    def __init__(self, runner):
        import vllm.v1.attention.backends.flashinfer as fi
        from vllm.model_executor.layers.attention.attention import Attention

        if TRTLLM_FORK:
            import flashinfer.trtllm_trace  # noqa: F401  (the fork's; stock FlashInfer has none)
        else:
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
        if not TRTLLM_FORK:
            trtllm_shim.install()
        fi._get_trtllm_workspace_buffer()
        # plain CUDA tensor attributes (each layer's kv_cache), the trtllm workspace (a module global) and counters
        self.attrs = [(m, k, v) for m in self.model.modules() for k, v in vars(m).items() if _cuda_tensor(v) and id(v) not in pb]
        self.attrs += [(fi, "trtllm_workspace_buffer", fi.trtllm_workspace_buffer)]
        if not TRTLLM_FORK:
            self.attrs += [(trtllm_shim, "COUNTERS", trtllm_shim.COUNTERS)]
        self.md_trace = METADATA == "trace" and runner.speculator is None and runner.pcp_manager is None and len(runner.attn_groups) == 1 and len(runner.attn_groups[0]) == 1
        if self.md_trace:
            # the runner buffers the prep's kernels read and write: block-table pointer tables (the gather's source rows
            # are read through block_table_ptrs only), the gathered tables, the slot mappings and the builder's indptr
            bt = runner.block_tables
            self.builder = runner.attn_groups[0][0].get_metadata_builder(0)
            assert bt.num_kv_cache_groups == 1
            if WIDE_SLOTS:
                bt.slot_mappings = bt.slot_mappings.new_zeros(bt.slot_mappings.shape[0], bt.slot_mappings.shape[1] + 1)
            self.attrs += [(bt, k, getattr(bt, k)) for k in ("block_table_ptrs", "block_table_strides", "input_block_table_ptrs", "slot_mappings",
                                                             "block_sizes_tensor", "kernel_block_sizes_tensor", "slot_mapping_enabled")]
            self.attrs += [(bt.input_block_tables, 0, bt.input_block_tables[0]), (self.builder.paged_kv_indptr, "gpu", self.builder.paged_kv_indptr.gpu)]
        if VP:
            s, rs = runner.sampler, runner.req_states
            self.vp_arange = torch.arange(runner.max_num_reqs + 1, dtype=torch.int32, device=runner.device)
            # temperature, seeds and prefill_len are UvaBackedTensors whose .gpu rotates through a UVA pool on every
            # write: per-call arguments (_vp_args), not attrs
            self.attrs += [(rs.num_computed_tokens, "gpu", rs.num_computed_tokens.gpu),
                           (rs, "last_sampled_tokens", rs.last_sampled_tokens), (rs.all_token_ids, "gpu", rs.all_token_ids.gpu),
                           (rs.total_len, "gpu", rs.total_len.gpu), (self, "vp_arange", self.vp_arange)]
            if s.penalties_state.output_bin_counts is not None:
                self.attrs.append((s.penalties_state, "output_bin_counts", s.penalties_state.output_bin_counts))
            self.vp_res = None  # (input_batch, (sampled, num_sampled, num_rejected)) of the step's VP entry
            self.vp_post = None
            if VP_UVA or VP_PINNED:
                M = runner.max_num_reqs
                self.vp_out_cpu = [(torch.zeros(M, dtype=torch.int64, pin_memory=True), torch.zeros(M, dtype=torch.int32, pin_memory=True))
                                   for _ in range(VP_SLOTS)]
                if VP_UVA:
                    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

                    self.vp_out_uva = [tuple(get_accelerator_view_from_cpu_tensor(t) for t in pair) for pair in self.vp_out_cpu]
                self.vp_slot = 0
                self.vp_host = {}  # id(device tensor handed to AsyncOutput) -> the pinned numpy the step wrote
        self.prefix = (*self.params, *(v for _, _, v in self.attrs))
        self.replay_kw = {"static_prefix": len(self.prefix)} if STATIC_PREFIX else {}
        if STATIC_SHAPES:
            # weights, KV layers and the fixed buffers keep their layout: constants in the trace (Dynamo's
            # force_parameter_static_shapes), so no guard reads their sizes (a KV view's numel != 0 at a relower)
            self.replay_kw["static_shapes"] = range(len(self.prefix))
        self.in_trace = self.md_trace and INPUTS == "trace"
        self.rec = None  # prepare_inputs' launches while it runs
        self.recorded = None  # its launches until prepare_attn takes them
        if self.in_trace:
            self.logits_buf = torch.empty(runner.max_num_reqs, dtype=torch.int64, device=runner.device)
        self.bad = set()  # exact keys (kind, T, _split, "md", sig) whose metadata form the step does not cover
        self.seen = set()  # FAMILY: exact keys whose vLLM metadata was checked against their family's template
        self.last_exact = None  # the exact key of the last deferred call (armv_serve's per-key record)
        if FAMILY and self.md_trace:
            KV_BLOCK[0] = None if NO_REJIT else triton.next_power_of_2(runner.max_num_reqs + 1)
        self.tmpl = {}  # traced-metadata key -> vLLM's metadata of the key's first call (policies applied)
        c, cache = torch.ops._C, torch.ops._C_cache_ops
        extern = [c.rms_norm.default, c.fused_add_rms_norm.default, c.rotary_embedding.default, c.silu_and_mul.default,
                  cache.reshape_and_cache_flash.default]
        if TRACED_IMPLS:
            import vllm_ops_trace

            names = [n for n in os.environ.get("ARMV_TRACED_IMPLS_NAMES", "").split(",") if n] or None  # diagnostics: a subset of the ports
            extern = [o for o in extern if o not in vllm_ops_trace.install(names)]
        if not TRTLLM_FORK:
            extern += [torch.ops.vllm_ht.trtllm_decode.default, torch.ops.vllm_ht.trtllm_context.default]
        extern = tuple(extern)
        # reshape_and_cache_flash writes rows of the interleaved K/V views of each layer's cache at slot = block * block_size
        # + row: verified on small caches (src_vllm1 CHANGES item 14)
        indexed = {cache.reshape_and_cache_flash.default: ("slot_mapping", "key_cache", "value_cache", 2)}
        indexed = {o: v for o, v in indexed.items() if o in extern}
        # src_vllmcpp item 15 (SGLang P1): an extern key at a new size learns without a model run; its large inputs are
        # views of the KV cache layers and the trtllm workspace, which it reads and does not write
        kw = {}
        if "lendable" in HarvestProvider.__init__.__code__.co_varnames:
            kw["lendable"] = tuple(v for _, k, v in self.attrs if k in ("kv_cache", "trtllm_workspace_buffer"))
        if LEND_GROUPS and "lend_alias_groups" in HarvestProvider.__init__.__code__.co_varnames:
            kw["lend_alias_groups"] = True
        self.md_stock_tried, self.md_stock_why = {}, {}  # MD_STOCK_DIAG
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed, **kw)
        self.entries = {}  # key -> HostTraceReplay
        self.calls = collections.Counter()
        self.host_s = collections.defaultdict(list)  # kind -> host seconds of the replayed call (entry to return, no sync)
        self.entry_bench = {}
        self.slow = collections.Counter()  # key -> post-trace calls through _call_slow
        self.slow_log = collections.defaultdict(list)  # per slow call: (replays, traces, variants) before and after
        self.pre_s = collections.defaultdict(list)  # kind -> of host_s, the seconds before the entry's call (traced metadata)
        self.check = False  # compare each call against the model's own eager forward
        self.checks = []
        self.hostcuts = []
        self._ctx = None

    def install(self):
        if PAD:
            self._install_pad()
        if VP:
            self._install_vp()
        if HOSTCUTS:
            import hostcuts
            self.hostcuts = hostcuts.install(self.runner, uva=VP_UVA, pinned=VP_PINNED)
        for layer in self.layers:
            layer.use_direct_call = DIRECT_CALL
        self.model.forward = self.forward
        if self.md_trace:
            import vllm.v1.worker.gpu.model_runner as mr

            r, ms = self.runner, self.runner.model_state
            self.prep = (r.prepare_attn, ms.prepare_attn, mr.build_slot_mappings_by_layer)
            orig_attn, orig_md, orig_slots = self.prep

            def prepare_attn(input_batch):
                p = _Pending(input_batch)
                p.inputs, self.recorded = self.recorded, None
                return p, p

            def build_slot_mappings_by_layer(slot_mappings, kv_cache_config):
                return slot_mappings if isinstance(slot_mappings, _Pending) else orig_slots(slot_mappings, kv_cache_config)

            def ms_prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture=False):
                if not isinstance(block_tables, _Pending):
                    return orig_md(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture=for_capture)
                block_tables.md_args = (cudagraph_mode, attn_groups, kv_cache_config, for_capture)
                return block_tables

            r.prepare_attn = prepare_attn
            if FAMILY:
                # the gather's num_reqs argument is the request count: Triton's == 1 / % 16 rules would guard it
                import vllm.v1.worker.gpu.block_table as btm

                if not NO_REJIT and not getattr(btm, "_armv_fixed", False):
                    btm._gather_block_tables_kernel = triton.jit(do_not_specialize=["num_reqs"])(btm._gather_block_tables_kernel.fn)
                    btm._armv_fixed = True
            if self.in_trace:
                self.inputs_orig = {n: getattr(mr, n) for n in INPUT_FNS}
                for n in INPUT_FNS:
                    setattr(mr, n, self._recorder(n))
                orig_pi = r.prepare_inputs

                def prepare_inputs(*args, **kw):
                    if self.rec is not None or self.recorded is not None:
                        raise AssertionError("arm V: the previous prepare_inputs' launches were never run")
                    self.rec = []
                    try:
                        ib = orig_pi(*args, **kw)
                    except BaseException:
                        self.rec = None
                        raise
                    self.recorded, self.rec = self.rec, None
                    return ib
                r.prepare_inputs = prepare_inputs
            ms.prepare_attn = ms_prepare_attn
            mr.build_slot_mappings_by_layer = build_slot_mappings_by_layer

    def _count_slow(self, key, entry):
        """Counts the calls the native path does not serve once the entry has a trace (where a learner variant runs)."""
        orig = entry._call_slow

        def counted(*a, **k):
            if entry.traces:
                self.slow[key] += 1
            before = (entry.replays, entry.traces, len(entry.variants))
            res = orig(*a, **k)
            self.slow_log[key].append(before + (entry.replays, entry.traces, len(entry.variants)))
            return res
        entry._call_slow = counted

    def _recorder(self, name):
        """mr.<name> while prepare_inputs runs: records the launch; combine_sampled_and_draft_tokens returns a view of
        logits_buf, which the launch fills."""
        fn = self.inputs_orig[name]

        def rec(*args, **kw):
            if self.rec is None:
                return fn(*args, **kw)
            out = None
            if name == "combine_sampled_and_draft_tokens":
                out = self.logits_buf[:args[8] if len(args) > 8 else kw["num_logits"]]
            self.rec.append((name, args, kw, out))
            return out
        return rec

    def _run_inputs(self, calls, tensors=None, family=False):
        """Runs recorded launches; with `tensors`, their tensors in _input_sig's order replace the recorded ones. With
        `family`, combine_sampled_and_draft_tokens' num_logits is its output's length (a size of the step's arguments)."""
        it = iter(tensors) if tensors is not None else None
        sub = (lambda a: next(it) if isinstance(a, torch.Tensor) else a) if it is not None else (lambda a: a)
        for name, args, kw, out in calls:
            a, k = [sub(x) for x in args], {kk: sub(v) for kk, v in kw.items()}
            o = sub(out) if out is not None else None
            if family and name == "combine_sampled_and_draft_tokens":
                if len(a) > 8:
                    a[8] = o.shape[0]
                else:
                    k["num_logits"] = o.shape[0]
            res = self.inputs_orig[name](*a, **k)
            if o is not None:
                o.copy_(res)

    @staticmethod
    def _sig_form(sig):
        """FAMILY: the input signature without combine_sampled_and_draft_tokens' num_logits (_run_inputs derives it)."""
        out = []
        for name, args, kw, has_out in sig:
            if name == "combine_sampled_and_draft_tokens":
                args = args[:8] + (None,) + args[9:] if len(args) > 8 else args
                kw = tuple((k, None if k == "num_logits" else v) for k, v in kw)
            out.append((name, args, kw, has_out))
        return tuple(out)

    @staticmethod
    def _bucket(T):
        return sum(T >= b for b in BS_SPLIT)

    def _family_key(self, kind, T, sig):
        return (kind, self._bucket(T), "fam", self._sig_form(sig))

    def _entry_key(self, kind, T, ints, sig):
        return self._family_key(kind, T, sig) if FAMILY else (kind, T, ints, "md", sig)

    @staticmethod
    def _fam_ints(kind, ints):
        """FAMILY: the step's top-level ints: the split (decode count, decode tokens) and the prefill max_q_len; with
        SYM_MAX_SEQ also the batch's max_seq_len (last)."""
        out = () if kind == "decode" else (ints[5],) if kind == "prefill" else (ints[1], ints[2], ints[5])
        return out + (ints[6],) if SYM_MAX_SEQ else out

    @staticmethod
    def _fixed_fields(md):
        """The metadata's non-tensor fields a family's step copies from its template: all but the counts it sets."""
        out = []
        ms = ("max_seq_len",) if SYM_MAX_SEQ else ()
        for part, skip in ((md, FAM_INTS + ("decode", "prefill")), (md.decode, ms), (md.prefill, ("max_q_len",) + ms)):
            out.append(None if part is None else (type(part), tuple((k, v) for k, v in vars(part).items() if k not in skip and not isinstance(v, torch.Tensor))))
        return out

    @staticmethod
    def _input_sig(calls):
        """The recorded launches' host values (part of the key) and their tensors, in order."""
        sig, tensors = [], []
        for name, args, kw, out in calls:
            for a in (*args, *kw.values(), out):
                if isinstance(a, torch.Tensor):
                    tensors.append(a)
            sig.append((name, tuple(None if isinstance(a, torch.Tensor) else a for a in args),
                        tuple((k, None if isinstance(v, torch.Tensor) else v) for k, v in kw.items()), out is not None))
        return tuple(sig), tensors

    def _flush(self, p):
        if p.inputs and not p.inputs_done:
            self._run_inputs(p.inputs)
        p.inputs_done = True

    def _build(self, input_batch, md_args):
        """vLLM's attention prep for input_batch: (attn_metadata dict, slot_mappings_by_layer)."""
        orig_attn, orig_md, orig_slots = self.prep
        cg, groups, kvc, for_capture = md_args
        block_tables, slot_mappings = orig_attn(input_batch)
        slots = orig_slots(slot_mappings, kvc)
        return orig_md(input_batch, cg, block_tables, slot_mappings, groups, kvc, for_capture=for_capture), slots

    def _live(self, fc):
        """The live context's metadata and slot mappings, built eagerly if deferred."""
        amd = fc.attn_metadata
        if not isinstance(amd, _Pending):
            return amd, fc.slot_mapping
        if amd.built is None:
            # vLLM's own build reads the staged inputs: copied here too (the step still copies them in)
            self._stage_eager(clear=False)
            self._flush(amd)
            amd.built = self._build(amd.input_batch, amd.md_args)
            md = next(iter(amd.built[0].values()), None)
            if md is not None and hasattr(md, "decode"):
                amd.stock = self._policy(md)
        return amd.built

    def _policy(self, md):
        """Applies the max_seq_len policies to md's decode / prefill parts in place: [(part, vLLM's value)]."""
        stock = []
        if md.decode is not None and DECODE_MAX_SEQ == "model" and md.decode.max_seq_len != self.max_model_len:
            stock.append((md.decode, md.decode.max_seq_len))
            md.decode.max_seq_len = self.max_model_len
        if md.prefill is not None and PREFILL_MAX_KV == "model" and md.prefill.max_seq_len != self.max_model_len:
            stock.append((md.prefill, md.prefill.max_seq_len))
            md.prefill.max_seq_len = self.max_model_len
        return stock

    def _split(self, ib):
        """split_decodes_and_prefills (vllm/v1/attention/backends/utils.py) on the host arrays, and the prefill max_q_len:
        the key's ints. The step checks them against the metadata it builds."""
        b, n, T = self.builder, ib.num_reqs, ib.num_tokens
        q = ib.num_scheduled_tokens[:n]
        thr, uniform = b.reorder_batch_threshold, not b.use_dedicated_xqa
        nd = n
        if not (q.max() <= thr and (not uniform or thr <= 1)):
            if q[0] > thr:
                nd = 0
            elif not (uniform and ((q == q[0]) | (q == 0)).all()):
                isp = q != q[0] if uniform else q > thr
                nd = int(isp.argmax()) if isp.any() else n
        nd_tok = int(q[:nd].sum())
        max_seq = None
        if DECODE_MAX_SEQ != "model" or PREFILL_MAX_KV != "model":
            max_seq = int(ib.seq_lens_cpu_upper_bound[:n].max())
        return (T, nd, nd_tok, n - nd, T - nd_tok, int(q[nd:].max()) if nd < n else None, max_seq)

    @staticmethod
    def _md_ints(md):
        return (md.num_actual_tokens, md.num_decodes, md.num_decode_tokens, md.num_prefills, md.num_prefill_tokens,
                md.prefill.max_q_len if md.prefill is not None else None)

    def _eager(self, why, fc, input_ids, positions, *rest):
        from vllm.forward_context import override_forward_context

        self.calls[why] += 1
        self._stage_eager()
        if not isinstance(fc.attn_metadata, _Pending):
            return self.orig_forward(input_ids, positions, *rest)
        amd, sm = self._live(fc)
        lfc = copy.copy(fc)
        lfc.attn_metadata, lfc.slot_mapping = amd, sm
        with override_forward_context(lfc):
            return self.orig_forward(input_ids, positions, *rest)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        from vllm.forward_context import get_forward_context

        t0 = time.perf_counter()
        fc = get_forward_context()
        if intermediate_tensors is not None or inputs_embeds is not None or input_ids is None:
            return self._eager("eager_inputs", fc, input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(fc.attn_metadata, _Pending):
            return self._forward_deferred(t0, fc, input_ids, positions)
        self._stage_eager()
        amd, sm = fc.attn_metadata, fc.slot_mapping
        if not isinstance(amd, dict) or not amd or not isinstance(sm, dict) or not sm:
            return self._eager("eager_no_metadata", fc, input_ids, positions)
        md = next(iter(amd.values()))
        slot = next(iter(sm.values()))
        dec, pre = md.decode, md.prefill
        if not self._form_ok(amd, sm, md, slot):
            return self._eager("eager_metadata_form", fc, input_ids, positions)
        stock = self._policy(md)
        kind = "mixed" if dec is not None and pre is not None else "decode" if dec is not None else "prefill"
        md_slot = None if _same_view(md.slot_mapping, slot) else md.slot_mapping
        ints = (md.num_actual_tokens, md.num_decodes, md.num_decode_tokens, md.num_prefills, md.num_prefill_tokens,
                dec.max_seq_len if dec is not None else None, (pre.max_q_len, pre.max_seq_len) if pre is not None else None)
        key = (kind, input_ids.shape[0], ints, md_slot is None)
        lifted = []
        if STOCK:
            lifted = [getattr(md, k) for k in _STOCK_INTS] + ([dec.max_seq_len] if dec is not None else []) + ([pre.max_q_len, pre.max_seq_len] if pre is not None else [])
            key = (kind, "stock", self._stock_form(md), md_slot is None)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step(key), opaque=(self.provider,), memory=MEMORY, **self.replay_kw)
        self.calls[kind] += 1
        args = [*self.prefix, input_ids, positions, slot]
        if md_slot is not None:
            args.append(md_slot)
        if dec is not None:
            args += [dec.block_tables, dec.seq_lens]
        if pre is not None:
            args += [pre.block_tables, pre.seq_lens, pre.cum_seq_lens_q, pre.cum_seq_lens_kv]
        args += lifted
        self._ctx = (fc, md)
        replays = entry.replays
        try:
            (out,) = entry(*args)
        finally:
            self._ctx = None
        if entry.replays > replays:
            self.host_s[kind].append(time.perf_counter() - t0)
        if self.check:
            self._check(kind, out, fc, input_ids, positions, amd, sm, stock)
        return out

    @staticmethod
    def _stock_form(md):
        """STOCK: the metadata's non-tensor fields other than the lifted ints (they must match within an entry)."""
        out = []
        for part, skip in ((md, _STOCK_INTS + ("decode", "prefill")), (md.decode, ("max_seq_len",)), (md.prefill, ("max_q_len", "max_seq_len"))):
            out.append(None if part is None else (type(part).__name__, tuple((k, v if isinstance(v, (int, float, bool, str, type(None))) else repr(v)[:80])
                                                                             for k, v in vars(part).items() if k not in skip and not isinstance(v, torch.Tensor))))
        return tuple(out)

    @staticmethod
    def _form_ok(amd, sm, md, slot):
        """The metadata form the step covers: one FlashInferMetadata and slot mapping shared by all layers, no cascade,
        trtllm-gen decode with one query token per request, trtllm prefill."""
        from vllm.v1.attention.backends.flashinfer import FlashInferDecodeKernel, FlashInferMetadata, FlashInferTrtllmAPIDecode, TRTLLMPrefill

        dec, pre = md.decode, md.prefill
        return (type(md) is FlashInferMetadata and not md.use_cascade and all(v is md for v in amd.values())
                and all(v is slot for v in sm.values())
                and (dec is None or type(dec) is FlashInferTrtllmAPIDecode and dec.kernel is FlashInferDecodeKernel.TRTLLM_GEN
                     and dec.q_len_per_req == 1 and dec.q_cu_seq_lens is None and dec.mask is None)
                and (pre is None or type(pre) is TRTLLMPrefill))

    def _forward_deferred(self, t0, fc, input_ids, positions):
        p = fc.attn_metadata
        ib = p.input_batch
        if p.md_args is None or p.md_args[0] != self.cg_none or p.md_args[3] or not self._same_inputs(ib, input_ids, positions):
            return self._eager("eager_prep_form", fc, input_ids, positions)
        ints = self._split(ib)
        kind = "mixed" if ints[1] and ints[3] else "decode" if ints[1] else "prefill"
        if VP and kind == "decode" and not self._vp_ok(ib):
            return self._eager("eager_vp_form", fc, input_ids, positions)
        sig, in_tensors = self._input_sig(p.inputs) if p.inputs else ((), [])
        T = input_ids.shape[0]
        exact = self.last_exact = (kind, T, ints, "md", sig)
        key, extra = self._entry_key(kind, T, ints, sig), (self._fam_ints(kind, ints) if FAMILY else ())
        if exact in self.bad:
            return self._eager("eager_metadata_form", fc, input_ids, positions)
        tmpl, verify = self.tmpl.get(key), None
        if tmpl is None or exact not in self.seen:
            # vLLM's own build, eagerly, once per exact key: the form and ints the step's metadata copies (FAMILY: and
            # the template's other host fields)
            amd, sm = self._live(fc)
            md = next(iter(amd.values()))
            if (not self._form_ok(amd, sm, md, next(iter(sm.values()))) or self._md_ints(md) != ints[:6]
                    or tmpl is not None and self._fixed_fields(md) != self._fixed_fields(tmpl)):
                self.bad.add(exact)
                return self._eager("eager_metadata_form", fc, input_ids, positions)
            self.seen.add(exact)
            if tmpl is None:
                tmpl = verify = self.tmpl[key] = md
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step_md(key), opaque=(self.provider,), memory=MEMORY, **self.replay_kw)
            self._count_slow(key, entry)
        self.calls[kind] += 1
        bt = self.runner.block_tables
        args = [*self.prefix, input_ids, positions, ib.idx_mapping, ib.query_start_loc, ib.seq_lens, bt.num_blocks.gpu, *in_tensors, *extra]
        if VP and kind == "decode":
            args += self._vp_args()
        stage = getattr(self.runner, "_ht_stage", None)
        if VP_PINNED and kind == "decode":
            # the step copies the staging views in itself (None, None: nothing staged)
            self.runner._ht_stage = None
            args += [stage[0][1], stage[1][1]] if stage is not None else [None, None]
        else:
            self._stage_eager()
        self._ctx = (fc, p, tmpl, verify)
        replays = entry.replays
        t1 = time.perf_counter()
        try:
            out, *vp = entry(*args)
            if vp:
                self.vp_res = (ib, vp, getattr(self, "vp_slot", None))
        except _FormMismatch:  # the step's metadata is not vLLM's (raised at the entry's first, eager, call)
            self.bad.add(exact)
            if FAMILY:
                del self.entries[key], self.tmpl[key]
            self.calls[kind] -= 1
            return self._eager("eager_metadata_form", fc, input_ids, positions)
        finally:
            self._ctx = None
        p.inputs_done = True
        if entry.replays > replays:
            self.host_s[f"{kind}/T{T}"].append(time.perf_counter() - t0)
            self.pre_s[f"{kind}/T{T}"].append(t1 - t0)
            if ENTRY_BENCH and entry.replays == 12 and key not in self.entry_bench:
                self.entry_bench[key] = self._bench_entry(entry, args, fc, p, tmpl)
        if self.check:
            amd, sm = self._live(fc)
            self._check(kind, out, fc, input_ids, positions, amd, sm, p.stock)
        return out

    def _bench_entry(self, entry, args, fc, p, tmpl):
        slow = [0]
        orig = entry._call_slow

        def counted(*a, **k):
            slow[0] += 1
            return orig(*a, **k)
        entry._call_slow = counted
        ts = []
        try:
            for _ in range(50):
                self._ctx = (fc, p, tmpl, None)
                torch.cuda.synchronize()
                t = time.perf_counter()
                entry(*args)
                ts.append(time.perf_counter() - t)
        finally:
            self._ctx = None
            entry._call_slow = orig
        ts.sort()
        return {"median_us": round(ts[25] * 1e6, 1), "p10_us": round(ts[5] * 1e6, 1), "slow_calls": slow[0], "variants": len(entry.variants),
                "nargs": len(args), "learns": [getattr(v, "learns", None) for v in entry.variants]}

    @property
    def cg_none(self):
        from vllm.config import CUDAGraphMode

        return CUDAGraphMode.NONE

    @staticmethod
    def _same_inputs(ib, input_ids, positions):
        """The model's input_ids and positions are the input batch's (no M-RoPE positions; no padding unless PAD)."""
        if PAD:
            return ib.positions is positions and ib.input_ids is input_ids and ib.num_tokens_after_padding == input_ids.shape[0]
        return ib.positions is positions and ib.input_ids is input_ids and ib.num_tokens == input_ids.shape[0] == ib.num_tokens_after_padding and ib.num_reqs == ib.num_reqs_after_padding

    def _vp_ok(self, ib):
        """The step's sampling is what the VP entry computes: no logits processing, logprobs, top-k/p, NaN count,
        sampling mask, trace replay, drafts or batch sharding."""
        s, idx_np = self.runner.sampler, ib.idx_mapping_np
        return (ib.num_draft_tokens == 0 and self.runner.batch_sharder is None and s.trace_replay_state is None and not s.compute_nans
                and not s.return_sampling_mask and not np.any(s.needs_logits_processing[idx_np]) and s.get_logprobs_dims(idx_np) is None
                and all(x is None for x in s.sampling_states.get_top_k_top_p(ib.expanded_idx_mapping, idx_np)))

    def _install_vp(self):
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        r = self.runner
        orig_sample, orig_post = r.sample, r.postprocess_sampled

        def sample(hidden_states, input_batch, grammar_output):
            vp, self.vp_res = self.vp_res, None
            if vp is None or vp[0] is not input_batch:
                return orig_sample(hidden_states, input_batch, grammar_output)
            if grammar_output is not None:
                raise AssertionError("arm VP: a grammar bitmask for a step the VP entry already sampled")
            sampled, ns, nr = vp[1][:3]
            self.vp_post = input_batch
            self.calls["vp_sampled"] += 1
            tok = sampled.view(-1, 1)
            if VP_UVA or VP_PINNED:
                n, (tc, nc) = sampled.shape[0], self.vp_out_cpu[vp[2]]
                self.vp_host[id(tok)], self.vp_host[id(ns)] = tc[:n].numpy().reshape(n, 1), nc[:n].numpy()
            return SamplerOutput(sampled_token_ids=tok, logprobs_tensors=None, num_nans=None, num_sampled=ns, num_rejected=nr), ns, nr

        def postprocess_sampled(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
            if self.vp_post is None:
                return orig_post(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
            self.vp_post = None  # post_update ran in the entry
            r.model_state.postprocess_state(idx_mapping, num_sampled, r.req_states.num_computed_tokens.gpu)
        r.sample, r.postprocess_sampled = sample, postprocess_sampled
        if VP_UVA or VP_PINNED:
            import vllm.v1.worker.gpu.async_utils as au

            orig_copy = au.async_copy_to_np

            def async_copy_to_np(x):
                # AsyncOutput's two copies for a VP step: the step already wrote them into pinned memory; AsyncOutput's
                # copy_event (on copy_stream after wait_stream(main)) still orders get_output after the step
                h = self.vp_host.pop(id(x), None)
                return orig_copy(x) if h is None else h
            au.async_copy_to_np = async_copy_to_np

    def _vp_args(self):
        """The VP step's per-call tensors (the current UVA buffers), appended after the step's other arguments; with
        VP_UVA also the step's output slot (rotated here, once per step)."""
        s, rs = self.runner.sampler, self.runner.req_states
        out = [s.sampling_states.temperature.gpu, s.sampling_states.seeds.gpu, rs.prefill_len.gpu]
        if VP_UVA or VP_PINNED:
            self.vp_slot = (self.vp_slot + 1) % VP_SLOTS
            out += list((self.vp_out_uva if VP_UVA else self.vp_out_cpu)[self.vp_slot])
        return out

    def _stage_eager(self, clear=True):
        """VP_PINNED: the staging copies hostcuts left to the step, run here for a step that does not take them
        (clear) or for an eager read before the step (not clear: the step still takes them)."""
        stage = getattr(self.runner, "_ht_stage", None)
        if stage is not None:
            if clear:
                self.runner._ht_stage = None
            for dst, src in stage:
                dst.copy_(src, non_blocking=True)

    def _vp_tail(self, out, input_ids, positions, idx_mapping, query_start_loc, seq_lens, temperature, seeds, prefill_len, tok_out=None, ns_out=None):
        """Inside the VP step, after the model: logits, sampler, num_sampled/num_rejected, post_update (as vLLM's
        sample + postprocess_sampled on a no-draft decode batch, whose logits_indices are arange(n))."""
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected, post_update
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        s, rs = self.runner.sampler, self.runner.req_states
        n = idx_mapping.shape[0]
        logits = self.model.compute_logits(out)
        sampled = _vp_gumbel(logits, idx_mapping, temperature, seeds, positions, s.use_fp64_gumbel)
        ns, nr = get_num_sampled_and_rejected(seq_lens.new_ones(n), seq_lens[:n], self.vp_arange[:n + 1], idx_mapping, prefill_len)
        post_update(idx_mapping, rs.num_computed_tokens.gpu, rs.last_sampled_tokens, s.penalties_state.output_bin_counts, sampled.view(-1, 1),
                    ns, nr, query_start_loc[:n + 1], rs.all_token_ids.gpu, rs.total_len.gpu)
        if tok_out is not None and VP_PINNED:  # the device-to-host results, as AsyncOutput's non_blocking copies
            tok_out[:n].copy_(sampled, non_blocking=True)
            ns_out[:n].copy_(ns, non_blocking=True)
        elif tok_out is not None:  # the device-to-host results, written by kernels into pinned memory (UVA)
            torch.add(sampled, 0, out=tok_out[:n])
            torch.add(ns, 0, out=ns_out[:n])
        return sampled, ns, nr

    def _install_pad(self):
        import bisect

        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        cgm = self.runner.cudagraph_manager
        orig = cgm.dispatch
        self.pad_rows = collections.Counter()  # "tokens" / "reqs": padded rows added over the run

        def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
            desc = orig(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=max_query_len)
            if num_tokens <= 0 or num_tokens > PAD_MAX_TOKENS:
                return desc
            t = PAD_SIZES[bisect.bisect_left(PAD_SIZES, num_tokens)]
            decode = uniform_token_count == 1 and num_tokens == num_reqs and t <= PAD_MAX_REQS
            self.pad_rows["tokens"] += t - num_tokens
            self.pad_rows["reqs"] += t - num_reqs if decode else 0
            return BatchExecutionDescriptor(cg_mode=desc.cg_mode, num_tokens=t, num_reqs=t if decode else None, num_active_loras=desc.num_active_loras)
        cgm.dispatch = dispatch

    def _check(self, kind, out, fc, input_ids, positions, amd, sm, stock):
        from vllm.forward_context import override_forward_context

        got = out.clone()
        torch.cuda.synchronize()
        lfc = copy.copy(fc)
        lfc.attn_metadata, lfc.slot_mapping = amd, sm
        with override_forward_context(lfc):
            want = self.orig_forward(input_ids, positions)
            torch.cuda.synchronize()
            full = got.shape == want.shape and torch.equal(got, want)
            if PAD:  # the real rows only: a padded row's output is whatever its zero-length request leaves
                n = next(iter(amd.values())).num_actual_tokens
                got, want = got[:n], want[:n]
            row = {"kind": kind, "shape": list(want.shape), "bitwise": bool(got.shape == want.shape and torch.equal(got, want)),
                   "max_abs": float((got.float() - want.float()).abs().max()) if got.shape == want.shape else None, "padded_rows_equal": bool(full)}
            if stock:  # vLLM's stock eager, with its own max_seq_len values
                for part, n in stock:
                    part.max_seq_len = n
                ref = self.orig_forward(input_ids, positions)
                torch.cuda.synchronize()
                for part, _ in stock:
                    part.max_seq_len = self.max_model_len
                row["stock_parts"] = [type(part).__name__ for part, _ in stock]
                ref = ref[:got.shape[0]]
                row["bitwise_vs_stock"] = bool(torch.equal(got, ref))
                row["max_abs_vs_stock"] = float((got.float() - ref.float()).abs().max())
        if VP and self.vp_res is not None and kind == "decode":  # the entry's tokens against vLLM's sampler on the eager logits
            from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

            sm = self.runner.sampler
            ib = self.vp_res[0]
            with torch.no_grad():
                ref = gumbel_sample(self.model.compute_logits(want), ib.idx_mapping, sm.sampling_states.temperature.gpu, sm.sampling_states.seeds.gpu,
                                    positions, apply_temperature=False, is_drafting=False, use_fp64=sm.use_fp64_gumbel)
            row["sampled_equal"] = bool(torch.equal(ref, self.vp_res[1][0]))
        self.checks.append(row)

    def _swapped(self, attrs):
        saved = [(m, k, _get(m, k)) for m, k, _ in self.attrs]
        for (m, k, _), t in zip(self.attrs, attrs):
            _set(m, k, t)
        return saved

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
            if STOCK:  # the lifted ints, symbolic
                for k in _STOCK_INTS:
                    setattr(shell, k, flat[i]); i += 1
                if md.decode is not None:
                    shell.decode.max_seq_len = flat[i]; i += 1
                if md.prefill is not None:
                    shell.prefill.max_q_len, shell.prefill.max_seq_len = flat[i:i + 2]; i += 2
            sfc = copy.copy(fc)
            sfc.attn_metadata = dict.fromkeys(fc.attn_metadata, shell)
            sfc.slot_mapping = dict.fromkeys(fc.slot_mapping, slot)
            saved = proxy._swapped(attrs)
            try:
                with override_forward_context(sfc), _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions)
            finally:
                for m, k, v in saved:
                    _set(m, k, v)
            return (out,)

        return step

    def _step_md(self, key):
        """The traced step with the attention prep inside. vLLM's prepare_attn (the block-table gather and slot-mapping
        Triton kernels) runs over a copy of the input batch holding the step's tensors, with the runner buffers swapped
        for the step's. The FlashInfer build reads host tensors (.item(), split_decodes_and_prefills), which a trace
        declines, so the step copies the key's template (vLLM's metadata of its first call) and recomputes its tensor
        fields as FlashInferMetadataBuilder.build does on the trtllm paths; at the first call each is checked against
        vLLM's."""
        from vllm.forward_context import override_forward_context

        kind = key[0]
        n_x = len(self._fam_ints(kind, (None,) * 7)) if FAMILY else 0
        n_p, n_a = len(self.params), len(self.attrs)
        page = self.builder.page_size
        layers = self.runner.attn_groups[0][0].layer_names
        proxy = self

        def metadata(tmpl, block_tables, slot_mappings, seq_lens, qo, nd, n):
            md = copy.copy(tmpl)
            bt0 = block_tables[0]
            md.slot_mapping = slot_mappings[0]
            if tmpl.decode is not None:
                md.decode = copy.copy(tmpl.decode)
                md.decode.block_tables, md.decode.seq_lens = bt0[:nd], seq_lens[:nd]
            if tmpl.prefill is not None:
                md.prefill = pre = copy.copy(tmpl.prefill)
                pre.block_tables, pre.seq_lens = bt0[nd:], seq_lens[nd:]
                pre.cum_seq_lens_q = qo[nd:] - qo[nd]
                pre.cum_seq_lens_kv = proxy.builder.paged_kv_indptr.gpu[nd:n + 1]
                _kv_indptr(pre.seq_lens, page, pre.cum_seq_lens_kv)
            for new, old in ((md, tmpl), (md.decode, tmpl.decode), (md.prefill, tmpl.prefill)):
                if old is not None and any(isinstance(v, torch.Tensor) and v is getattr(new, k) for k, v in vars(old).items()):
                    raise _FormMismatch(f"{type(old).__name__} has a tensor field the step does not recompute")
            return md

        def verify(md, live):
            for new, old in ((md, live), (md.decode, live.decode), (md.prefill, live.prefill)):
                if old is None:
                    continue
                for k, v in vars(old).items():
                    w = getattr(new, k)
                    if isinstance(v, torch.Tensor):
                        if not (w.shape == v.shape and w.dtype == v.dtype and torch.equal(w, v)):
                            raise _FormMismatch(f"{type(old).__name__}.{k} differs from vLLM's")
                    elif k not in ("decode", "prefill") and w != v:
                        raise _FormMismatch(f"{type(old).__name__}.{k} {w} != {v}")

        def step(*flat):
            fc, pending, tmpl, live = proxy._ctx
            params = flat[:n_p]
            attrs = flat[n_p:n_p + n_a]
            input_ids, positions, idx_mapping, query_start_loc, seq_lens, num_blocks = flat[n_p + n_a:n_p + n_a + 6]
            rest = flat[n_p + n_a + 6:]
            vp_t = ()
            if VP and kind == "decode":
                if VP_PINNED:
                    # the pinned staging views: into the device buffers before anything reads them
                    rest, (h_idx, h_qsl) = rest[:-2], rest[-2:]
                    if h_idx is not None:
                        idx_mapping.copy_(h_idx, non_blocking=True)
                        query_start_loc.copy_(h_qsl, non_blocking=True)
                nv = 5 if VP_UVA or VP_PINNED else 3
                rest, vp_t = rest[:-nv], rest[-nv:]
            tensors, extra = (rest[:-n_x], rest[-n_x:]) if n_x else (rest, ())
            if pending.inputs:
                proxy._run_inputs(pending.inputs, tensors, family=FAMILY)
            ib = copy.copy(pending.input_batch)
            ib.input_ids, ib.positions, ib.idx_mapping, ib.query_start_loc, ib.seq_lens = input_ids, positions, idx_mapping, query_start_loc, seq_lens
            if FAMILY:
                T, n = input_ids.shape[0], idx_mapping.shape[0]
                ex = extra[:-1] if SYM_MAX_SEQ else extra
                nd, nd_tok, mq = (n, T, None) if kind == "decode" else (0, 0, ex[0]) if kind == "prefill" else ex
                ib.num_reqs = ib.num_reqs_after_padding = n
                ib.num_tokens = ib.num_tokens_after_padding = T
            else:
                ints = key[2]
                nd, n = ints[1], ints[1] + ints[3]
            nb = proxy.runner.block_tables.num_blocks
            saved = proxy._swapped(attrs) + [(nb, "gpu", nb.gpu)]
            nb.gpu = num_blocks
            try:
                block_tables, slot_mappings = proxy.prep[0](ib)
                if MD_STOCK_DIAG and not proxy.md_stock_tried.get(kind):
                    # diagnostics: vLLM's own metadata build inside the trace; record why the trace declines or raises
                    proxy.md_stock_tried[kind] = True
                    try:
                        cg, groups, kvc, for_capture = pending.md_args
                        proxy.prep[1](ib, cg, block_tables, slot_mappings, groups, kvc, for_capture=for_capture)
                        proxy.md_stock_why[kind] = "built without a decline"
                    except BaseException as ex:  # noqa: B036 (a Declined is the answer)
                        proxy.md_stock_why[kind] = f"{type(ex).__name__}: {str(ex)[:600]} @ " + " <- ".join(
                            f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in traceback.extract_tb(ex.__traceback__)[-8:])
                md = metadata(tmpl, block_tables, slot_mappings, seq_lens[:n], query_start_loc[:n + 1], nd, n)
                if FAMILY:
                    for k, v in zip(FAM_INTS, (T, nd, nd_tok, n - nd, T - nd_tok)):
                        setattr(md, k, v)
                    if md.prefill is not None:
                        md.prefill.max_q_len = mq
                    if SYM_MAX_SEQ:
                        for part in (md.decode, md.prefill):
                            if part is not None:
                                part.max_seq_len = extra[-1]
                if live is not None:
                    verify(md, live)
                sfc = copy.copy(fc)
                sfc.attn_metadata, sfc.slot_mapping = dict.fromkeys(layers, md), dict.fromkeys(layers, md.slot_mapping)
                if MD_STOCK:
                    # diagnostics: vLLM's own metadata build inside the trace in place of the template copy (its host
                    # numpy / CPU-tensor math is invisible to the trace: replays reuse the traced call's host values)
                    cg, groups, kvc, for_capture = pending.md_args
                    sfc.attn_metadata = proxy.prep[1](ib, cg, block_tables, slot_mappings, groups, kvc, for_capture=for_capture)
                    sfc.slot_mapping = proxy.prep[2](slot_mappings, kvc)
                with override_forward_context(sfc), _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions)
                    vp = proxy._vp_tail(out, input_ids, positions, idx_mapping, query_start_loc, seq_lens, *vp_t) if vp_t else ()
            finally:
                for m, k, v in saved:
                    _set(m, k, v)
            return (out, *vp)

        return step

    def _trtllm_dump(self):
        """Per trtllm decode binding: its key's scalars and sizes, and per kernel node the function, grid, block, smem and
        the bytes outside slots; then, for keys that differ only in scalars, which parameter bytes differ."""
        from torch.cuda._host_trace_opaque import OpaqueKernel

        rows = []
        for k, b in self.provider.bindings.items():
            if "trtllm_decode" not in str(k[1]):
                continue
            ks = [n for n in b.nodes if isinstance(n, OpaqueKernel)]
            rows.append((k, ks))
        out = {"n": len(rows), "keys": [], "diffs": []}
        for k, ks in rows[:12]:
            out["keys"].append({"key": str(k)[:600], "kernels": [{"function": n.function, "grid": n.grid, "block": n.block, "smem": n.smem} for n in ks]})
        by_shape = {}
        for k, ks in rows:
            sig = tuple((n.function, n.grid, n.block, n.smem) for n in ks)
            by_shape.setdefault(sig, []).append((k, ks))
        for sig, group in by_shape.items():
            if len(group) < 2:
                continue
            (k0, a), (k1, b) = group[0], group[1]
            d = []
            for na, nb in zip(a, b):
                for p, (ia, ib) in enumerate(zip(na.images, nb.images)):
                    if ia != ib:
                        idx = [i for i in range(min(len(ia), len(ib))) if ia[i] != ib[i]]
                        d.append({"param": p, "bytes": idx[:32], "a": ia[idx[0]:idx[-1] + 1].hex() if idx else "", "b": ib[idx[0]:idx[-1] + 1].hex() if idx else ""})
            out["diffs"].append({"same_launch_keys": len(group), "key_a": str(k0)[:400], "key_b": str(k1)[:400], "differing_param_bytes": d[:12]})
        out["distinct_launch_configs"] = len(by_shape)
        return out

    def _plan_dump(self):
        from torch.cuda._host_trace_lower_tape import LoweredLaunch, PointerSlot
        from torch.cuda._host_trace_memory import plan_memory

        out = {}
        for key, e in self.entries.items():
            if key[0] != "decode":
                continue
            per = []
            for v in e.variants:
                lo = v.captured.lowered
                vals = lo.program.values
                plan = plan_memory(lo, "run_buffer")
                n_alloc = len(lo.allocations)
                uses = {}  # allocation -> [(seq, launch name)]
                desc = []
                for la in lo.launches:
                    if isinstance(la, LoweredLaunch):
                        for sl in la.slots:
                            if isinstance(sl, PointerSlot) and sl.base is not None and sl.base < n_alloc:
                                uses.setdefault(sl.base, []).append((la.seq, la.launch.name[:40]))
                        if la.launch.descriptors or la.launch.packed:
                            desc.append((la.seq, la.launch.name[:50], len(la.launch.descriptors), la.launch.packed,
                                         [(sl.root[0], sl.base) for sl in la.slots if isinstance(sl, PointerSlot)]))
                # the launches around the first packed one: their pointer slots' bases
                firstp = next((i for i, la in enumerate(lo.launches) if isinstance(la, LoweredLaunch) and la.launch.packed), None)
                around = []
                if firstp is not None:
                    for la in lo.launches[max(0, firstp - 6):firstp + 3]:
                        if isinstance(la, LoweredLaunch):
                            around.append((la.seq, la.launch.name[:40], [(sl.root[0], sl.base) for sl in la.slots if isinstance(sl, PointerSlot)]))
                        else:
                            around.append((getattr(la, "seq", None), type(la).__name__, [(sl.root[0], sl.base) for sl in getattr(la, "slots", ()) if isinstance(sl, PointerSlot)]))
                temps = []
                for i, st in enumerate(plan.steps):
                    for k, seq, last in st.temporaries:
                        nb = vals[lo.allocations[k].nbytes] if isinstance(lo.allocations[k].nbytes, int) and lo.allocations[k].nbytes < len(vals) else None
                        u = uses.get(k, [])
                        temps.append((i, k, seq, last, nb, len(u), u[-1] if u else None))
                live = {k: (seq, last) for st in plan.steps for k, seq, last in st.temporaries}
                first_use = {}
                for la in lo.launches:
                    if isinstance(la, LoweredLaunch):
                        for sl in la.slots:
                            if isinstance(sl, PointerSlot) and sl.base is not None and sl.base < n_alloc and sl.base not in first_use:
                                first_use[sl.base] = (la.seq, la.launch.name[:40])
                early = [(k, lo.allocations[k].seq, fu[0], fu[1], str(lo.allocations[k].sizes)[:40]) for k, fu in sorted(first_use.items())
                         if fu[0] < lo.allocations[k].seq]
                ops = [(i, getattr(op, "func", None) and str(op.func)[:40], getattr(op, "launches", None) and (op.launches.start, op.launches.stop))
                       for i, op in enumerate(getattr(v.tape, "ops", [])[:40])]
                per.append({"n_alloc": n_alloc, "n_launches": len(lo.launches), "temporaries": temps[:200], "packed_or_desc_launches": desc[:6],
                            "around_first_packed": around, "live_of_around": {str(b): live.get(b) for a in around for _, b in a[2] if b is not None},
                            "used_before_allocated": early[:40], "n_used_before_allocated": len(early), "tape_ops_head": ops,
                            "tape_alloc_seqs_head": [(k, a.seq) for k, a in enumerate(lo.allocations[:30])]})
            per.append({"entry_counters": {c: getattr(e, c, None) for c in ("traces", "respecs", "redispatches", "folds", "eager_sites", "oob_binds", "relowers")}})
            out[f"{key[0]}/T{key[1]}/{key[2]}"] = per
        return out

    def counters(self):
        es = list(self.entries.values())
        out = {k: sum(getattr(e, k, 0) for e in es) for k in ("traces", "replays", "eager", "uncaptured", "structural", "guards", "folds")}
        out["variants"] = sum(len(e.variants) for e in es)
        out["calls"] = dict(self.calls)
        return out

    def summary(self):
        import torch.cuda._host_trace_tape as T
        from torch.cuda._host_trace_launch import KernelLaunch

        res = {"families": list(FAMILIES), "decode_max_seq": DECODE_MAX_SEQ, "prefill_max_kv": PREFILL_MAX_KV, "metadata": "trace" if self.md_trace else "eager",
               "args": {"params": len(self.params), "attrs": len(self.attrs), "attr_names": sorted({f"{type(m).__name__}.{k}" for m, k, _ in self.attrs})}, "bad_keys": [str(k) for k in self.bad]}
        res["hostcuts"] = self.hostcuts
        res["replay_kw"] = dict(self.replay_kw)
        if MD_STOCK_DIAG:
            res["md_stock_why"] = dict(self.md_stock_why)
        res["pad"] = {"on": PAD, "rows": dict(getattr(self, "pad_rows", {}))}
        res.update(self.counters())
        res["harvests"] = self.provider.harvests
        res["bindings"] = len(self.provider.bindings)
        res["traced_impls"], res["trtllm_fork"] = TRACED_IMPLS, TRTLLM_FORK
        res["learned_by_op"] = dict(collections.Counter(str(k[1]) for k in self.provider.bindings).most_common())
        # replayed calls only: median and count
        res["host_ms"] = {k: (sorted(v)[len(v) // 2] * 1e3, len(v)) for k, v in self.host_s.items() if v}
        res["entry_bench"] = {str(k): v for k, v in self.entry_bench.items()}
        res["pre_ms"] = {k: (sorted(v)[len(v) // 2] * 1e3, len(v)) for k, v in self.pre_s.items() if v}
        res["entries"] = {}
        for key, e in self.entries.items():
            ops, whys, host_ops, per_variant = collections.Counter(), collections.Counter(), collections.Counter(), []
            for v in e.variants:
                k = n_e = o = n_h = 0
                vops, oops = collections.Counter(), collections.Counter()
                for _, rec in v.tape.launches:
                    if isinstance(rec, KernelLaunch):
                        k += 1
                    elif isinstance(rec, T.OpaqueCall):
                        o += 1
                        oops[rec.name] += 1
                    elif isinstance(rec, T.EagerCall) and rec.host:
                        n_h += 1
                        host_ops[rec.name] += 1
                    elif isinstance(rec, T.EagerCall):
                        n_e += 1
                        ops[rec.name] += 1
                        vops[rec.name] += 1
                        whys[(rec.name, (rec.reason or "no traced host")[:200])] += 1
                steps = v.captured.lowered.steps
                if DUMP_LAUNCH:  # diagnostics: the matching launches of each variant, with their neighbours' names
                    recs = [rec for _, rec in v.tape.launches]
                    dump = []
                    for i, rec in enumerate(recs):
                        if isinstance(rec, KernelLaunch) and DUMP_LAUNCH in rec.name:
                            near = [getattr(recs[j], "name", type(recs[j]).__name__)[:60] for j in (i - 1, i + 1) if 0 <= j < len(recs)]
                            dump.append({"i": i, "name": rec.name[:80], "grid": str(rec.grid), "block": str(rec.block), "smem": str(rec.smem),
                                         "slots": str(rec.slots)[:400], "programmatic": rec.programmatic, "attributes": str(rec.attributes)[:200], "near": near})
                    DUMPS.setdefault(f"{key[0]}/T{key[1]}/{key[2]}", []).append(dump[:6])
                per_variant.append({"kernels": k, "eager_calls": n_e, "host": n_h, "opaque": o,
                                    "boundaries": sum(not isinstance(s, range) and not s.call.host for s in steps),
                                    "segments": len(v.captured.segments), "learns": getattr(v, "learns", None), "eager_ops": dict(vops), "opaque_ops": dict(oops)})
            res["entries"][f"{key[0]}/T{key[1]}/{key[2]}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants), "slow_after_trace": self.slow[key], "slow_log": self.slow_log[key][:8],
                "eager_ops": dict(ops.most_common(30)), "host_ops": dict(host_ops.most_common(30)),
                "eager_reasons": [f"{n}x {op}: {why}" for (op, why), n in whys.most_common(20)],
                "per_variant": per_variant, "declines": sorted({str(d)[:400] for d in e.declines})[:30],
                "reasons": list(getattr(e, "_reasons", {}))[:20], "triton_fallbacks": list(e.triton_fallbacks)[:20],
                "retrace_causes": {str(c): n for c, n in getattr(e, "retrace_causes", {}).items()},
                "respecs": getattr(e, "respecs", None), "redispatches": getattr(e, "redispatches", None)}
        if DUMP_LAUNCH:
            res["launch_dump"] = DUMPS
        if os.environ.get("ARMV_DUMP_TRTLLM"):  # diagnostics: the closed trtllm decode bindings across max_seq_len values
            res["trtllm_dump"] = self._trtllm_dump()
        if os.environ.get("ARMV_DUMP_PLAN"):  # diagnostics: each decode variant's run_buffer plan and its launches' uses
            res["plan_dump"] = self._plan_dump()
        res["decline_sites"] = [f"{n}x {m} @ {w}" for (m, w), n in DECLINE_SITES.most_common(20)]
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["checks"] = self.checks
        return res
