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
METADATA = os.environ.get("ARMV_METADATA", "trace")
ENTRY_BENCH = bool(os.environ.get("ARMV_ENTRY_BENCH"))  # time 50 extra (idempotent) calls of each key's entry at its 12th replay
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


def _kv_indptr(seq_lens, page, out):
    """out[0] = 0, out[1:] = cumsum(ceil(seq_lens / page)) in one Triton launch (sglang/armF change 9): torch.cumsum's
    two CUB kernels are an eager boundary of the trace."""
    n = seq_lens.shape[0]
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


class _FormMismatch(Exception):
    pass


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
        self.md_trace = METADATA == "trace" and runner.speculator is None and runner.pcp_manager is None and len(runner.attn_groups) == 1 and len(runner.attn_groups[0]) == 1
        if self.md_trace:
            # the runner buffers the prep's kernels read and write: block-table pointer tables (the gather's source rows
            # are read through block_table_ptrs only), the gathered tables, the slot mappings and the builder's indptr
            bt = runner.block_tables
            self.builder = runner.attn_groups[0][0].get_metadata_builder(0)
            assert bt.num_kv_cache_groups == 1
            self.attrs += [(bt, k, getattr(bt, k)) for k in ("block_table_ptrs", "block_table_strides", "input_block_table_ptrs", "slot_mappings",
                                                             "block_sizes_tensor", "kernel_block_sizes_tensor", "slot_mapping_enabled")]
            self.attrs += [(bt.input_block_tables, 0, bt.input_block_tables[0]), (self.builder.paged_kv_indptr, "gpu", self.builder.paged_kv_indptr.gpu)]
        self.prefix = (*self.params, *(v for _, _, v in self.attrs))
        self.bad = set()  # traced-metadata keys whose metadata form the step does not cover
        self.tmpl = {}  # traced-metadata key -> vLLM's metadata of the key's first call (policies applied)
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
        self.entry_bench = {}
        self.pre_s = collections.defaultdict(list)  # kind -> of host_s, the seconds before the entry's call (traced metadata)
        self.check = False  # compare each call against the model's own eager forward
        self.checks = []
        self._ctx = None

    def install(self):
        for layer in self.layers:
            layer.use_direct_call = True
        self.model.forward = self.forward
        if self.md_trace:
            import vllm.v1.worker.gpu.model_runner as mr

            r, ms = self.runner, self.runner.model_state
            self.prep = (r.prepare_attn, ms.prepare_attn, mr.build_slot_mappings_by_layer)
            orig_attn, orig_md, orig_slots = self.prep

            def prepare_attn(input_batch):
                p = _Pending(input_batch)
                return p, p

            def build_slot_mappings_by_layer(slot_mappings, kv_cache_config):
                return slot_mappings if isinstance(slot_mappings, _Pending) else orig_slots(slot_mappings, kv_cache_config)

            def ms_prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture=False):
                if not isinstance(block_tables, _Pending):
                    return orig_md(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture=for_capture)
                block_tables.md_args = (cudagraph_mode, attn_groups, kv_cache_config, for_capture)
                return block_tables

            r.prepare_attn = prepare_attn
            ms.prepare_attn = ms_prepare_attn
            mr.build_slot_mappings_by_layer = build_slot_mappings_by_layer

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
            self._check(kind, out, fc, input_ids, positions, amd, sm, stock)
        return out

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
        key = (kind, input_ids.shape[0], ints, "md")
        if key in self.bad:
            return self._eager("eager_metadata_form", fc, input_ids, positions)
        tmpl, verify = self.tmpl.get(key), None
        if tmpl is None:
            # vLLM's own build, eagerly, once per key: the form and ints the step's metadata copies
            amd, sm = self._live(fc)
            md = next(iter(amd.values()))
            if not self._form_ok(amd, sm, md, next(iter(sm.values()))) or self._md_ints(md) != ints[:6]:
                self.bad.add(key)
                return self._eager("eager_metadata_form", fc, input_ids, positions)
            tmpl = verify = self.tmpl[key] = md
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step_md(key), opaque=(self.provider,), memory=MEMORY)
        self.calls[kind] += 1
        bt = self.runner.block_tables
        args = [*self.prefix, input_ids, positions, ib.idx_mapping, ib.query_start_loc, ib.seq_lens, bt.num_blocks.gpu]
        self._ctx = (fc, p, tmpl, verify)
        replays = entry.replays
        t1 = time.perf_counter()
        try:
            (out,) = entry(*args)
        except _FormMismatch:  # the step's metadata is not vLLM's (raised at the entry's first, eager, call)
            self.bad.add(key)
            self.calls[kind] -= 1
            return self._eager("eager_metadata_form", fc, input_ids, positions)
        finally:
            self._ctx = None
        if entry.replays > replays:
            self.host_s[f"{kind}/T{key[1]}"].append(time.perf_counter() - t0)
            self.pre_s[f"{kind}/T{key[1]}"].append(t1 - t0)
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
            del entry._call_slow
        ts.sort()
        return {"median_us": round(ts[25] * 1e6, 1), "p10_us": round(ts[5] * 1e6, 1), "slow_calls": slow[0], "variants": len(entry.variants),
                "nargs": len(args), "learns": [getattr(v, "learns", None) for v in entry.variants]}

    @property
    def cg_none(self):
        from vllm.config import CUDAGraphMode

        return CUDAGraphMode.NONE

    @staticmethod
    def _same_inputs(ib, input_ids, positions):
        """The model's input_ids and positions are the input batch's (no M-RoPE positions, no padding)."""
        return ib.positions is positions and ib.input_ids is input_ids and ib.num_tokens == input_ids.shape[0] == ib.num_tokens_after_padding and ib.num_reqs == ib.num_reqs_after_padding

    def _check(self, kind, out, fc, input_ids, positions, amd, sm, stock):
        from vllm.forward_context import override_forward_context

        got = out.clone()
        torch.cuda.synchronize()
        lfc = copy.copy(fc)
        lfc.attn_metadata, lfc.slot_mapping = amd, sm
        with override_forward_context(lfc):
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

        kind, T, ints, _ = key
        nd, n = ints[1], ints[1] + ints[3]
        n_p, n_a = len(self.params), len(self.attrs)
        page = self.builder.page_size
        layers = self.runner.attn_groups[0][0].layer_names
        proxy = self

        def metadata(tmpl, block_tables, slot_mappings, seq_lens, qo):
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
            input_ids, positions, idx_mapping, query_start_loc, seq_lens, num_blocks = flat[n_p + n_a:]
            ib = copy.copy(pending.input_batch)
            ib.input_ids, ib.positions, ib.idx_mapping, ib.query_start_loc, ib.seq_lens = input_ids, positions, idx_mapping, query_start_loc, seq_lens
            nb = proxy.runner.block_tables.num_blocks
            saved = proxy._swapped(attrs) + [(nb, "gpu", nb.gpu)]
            nb.gpu = num_blocks
            try:
                block_tables, slot_mappings = proxy.prep[0](ib)
                md = metadata(tmpl, block_tables, slot_mappings, seq_lens[:n], query_start_loc[:n + 1])
                if live is not None:
                    verify(md, live)
                sfc = copy.copy(fc)
                sfc.attn_metadata, sfc.slot_mapping = dict.fromkeys(layers, md), dict.fromkeys(layers, md.slot_mapping)
                with override_forward_context(sfc), _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions)
            finally:
                for m, k, v in saved:
                    _set(m, k, v)
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

        res = {"families": list(FAMILIES), "decode_max_seq": DECODE_MAX_SEQ, "prefill_max_kv": PREFILL_MAX_KV, "metadata": "trace" if self.md_trace else "eager",
               "args": {"params": len(self.params), "attrs": len(self.attrs), "attr_names": sorted({f"{type(m).__name__}.{k}" for m, k, _ in self.attrs})}, "bad_keys": [str(k) for k in self.bad]}
        res.update(self.counters())
        res["harvests"] = self.provider.harvests
        res["bindings"] = len(self.provider.bindings)
        # replayed calls only: median and count
        res["host_ms"] = {k: (sorted(v)[len(v) // 2] * 1e3, len(v)) for k, v in self.host_s.items() if v}
        res["entry_bench"] = {str(k): v for k, v in self.entry_bench.items()}
        res["pre_ms"] = {k: (sorted(v)[len(v) // 2] * 1e3, len(v)) for k, v in self.pre_s.items() if v}
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
