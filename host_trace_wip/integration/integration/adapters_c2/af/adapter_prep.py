# Arm Af: stock SGLang (its CUDA graphs on, no host trace) with arm P's fused decode-prep Triton kernel run eagerly
# before SGLang's own forward. It stands in under driveF_prep.py's module name (ARMF_ADAPTER_DIR=adapters_c2/af,
# class PrepF) so the P drivers (PREP_SEQCHECK, PREP_STAMPS, PREP_ASYNC, profiles) run it unchanged.
# For an eligible plain decode batch, one_batch.decode is replaced by: prepare_for_decode's host bookkeeping exactly as
# arm P does it (armF/prep/adapter_prep.py PrepF.decode), then _prep_decode_kernel (a verbatim copy of P's kernel)
# eagerly in place of prep kernels #1-#10 (last_loc sub/gather, add, alloc_decode_kernel, clone, int cast, req_to_token
# index_put, seq_lens + 1, orig_seq_lens + 1, clamp_position), then SGLang's stock ForwardBatch.init_new (its
# clamp_position call returns the kernel's positions for this step's seq_lens), ModelRunner.forward (cuda graph runner:
# _small_copy_kernel, update_trtllm_mha_graph_metadata_kernel, replay) and the eager sampler. Other batches run
# SGLang's decode unchanged. Nothing from torch.cuda._host_trace is imported.
import collections

import torch
import triton
import triton.language as tl

BRK = None  # stamps_prep.py reads adapter_prep.BRK


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


class PrepF:
    def __init__(self, model_runner):
        self.mr = model_runner
        self.model = model_runner.model
        self.calls = collections.Counter()
        self.checks = []  # no in-process check: bitwise is PREP_SEQCHECK against an eager run
        self.check = False
        self.entries = {}
        self.eligible = {}  # (batch, reqs, len, sampling_info ids) -> bool
        self.orig_decode = None
        self._pos = None  # (this step's new seq_lens tensor, the kernel's positions)
        self._free_view = (None, None)  # (allocator free_pages object, its [:bs] view)

    def install(self):
        import sglang.benchmark.one_batch as ob
        import sglang.srt.model_executor.forward_batch_info as fbi
        from sglang.srt.mem_cache.allocation import alloc_token_slots
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        self._alloc_token_slots, self._evict_from_tree_cache = alloc_token_slots, evict_from_tree_cache
        orig_clamp = fbi.clamp_position

        def clamp_position(seq_lens):
            p = self._pos
            return p[1] if p is not None and p[0] is seq_lens else orig_clamp(seq_lens)

        fbi.clamp_position = clamp_position
        self.orig_decode = ob.decode
        ob.decode = self.decode

    def _eligible(self, batch, mr):
        """arm P's eligibility (PrepF._eligible), verbatim."""
        from sglang.srt.managers import schedule_batch as sbm
        from sglang.srt.mem_cache.allocation import _alloc_page_size
        import sglang.benchmark.one_batch as ob

        si, tc = batch.sampling_info, batch.tree_cache
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
            self.calls["af_ineligible:" + ",".join(bad)] += 1
        return not bad

    def decode(self, tok, batch, mr):
        from sglang.srt.mem_cache.allocation import _alloc_page_size
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

        k = (id(batch), id(batch.reqs), len(batch.reqs), id(batch.sampling_info))
        ok = self.eligible.get(k)
        if ok is None:
            ok = self.eligible[k] = self._eligible(batch, mr)
        if not ok:
            self.calls["decode"] += 1
            return self.orig_decode(tok, batch, mr)
        self.calls["af"] += 1
        bs = len(batch.reqs)
        page = _alloc_page_size(batch)
        paged = page > 1
        # prepare_for_decode's host part, as arm P (PrepF.decode) does it
        batch.forward_mode = ForwardMode.DECODE
        batch.mamba_track_seqlens_cpu = batch.mamba_prefill_track_mask_cpu = batch.input_embeds = None
        batch.mamba_cow_src_indices = batch.mamba_cow_dst_indices = batch.mamba_clear_indices = None
        if getattr(batch, "attn_cp_metadata", None) is not None:
            batch.attn_cp_metadata = None
        tc = batch.tree_cache
        nxt = batch.seq_lens_cpu + 1
        nl = nxt.tolist()
        if paged:
            alloc = tc.token_to_kv_pool_allocator
            if tc is not None and not tc.is_chunk_cache() and bs * page > alloc.available_size():
                self._evict_from_tree_cache(tc, bs * page)
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
            view = self._alloc_token_slots(tc, bs)
        batch.req_to_token_pool.alloc_aux_to_lengths(req_pool_indices_cpu=batch.req_pool_indices_cpu, target_seq_lens_cpu=nxt)
        for req in batch.reqs:
            req.decode_batch_idx += 1
            kv = req.kv
            kv.kv_allocated_len += 1
            kv.kv_committed_len += 1
        batch.seq_lens_cpu = nxt
        batch.seq_lens_sum = sum(nl)
        # prep kernels #1-#10 as one eager launch (arm P runs the same kernel inside its trace)
        seq, orig = batch.seq_lens, batch.orig_seq_lens
        ids = torch.empty(bs, dtype=torch.int64, device=seq.device)
        new_seq, new_orig, pos = torch.empty_like(seq), torch.empty_like(orig), torch.empty_like(seq)
        loc = torch.empty(bs, dtype=torch.int64, device=seq.device) if paged else view
        r2t = batch.req_to_token_pool.req_to_token
        _prep_decode_kernel[(bs,)](tok, ids, seq, new_seq, orig, new_orig, pos, batch.req_pool_indices, r2t, r2t.stride(0), view, loc,
                                   BLOCK=triton.next_power_of_2(bs), PAGE=page, PAGED=paged)
        batch.input_ids, batch.seq_lens, batch.orig_seq_lens, batch.out_cache_loc = ids, new_seq, new_orig, loc
        # one_batch.decode from here on, unchanged
        self._pos = (new_seq, pos)
        try:
            forward_batch = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
        finally:
            self._pos = None
        logits_output = mr.forward(forward_batch).logits_output
        next_token_ids = mr.sample(logits_output, forward_batch)
        return next_token_ids, logits_output.next_token_logits

    def counters(self):
        return {"traces": 0, "replays": 0, "eager": 0, "uncaptured": 0, "structural": 0, "guards": 0, "folds": 0, "variants": 0,
                "calls": dict(self.calls)}

    def summary(self):
        return {"arm": "Af", "calls": dict(self.calls), "traces": 0, "replays": 0, "eager": 0, "variants": 0, "entries": {},
                "refused": [], "decline_sites": [], "checks": []}
