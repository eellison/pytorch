# Arm V host cuts (phase 6 step 2): V-only patches of the vLLM V2 runner's per-step host work, applied by ArmV.install()
# unless ARMV_HOSTCUTS=0. vLLM's own code is unchanged; the default arm never sees these.
#
# 1. STAGE (after SGLang change 11): prepare_inputs' per-step host arrays (idx_mapping, query_start_loc) go into one
#    pinned staging slot and reach the device in one memcpy into a fixed device buffer; idx_mapping and query_start_loc
#    are slices of it (input_buffers.query_start_loc is re-pointed at its slice). cu_num_logits and expanded_local_pos,
#    an arange and zeros without drafts, are slices of fixed device tensors. Replaces two pin_memory()+copy pairs, a
#    device allocation, an arange and a zeros. The no-draft path only (no spec decode, DCP, PCP, PP, R-SWA); the rest
#    goes to vLLM's prepare_inputs. Pinned slots rotate like vLLM's UvaBufferPool (max_concurrency). With
#    VLLM_MOE_SKIP_PADDING (vLLM's default) the is_padding buffer is a slice of the stage too, so its two fill_ launches
#    become host writes (only [:num_tokens_after_padding] is ever read).
# 2. Sampler.apply_staged_writes runs every step and copies about a dozen per-request arrays into UVA buffers even when
#    no request was added; every one of its writes is made by Sampler.add_request, so it is skipped until one is.
# 3. RequestState.apply_staged_writes applies total_len, all_token_ids and num_computed_tokens with one
#    _apply_write_kernel launch (MULTI_GROUP, as BlockTables does) whose index/content arrays are UVA buffers: no H2D
#    copy and no torch.tensor(list); three launches and three H2D copies before.
import time

import numpy as np
import torch

PROF = None  # bench/drive.py --stamps-fine: its STAMPS dict; STAGE prepare_inputs adds its sections ("pi.*") to it


def _mark(name, t):
    now = time.perf_counter()
    PROF[name] = PROF.get(name, 0.0) + now - t
    return now


def install(runner, uva=False):
    """uva (arm VP's single-launch step): the staging slots are pinned host buffers that the step's kernels read
    through their UVA views (no host-to-device memcpy); idx_mapping / query_start_loc / is_padding are slices of the
    slot's view, so they move with the slot."""
    import vllm.envs as envs
    import vllm.v1.worker.gpu.model_runner as mr
    from vllm.v1.worker.gpu.buffer_utils import _DEFAULT_MAX_CONCURRENCY, UvaBufferPool, _apply_write_kernel
    from vllm.v1.worker.gpu.input_batch import InputBatch

    r, dev = runner, runner.device
    M = r.max_num_reqs
    done = []

    # 1. STAGE
    if not (r.use_dcp or r.use_pp or r.pcp_manager is not None or r.model_config.rswa_window is not None):
        pad = envs.VLLM_MOE_SKIP_PADDING
        T = r.input_buffers.is_padding.numel() if pad else 0
        q0 = 8 * M
        p0 = (q0 + 4 * (M + 1) + 15) // 16 * 16
        K, nbytes = _DEFAULT_MAX_CONCURRENCY, p0 + T
        host = torch.zeros(K, nbytes, dtype=torch.uint8, pin_memory=True)
        h_idx = [host[k, :q0].numpy().view(np.int64) for k in range(K)]
        h_qsl = [host[k, q0:q0 + 4 * (M + 1)].numpy().view(np.int32) for k in range(K)]
        h_pad = [host[k, p0:].numpy().view(np.bool_) for k in range(K)]
        if uva:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            views = [get_accelerator_view_from_cpu_tensor(host[k]) for k in range(K)]
            slots_dev = [(v[:q0].view(torch.int64), v[q0:q0 + 4 * (M + 1)].view(torch.int32), v[p0:].view(torch.bool)) for v in views]
            stage = None
        else:
            stage = torch.zeros(nbytes, dtype=torch.uint8, device=dev)
            d_idx, d_qsl = stage[:q0].view(torch.int64), stage[q0:q0 + 4 * (M + 1)].view(torch.int32)
            r.input_buffers.query_start_loc = d_qsl
            if pad:
                r.input_buffers.is_padding = stage[p0:].view(torch.bool)
        d_arange = torch.arange(M + 1, dtype=torch.int32, device=dev)
        d_zeros = torch.zeros(M, dtype=torch.int32, device=dev)
        slot = [0]
        orig_pi = r.prepare_inputs
        ib_ = r.input_buffers
        rs = r.req_states

        def prepare_inputs(scheduler_output, batch_req_state, batch_desc):
            if scheduler_output.scheduled_spec_decode_tokens:
                return orig_pi(scheduler_output, batch_req_state, batch_desc)
            t = time.perf_counter() if PROF is not None else None
            num_tokens = batch_req_state.num_tokens
            assert num_tokens > 0
            req_ids, nst, idx_np = batch_req_state.req_ids, batch_req_state.num_scheduled_tokens, batch_req_state.idx_mapping_np
            n = len(req_ids)
            nrp = batch_desc.num_reqs or n
            k = slot[0] = (slot[0] + 1) % K
            h_idx[k][:n] = idx_np
            qsl_np = np.empty(M + 1, dtype=np.int32)
            qsl_np[0] = 0
            np.cumsum(nst, out=qsl_np[1:n + 1])
            qsl_np[n + 1:] = num_tokens
            h_qsl[k][:] = qsl_np
            num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
            if pad:
                h_pad[k][:num_tokens] = False
                h_pad[k][num_tokens:num_tokens_after_padding] = True
            if t is not None:
                t = _mark("pi.host_fill", t)
            if stage is not None:
                stage.copy_(host[k], non_blocking=True)
                s_idx, s_qsl = d_idx, d_qsl
            else:
                s_idx, s_qsl, s_pad = slots_dev[k]
                ib_.query_start_loc = s_qsl
                if pad:
                    ib_.is_padding = s_pad
            if t is not None:
                t = _mark("pi.memcpy", t)
            idx_mapping, qsl, cu_num_logits = s_idx[:n], s_qsl[:nrp + 1], d_arange[:n + 1]
            if batch_req_state.has_prefill:
                mr.prepare_prefill_inputs(ib_.input_ids, rs.next_prefill_tokens, idx_mapping, qsl, rs.all_token_ids.gpu, rs.prefill_len.gpu,
                                          rs.num_computed_tokens.gpu)
            mr.prepare_pos_seq_lens(idx_mapping, qsl, rs.num_computed_tokens.gpu, ib_.positions, ib_.seq_lens)
            seq_lens = ib_.seq_lens[:nrp]
            logits_indices = mr.combine_sampled_and_draft_tokens(ib_.input_ids, idx_mapping, rs.last_sampled_tokens, qsl, seq_lens, rs.prefill_len.gpu,
                                                                 rs.draft_tokens, cu_num_logits, n, r.model_state.num_new_sampled_tokens_per_step)
            if t is not None:
                t = _mark("pi.launches", t)
            nct_np = rs.num_computed_tokens_np[idx_np]
            ub = np.zeros(nrp, dtype=np.int32)
            np.add(nct_np, nst, out=ub[:n])
            if t is not None:
                t = _mark("pi.ub", t)
            ib = InputBatch(
                req_ids=req_ids, num_reqs=n, num_reqs_after_padding=nrp, idx_mapping=idx_mapping, idx_mapping_np=idx_np,
                expanded_idx_mapping=idx_mapping, expanded_local_pos=d_zeros[:n], num_scheduled_tokens=nst, num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding, num_draft_tokens=0, num_draft_tokens_per_req=None,
                query_start_loc=qsl, query_start_loc_np=qsl_np[:nrp + 1], seq_lens=seq_lens, seq_lens_cpu_upper_bound=torch.from_numpy(ub),
                dcp_local_seq_lens=None, num_computed_tokens_np=nct_np, prefill_len_np=batch_req_state.prefill_len_np,
                num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np, is_prefilling_np=batch_req_state.is_prefilling_np,
                has_prefill=batch_req_state.has_prefill, max_seq_len_np=None, input_ids=ib_.input_ids[:num_tokens_after_padding],
                positions=ib_.positions[:num_tokens_after_padding], is_padding=ib_.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices, cu_num_logits=cu_num_logits, cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
                has_structured_output_reqs=scheduler_output.has_structured_output_requests, prompt_lens=None, max_query_len=None)
            if t is not None:
                _mark("pi.input_batch", t)
            return ib
        r.prepare_inputs = prepare_inputs
        done.append("stage")

    # 2. sampler staged writes only after an add_request
    smp = r.sampler
    if smp is not None:
        dirty = [True]
        orig_add, orig_apply = smp.add_request, smp.apply_staged_writes

        def add_request(*args, **kw):
            dirty[0] = True
            return orig_add(*args, **kw)

        def apply_staged_writes():
            if dirty[0]:
                orig_apply()
                dirty[0] = False
        smp.add_request, smp.apply_staged_writes = add_request, apply_staged_writes
        done.append("sampler_dirty")

    # 3. one launch for RequestState's staged writes
    rs = r.req_states
    tensors = (rs.total_len, rs.all_token_ids, rs.num_computed_tokens)
    if all(t.dtype == torch.int32 for t in tensors):
        ptrs = torch.tensor([t.gpu.data_ptr() for t in tensors], dtype=torch.uint64, device=dev)
        strides = torch.tensor([t.gpu.stride(0) for t in tensors], dtype=torch.int64, device=dev)
        W, C = 3 * M, 1 << 20
        pools = {k: UvaBufferPool(W, torch.int32) for k in ("group", "index", "start", "cu")}
        contents = UvaBufferPool(C, torch.int32)
        orig_rs_apply = rs.apply_staged_writes

        def rs_apply_staged_writes():
            nw = sum(len(t._staged_write_indices) for t in tensors)
            nc = sum(len(t._staged_write_contents) for t in tensors)
            if nw > W or nc > C:
                return orig_rs_apply()
            rs.prompt_len.copy_to_uva()
            rs.prefill_len.copy_to_uva()
            if nw == 0:
                return
            group, index, start, cu = [], [], [], []
            pool = contents
            pool._curr = (pool._curr + 1) % pool.max_concurrency
            buf = pool._uva_bufs[pool._curr]
            off = 0
            for g, t in enumerate(tensors):
                m = len(t._staged_write_indices)
                if m == 0:
                    continue
                group += [g] * m
                index += t._staged_write_indices
                start += t._staged_write_starts
                cu += [off + c for c in t._staged_write_cu_lens]
                c = t._staged_write_contents
                buf.np[off:off + len(c)] = c
                off += len(c)
                t.clear_staged_writes()
            _apply_write_kernel[(nw,)](ptrs, strides, pools["index"].copy_to_uva(index), pools["start"].copy_to_uva(start), buf.uva[:off],
                                       pools["cu"].copy_to_uva(cu), pools["group"].copy_to_uva(group), BLOCK_SIZE=1024, MULTI_GROUP=True)
        rs.apply_staged_writes = rs_apply_staged_writes
        done.append("req_states_fused")
    return done
