#!/usr/bin/env bash
# Open-variant NaN diagnostics (fork + traced impls) inside a GPU 0 hold (GPU 1 is reserved for candidate 3's gates):
# PDL tracing / programmatic edges off, and the ports bisected under the fork.
source $(dirname $0)/timing_lib.sh
D=(--arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0)
trun diag_both CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_both.json
trun diag_both_nopdl CAND_LINE=cpp "${OPEN[@]}" ARMV_HT_SET=torch.cuda._host_trace.trace_pdl=0,torch.cuda._host_trace_capture.keep_programmatic_edges=0 -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_both_nopdl.json
trun diag_fork_cache CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=reshape_and_cache_flash -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_fork_cache.json
trun diag_fork_nocache CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=rms_norm,fused_add_rms_norm,rotary_embedding,silu_and_mul -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_fork_nocache.json
echo ALLDONE
