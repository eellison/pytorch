#!/usr/bin/env bash
# run_buffer NaN (fork + silu port): each decode variant's run_buffer plan (temporaries, their recorded last use and
# users) and the packed/descriptor launches, with the silu port and without it (fork only).
source $(dirname $0)/timing_lib.sh
D=(--arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0)
trun diag_plan3_fork_silu CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=silu_and_mul ARMV_DUMP_PLAN=1 -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_plan3_fork_silu.json
trun diag_plan3_fork_cache CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=reshape_and_cache_flash ARMV_DUMP_PLAN=1 -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_plan3_fork_cache.json
echo ALLDONE
