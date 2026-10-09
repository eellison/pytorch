#!/usr/bin/env bash
# Open-variant NaN = trtllm fork + the silu_and_mul port: dump each variant's silu launches (grid/block/slots).
source $(dirname $0)/timing_lib.sh
trun diag_fork_silu_dump CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=silu_and_mul ARMV_DUMP_LAUNCH=silu -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_fork_silu_dump.json
trun diag_silu_only_dump CAND_LINE=cpp ARMV_TRACED_IMPLS=1 ARMV_TRACED_IMPLS_NAMES=silu_and_mul ARMV_DUMP_LAUNCH=silu -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_silu_only_dump.json
echo ALLDONE
