#!/usr/bin/env bash
# The run_buffer NaN repro (fork + silu port, decode bs 5/8/64, --check) on candidate 3: the int6 even top step246
# (build_cpp8) and the pinned lane (step242-based). Respec is removed there; check_plan rejects bad intervals.
source $(dirname $0)/timing_lib.sh
D=(--arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0)
for line in c3 pinned; do
  trun c3nan_$line CAND_LINE=$line "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=silu_and_mul ARMV_DUMP_PLAN=1 -- vllm/bench/drive.py "${D[@]}" --out vllm/out/c3nan_$line.json
done
echo ALLDONE
