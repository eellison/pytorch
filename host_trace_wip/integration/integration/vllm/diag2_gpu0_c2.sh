#!/usr/bin/env bash
# Open-variant NaN: the fork with one non-cache port at a time (rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul).
source $(dirname $0)/timing_lib.sh
D=(--arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0)
for n in rms_norm fused_add_rms_norm rotary_embedding silu_and_mul; do
  trun diag_fork_$n CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=$n -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_fork_$n.json
done
echo ALLDONE
