#!/usr/bin/env bash
# swpe (serve lane, 10-15, main: the talk uses only the compile-OFF forward): swph1's eager decode curves rerun, ~1.1K and ~8.3K context,
# every bs 1-128, 128 measured full-batch steps per point (swph1's grid and harness). Arms fullnc (vLLM FULL decode graphs), bnc (vLLM
# breakable graphs), vrs (ours, eager forward + sampling in the step; warm-range + freeze + ARMV_REFREEZE, strict windows). Every arm:
# DRIVE_GC_LOG (gc.callbacks: every collection's generation and ms), torch.cuda.empty_cache() once before the first measured point, per-point
# tail (p99, max, steps > 10x the median with the GC passes inside them, PSI) and flag (PSI > 2% of the point or outlier excess > 1%).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_GC_LOG=1 DRIVE_EMPTY_CACHE=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
VR=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_REFREEZE=1)
SZ=($(seq 1 ${SWPE_BSMAX:-128}))
W=(--max-model-len 16384 --batch-size 1 --decode-tokens 8 --dec-ctx ${SWPE_CTX:-1100 8300} --dec-ctx-sizes "${SZ[@]}" --dec-ctx-steps ${SWPE_STEPS:-128} --prefill-lens 64 --mixed 0)
arms=(${ARMS:-fullnc bnc vrs}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  t=t_${SET}_${a}_r$R
  case $a in
    fullnc) trun $t CAND_LINE=$L -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/$t.json ;;
    bnc) trun $t CAND_LINE=$L VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;
    vrs) trun $t "${VR[@]}" ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --warm-range ${SWPE_WARM:-1 2048 128} --out vllm/out/$t.json ;;
  esac
done
echo ALLDONE
