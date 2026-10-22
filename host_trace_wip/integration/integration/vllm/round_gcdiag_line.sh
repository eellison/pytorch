#!/usr/bin/env bash
# gcdiag (serve lane, 10-14): untimed GC diagnostic: vcsl (freeze, both helpers) over the ~1K-context decode curve bs 1-64 with
# DRIVE_GC_DIAG=1 (DEBUG_SAVEALL + per-pass garbage histogram, allocation sites, referrers, finalizer / deallocation time).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
VC=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1)
trun t_gcdiag_vcsl_r1 "${VC[@]}" ARMV_VPS=1 DRIVE_GC_FREEZE=1 DRIVE_GC_DIAG=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl -- vllm/bench/drive_sweep.py --arm Vc --max-model-len 16384 --batch-size 1 --decode-tokens 8 --dec-ctx 1100 --dec-ctx-sizes $(seq 1 64) --dec-ctx-steps 128 --prefill-lens 64 --mixed 0 --warm-range 1 2048 128 --out vllm/out/t_gcdiag_vcsl_r1.json
echo ALLDONE
