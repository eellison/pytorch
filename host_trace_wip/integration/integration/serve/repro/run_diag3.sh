#!/usr/bin/env bash
# diag3 (10-14, untimed): main's three questions on swph1. ARM=vcsl|default. Fixed bs 37/63/65 and alternating 63<->65 at the
# swph1 alternating KV (1024-2048) with torch.profiler traces of fixed steps 12-13 and alternating steps 24-27 (DRIVE_IXPROF_DIR),
# plus the ~1K decode curve at bs 29 and 63 (128 steps); every step's cyclic garbage attributed to its step class with our trace /
# learn / slow-call counter deltas (DRIVE_GC_STEPS). swph1's configs otherwise (vcsl: warm-range + freeze, bindings preloaded).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${ARM:?}
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_IXPROF_DIR=$PWD/serve/out/prof_diag3 DRIVE_PROFILE_TAG=$ARM DRIVE_GC_STEPS=1
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
   --dec-ctx 1100 --dec-ctx-sizes 29 63 --dec-ctx-steps 128
   --ix-decode 37:1024-2048 63:1024-2048 65:1024-2048 --alt-decode 63:65@1024-2048 --ix-reps 6)
BASE=(CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
case $ARM in
  vcsl) env "${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 DRIVE_SETLOG=1 DRIVE_ENTRY_SYNC_DEBUG=1 \
          ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 2048 128 \
          --out vllm/out/t_diag3_vcsl_r2.json > serve/logs/diag3_vcsl_r2_run.log 2>&1 ;;
  default) env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm default "${W[@]}" \
          --out vllm/out/t_diag3_default_r1.json > serve/logs/diag3_default_run.log 2>&1 ;;
esac
echo "diag3 $ARM rc=$?"
