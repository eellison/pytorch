#!/usr/bin/env bash
# synccheck (main 10-14, untimed): CUDA synchronize calls per step in swph1's timed configuration, every arm: one profiled full step
# (execute_model .. sample_tokens; DRIVE_PROFILE_DIR) of prefill 512 and of mixed 1024 + 64 decodes at 1-2K KV, at the 5th occurrence.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_PROFILE_DIR=$PWD/serve/out/prof_synccheck DRIVE_PROFILE_CLASSES=1:512,65:1088 DRIVE_PROFILE_AT=5
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 512 --prefill-reps 9 --mixed 0 --ix-mixed 1024x64@1024-2048 --ix-reps 6)
BASE=(CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1)
for arm in default fullnc bnc vcsl vrs; do
  case $arm in
    default) env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_PROFILE_TAG=$arm bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/t_synccheck_$arm.json ;;
    fullnc) env CAND_LINE=c5h DRIVE_PROFILE_TAG=$arm bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/t_synccheck_$arm.json ;;
    bnc) env CAND_LINE=c5h VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' DRIVE_PROFILE_TAG=$arm bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/t_synccheck_$arm.json ;;
    vcsl) env "${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl DRIVE_PROFILE_TAG=$arm bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --warm-range 1 1100 8 --out vllm/out/t_synccheck_$arm.json ;;
    vrs) env "${BASE[@]}" DRIVE_PROFILE_TAG=$arm bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm V "${W[@]}" --warm-range 1 1100 8 --out vllm/out/t_synccheck_$arm.json ;;
  esac > serve/logs/synccheck_$arm.log 2>&1
  echo "synccheck $arm rc=$?"
done
