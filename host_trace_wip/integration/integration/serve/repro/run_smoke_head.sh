#!/usr/bin/env bash
# smoke (10-14, untimed, about 2-3 min per arm): do vcsl / vrs run as-is on the new candidate head (CAND_LINE=ms3: snap/step_ms2b +
# build_cpp_ms3)? A short churn (C = 16, 60 measured steps) after a small warm-up; prints SMOKE_HEAD_OK when both exit 0.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1
BASE=(CAND_LINE=ms3 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1)
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 --cb-steps 60 --cb-warm 20 --cb-out 64 256 --cb-prompt 64 512 --warm-range 1 256 32)
env "${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/t_smoke_ms3_vcsl.json > serve/logs/smoke_ms3_vcsl.log 2>&1
r1=$?
env "${BASE[@]}" timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/t_smoke_ms3_vrs.json > serve/logs/smoke_ms3_vrs.log 2>&1
r2=$?
echo "smoke ms3 vcsl rc=$r1 vrs rc=$r2"
[ $r1 = 0 ] && [ $r2 = 0 ] && echo SMOKE_HEAD_OK
