#!/usr/bin/env bash
# smoke (10-15, untimed, GPU 3 shared): does vcsl run as-is on head ms7 (snap/step_ms6 + build_cpp_ms7), split off and split 64? Short
# unsynced churn (C = 16, 60 measured steps) after a small warm-up, small KV (20 GiB) so it fits beside the other lanes' processes.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.40 INTEG_KV_BYTES=21474836480 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_EMPTY_CACHE=1 DRIVE_HOSTFLAG_FAIL=0
BASE=(CAND_LINE=ms7 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1)
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 --cb-steps 60 --cb-warm 20 --cb-out 64 256 --cb-prompt 64 512 --warm-range 1 256 32)
for sp in -1 64; do
  env "${BASE[@]}" DRIVE_SPLIT=$sp timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/t_smoke_ms7_split$sp.json > serve/logs/smoke_ms7_split$sp.log 2>&1
  echo "smoke ms7 split=$sp rc=$?"
done
