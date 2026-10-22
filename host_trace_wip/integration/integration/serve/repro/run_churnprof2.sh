#!/usr/bin/env bash
# churnprof2 (main 10-14, untimed): the churn truly unsynced (the forward wrappers no longer sync in NOSYNC windows), C = 64, vcsl and
# default: torch.profiler timeline of churn steps 200-260 (where the host blocks, our phases vs the wait, GPU gaps between steps), and
# for vcsl every cuGraphExecKernelNodeSetParams over steps 300-360 with its wall duration, plus per step whether the stream was still
# busy at execute_model entry and at our forward call (no profiler in that window).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_CBPROF=200:260 DRIVE_CBSETLOG=300:360 DRIVE_IXPROF_DIR=$PWD/serve/out/prof_churn2
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 --cb-steps 400 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 DRIVE_SETLOG=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_cbh4.pkl DRIVE_PROFILE_TAG=vcsl \
    bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 3072 160 --out vllm/out/t_churnprof2_vcsl_r1.json > serve/logs/churnprof2_vcsl_run.log 2>&1
echo "churnprof2 vcsl rc=$?"
env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_PROFILE_TAG=default bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm default "${W[@]}" \
    --out vllm/out/t_churnprof2_default_r1.json > serve/logs/churnprof2_default_run.log 2>&1
echo "churnprof2 default rc=$?"
