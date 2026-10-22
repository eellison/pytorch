#!/usr/bin/env bash
# churnprof (main 10-14, untimed): the cbh4 churn at C = 64 (unsynced, async scheduling), vrs and bnc: torch.profiler timeline of churn
# steps 200-260 (DRIVE_CBPROF; mixed and size-changed decode steps among them), per-step phase clocks and node sets (ours), and every
# cuGraphExecKernelNodeSetParams in those steps (ours, DRIVE_SETLOG: no-op row switches counted). cbh4 configs (vrs: warm-range 1-3072/160,
# mixed warm-up, refreeze).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_CBPROF=200:260 DRIVE_IXPROF_DIR=$PWD/serve/out/prof_churn
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 --cb-steps 400 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 DRIVE_SETLOG=1 \
    DRIVE_PROFILE_TAG=vrs bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm V "${W[@]}" --warm-range 1 3072 160 --out vllm/out/t_churnprof_vrs_r1.json > serve/logs/churnprof_vrs_run.log 2>&1
echo "churnprof vrs rc=$?"
env CAND_LINE=c5h VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' DRIVE_PROFILE_TAG=bnc bash python_vllm_cand2.sh vllm/bench/drive_diag.py \
    --arm fullnc_full "${W[@]}" --out vllm/out/t_churnprof_bnc_r1.json > serve/logs/churnprof_bnc_run.log 2>&1
echo "churnprof bnc rc=$?"
