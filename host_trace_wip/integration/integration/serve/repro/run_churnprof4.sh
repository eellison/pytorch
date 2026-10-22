#!/usr/bin/env bash
# churnprof4 (main 10-14, untimed): churnprof3 (a) WITHOUT the set-logging shim (its per-call bookkeeping inflated churnprof2/3's patch
# phase): vcsl unsynced C = 64, per-step phase clocks + kernel packs / TMA counters, perf record -g --call-graph dwarf over churn steps 100-400.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_CBPROF=100:400 DRIVE_CBPERF=100:400 DRIVE_IXPROF_DIR=
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 --cb-steps 400 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_cbh4.pkl DRIVE_PROFILE_TAG=vcsl \
    bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 3072 160 --out vllm/out/t_churnprof4_vcsl_r1.json > serve/logs/churnprof4_vcsl_run.log 2>&1
echo "churnprof4 vcsl rc=$?"
