#!/usr/bin/env bash
# cprof (main 10-14, untimed): cProfile of execute_model .. sample_tokens, 3 bs-63 decode steps at the ~1.1K curve point and at fixed
# bs 63 at the alternating KV, ours (vcsl config) and default (DRIVE_CPROF).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CPROF=1
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
   --dec-ctx 1100 --dec-ctx-sizes 63 --dec-ctx-steps 32 --ix-decode 63:1024-2048 --ix-reps 6)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 2048 128 \
    --out vllm/out/t_cprof_vcsl_r1.json > serve/logs/cprof_vcsl_run.log 2>&1
echo "cprof vcsl rc=$?"
env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm default "${W[@]}" --out vllm/out/t_cprof_default_r1.json > serve/logs/cprof_default_run.log 2>&1
echo "cprof default rc=$?"
