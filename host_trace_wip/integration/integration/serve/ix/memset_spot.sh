#!/usr/bin/env bash
# memset spot check (main 10-14, untimed): vcsl config (warm-range, bindings preloaded); the memset widths our decode / mixed replays
# hold (land/core/memset/check_held_memsets.py, DRIVE_MEMSET_CHECK) at decode bs 3-100 and mixed 512 x 16 / 64, compared with
# FlashInfer's round_up(max(bs * 32, 152), 8) * 4 by serve/ix/memset_spot_table.py.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_MEMSET_CHECK=1
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc --model /data/eellison/models/Qwen3-8B \
    --max-model-len 16384 --batch-size 3 6 13 20 27 40 50 60 100 --decode-tokens 8 --prefill-lens 64 --mixed 0 \
    --ix-mixed 512x16@1024-2048 512x64@1024-2048 --ix-reps 2 --warm-range 1 2048 128 --out vllm/out/t_memspot_vcsl_r1.json > serve/logs/memspot_vcsl_run.log 2>&1
echo "memspot rc=$?"
python3 serve/ix/memset_spot_table.py vllm/out/t_memspot_vcsl_r1.json
