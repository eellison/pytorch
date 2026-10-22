#!/usr/bin/env bash
# q1p memory question: where does V-stock's +473 MiB bench peak (2356 vs default-nc 1883 MiB over init; end-of-run held memory equal,
# 49.8 MiB in every arm) come from? The q1p V-stock workload with ARMV_MEM_PROBE=1: peak allocated over each HostTraceReplay call's
# start, by call kind (trace / replay / eager).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.6
W=(--model /data/eellison/models/Qwen3.5-9B --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 8 --prefill-lens 64 --mixed 0 --async-decode 1 8
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000
              4096x64@4096-8000 8192x32@4096-8000 8192x64@4096-8000 8192x128@4096-8000
   --ix-decode 64:6000-8000 128:6000-8000 256:6000-8000 --ix-reps 6)
env CAND_LINE=trtcpp ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_HOST_ARGS=pinned \
    INTEG_ARMV=$PWD/vllm/armV_stock_fast_cpp_m ARMV_MEM_PROBE=1 bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V "${W[@]}" --out serve/out/q1p_mem.json
echo "q1p_mem rc=$?"
python3 -c "import json; d=json.load(open('serve/out/q1p_mem.json')); print('Q1PMEM', d.get('mem_probe'), d['mem'])"
