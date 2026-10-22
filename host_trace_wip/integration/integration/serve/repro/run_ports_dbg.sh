#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.15 INTEG_KV_BYTES=4294967296
env CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m timeout 900 bash python_vllm_cand2.sh serve/repro/ports_dbg.py --arm V --model /data/eellison/models/Qwen3-8B --max-model-len 4096 --batch-size 1 --prefill-lens 64 --mixed 0 --out serve/out/ports_dbg.json 2>&1 | grep -v "^INFO\|WARNING" | tail -40
