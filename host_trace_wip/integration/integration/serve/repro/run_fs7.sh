#!/usr/bin/env bash
# fs7: fs6 + ARMV_EAGER_KINDS=prefill,mixed (fs6: a prefill replay left SharedExperts._output set -> AssertionError). fs6 was fs5 (Flash-Next V-stock --check on candidate 4) with serve/armV_stock_m: the unpinned metadata host tensors' meta
# stand-ins made outside the trace (fs5's prefill traces all declined "an allocation on meta in a trace on cuda:0").
export CAND_LINE=c4 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 240000 ]; do sleep 60; done
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false}'
ARMV_EAGER_KINDS=prefill,mixed ARMV_EXTRA_EXTERN=scaled_fp4_quant.out,static_scaled_fp8_quant.default,cooperative_topk.default,persistent_topk.default bash python_vllm_cand2.sh serve/r1_probe.py --out serve/out/fs7.json --adapter-dir armV_stock_m --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --max-model-len 16384 --kv-gib 30 --kv-dtype auto --cute-observe --sym-origins --check --prompts many --llm-kw "$KW"
echo "fs7 rc=$?"
