#!/usr/bin/env bash
# Qwen3.8-Flash-Next V-stock on int6 step280 (candidate 4 even top) + FZ12 (serve/wt_s280fz12 on build_cpp9; launcher
# serve/python_vllm_s280.sh), --check bitwise vs stock eager, counts, retrace causes. Same args as fn5 (bf16 KV,
# --moe-backend flashinfer_cutlass, recipe single-GB300 sizes). No trtllm fork (Flash-Next's attention is vLLM's QSA Triton).
# fs2: vLLM's top-k _C ops (cooperative_topk / persistent_topk, the QSA indexer) harvested like the other _C ops.
# and the metadata host tensors as meta tensors (ARMV_HOST_ARGS=meta, r1gaps): only what the region reads is lifted.
M=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false}'
export CAND_LINE=cpp ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
echo "gpu1 used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
ARMV_HOST_ARGS=meta ARMV_EXTRA_EXTERN=scaled_fp4_quant.out,static_scaled_fp8_quant.default,cooperative_topk.default,persistent_topk.default bash serve/python_vllm_s280.sh serve/r1_probe.py --out serve/out/fs2.json --adapter-dir armV_stock --model $M --max-model-len 16384 --kv-gib 30 --kv-dtype auto --moe-backend flashinfer_cutlass --cute-observe --sym-origins --check --prompts many --llm-kw "$KW"
echo "fs2 rc=$?"
