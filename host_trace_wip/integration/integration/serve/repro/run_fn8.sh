#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 bring-up (GPU 1 correctness), recipe single-GB300 args minus graphs (the eager kernels both arms
# fn8 (diagnosis): reshape_and_cache_flash left eager (not harvested), to look past core gap_5 (the cache write inside
# vllm.qwen4_exp_qsa_with_output claimed by a keyed site and the op's selector).
# backend FLASHINFER_TRTLLM needs trtllm-gen batched_gemm cubins not on disk -> --moe-backend flashinfer_cutlass (JIT, no download).
# run): max_num_seqs 32, max_num_batched_tokens 8192, fp8 KV, indexer_kv_dtype fp8, no FlashInfer autotune, text-only,
# no prefix caching (InferenceX fixed-seq); fn1 = stock eager, fn2 = V-stock (pinned + trtllm-gen fork, serve/armV_stock)
# with --check (bitwise vs stock eager on the same metadata, NaN-aware, recurrent state compared).
M=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false}'
export CAND_LINE=pinned CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
L=python_vllm_cand2.sh
echo "gpu1 used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
ARMV_EXTERN_DROP=reshape_and_cache_flash bash $L serve/r1_probe.py --out serve/out/fn8.json --adapter-dir armV_stock --model $M --max-model-len 16384 --kv-gib 30 --kv-dtype auto --moe-backend flashinfer_cutlass --cute-observe --sym-origins --check --llm-kw "$KW"
echo "fn8 rc=$?"
