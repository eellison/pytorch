#!/usr/bin/env bash
# V-stock (CORE_TASKS.md's criterion-5 configuration) on the pinned build, serve/armV_stock (= vllm/armV_cand2 10-08
# 15:00 + the hybrid metadata path with ints lifted, fp8 KV in the trtllm shim, _C FP4/FP8 quant externs), in-process
# probe with --check (bitwise vs stock eager on the same metadata, NaN-aware, recurrent state compared too):
#   s1  R1 Qwen3.8-27B NVFP4 (cute.compile observed + serve/fi_ws_ht FlashInfer workspace)
#   s2  Qwen3.5-35B-A3B bf16, fp8 KV, --moe-backend triton (vLLM default flashinfer_trtllm needs MoE cubins not on disk)
export CAND_LINE=pinned ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
L=python_vllm_cand2.sh
bash $L serve/r1_probe.py --out serve/out/s1b.json --adapter-dir armV_stock --cute-observe --sym-origins --check --max-model-len 16384
echo "s1 rc=$?"
bash $L serve/r1_probe.py --out serve/out/s2b.json --adapter-dir armV_stock --cute-observe --sym-origins --check --max-model-len 16384 \
  --model /data/eellison/models/Qwen3.5-35B-A3B --kv-gib 40 --moe-backend triton
echo "s2 rc=$?"
CAND_LINE=pinned bash python_vllm_cand2.sh serve/repro/core_gaps_repro.py; echo "core rc=$?"
