#!/usr/bin/env bash
# V-stock round 3 (pinned build, serve/armV_stock):
#   s1c  R1 with _C.static_scaled_fp8_quant left eager (ARMV_EXTRA_EXTERN=scaled_fp4_quant.out): past core gap 2, to see
#        what follows under V-stock
#   s2c  Qwen3.5-35B-A3B bf16, bf16 KV, --moe-backend triton --attention-backend TRITON_ATTN: the no-download
#        configuration (vLLM's defaults need FlashInfer trtllm-gen MoE batched_gemm and fp8 fmha P16 cubins not on disk)
export CAND_LINE=pinned ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
L=python_vllm_cand2.sh
ARMV_EXTRA_EXTERN=scaled_fp4_quant.out bash $L serve/r1_probe.py --out serve/out/s1c.json --adapter-dir armV_stock --cute-observe --sym-origins --check --max-model-len 16384
echo "s1c rc=$?"
bash $L serve/r1_probe.py --out serve/out/s2c.json --adapter-dir armV_stock --cute-observe --sym-origins --check --max-model-len 16384 \
  --model /data/eellison/models/Qwen3.5-35B-A3B --kv-gib 40 --kv-dtype auto --moe-backend triton --attn TRITON_ATTN
echo "s2c rc=$?"
