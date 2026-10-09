#!/usr/bin/env bash
# V-stock round 4 (pinned build, serve/armV_stock with ARMV_HOST_ARGS=1: GDN prefill's CPU tensors as arguments):
#   s2d  Qwen3.5-35B-A3B no-download configuration (bf16 KV, triton MoE, TRITON_ATTN), 16 single-request prompts
#   core repros (gap_1b, gap_2, gap_3 side stream)
export CAND_LINE=pinned ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
L=python_vllm_cand2.sh
bash $L serve/r1_probe.py --out serve/out/s2d.json --adapter-dir armV_stock --cute-observe --sym-origins --check --max-model-len 16384 \
  --model /data/eellison/models/Qwen3.5-35B-A3B --kv-gib 40 --kv-dtype auto --moe-backend triton --attn TRITON_ATTN --prompts many
echo "s2d rc=$?"
bash $L serve/repro/core_gaps_repro.py; echo "core rc=$?"
