#!/usr/bin/env bash
# n27c5s (ARMV_SOUND=1: only cuBLAS harvested; open _C ops, the trtllm shim ops and top-k as eager steps): Qwen3.8-27B NVFP4 V-stock --check on candidate 5 PRE-GATE (CAND_LINE=c5: int6 step310 = mm_fp4 stack 97b_c3 + R1_ii + 99,
# build_cpp10) with serve/armV_stock_m + ARMV_HOST_ARGS=pinned; recipe fp8 KV; trtllm-gen attention through the adapter shim.
# Question: does decode replay, and is any mm_fp4 (FlashInfer CuTe dense_blockscaled_gemm) left as an eager step?
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export CAND_LINE=c5 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned ARMV_SOUND=1
export FLASHINFER_WORKSPACE_BASE=$PWD/serve/fi_ws_ht
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 100000 ]; do sleep 60; done
bash python_vllm_cand2.sh serve/r1_probe.py --out serve/out/n27c5s.json --adapter-dir armV_stock_m --model /data/eellison/models/Qwen3.8-27B-NVFP4 --max-model-len 16384 --kv-gib 40 --kv-dtype fp8 --cute-observe --sym-origins --check --prompts many --llm-kw '{"max_num_seqs": 64}'
echo "n27c5s rc=$?"
