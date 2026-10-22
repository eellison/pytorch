#!/usr/bin/env bash
# NVFP4 sweep line check (candidate 4: CAND_LINE=c4 = snap/step290 + build_cpp9, the sweep's line), one GPU 1 hold:
# fs5 = Flash-Next V-stock --check (fs4's config: pinned host args, top-k externs, default FLASHINFER_TRTLLM MoE);
# n27 = Qwen3.8-27B V-stock --check (recipe fp8 KV, trtllm-gen attention through the adapter shim, FP4/FP8 quant externs).
# Each waits (inside the hold) for the GPU memory it needs: another user's job can hold GPU 1 without the lock.
L=../python_vllm_cand2.sh
export CAND_LINE=c4 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
waitfree() { until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 1 | tr -d , | awk '{print $1 - $2}') -ge $1 ]; do sleep 60; done; echo "$(date +%T) gpu1 free >= $1 MiB"; }
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
waitfree 240000
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false}'
ARMV_EXTRA_EXTERN=scaled_fp4_quant.out,static_scaled_fp8_quant.default,cooperative_topk.default,persistent_topk.default bash python_vllm_cand2.sh serve/r1_probe.py --out serve/out/fs5.json --adapter-dir armV_stock --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --max-model-len 16384 --kv-gib 30 --kv-dtype auto --cute-observe --sym-origins --check --prompts many --llm-kw "$KW"
echo "fs5 rc=$?"
waitfree 100000
bash python_vllm_cand2.sh serve/r1_probe.py --out serve/out/n27.json --adapter-dir armV_stock --model /data/eellison/models/Qwen3.8-27B-NVFP4 --max-model-len 16384 --kv-gib 40 --kv-dtype fp8 --cute-observe --sym-origins --check --prompts many --llm-kw '{"max_num_seqs": 64}'
echo "n27 rc=$?"
