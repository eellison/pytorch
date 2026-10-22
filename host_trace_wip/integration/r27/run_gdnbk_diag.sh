#!/usr/bin/env bash
# Diagnosis (not timing-grade, correctness GPU): vLLM's own arms' mixed-step time with FlashInfer GDN prefill (stock) vs Triton GDN
# prefill (our diagnosis config), to see whether vLLM's ~190-260 ms flat mixed step under Triton GDN is that backend's host cost.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step324 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp12/install R27_OVERLAY=$S INTEG_GPU_UTIL=0.35 INTEG_KV_BYTES=$(( 40 << 30 ))
for be in flashinfer triton; do
  bash $R/python_r27.sh vllm/bench/drive_fn.py --arm fullnc --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 \
    --batch-size 1 --prefill-lens 64 512 --mixed 0 --ix-mixed 1024x8@1024-2048 2048x8@1024-2048 --ix-reps 6 \
    --llm-kw "{\"kv_cache_dtype\": \"fp8\", \"max_num_seqs\": 128, \"gdn_prefill_backend\": \"$be\"}" --out $R/out/gdnbk_$be.json 2>&1 | grep -E "^ix|^prefill|^decode"
done
