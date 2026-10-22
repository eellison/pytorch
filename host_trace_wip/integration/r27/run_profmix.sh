#!/usr/bin/env bash
# Diagnosis: torch.profiler trace of vLLM default-nc's mixed steps (8 decoders + one 128-token prompt), Triton GDN; where ~170 ms goes
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step324 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp12/install R27_OVERLAY=$S INTEG_GPU_UTIL=0.35 INTEG_KV_BYTES=$(( 40 << 30 ))
bash $R/python_r27.sh vllm/bench/drive_fn.py --arm fullnc --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 \
  --batch-size 1 --prefill-lens 64 --mixed 6 --prof-mixed $R/out/prof/mixed --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}' --out $R/out/profmix_fullnc.json 2>&1 | grep -E "^mixed|^prefill|Error"
