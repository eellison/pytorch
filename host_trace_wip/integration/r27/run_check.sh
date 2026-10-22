#!/usr/bin/env bash
# run_check.sh TAG [extra drive_fn args]: Qwen3.8-27B NVFP4 V-stock --check, SOUND (ARMV_SOUND=1: only cuBLAS harvested) +
# ARMV_TRTLLM_CPP=1 (FlashInfer's trtllm-gen launcher traced) + r27/armV_r27 (pinned host args, AutoTuner state local to a trace)
# on CAND_LINE c5 (python_r27.sh). Small KV (R27_KV_GIB, default 6) and max_model_len 9216 so it fits beside another user's job.
# Decode bs 1/8/64 (short), decode bs 1 and 8 at KV 6000-8000, mixed 1024x8 @1-2K and 2048x8 @6-8K, prefills 64 / 512.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
TAG=$1; shift
cd $G
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=${ARMV_HOST_ARGS:-pinned} ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=${ARMV_PORTS:-0}
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.04} INTEG_KV_BYTES=$(( ${R27_KV_GIB:-6} * 1073741824 ))
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 9216 --batch-size 1 8 64 --decode-tokens 12
   --prefill-lens 64 512 --mixed 0 --ix-mixed 1024x8@1024-2048 2048x8@6000-8000 --ix-decode 1:7000-8000 8:6000-8000 --ix-reps 3
   --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 64}')
bash $R/python_r27.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" "$@" --out $R/out/$TAG.json
echo "$TAG rc=$?"
python3 $R/summ.py $R/out/$TAG.json
