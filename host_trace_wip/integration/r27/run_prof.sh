#!/usr/bin/env bash
# run_prof.sh ARM (default|fullnc|V|Vc): profiled decode bursts (bs 1 and 64, steps 12-14 each, torch.profiler CPU+CUDA) in the n27t
# configuration (gdn_prefill_backend=triton, KV 40 GiB, mml 16384), traces into out/prof/<ARM>/
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
ARM=$1
VS="ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed"
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
case $ARM in
  default) E="R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0" ;;
  fullnc) E="R27_OVERLAY=$S" ;;
  V) E="$VS" ;;
  Vc) E="$VS ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0"; KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}' ;;
esac
mkdir -p $R/out/prof/$ARM
env $E INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 )) bash $R/python_r27.sh vllm/bench/drive_fn.py --arm $ARM --model /data/eellison/models/Qwen3.8-27B-NVFP4 \
  --language-model-only --max-model-len 16384 --batch-size 1 64 --decode-tokens 20 --prefill-lens 64 --mixed 0 \
  --profile-decode 1 64 --profile-dir $R/out/prof/$ARM --llm-kw "$KW" --out $R/out/prof/$ARM.json
echo "prof $ARM rc=$?"
