#!/usr/bin/env bash
# The exact timing configuration of round_27s_line.sh's V-stock arm, with --check (clean-row verdict after)
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed ARMV_CHECK_TWICE=1
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step318 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp11/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/build_c11 TORCH_EXTENSIONS_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/torch_ext_c11 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 ))
bash $R/python_r27.sh vllm/bench/drive_fn.py --arm Vc --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 \
  --batch-size 1 8 64 --prefill-lens 64 --mixed 0 --ix-decode 8:6000-8000 64:6000-8000 --ix-reps 6 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}' --out $R/out/vcvp_c11.json
echo "vcvp_c11 rc=$?"
python3 $R/b200_bundle/tools/verdict.py $R/out/vcvp_c11.json
