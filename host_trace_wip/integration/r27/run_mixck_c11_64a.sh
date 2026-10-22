#!/usr/bin/env bash
# c5g mixed-step check, the timed arm (Vc+VP, VP_STOCK, every kind replayed) with the unmerged patches/cute_arith shim
# (ARMV_DIAG_CUTE_ARITH=1); only mixed + decode steps compared (ARMV_CHECK_KINDS). Shapes: 1024 / 2048-token chunk + 64 decodes at KV 1-2K, 6-8K.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_CHECK_KINDS=mixed,decode ARMV_CHECK_STATE_CPU=0
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step318 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp11/install FI_HT_BUILD=$R/build_c11 TORCH_EXTENSIONS_DIR=$R/torch_ext_c11
export ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_DIAG_CUTE_ARITH=1 ARMV_SYM_ORIGINS=1 INTEG_GPU_UTIL=0.35 INTEG_KV_BYTES=$(( 40 << 30 ))
bash $R/python_r27.sh vllm/bench/drive_fn.py --arm Vc --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 \
  --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 2048x64@1024-2048 --ix-reps 4 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}' --out $R/out/mixck_c11_64a.json
echo "mixck_c11_64a rc=$?"
python3 $R/b200_bundle/tools/verdict.py $R/out/mixck_c11_64a.json
