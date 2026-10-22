#!/usr/bin/env bash
# c5h churn check (round_27cb_line.sh design, C = 16, 60 warm + 300 measured steps, outputs 64-256 so requests finish and new prefills land as mixed steps): Vc+VP with VP_STOCK, every kind replayed,
# mixed + decode steps compared against eager (ARMV_CHECK_KINDS), vLLM's sampler tokens checked on decode rows.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_CHECK_KINDS=mixed,decode ARMV_CHECK_STATE_CPU=0
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step324 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
export ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_SYM_ORIGINS=1 INTEG_GPU_UTIL=0.35 INTEG_KV_BYTES=$(( 40 << 30 ))
bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm Vc --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 \
  --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --cb 16 --cb-steps 300 --cb-warm 60 --cb-out 64 256 --cb-prompt 64 512 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}' --out $R/out/cbck2_c12.json
echo "cbck2_c12 rc=$?"
python3 $R/b200_bundle/tools/verdict.py $R/out/cbck2_c12.json
