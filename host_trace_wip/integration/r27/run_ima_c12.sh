#!/usr/bin/env bash
# Repro of the t_n27w_learn illegal memory access (c5h, Vc+VP, decode sweep died after bs 5): the round_27w_line.sh vcvp env, decode
# bs 1-8 only. IMA_CHECK=1 adds --check (decode compared against eager, vLLM's sampler tokens checked).
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_CHECK_KINDS=decode ARMV_CHECK_STATE_CPU=0
export INTEG_GPU_UTIL=${IMA_UTIL:-0.9} INTEG_KV_BYTES=$(( 40 << 30 )) ${IMA_ENV:-}
bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm Vc ${IMA_CHECK:+--check} --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
  --max-model-len 16384 --batch-size ${IMA_BS:-1 2 3 4 5 6 7 8} --decode-tokens 20 --prefill-lens ${IMA_PL:-64} --mixed 0 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}' --out $R/out/${IMA_TAG:-ima_c12}.json
echo "${IMA_TAG:-ima_c12} rc=$?"
