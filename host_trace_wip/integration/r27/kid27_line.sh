#!/usr/bin/env bash
# kid27_line.sh CFG ARM: kernel-identity profile run (untimed): one decode (bs 8), one prefill (1x512) and one mixed (512 + 16
# decoders) full step profiled at the class's 7th occurrence (ours: a replay), into out/kid/$CFG/$ARM_<reqs>x<tokens>.json.
# CFG fi = vLLM default (FlashInfer GDN), tr = gdn_prefill_backend triton. Run inside a GPU hold (kid27.sh).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
CFG=$1 ARM=$2
export DRIVE_STEP_WITH_SAMPLE=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export DRIVE_TACTIC_LOG=1 DRIVE_PROFILE_DIR=$R27/out/kid/$CFG DRIVE_PROFILE_CLASSES=8:8,1:512,17:528 DRIVE_PROFILE_AT=7 DRIVE_PROFILE_TAG=$ARM
COMMON=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
case $CFG in
  fi) KW='{"kv_cache_dtype": "fp8"}'; KWC='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'; B=n27wf ;;
  tr) KW='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}'
      KWC='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'; B=n27wf ;;
esac
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 8 --decode-tokens 16
   --prefill-lens 512 --mixed 0 --ix-mixed 512x16@1024-2048 --ix-reps 8)
D=$R27/bench/drive_sweep27.py; O=$R27/out/kid/$CFG; mkdir -p $O
case $ARM in
  default) envs=(R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0); args=(--arm default --llm-kw "$KW") ;;
  fullnc) envs=(R27_OVERLAY=$S); args=(--arm fullnc --llm-kw "$KW") ;;
  vcvp) envs=("${COMMON[@]}" ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R27/bindings/$B.pkl); args=(--arm Vc --llm-kw "$KWC") ;;
  vsvp) envs=("${COMMON[@]}" ARMV_BINDINGS_LOAD=$R27/bindings/${B}_v.pkl); args=(--arm V --llm-kw "$KW") ;;
esac
env "${envs[@]}" INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) bash $R27/python_r27.sh $D "${W[@]}" "${args[@]}" --out $O/$ARM.json
