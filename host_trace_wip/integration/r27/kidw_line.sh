#!/usr/bin/env bash
# kidw_line.sh CFG ARM: every-size kernel identity for the finished n27wf (CFG fi: FlashInfer GDN, vLLM defaults) / n27wt (CFG tr:
# gdn_prefill_backend triton) sets: the set's workload (prefill 64-16000, the 10 ix mixed specs) once, DRIVE_KID=1 (one profiled
# step per size after its window) and DRIVE_TACTIC_LOG=1; out/kid/t_kidw<CFG>_<ARM>_r1.json. Run inside a GPU hold (kidw_queue.sh).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
CFG=$1 ARM=$2
export DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=1 DRIVE_TACTIC_LOG=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
COMMON=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
case $CFG in
  fi) KW='{"kv_cache_dtype": "fp8"}' ;;
  tr) KW='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}' ;;
esac
KWC=${KW%\}}', "compilation_config": {"cudagraph_mode": "NONE"}}'
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8
   --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}" --ix-reps 6)
D=$R27/bench/drive_sweep27.py; O=$R27/out/kid; mkdir -p $O
case $ARM in
  default) envs=(R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0); args=(--arm default --llm-kw "$KW") ;;
  fullnc) envs=(R27_OVERLAY=$S); args=(--arm fullnc --llm-kw "$KW") ;;
  breakable) envs=(R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); args=(--arm fullnc_full --llm-kw "$KW") ;;
  breakablegb) envs=(R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); args=(--arm fullnc_full --llm-kw "$KW") ;;
  vcvp) envs=("${COMMON[@]}" ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27wf.pkl); args=(--arm Vc --llm-kw "$KWC") ;;
  vsvp) envs=("${COMMON[@]}" ARMV_BINDINGS_LOAD=$R27/bindings/n27wf_v.pkl); args=(--arm V --llm-kw "$KW") ;;
esac
env "${envs[@]}" INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) bash $R27/python_r27.sh $D "${W[@]}" "${args[@]}" --out $O/t_kidw${CFG}_$ARM.json_tmp \
  && mv $O/t_kidw${CFG}_$ARM.json_tmp $O/t_kidw${CFG}_${ARM}_r1.json
