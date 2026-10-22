#!/usr/bin/env bash
# run_brk3.sh CFG ARM (fi | tr) (breakable | fullnc): breakable GDN check through drive_sweep27 (the harness breakable runs in): one
# warm-up generate, then DRIVE_LOGIT_PROBE prefills of 64 / 512 / 1024 (breakable PIECEWISE sizes) and 2048 (eager in breakable:
# control); logits to out/brk3/<CFG>_<ARM>_<L>.pt. vLLM's defaults otherwise (the n27wf / n27wt configs). Run inside a GPU hold.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
CFG=$1 ARM=$2
export DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=0 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export DRIVE_LOGIT_PROBE="64 512 1024 2048" DRIVE_LOGIT_PROBE_DIR=$R27/out/brk3 DRIVE_LOGIT_PROBE_TAG=${CFG}_$ARM R27_OVERLAY=$S
case $CFG in fi) KW='{"kv_cache_dtype": "fp8"}' ;; tr) KW='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}' ;; esac
case $ARM in
  breakable) envs=(VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); a=fullnc_full ;;
  breakablegb) envs=(VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); a=fullnc_full ;;
  fullnc) envs=(); a=fullnc ;;
esac
env "${envs[@]}" INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) bash $R27/python_r27.sh $R27/bench/drive_sweep27.py --arm $a \
  --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 12 \
  --prefill-lens 64 --mixed 0 --llm-kw "$KW" --out $R27/out/brk3/${CFG}_$ARM.json
