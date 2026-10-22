#!/usr/bin/env bash
# lp_gdnple.sh: logits check for the gdnple arm (coordinator 10-15): DRIVE_LOGIT_PROBE prefills of 64 / 512 / 1024 (breakable PIECEWISE)
# and 2048 (eager in breakable: control), last-row fp32 logits to out/lp/<tag>_<L>.pt, for fullnc (twice: the eager spread) and gdnple.
# Untimed; gb300-box GPU 2, gpu2.lock per run (slots between chain_g2c's arms), CPUs 36-71, the timed sets' build / settings.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
echo $$ > $D/logs/mon/lp_gdnple.pid
mkdir -p $D/out/lp
export FN_WT=$L/int6/snap/step334_ss FN_INSTALL=$L/int6/build_cpp15/install TORCH_EXTENSIONS_DIR=$D/cache/torch_ext_c15
export SGL_OPS_TRACE_DIR=$L/core/sidestreams/port_overlay FN_FI_AUTOTUNE_FILE=$D/cache/fi_autotune/f2_autotune.json HOSTTRACE_TORCH_DISABLE_CACHES=0
export FN_MOE_OVERLAY=$L/scratch/sglang/site DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=0 INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((40 << 30)) CUDA_VISIBLE_DEVICES=2
export DRIVE_LOGIT_PROBE="64 512 1024 2048" DRIVE_LOGIT_PROBE_DIR=$D/out/lp
one() {  # tag arm env...
  local tag=$1 arm=$2; shift 2
  ( flock -x 9 || { echo "$(date +%T) $1: flock failed, not running"; exit 1; }; echo "$(date +%T) $tag holding gpu2.lock"
    env "$@" DRIVE_LOGIT_PROBE_TAG=$tag timeout --foreground 1800 taskset -c 36-71 bash $D/python_fn.sh $D/scripts/with_fi_autotune.py $D/bench/drive_sweep27.py \
      --arm $arm --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-model-only --max-model-len 16384 \
      --llm-kw '{"gdn_prefill_backend": "triton"}' --batch-size 1 --decode-tokens 12 --prefill-lens 64 --mixed 0 --out $D/out/lp/$tag.json \
      > $D/logs/lp_$tag.log 2>&1; echo "$(date +%T) $tag rc=$?" ) 9>$L/scratch/eager/gpu2.lock
}
one fullnc_a fullnc
one gdnple fullnc_full VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 FN_PLE_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
one fullnc_b fullnc
one breakable fullnc_full VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
for x in gdnple fullnc_b breakable; do echo "== fullnc_a vs $x"; bash $D/python_fn.sh $L/core/r27/scripts/logit_cmp.py $D/out/lp fullnc_a $x 2>&1 | grep "^L="; done
echo "$(date +%T) lp_gdnple done"
