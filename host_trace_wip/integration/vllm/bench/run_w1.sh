#!/usr/bin/env bash
# Phase 6 step 1: pre-launch window (launchshim.so) for V and default, then Kineto decode profiles for the GPU span.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=${TAG:-w1} DRIVE_ARGS="--launch-stamps --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8" bash bench/run_arms.sh ${ARMS:-V default}
unset LD_PRELOAD
[ -n "$NOPROF" ] || TAG=${TAG:-w1}p DRIVE_ARGS="--profile-decode 1 8 64 --profile-dir $V/bench/out/prof_${TAG:-w1} --prefill-lens 64 --mixed 0" bash bench/run_arms.sh ${ARMS:-V default}
echo ALLDONE
