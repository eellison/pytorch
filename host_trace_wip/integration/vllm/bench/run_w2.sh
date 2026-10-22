#!/usr/bin/env bash
# Phase 6 step 2 baseline: pre-launch window with unsynced section stamps (V only unless ARMS).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=${TAG:-w2} DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8 ${EXTRA:-}" bash bench/run_arms.sh ${ARMS:-V}
echo ALLDONE
