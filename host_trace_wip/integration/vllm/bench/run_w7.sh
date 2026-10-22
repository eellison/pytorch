#!/usr/bin/env bash
# Phase 6 step 2/3: finer sections of the STAGE prepare_inputs and the remaining execute_model glue (arm Vcpp).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=w7s DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 --prefill-reqs 4" bash bench/run_arms.sh Vcpp
echo ALLDONE
