#!/usr/bin/env bash
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=w5s DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 512 --prefill-reqs 4" bash bench/run_arms.sh V
echo ALLDONE
