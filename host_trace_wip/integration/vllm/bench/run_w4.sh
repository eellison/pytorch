#!/usr/bin/env bash
# Phase 6 step 2: V with armV/hostcuts.py: decode dump (bitwise vs dump_fullnc.pt), then the pre-launch window with and
# without section stamps.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=dumphc2 DRIVE_ARGS="--prefill-lens 64 --mixed 0 --dump-decode bench/out/dump_Vhc2.pt" bash bench/run_arms.sh V
CUDA_VISIBLE_DEVICES= bash python_vllm.sh armV/dumpcmp.py bench/out/dump_fullnc.pt bench/out/dump_Vhc2.pt
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=w4s DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8" bash bench/run_arms.sh V
TAG=w4 DRIVE_ARGS="--launch-stamps --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8" bash bench/run_arms.sh V
echo ALLDONE
