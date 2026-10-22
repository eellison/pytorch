#!/usr/bin/env bash
# Phase 6 step 3: arm Vcpp (install_sgl_cpp, static-prefix skip + O(1) variant selection, Python adapter): decode dump
# (bitwise vs dump_fullnc.pt), then the pre-launch window, Vcpp and V interleaved.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
TAG=dumpcpp DRIVE_ARGS="--prefill-lens 64 --mixed 0 --dump-decode bench/out/dump_Vcpp.pt" bash bench/run_arms.sh Vcpp
CUDA_VISIBLE_DEVICES= bash python_vllm.sh armV/dumpcmp.py bench/out/dump_fullnc.pt bench/out/dump_Vcpp.pt
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=w6s DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 512 --prefill-reqs 4" bash bench/run_arms.sh Vcpp
TAG=w6 DRIVE_ARGS="--launch-stamps --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8" bash bench/run_arms.sh Vcpp V
echo ALLDONE
