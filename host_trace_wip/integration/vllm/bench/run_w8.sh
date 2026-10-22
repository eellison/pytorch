#!/usr/bin/env bash
# Phase 6 step 3 level 2: arm Vcpp with the bound decode entry (armV/adapter_cpp.py): decode dump with ARMV_BOUND_VERIFY=1
# (bitwise vs dump_fullnc.pt), then sections and the window, Vcpp and V interleaved.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
ARMV_BOUND_VERIFY=1 TAG=dumpbd DRIVE_ARGS="--prefill-lens 64 --mixed 0 --dump-decode bench/out/dump_Vbd.pt" bash bench/run_arms.sh Vcpp
CUDA_VISIBLE_DEVICES= bash python_vllm.sh armV/dumpcmp.py bench/out/dump_fullnc.pt bench/out/dump_Vbd.pt
python3 -c "import json; print('bound', json.load(open('bench/out/dumpbd_Vcpp.json'))['V'].get('bound'))"
export LD_PRELOAD=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
TAG=w8s DRIVE_ARGS="--launch-stamps --stamps --stamps-fine --decode-tokens 45 --prefill-lens 64 --prefill-reqs 4" bash bench/run_arms.sh Vcpp
TAG=w8 DRIVE_ARGS="--launch-stamps --decode-tokens 45 --prefill-lens 64 512 2048 --prefill-reqs 4 8" bash bench/run_arms.sh Vcpp V
ARMV_BOUND=0 TAG=w8n DRIVE_ARGS="--launch-stamps --decode-tokens 45 --prefill-lens 64 512 --prefill-reqs 4" bash bench/run_arms.sh Vcpp
echo ALLDONE
