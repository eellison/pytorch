#!/usr/bin/env bash
# Phase 6 step 4b: bench/missdiag.py under gpu2.lock. Args: TAG, then newkey args.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=$1; shift
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
echo "=== $TAG $* $(date)"
CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock \
  timeout 3000 taskset -c 108-143 bash python_vllm_cpp.sh bench/missdiag.py --out bench/out/newkey/$TAG.json "$@"
echo "rc=$? $(date)"
echo MISSDIAGDONE
