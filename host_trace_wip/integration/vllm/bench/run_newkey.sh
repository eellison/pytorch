#!/usr/bin/env bash
# Phase 6 step 4b: bench/newkey.py under gpu2.lock. Args: TAG, then VAR=value env settings (e.g. ARMV_FAMILY=0).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=$1; shift
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
echo "=== $TAG $* $(date)"
env "$@" CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock \
  timeout 3600 taskset -c 108-143 bash python_vllm_cpp.sh bench/newkey.py --out bench/out/newkey/$TAG.json
echo "rc=$? $(date)"
echo NEWKEYDONE
