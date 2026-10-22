#!/usr/bin/env bash
# Phase 6 step 4c: bench/missreason.py under gpu2.lock. Args: TAG, then VAR=value env settings, then -- rangesweep args.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=$1; shift
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
echo "=== $TAG ${ENVS[*]} $* $(date)"
env "${ENVS[@]}" CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock \
  timeout 3600 taskset -c 108-143 bash python_vllm_cpp.sh bench/missreason.py --out bench/out/range/$TAG.json "$@"
echo "rc=$? $(date)"
echo MRDONE
