#!/usr/bin/env bash
# Items 1-3: bench/rangesweep.py under gpu2.lock with a chosen launcher. Args: TAG LAUNCHER(cpp|fork), then VAR=value env
# settings, then -- rangesweep args. fork = python_vllm_cpp_fork.sh (the trtllm fork's flashinfer; pair with ARMV_TRTLLM_FORK=1).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=$1; L=$([ "$2" = fork ] && echo python_vllm_cpp_fork.sh || echo python_vllm_cpp.sh); shift 2
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
echo "=== $TAG $L ${ENVS[*]} $* $(date)"
env "${ENVS[@]}" CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock \
  timeout 3600 taskset -c 108-143 bash $L bench/rangesweep.py --out bench/out/range/$TAG.json "$@"
echo "rc=$? $(date)"
