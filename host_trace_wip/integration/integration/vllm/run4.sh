#!/usr/bin/env bash
# Round 3: the range sweep with --check (bitwise vs eager on every forward) under int6 default key budget, even line.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
cd $G
echo "$(date +%T) start even_range_key_check"
INTEG_GPU_UTIL=0.072 ARMV_STATIC_SHAPES=1 CUDA_VISIBLE_DEVICES=1 flock -x $LOCK bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1; exec timeout 3600 taskset -c 72-107 "$@"' _ \
  bash python_vllm_int6_even.sh vllm/bench/rangesweep.py --check --out vllm/out/even_range_key_check.json > vllm/logs/even_range_key_check.log 2>&1
echo "$(date +%T) end even_range_key_check rc=$?"
