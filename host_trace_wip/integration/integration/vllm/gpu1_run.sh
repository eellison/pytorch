#!/usr/bin/env bash
# One GPU 1 correctness run under gpu1.lock: gpu1_run.sh TAG [ENV=..] -- script args. util 0.6 on an empty GPU 1, else util 0.1 with an explicit 6 GiB KV pool (kv_cache_memory_bytes: no profiling against a moving foreign job).
# Retries (up to 2 more times) when vLLM's init fails because a foreign job's memory moved under it.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
tag=$1; shift; envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
for try in 1 2 3; do
  echo "$(date +%T) start $tag try $try"
  CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env "${envs[@]}" bash -c '
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
    if [ "$used" -lt 2048 ]; then export INTEG_GPU_UTIL=0.6; else export INTEG_GPU_UTIL=0.1 INTEG_KV_BYTES=$((6 << 30)); fi
    echo "gpu1 used $used util $INTEG_GPU_UTIL kv_bytes ${INTEG_KV_BYTES:-auto}"
    exec timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh "$@"' _ "$@" > vllm/logs/$tag.log 2>&1
  rc=$?
  grep -q "Error in memory profiling\|less than desired GPU memory utilization\|No available memory for the cache blocks" vllm/logs/$tag.log || break
  echo "$(date +%T) $tag: init failed on foreign GPU 1 memory, retrying"
done
echo "$(date +%T) end $tag rc=$rc"
