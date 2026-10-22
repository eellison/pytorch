#!/usr/bin/env bash
# gpu_run.sh TAG MIN_FREE_MIB script [args]: wait until one of GPUs ${GPUS:-1 2 3} has >= MIN_FREE_MIB free and its lock
# (land/scratch/eager/gpuN.lock) is free, then run `bash script args` there under the lock (one job, timeout ${CMD_TIMEOUT:-1200} s),
# CUDA_VISIBLE_DEVICES=N GPU_IDX=N on that GPU's CPUs. Never GPU 0. Another user's resident job (another user) is tolerated only while the
# free memory covers this run (coordinator rule 10-12); the free check repeats right before the job starts.
TAG=$1 MIN=$2; shift 2
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 | tr -d , | awk '{print $1 - $2}'; }
echo "$(date +%T) $TAG waiting for >= $MIN MiB on one of ${GPUS:-1 2 3}"
while :; do
  for g in ${GPUS:-1 2 3}; do
    [ $g = 0 ] && continue
    [ "$(free_mib $g)" -ge $MIN ] || continue
    # CPU sets (coordinator 10-13): GPU 1 -> 72-107, GPU 2 -> 36-71; never 0-35 (GPU 0 sets) or 108-143 (GPU 3 / vLLM lane)
    case $g in 1) cpus=72-107 ;; 2|3) cpus=36-71 ;; esac
    cpus=${CPUS:-$cpus}
    exec 9>$E/gpu$g.lock
    if flock -n -x 9; then
      # LANES.md (10-14): inside the hold, wait for the GPU's used memory to be < 20 GiB before starting
      while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')" -ge 20480 ]; do sleep 30; done
      if [ "$(free_mib $g)" -ge $MIN ]; then
        echo "$(date +%T) $TAG holding gpu$g.lock (free $(free_mib $g) MiB) cpus $cpus"
        CUDA_VISIBLE_DEVICES=$g GPU_IDX=$g timeout ${CMD_TIMEOUT:-1200} taskset -c $cpus bash "$@" > $R/logs/$TAG.log 2>&1
        echo "$(date +%T) $TAG rc=$? on gpu$g"
        exit 0
      fi
    fi
    exec 9>&-
  done
  sleep 30
done
