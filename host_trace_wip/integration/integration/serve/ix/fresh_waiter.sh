#!/usr/bin/env bash
# the fresh-process control on the first of GPUs 1/2/3 with >= 150 GB free, stable for 2 minutes (correctness only, beside another user if needed)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
while :; do
  for g in 3 2 1; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | tr -d , | awk '{print $1 - $2}')
    [ "$free" -ge 150000 ] || continue
    sleep 120  # still free two minutes later (another user's job grows in steps)
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | tr -d , | awk '{print $1 - $2}')
    [ "$free" -ge 150000 ] || continue
    case $g in 1) cpus=72-107 ;; 2) cpus=108-125 ;; 3) cpus=126-143 ;; esac
    exec 9>$E/gpu$g.lock
    if flock -n -x 9; then
      echo "$(date +%T) gpu$g free $free MiB"
      CUDA_VISIBLE_DEVICES=$g GPU_IDX=$g timeout 2000 taskset -c $cpus bash serve/repro/run_q35_fresh.sh > serve/logs/q35_fresh.log 2>&1
      echo "$(date +%T) done rc=$?"; exit 0
    fi
    exec 9>&-
  done
  sleep 60
done
