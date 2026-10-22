#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
while :; do
  for g in 1 2 3; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | tr -d , | awk '{print $1 - $2}')
    [ "$free" -ge 190000 ] || continue
    case $g in 1) cpus=72-107 ;; 2) cpus=108-125 ;; 3) cpus=126-143 ;; esac
    exec 9>$E/gpu$g.lock
    if flock -n -x 9; then
      echo "$(date +%T) gpu$g free $free MiB: q8vcs${TAG}"
      CUDA_VISIBLE_DEVICES=$g GPU_IDX=$g timeout 1800 taskset -c $cpus bash serve/repro/run_q8vcs_check.sh > serve/logs/q8vcs${TAG:?}.log 2>&1
      echo "$(date +%T) q8vcs${TAG} rc=$?"; exit 0
    fi
    exec 9>&-
  done
  sleep 60
done
