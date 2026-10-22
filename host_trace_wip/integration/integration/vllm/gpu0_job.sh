#!/usr/bin/env bash
# One GPU 0 timing job under a single flock -x gpu0.lock hold: bash gpu0_job.sh <inner.sh> [args]. CUDA_VISIBLE_DEVICES=0, CPUs 0-71.
# Foreign compute processes on GPU 0 are not under the lock: wait for an empty GPU 0 before taking the lock, and if one
# appears by the time the lock is ours, release it and wait again (exit 75) instead of running the job into skips.
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
while :; do
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)" != 0 ]; do sleep 60; done
  echo "$(date +%T) waiting for gpu0.lock"
  env CUDA_VISIBLE_DEVICES=0 flock -x $LOCK bash -c '
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)" != 0 ]; then echo "$(date +%T) lock held but GPU 0 busy: releasing"; exit 75; fi
    echo "$(date +%T) holding gpu0.lock"; exec taskset -c 0-71 bash "$@"' _ "$@"
  rc=$?
  [ $rc != 75 ] && exit $rc
  sleep 60
done
