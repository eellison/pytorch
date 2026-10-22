#!/usr/bin/env bash
# One GPU 0 timing job under a single flock -x gpu0.lock hold: bash gpu0_job2.sh <inner.sh> [args]. CUDA_VISIBLE_DEVICES=0, CPUs 0-71.
# As gpu0_job.sh, but queued on the lock itself (blocking flock) instead of polling for an empty GPU first, so a lane that
# re-takes the lock right after releasing it does not starve this one. Inside the lock: a foreign compute process on GPU 0 ->
# release and retry (exit 75).
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
while :; do
  echo "$(date +%T) waiting for gpu0.lock"
  env CUDA_VISIBLE_DEVICES=0 flock -x $LOCK bash -c '
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)" != 0 ]; then echo "$(date +%T) lock held but GPU 0 busy: releasing"; exit 75; fi
    echo "$(date +%T) holding gpu0.lock"; exec taskset -c 0-71 bash "$@"' _ "$@"
  rc=$?
  [ $rc != 75 ] && exit $rc
  sleep 20
done
