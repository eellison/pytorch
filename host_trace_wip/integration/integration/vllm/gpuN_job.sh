#!/usr/bin/env bash
# One timing job on GPU $GPU under a single flock -x gpu$GPU.lock hold, CPUs $CPUS (10-13 per-GPU pinning: GPU 0 0-35):
#   GPU=0 CPUS=0-35 bash gpuN_job.sh <inner.sh> [args]
# The job runs as flock's child, so it never starts without the lock; a flock failure (or kill) exits non-zero and is reported
# (LANES.md 10-14). Inside the hold: a foreign compute process on the GPU -> release and retry (exit 75); the GPU's used memory must
# be < 20 GiB before the job starts (waits up to 20 min, then releases and retries). HT_GPU / HT_CPUS tell timing_lib which GPU / cores.
GPU=${GPU:?} CPUS=${CPUS:?}
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$GPU.lock
export HT_GPU=$GPU HT_CPUS=$CPUS
while :; do
  echo "$(date +%T) waiting for gpu$GPU.lock"
  env CUDA_VISIBLE_DEVICES=$GPU flock -x $LOCK bash -c '
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $HT_GPU | wc -l)" != 0 ]; then echo "$(date +%T) lock held but GPU $HT_GPU busy: releasing"; exit 75; fi
    w=0
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $HT_GPU | tr -d " ")" -ge 20480 ]; do
      [ $w -ge 1200 ] && { echo "$(date +%T) GPU $HT_GPU memory still >= 20 GiB after 20 min: releasing"; exit 75; }
      sleep 20; w=$((w + 20))
    done
    echo "$(date +%T) holding gpu$HT_GPU.lock"; exec taskset -c '"$CPUS"' bash "$@"' _ "$@"
  rc=$?
  if [ $rc = 75 ]; then sleep 20; continue; fi
  [ $rc != 0 ] && echo "$(date +%T) gpuN_job: flock or job exited $rc (job not started if flock itself failed)"
  exit $rc
done
