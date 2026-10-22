#!/usr/bin/env bash
# One correctness/profile job on GPU 3 (weekend rule, LANES.md): gpu_idle_ok.sh 3 before taking gpu3.lock and again under it,
# CUDA_VISIBLE_DEVICES=3, HT_GPU=3 (timing_lib's GPU checks), CPUs 126-143. Not for timing (GPU 0 stays the timing GPU).
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
while :; do
  while ! bash $E/gpu_idle_ok.sh 3; do sleep 60; done
  echo "$(date +%T) waiting for gpu3.lock"
  env CUDA_VISIBLE_DEVICES=3 HT_GPU=3 flock -x $E/gpu3.lock bash -c '
    bash '$E'/gpu_idle_ok.sh 3 || { echo "$(date +%T) GPU 3 has another user: releasing"; exit 75; }
    echo "$(date +%T) holding gpu3.lock"; exec taskset -c 126-143 bash "$@"' _ "$@"
  rc=$?
  [ $rc != 75 ] && exit $rc
  sleep 60
done
