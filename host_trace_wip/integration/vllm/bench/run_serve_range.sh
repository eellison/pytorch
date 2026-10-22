#!/usr/bin/env bash
# Phase 6 step 5: bench/serve_range.sh per arm, one gpu2.lock hold per arm.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
for a in ${@:-V eagerm default}; do
  if pgrep -af "pip|ninja" | grep -q wt-sgl-cpp; then echo "wt-sgl-cpp build running; abort"; exit 1; fi
  echo "=== $a $(date)"
  CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock timeout 7000 bash bench/serve_range.sh $a
  echo "rc=$? $(date)"
done
echo SRDONE
