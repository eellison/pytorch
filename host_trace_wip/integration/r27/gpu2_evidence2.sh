#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=../../scratch/eager
flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_initcheck.sh >> logs/fi_evidence.out 2>&1
flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_hashseed.sh >> logs/fi_evidence.out 2>&1
echo "gpu2_evidence2 done $(date +%T)" >> logs/fi_evidence.out
