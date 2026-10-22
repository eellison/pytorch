#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
flock -x ../../scratch/eager/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_ficheckP.sh c 2 >> logs/ficheckP.out 2>&1
echo "ficheckP done $(date +%T)" >> logs/ficheckP.out
