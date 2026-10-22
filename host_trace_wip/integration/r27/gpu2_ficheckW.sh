#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=../../scratch/eager
for fwd in c e; do flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_ficheckW.sh $fwd 2 >> logs/ficheckW.out 2>&1; done
echo "ficheckW done $(date +%T)" >> logs/ficheckW.out
