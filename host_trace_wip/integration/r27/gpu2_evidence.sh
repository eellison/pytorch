#!/usr/bin/env bash
# GPU 2 queue (10-13 evening): eager-twice checks (stock), FlashInfer GDN evidence (initcheck, racecheck, fence NaN rates), fenced strict
# checks; then STOP (no timing until the coordinator decides)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=../../scratch/eager
while pgrep -f "run_ficheck2.sh c" > /dev/null; do sleep 30; done
flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_ficheck2.sh e 2 >> logs/ficheck2.out 2>&1
flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_fi_evidence.sh >> logs/fi_evidence.out 2>&1
for fwd in c e; do flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_ficheckF.sh $fwd 2 >> logs/ficheckF.out 2>&1; done
echo "gpu2_evidence done $(date +%T)" >> logs/fi_evidence.out
