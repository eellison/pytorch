#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f gpu2_ficheckW.sh > /dev/null; do sleep 30; done
flock -x ../../scratch/eager/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_gdn_ours2.sh > logs/gdn_ours2.log 2>&1
