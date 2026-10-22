#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
flock -x ../../scratch/eager/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 timeout 2400 bash run_vsvp_check2.sh > logs/vsvp_check2.out 2>&1
