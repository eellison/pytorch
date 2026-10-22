#!/usr/bin/env bash
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 TPL_LAYER_KEYOUT=1 DRIVE_RANGE_PHASES=decode,prefill2 DRIVE_STARTUP_STEPS=10
echo "$(date +%T) memo=0"; DIAG27_REFUSAL_MEMO=0 ./gpulock.sh 3 3600 ./startup27.sh run vt rm0 > logs/rw_vtrm0.log 2>&1
echo "$(date +%T) memo=0 rc=$?"
