#!/usr/bin/env bash
# q_rmemo.sh: refusal memo equality (coordinator 10-15): vt RANGE=1 phases decode,prefill2 + 10 stream steps on step_ms10, memo off
# then on; compare trace count and refusal set. GPU 3, untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=${UTIL:-0.6} HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 TPL_LAYER_KEYOUT=1 DRIVE_RANGE_PHASES=${PHASES:-decode,prefill2} DRIVE_STARTUP_STEPS=10
for m in 0 1; do
  echo "$(date +%T) memo=$m"; DIAG27_REFUSAL_MEMO=$m ./gpulock.sh 3 3600 ./startup27.sh run vt rm$m > logs/rw_vtrm$m.log 2>&1
  echo "$(date +%T) memo=$m rc=$?"
done
touch logs/q_rmemo.done
