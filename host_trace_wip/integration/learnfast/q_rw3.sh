#!/usr/bin/env bash
# q_rw3.sh: vt RANGE=1 sizing with sections on (Delta 3 snap), cross-layer templates (key-outs + confirmation), refusal memo.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=1 HEAD_WT=${HEAD_WT:-$L/sec_snap} LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 DRIVE_STARTUP_STEPS=${STEPS:-800}
echo "$(date +%T) vt RANGE=1 sections memo"; ./gpulock.sh 3 3600 ./startup27.sh run vt ${R:-s1} > logs/rw_vt${R:-s1}.log 2>&1
echo "$(date +%T) rc=$?"
