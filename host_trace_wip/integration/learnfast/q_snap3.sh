#!/usr/bin/env bash
# q_snap3.sh: the candidate fix, ARMV_MEMORY=eager (sections' temporaries made and freed at their op in eager order), decode then
# the prefill sweep; GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while pgrep -f "q_snap.sh" >/dev/null; do sleep 20; done
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 DRIVE_RANGE_WARM=1 DRIVE_RANGE_PHASES=decode,psweep DRIVE_STARTUP_STEPS=5 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
echo "$(date +%T) eager-mem"; ARMV_MEMORY=eager ./gpulock.sh 0 3600 ./startup27.sh run vt snapE > logs/snapE.log 2>&1
echo "$(date +%T) rc=$? $(grep PSWEEP logs/snapE.log)"
