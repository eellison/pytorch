#!/usr/bin/env bash
# q_prof_rd.sh: cProfile of every redispatch attempt over the vt RANGE=1 warm-up (sections on, templates, memo), untimed, GPU 3.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while pgrep -f "q_secchk.sh" >/dev/null; do sleep 30; done
export UTIL=0.6 HEAD=1 HEAD_WT=$L/sec_snap LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 DRIVE_STARTUP_STEPS=50
export DIAG27_PROFILE_REDISPATCH=$L/out/prof_redispatch.pstats
echo "$(date +%T) prof"; ./gpulock.sh 3 3600 ./startup27.sh run vt prof1 > logs/rw_vtprof1.log 2>&1
echo "$(date +%T) rc=$?"
