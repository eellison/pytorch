#!/usr/bin/env bash
# q_diag_fg.sh: warm-up coverage diagnostic: failing guards per op at every redispatch + guard stacks for the GDN pin, vt RANGE=1
# (sections, templates, memo) + 300 stream steps. GPU 3, untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while pgrep -f "q_prof_rd.sh" >/dev/null; do sleep 30; done
export UTIL=0.6 HEAD=1 HEAD_WT=$L/sec_snap LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 DRIVE_STARTUP_STEPS=300
export DIAG27_FAILING_GUARDS=1 DIAG27_GUARD_STACKS=1
echo "$(date +%T) fg"; ./gpulock.sh 3 3600 ./startup27.sh run vt fg1 > logs/rw_vtfg1.log 2>&1
echo "$(date +%T) rc=$?"
