#!/usr/bin/env bash
# q_val.sh (after the memo gate), GPU 3 untimed, head step_ms11 (Delta 3 + desync fix):
#  rs2: the rs1 configuration (vt RANGE=1, sections, templates, memo) with the prototype costs cut (FNRESOLVE shallow resolve,
#       opdeps classify O(ops), DIAG27_QUIET) -> ready vs rs1's 453 s
#  val1: + TPL_CONFIRM=oplayer TPL_CONFIRM_VALIDATE=1 + entry-form diagnostic, 300 stream steps
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while pgrep -f "gate_tpl.sh $L/memo_snap" >/dev/null; do sleep 30; done
export UTIL=0.6 HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1
echo "$(date +%T) rs2"; DRIVE_STARTUP_STEPS=800 DRIVE_ENTRY_FORMS=1 ./gpulock.sh 3 3600 ./startup27.sh run vt rs2 > logs/rw_vtrs2.log 2>&1
echo "$(date +%T) rs2 rc=$?"
echo "$(date +%T) val1"; DRIVE_STARTUP_STEPS=300 TPL_CONFIRM=oplayer TPL_CONFIRM_VALIDATE=1 ./gpulock.sh 3 3600 ./startup27.sh run vt val1 > logs/rw_vtval1.log 2>&1
echo "$(date +%T) val1 rc=$?"
