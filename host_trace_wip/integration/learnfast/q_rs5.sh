#!/usr/bin/env bash
# q_rs5.sh: wip14 preview sizing (step_ms12 + core memo + symbolic masked setitem + per-attempt cuts), build_cpp_ms9, WITHOUT the
# gdn_maskfill overlay; sections, templates, pin refinement, ARMV_PIN_UNPINNED (labelled). GPU 0 untimed (idle otherwise).
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_ENTRY_FORMS=1 DRIVE_STARTUP_STEPS=800
echo "$(date +%T) rs5"; ./gpulock.sh 0 3600 ./startup27.sh run vt rs5 > logs/rw_vtrs5.log 2>&1
echo "$(date +%T) rs5 rc=$?"
