#!/usr/bin/env bash
# q_rs4.sh: vt RANGE=1 sizing on HEAD=2 (step_ms12 / build_cpp_ms9): sections, templates, memo, pin refinement, GDN nums pinned
# argument (ARMV_PIN_UNPINNED, labelled), proposed GDN masked_fill_ overlay (DIAG27_GDN_MASKFILL, labelled). GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DIAG27_GDN_MASKFILL=1 DRIVE_ENTRY_FORMS=1
echo "$(date +%T) rs4"; DRIVE_STARTUP_STEPS=800 ./gpulock.sh 3 3600 ./startup27.sh run vt rs4 > logs/rw_vtrs4.log 2>&1
echo "$(date +%T) rs4 rc=$?"
