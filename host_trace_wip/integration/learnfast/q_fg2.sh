#!/usr/bin/env bash
# q_fg2.sh: warm-up coverage data for the derived warm-up set: rs5's config (wip14, no overlay) + failing guards at every
# redispatch (DIAG27_FAILING_GUARDS, 400 per op), per-phase / stream peak memory. GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=800 DIAG27_FAILING_GUARDS=1 DIAG27_FAILING_GUARDS_N=400
echo "$(date +%T) fg2"; ./gpulock.sh 3 3600 ./startup27.sh run vt fg2 > logs/rw_vtfg2.log 2>&1
echo "$(date +%T) fg2 rc=$?"
