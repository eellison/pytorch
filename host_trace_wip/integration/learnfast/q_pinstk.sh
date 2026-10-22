#!/usr/bin/env bash
# q_pinstk.sh: where the GDN sequence-count pins (s == k on the pinned nums length) are recorded: decode phase of vt RANGE=1 with
# ARMV_PIN_UNPINNED=1 + DIAG27_GUARD_STACKS=1 + PROV_PIN_ATEN=1, HEAD=2 (step_ms12 / build_cpp_ms9). GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DIAG27_GUARD_STACKS=1
export DRIVE_RANGE_PHASES=decode DRIVE_STARTUP_STEPS=10
# HEAD=2 has no v bindings yet: load the ms8 ones (a diagnostic; harvests now work for any miss)
ln -sf $L/out/bind_startup27_ht_v.pkl $L/out/bind_startup27_h2t_v.pkl
./gpulock.sh 3 3600 ./startup27.sh run vt pinstk > logs/rw_vtpinstk.log 2>&1
echo "$(date +%T) rc=$?"
