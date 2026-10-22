#!/usr/bin/env bash
# q_rs6.sh: wip14 (no overlay) + derived warm-up phases (attn split classes, mixed tile classes) after the edge phases; GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=${HEAD_WT:-/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14} LF_GPU=${G:-3} LF_CPUS=${CPUS:-108-143} HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=800 DIAG27_FAILING_GUARDS=1 DIAG27_FAILING_GUARDS_N=400
export DRIVE_RANGE_PHASES=${PHASES:-decode,prefill1,prefill2,mixed,attn,mixedx}
echo "$(date +%T) ${R:-rs6}"; ./gpulock.sh ${G:-3} 3600 ./startup27.sh run vt ${R:-rs6} > logs/rw_vt${R:-rs6}.log 2>&1
echo "$(date +%T) rc=$?"
