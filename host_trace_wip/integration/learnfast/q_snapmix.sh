#!/usr/bin/env bash
# q_snapmix.sh (low priority): allocator snapshots over the mixed phase (after decode + prefill1), ours ms15 run_buffer + sections
# and vLLM eager, to name ours' extra ~300 MiB at the mixed peak. GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=3 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 DRIVE_RANGE_WARM=1 DRIVE_RANGE_PHASES=decode,prefill1,mixed DRIVE_STARTUP_STEPS=5
export DRIVE_MEM_SNAPSHOT_PHASE=mixed
DRIVE_MEM_SNAPSHOT=$L/out/memsnap_mixed_ours15.pickle SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ./gpulock.sh 0 3600 ./startup27.sh run vt snapmix > logs/snapmix_ours.log 2>&1
echo "$(date +%T) ours rc=$?"
DRIVE_MEM_SNAPSHOT=$L/out/memsnap_mixed_eager15.pickle ./gpulock.sh 0 3600 ./startup27.sh run eager snapmix > logs/snapmix_eager.log 2>&1
echo "$(date +%T) eager rc=$?"
