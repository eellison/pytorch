#!/usr/bin/env bash
# q_mem2.sh: is the 199 GiB prefill1 peak (rs5/rs6/fg2) an interaction with the decode phase? ours V run_buffer, phases
# decode,prefill1, 20 stream steps; then vLLM eager with the same phases. GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 DRIVE_RANGE_WARM=1 DRIVE_RANGE_PHASES=decode,prefill1 DRIVE_STARTUP_STEPS=20
echo "$(date +%T) ours"; SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ./gpulock.sh 0 3600 ./startup27.sh run vt mem2 > logs/mem2_ours.log 2>&1; echo "$(date +%T) rc=$?"
echo "$(date +%T) eager"; ./gpulock.sh 0 3600 ./startup27.sh run eager mem2 > logs/mem2_eager.log 2>&1; echo "$(date +%T) rc=$?"
