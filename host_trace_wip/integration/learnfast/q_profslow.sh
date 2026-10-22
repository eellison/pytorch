#!/usr/bin/env bash
# q_profslow.sh: at 07:00 (after the quiet window), cProfile of _call_slow on step_ms14, rs5-style config with ARMV_MEMORY=eager,
# then the same with ARMV_MEMORY=run_buffer for the A/B (both untimed, GPU 0, load gate before each lock).
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
while [ "$(date +%H%M)" -lt 0700 ] || [ "$(date +%H%M)" -gt 2200 ]; do sleep 60; done
export UTIL=0.6 HEAD=2 HEAD_WT=$I/snap/step_ms14 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
export DRIVE_STARTUP_STEPS=50
for m in eager run_buffer; do
  w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  echo "$(date +%T) $m"
  ARMV_MEMORY=$m DIAG27_PROFILE_SLOW=$L/out/prof_slow_ms14_$m.pstats ./gpulock.sh 0 3600 ./startup27.sh run vt profslow_$m > logs/rw_vtprofslow_$m.log 2>&1
  echo "$(date +%T) $m rc=$?"
done
