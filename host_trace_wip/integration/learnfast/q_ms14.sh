#!/usr/bin/env bash
# q_ms14.sh: rs5-style cold start (vt RANGE=1, edge phases only, 800-step stream; sections on with ARMV_MEMORY=eager, pin refinement,
# ARMV_PIN_UNPINNED) on build_cpp_ms9, GPU 0 untimed, load gate before each lock:
#   ms14: step_ms14 (core memo, per-attempt cuts, symbolic masked setitem), no overlay
#   ms13: step_ms13 + gdn_maskfill overlay + the prototype refusal memo (ms13 has no core memo)
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=2 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=800
for R in ${ROUNDS:-1}; do
  for t in ms14 ms13; do
    w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
    echo "$(date +%T) $t r$R load $(cut -d' ' -f1 /proc/loadavg)"
    if [ $t = ms14 ]; then
      HEAD_WT=$I/snap/step_ms14 MEMO=0 ./gpulock.sh 0 3600 ./startup27.sh run vt ${t}_r$R > logs/rw_vt${t}_r$R.log 2>&1
    else
      HEAD_WT=$I/snap/step_ms13 MEMO=1 DIAG27_GDN_MASKFILL=1 ./gpulock.sh 0 3600 ./startup27.sh run vt ${t}_r$R > logs/rw_vt${t}_r$R.log 2>&1
    fi
    echo "$(date +%T) $t r$R rc=$?"
  done
done
