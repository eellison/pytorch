#!/usr/bin/env bash
# q_splitab.sh: 27B split A/B on step_ms16 / build_cpp_ms11 in two processes (walls; then the profiler busy pass, CUPTI attached),
# DRIVE_SPLIT_STEPS=16, decode + mixed warm-up first. GPU 0 untimed; stop on a CUDA error.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_RANGE_PHASES=decode,mixed DRIVE_SPLIT_STEPS=16
for ps in walls busy; do
  w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  echo "$(date +%T) $ps"
  DRIVE_SPLIT_PASS=$ps ./gpulock.sh 0 3600 ./startup27.sh run vt splitab_$ps > logs/rw_vtsplitab_$ps.log 2>&1
  echo "$(date +%T) $ps rc=$? rows $(grep -c SPLIT_AB_ROW logs/rw_vtsplitab_$ps.log)"
  grep -qiE "CUDA error|illegal instruction|illegal memory access" logs/rw_vtsplitab_$ps.log && { echo "CUDA ERROR: stop"; exit 1; }
done
