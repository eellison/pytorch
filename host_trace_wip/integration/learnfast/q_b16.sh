#!/usr/bin/env bash
# q_b16.sh [TAG]: rs5-style 27B cold start on the head (step_ms16 / build_cpp_ms11): ours vt, RANGE=1 edge warm-up, sections on,
# run_buffer memory, 800-step stream. GPU 3 (learnfast lane after the 10-17 reboot), untimed-shared rules do not apply: we own GPU 3.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
TAG=${1:-b16_r1} SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/$SNAP HEAD3_INSTALL=$INST HEAD3_EXT=$L/torch_ext_ms16 HOLD_S=2700 RUN_TIMEOUT=2600
export RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=800
gates; echo "$(date +%T) $TAG"
p0=$(awk '/^full/{split($5,a,"="); print a[2]}' /proc/pressure/memory)
./gpulock.sh 3 3600 ./startup27.sh run vt $TAG > logs/rw_vt$TAG.log 2>&1
rc=$?
p1=$(awk '/^full/{split($5,a,"="); print a[2]}' /proc/pressure/memory)
echo "$(date +%T) $TAG rc=$rc psi_full_s_during $(python3 -c "print(round(($p1 - $p0) / 1e6, 1))")"
cudaerr logs/rw_vt$TAG.log && echo "CUDA ERROR in $TAG"
