#!/usr/bin/env bash
# q_splitab2.sh: split A/B at decode bs 1 and 32 only, walls process with the launch shim (time inside each cudaGraphLaunch and the
# host gap before it), graph launches / kernel + memory node sets / piece skips per step; no profiler. step_ms16, GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_RANGE_PHASES=decode DRIVE_SPLIT_STEPS=32
export DRIVE_SPLIT_PASS=walls DRIVE_SPLIT_BS=1,32 DRIVE_SPLIT_ONLY_DECODE=1 LD_PRELOAD=$L/shim/launchshim2.so
w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
./gpulock.sh 0 3600 ./startup27.sh run vt splitab2 > logs/rw_vtsplitab2.log 2>&1
echo "$(date +%T) rc=$? rows $(grep -c SPLIT_AB_ROW logs/rw_vtsplitab2.log) cuda_err $(grep -ciE 'CUDA error|illegal instruction|illegal memory' logs/rw_vtsplitab2.log)"
