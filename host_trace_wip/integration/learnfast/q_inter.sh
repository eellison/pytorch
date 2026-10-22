#!/usr/bin/env bash
# q_inter.sh: split off/on interleaved x3 (alternating 9:16 and mixed 8x512 only), walls + launch shim + pre-launch breakdown, no
# profiler; step_ms16 / build_cpp_ms11, GPU 2 (CPUs 36-71) untimed; stop on a CUDA error.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=2 LF_CPUS=36-71 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_SPLIT_PASS=walls
export DRIVE_RANGE_PHASES=decode,mixed DRIVE_SPLIT_BS= DRIVE_SPLIT_MODES=off,on,off,on,off,on DRIVE_SPLIT_STEPS=24 LD_PRELOAD=$L/shim/launchshim2.so
w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
./gpulock.sh 2 3600 ./startup27.sh run vt inter > logs/rw_vtinter.log 2>&1
echo "$(date +%T) rc=$? rows $(grep -c SPLIT_AB_ROW logs/rw_vtinter.log) cuda_err $(grep -ciE 'CUDA error|illegal instruction|illegal memory' logs/rw_vtinter.log)"
