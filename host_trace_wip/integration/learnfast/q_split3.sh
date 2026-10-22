#!/usr/bin/env bash
# q_split3.sh (coordinator 10-16), step_ms16 / build_cpp_ms11, GPU 0 untimed, gates; stop on a CUDA error:
#  1. nsys: decode bs 1, split off then on, the last 6 measured steps of each in a cudaProfiler range (capture-range repeat)
#  2. interleaved off/on x3: alternating 9:16 and mixed 8x512 only, walls + launch shim + pre-launch breakdown (no profiler)
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
cd $L
mkdir -p $L/tmp_nsys out/nsys
export TMPDIR=$L/tmp_nsys
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_SPLIT_PASS=walls
gate() { w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done; }
cudaerr() { grep -qiE "CUDA error|illegal instruction|illegal memory access" "$@"; }
gate; echo "$(date +%T) nsys"
# nsys wraps the vt run line (startup27.sh PY_PREFIX)
DRIVE_RANGE_PHASES=decode DRIVE_SPLIT_BS=1 DRIVE_SPLIT_ONLY_DECODE=1 DRIVE_SPLIT_STEPS=16 DRIVE_SPLIT_NSYS_STEPS=6 \
  PY_PREFIX="$NSYS profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --force-overwrite=true -o $L/out/nsys/split_bs1" \
  ./gpulock.sh 0 3600 ./startup27.sh run vt nsys1 > logs/rw_vtnsys1.log 2>&1
echo "$(date +%T) nsys rc=$? $(ls out/nsys)"; cudaerr logs/rw_vtnsys1.log && { echo "CUDA ERROR: stop"; exit 1; }
gate; echo "$(date +%T) interleaved"
DRIVE_RANGE_PHASES=decode,mixed DRIVE_SPLIT_BS= DRIVE_SPLIT_MODES=off,on,off,on,off,on DRIVE_SPLIT_STEPS=24 LD_PRELOAD=$L/shim/launchshim2.so \
  ./gpulock.sh 0 3600 ./startup27.sh run vt split3 > logs/rw_vtsplit3.log 2>&1
echo "$(date +%T) interleaved rc=$? rows $(grep -c SPLIT_AB_ROW logs/rw_vtsplit3.log)"; cudaerr logs/rw_vtsplit3.log && echo "CUDA ERROR"
