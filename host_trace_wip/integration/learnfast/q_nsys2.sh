#!/usr/bin/env bash
# q_nsys2.sh: nsys timeline, decode bs 1, split off then on: graph-level CUDA tracing (--cuda-graph-trace=graph, no osrt), the last 6
# measured steps of each mode in a cudaProfiler range; nsys stops the app after 900 s. GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
cd $L
export TMPDIR=$L/tmp_nsys
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_SPLIT_PASS=walls
export DRIVE_RANGE_PHASES=decode DRIVE_SPLIT_BS=1 DRIVE_SPLIT_ONLY_DECODE=1 DRIVE_SPLIT_STEPS=12 DRIVE_SPLIT_NSYS_STEPS=6
export PY_PREFIX="$NSYS profile --trace=cuda --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --duration=900 --kill=sigkill --force-overwrite=true -o $L/out/nsys/split_bs1"
./gpulock.sh 0 3600 ./startup27.sh run vt nsys2 > logs/rw_vtnsys2.log 2>&1
echo "$(date +%T) rc=$? $(ls out/nsys) cuda_err $(grep -ciE 'CUDA error|illegal instruction|illegal memory' logs/rw_vtnsys2.log)"
