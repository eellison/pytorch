#!/usr/bin/env bash
# q_trace.sh: device timeline of decode bs 1, split off then on, from torch.profiler (CUPTI kernel activity; nsys hangs the native
# replay): 6-step windows exported as chrome traces (out/split_trace_{1,2}.json = off, on). GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms16 HEAD3_INSTALL=$I/build_cpp_ms11/install HEAD3_EXT=$L/torch_ext_ms16 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_SPLIT_PASS=busy
export DRIVE_RANGE_PHASES=decode DRIVE_SPLIT_BS=1 DRIVE_SPLIT_ONLY_DECODE=1 DRIVE_SPLIT_TRACE=$L/out/split_trace
./gpulock.sh 0 3600 ./startup27.sh run vt trace1 > logs/rw_vttrace1.log 2>&1
echo "$(date +%T) rc=$? $(ls out/split_trace_* 2>/dev/null)"
