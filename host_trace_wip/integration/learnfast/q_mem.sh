#!/usr/bin/env bash
# q_mem.sh (coordinator 10-15: "live memory = eager" check): 27B prefill1 (edges to 16376) + mixed (8 decoders + T) phases + a
# 100-step stream; arms vLLM eager, ours V run_buffer (adapter default), ours V eager memory. wip14 / build_cpp_ms9, KV 64 GiB,
# max_memory_allocated / reserved per phase. GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150
export RANGE=1 DRIVE_RANGE_WARM=1 DRIVE_RANGE_PHASES=prefill1,mixed DRIVE_STARTUP_STEPS=100
echo "$(date +%T) eager"; ./gpulock.sh 3 3600 ./startup27.sh run eager mem1 > logs/mem_eager.log 2>&1; echo "$(date +%T) rc=$?"
export SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
for m in run_buffer eager; do
  echo "$(date +%T) ours $m"; ARMV_MEMORY=$m ./gpulock.sh 3 3600 ./startup27.sh run vt mem_$m > logs/mem_ours_$m.log 2>&1; echo "$(date +%T) rc=$?"
done
