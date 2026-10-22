#!/usr/bin/env bash
# q_wip23.sh: gc full-walk trigger A/B (core agent 10-18): snap/wip23 (= ms22 + full walk by objects untracked) vs ms22's c22 runs,
# rs5-style, core versions only, per-settle gc log. GPU 0. Waits for q_ms22.done.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_ms22.done ]; do sleep 60; done
export SNAP=wip23 INST=$I/build_cpp_ms12b/install
EXT=$L/torch_ext_ms20
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
startup w23_r1 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
SNAP=step_ms22 startup c22_r3 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
startup w23_r2 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
touch logs/q_wip23.done
