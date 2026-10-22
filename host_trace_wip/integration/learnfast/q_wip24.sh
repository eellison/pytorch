#!/usr/bin/env bash
# q_wip24.sh: gc incremental-walk A/B (core agent 10-18): snap/wip24 on build_cpp_ms12c (sealed referents take no node), rs5-style,
# core versions only, per-settle gc log; a smoke check first (n27wf --check a). GPU 0. Waits for q_wip23.done and logs/wip24.go.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_wip23.done ] || [ ! -e logs/wip24.go ]; do sleep 60; done
export SNAP=${W24_SNAP:-wip24} INST=$I/build_cpp_ms12c/install
EXT=$L/torch_ext_ms12c
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk a w24chk_a
startup w24_r1 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
startup w24_r2 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
touch logs/q_wip24.done
