#!/usr/bin/env bash
# q_ms27.sh: step_ms27 (static-prefix pins one row per lowering) on build_cpp_ms12c, GPU 0: n27wf --check b, then rs5-style w28_r1
# (DIAG27_RDCOST timers) and m27_r2. Waits for q_nsysd.done.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_nsysd.done ]; do sleep 60; done
export SNAP=step_ms27 INST=$I/build_cpp_ms12c/install
EXT=$L/torch_ext_ms12c
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk b m27chk_b
grep -q "confirmation mismatch" logs/m27chk_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
startup w28_r1 DIAG27_GC_TIME=1 DIAG27_RDCOST=1
startup m27_r2 DIAG27_GC_TIME=1
touch logs/q_ms27.done
