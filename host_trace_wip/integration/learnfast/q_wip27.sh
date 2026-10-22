#!/usr/bin/env bash
# q_wip27.sh: snap/wip27 (= ms25 + check_section caching) on build_cpp_ms12c: n27wf --check b first (GDN templates; bitwise
# + tpl confirmations), then rs5-style w27_r1 with DIAG27_RDCOST (pair with wip25's w25_r1); then the nsys retry on step_ms24
# (coordinator 10-18: 27B decode, node-level CUDA graph trace, --duration 600 --kill, TMPDIR on /data, untimed). GPU 0.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_nsys24.done ]; do sleep 60; done
export SNAP=wip27 INST=$I/build_cpp_ms12c/install
EXT=$L/torch_ext_ms12c
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk b w27chk_b
grep -q "confirmation mismatch" logs/w27chk_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
startup w27_r1 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1 DIAG27_RDCOST=1
touch logs/q_wip27.done
