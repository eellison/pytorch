#!/usr/bin/env bash
# q_ms22.sh: the in-core number (coordinator 10-18): step_ms22 (core xvr = ms21 + core GDN template-key fix) on build_cpp_ms12b,
# GPU 0, CORE versions only (xvr.py and TPL_DECLVAL off; tpl_layer keyout as in every run since 10-15). Waits for the snap and
# for logs/ms22.go. n27wf --check a / b first (bitwise + tpl confirmations), then rs5-style c22_r1, baseline b20_r3 (step_ms20,
# same GPU), c22_r2.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -d $I/snap/step_ms22 ] || [ ! -e logs/ms22.go ]; do sleep 60; done
export SNAP=step_ms22 INST=$I/build_cpp_ms12b/install
EXT=$L/torch_ext_ms20
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk a c22chk_a
grep -q "confirmation mismatch" logs/c22chk_a.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
chk b c22chk_b
grep -q "confirmation mismatch" logs/c22chk_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
startup c22_r1 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
SNAP=step_ms20 startup b20_r3 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
startup c22_r2 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1
touch logs/q_ms22.done
