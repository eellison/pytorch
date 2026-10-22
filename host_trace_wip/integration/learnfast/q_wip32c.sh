#!/usr/bin/env bash
# q_wip32c.sh: check b on the final wip32 (17:34:55, memo scoped to _Trace.op) + wip31 (= ms28) round for the memory/time pair. Waits for q_wip32b.done.
# default on) on build_cpp_ms12d. GPU 2 (gpu2.lock, CPUs 36-71; gpulock waits while another user's process is on GPU 2; STEP_OFF=1).
# n27wf --check b on wip32 (defaults, split on) first, then rs5-style: w32s0 (split off: isolates the memo vs ms27's m27/w28),
# w31 (split on only), w32 (both).
export LANE_GPU=2 LANE_CPUS=36-71 STEP_OFF=1
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_wip32b.done ]; do sleep 30; done
export INST=$I/build_cpp_ms12d/install SNAP=wip32
EXT=$L/torch_ext_ms12d
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 2/g")
chk b w32chk3_b
grep -q "confirmation mismatch" logs/w32chk3_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
SNAP=wip31 startup w31_r2 DIAG27_GC_TIME=1
touch logs/q_wip32c.done
