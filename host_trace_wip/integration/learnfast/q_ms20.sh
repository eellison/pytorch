#!/usr/bin/env bash
# q_ms20.sh [SNAP INST EXT TAGSFX]: GPU 0 (gpu0.lock, CPUs 0-35; coordinator 10-18) on a head (default step_ms20 /
# build_cpp_ms12b): n27wf --check a / b with xvr + TPL_DECLVAL + XVR_VALIDATE, then rs5-style baseline / xvr / xvr+declval x2.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
export SNAP=${1:-step_ms20} INST=${2:-$I/build_cpp_ms12b/install}
EXT=${3:-$L/torch_ext_ms20} S=${4:-20}
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk a x${S}chk_a DIAG27_XVR=1 XVR_VALIDATE=1 TPL_DECLVAL=1
grep -q "validate_mismatch\|validate_fresh_raised\|confirmation mismatch" logs/x${S}chk_a.log && { echo "MISMATCH: stop"; exit 1; }
chk b x${S}chk_b DIAG27_XVR=1 XVR_VALIDATE=1 TPL_DECLVAL=1
grep -q "validate_mismatch\|validate_fresh_raised\|confirmation mismatch" logs/x${S}chk_b.log && { echo "MISMATCH: stop"; exit 1; }
for r in 1 2; do
  startup b${S}_r$r DIAG27_GC_TIME=1
  startup x${S}_r$r DIAG27_XVR=1 DIAG27_GC_TIME=1
  startup xd${S}_r$r DIAG27_XVR=1 TPL_DECLVAL=1 DIAG27_GC_TIME=1
done
touch logs/q_ms${S}.done
