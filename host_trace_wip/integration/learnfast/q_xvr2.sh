#!/usr/bin/env bash
# q_xvr2.sh: xvr with the extent-read watch, validation mode in the --check runs (every hit also runs the body and compares
# the renamed entry); then rs5-style x16_r3 / x16_r4 (watch on, validation off). Waits for q_gcu.done.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
while [ ! -e logs/q_gcu.done ]; do sleep 30; done
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh)
chk a xvr2chk_a DIAG27_XVR=1 XVR_VALIDATE=1
grep -q "validate_mismatch\|validate_fresh_raised" logs/xvr2chk_a.log && { echo "VALIDATE MISMATCH: stop"; exit 1; }
chk b xvr2chk_b DIAG27_XVR=1 XVR_VALIDATE=1
grep -q "validate_mismatch\|validate_fresh_raised" logs/xvr2chk_b.log && { echo "VALIDATE MISMATCH: stop"; exit 1; }
startup x16_r3 DIAG27_XVR=1
startup x16_r4 DIAG27_XVR=1
touch logs/q_xvr2.done
