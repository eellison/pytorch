#!/usr/bin/env bash
# q_ms19.sh: the head step_ms19 / build_cpp_ms12a (incremental gc_untrack): xvr soundness first (n27wf --check a / b with
# XVR_VALIDATE + TPL_DECLVAL; the first also builds the fresh torch extensions dir), then rs5-style baseline / xvr / xvr+declval x2.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
export SNAP=step_ms19 INST=$I/build_cpp_ms12a/install
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed 's/torch_ext_ms16/torch_ext_ms19/g')
chk a x19chk_a DIAG27_XVR=1 XVR_VALIDATE=1 TPL_DECLVAL=1 XVR_KEYDIFF=1
grep -q "validate_mismatch\|validate_fresh_raised" logs/x19chk_a.log && { echo "VALIDATE MISMATCH: stop"; exit 1; }
chk b x19chk_b DIAG27_XVR=1 XVR_VALIDATE=1 TPL_DECLVAL=1
grep -q "validate_mismatch\|validate_fresh_raised" logs/x19chk_b.log && { echo "VALIDATE MISMATCH: stop"; exit 1; }
grep -q "confirmation mismatch" logs/x19chk_a.log logs/x19chk_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
startup b19_r1 DIAG27_GC_TIME=1
startup x19_r1 DIAG27_XVR=1 DIAG27_GC_TIME=1
startup xd19_r1 DIAG27_XVR=1 TPL_DECLVAL=1 DIAG27_GC_TIME=1
startup b19_r2 DIAG27_GC_TIME=1
startup x19_r2 DIAG27_XVR=1 DIAG27_GC_TIME=1
startup xd19_r2 DIAG27_XVR=1 TPL_DECLVAL=1 DIAG27_GC_TIME=1
touch logs/q_ms19.done
