#!/usr/bin/env bash
# q_combo.sh: "how far" arm: xvr prototype + gc_untrack walk off (an existing core switch, labelled), rs5-style x2. Waits for q_prof16.done.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
while [ ! -e logs/q_prof16.done ]; do sleep 30; done
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh)
startup xg16_r1 DIAG27_XVR=1 DIAG27_GC_UNTRACK=0 DIAG27_GC_TIME=1
startup xg16_r2 DIAG27_XVR=1 DIAG27_GC_UNTRACK=0 DIAG27_GC_TIME=1
touch logs/q_combo.done
