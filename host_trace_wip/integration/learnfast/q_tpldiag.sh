#!/usr/bin/env bash
# q_tpldiag.sh: GDN template predicate diagnosis (untimed): decode + prefill1 phases, 5 stream steps, xvr + XVR_TPLDIAG=1.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh)
STEPS=5 startup tpldiag DIAG27_XVR=1 XVR_TPLDIAG=1 DRIVE_RANGE_PHASES=decode,prefill1
touch logs/q_tpldiag.done
