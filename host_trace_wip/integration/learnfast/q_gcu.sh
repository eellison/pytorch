#!/usr/bin/env bash
# q_gcu.sh: gc_untrack walk A/B on the head (rs5-style, GPU 3): g16_r1 (DIAG27_GC_UNTRACK=0), b16_r4 (on, timer), g16_r2 (off).
# GC seconds per run via DIAG27_GC_TIME=1 in all three (gen-2 cost of what stays tracked).
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh)
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
startup g16_r1 DIAG27_GC_UNTRACK=0 DIAG27_GC_TIME=1
startup b16_r4 DIAG27_GC_TIME=1
startup g16_r2 DIAG27_GC_UNTRACK=0 DIAG27_GC_TIME=1
touch logs/q_gcu.done
