#!/usr/bin/env bash
# q_prof16.sh: untimed cProfile of _call_slow on the head with xvr on (rs5-style edge phases + 50 stream steps, run_buffer +
# sections), for the per-attempt redispatch overhead and trace cost hand-off. Waits for q_xvr2.done.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
while [ ! -e logs/q_xvr2.done ]; do sleep 30; done
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh)
STEPS=50 startup prof16x DIAG27_XVR=1 DIAG27_PROFILE_SLOW=$L/out/prof_slow_ms16_xvr.pstats
touch logs/q_prof16.done
