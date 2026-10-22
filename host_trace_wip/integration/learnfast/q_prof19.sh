#!/usr/bin/env bash
# q_prof19.sh: untimed per-part cProfile (trace / redispatch / settle) on step_ms19 with xvr, edge phases + 50 steps. Waits for q_ms19.done.
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
while [ ! -e logs/q_ms19.done ]; do sleep 30; done
export SNAP=step_ms19 INST=$I/build_cpp_ms12a/install
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh | sed 's/torch_ext_ms16/torch_ext_ms19/g')
STEPS=50 startup prof19x DIAG27_XVR=1 DIAG27_PROFILE_PARTS=$L/out/prof19x
touch logs/q_prof19.done
