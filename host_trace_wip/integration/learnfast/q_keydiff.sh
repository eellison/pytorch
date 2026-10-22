#!/usr/bin/env bash
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
SNAP=step_ms16 INST=$I/build_cpp_ms11/install
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh)
STEPS=5 startup keydiff DIAG27_XVR=1 XVR_KEYDIFF=1 DRIVE_RANGE_PHASES=decode,prefill1
