#!/usr/bin/env bash
# chain13_g2.sh: gb300-box GPU 2, the w set (all 3 rounds; coordinator 15:3x), in parallel with GPU 0's d set and the 8B lane.
# Starts when logs/mon/g2.go exists (created once the 27B lane's churn rerun has released gpu2.lock) and the fast cuBLAS
# bindings exist (made by d r1's v arm on GPU 0; no second learner). CPUs 36-71, gpu2.lock per arm run after the own-core gate.
# Set g2w = gb300-box GPU 2. DRIVE_FWD_SYNC=0 (round_fn_line w). Touches out/timing/CHAIN13_W1_DONE after w r1.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain13_g2.pid
while [ ! -f $D/logs/mon/g2.go ] || [ ! -f $D/bindings/fn_v.fast.pkl ]; do sleep 60; done
echo "$(date +%T) chain13_g2: go"
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain13_g2: $*"; env "$@" bash $D/round_fn_line2.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain13_g2: round rc $rc ($*)"; return 0; }
run SET=g2w KIND=w R=1
touch $D/out/timing/CHAIN13_W1_DONE; echo "$(date +%T) chain13_g2: w r1 done, CHAIN13_W1_DONE touched"
for r in 2 3; do run SET=g2w KIND=w R=$r; done
echo "$(date +%T) chain13_g2 done"
