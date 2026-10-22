#!/usr/bin/env bash
# chain13_g0.sh: gb300-box GPU 0, d set rounds 2 / 3 (d20; d r1 = the round chain13 started, PID $1, left to finish). Paused
# (no lock taken) until that round has exited AND the 8B lane's cbh4b is done (serve/out/CBH4B_DONE; coordinator 15:3x).
# CPUs 0-35, gpu0.lock per arm run taken after the own-core gate (round_fn_line2.sh). Sets g0d / g0d20 = gb300-box GPU 0.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain13_g0.pid
DONE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/out/CBH4B_DONE
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep round_fn_line)" ] || [ ! -f $DONE ]; do sleep 60; done
echo "$(date +%T) chain13_g0: d r1 round exited and CBH4B_DONE exists"
export HT_GPU=0 HT_CPUS=0-35 FN_FI_TAG=f0 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain13_g0: $*"; env "$@" bash $D/round_fn_line2.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain13_g0: round rc $rc ($*)"; return 0; }
run SET=g0d KIND=d R=1   # resumes d r1 if any arm left no JSON (outputs that exist are skipped)
for r in 2 3; do run SET=g0d20 KIND=d20 R=$r; done
echo "$(date +%T) chain13_g0 done"
