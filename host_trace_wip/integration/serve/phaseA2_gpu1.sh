#!/usr/bin/env bash
# Phase A correctness, round 2 (GPU 1, one gpu1.lock hold): eager and default (round 1 failed at startup: foreign
# processes on GPU 1 held/released ~180 GB around vLLM's memory check). Fixed KV pool (--kv-cache-memory-bytes 32 GiB)
# and util 0.15 (only the startup free-memory check uses it then), so foreign memory swings do not break startup.
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
export TAG=${TAG:-a2} GPU=1 UTIL=0.15 SERVER_EXTRA="--kv-cache-memory-bytes 34359738368"
echo "$(date +%T) waiting for gpu1.lock"
exec flock -x $LOCK bash -c '
echo "$(date +%T) holding gpu1.lock"
S=$0
timeout 1800 bash $S/session.sh eager check1 burst8
timeout 2400 bash $S/session.sh default check1 burst8
timeout 1800 bash $S/session.sh eagerm burst8
echo PHASEA2_GPU1_DONE' $SV
