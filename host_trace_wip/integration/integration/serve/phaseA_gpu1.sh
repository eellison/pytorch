#!/usr/bin/env bash
# Phase A correctness batch on GPU 1 (one gpu1.lock hold for all arms): Qwen3-8B under `vllm serve`, fixed prompt set at
# concurrency 1 and 8, V counters, V adapter check during a burst, a small bench-serve smoke per shape.
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
export TAG=${TAG:-a1} GPU=1
echo "$(date +%T) waiting for gpu1.lock"
exec flock -x $LOCK bash -c '
echo "$(date +%T) holding gpu1.lock"
S=$0
timeout 3000 bash $S/session.sh V check1 ctr:c1 checkon burst8:on ctr:burst checkoff burst8 ctr:burst2 sweep:s1k:922:922:0.1111:2 sweep:s8k:7373:922:0.1111:2 ctr:end summary:end
timeout 1800 bash $S/session.sh eagerm check1 burst8 ctr:end
timeout 1800 bash $S/session.sh eager check1 burst8
timeout 2400 bash $S/session.sh default check1 burst8
echo PHASEA_GPU1_DONE' $SV
