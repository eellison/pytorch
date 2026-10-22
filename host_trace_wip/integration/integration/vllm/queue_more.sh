#!/usr/bin/env bash
# After startup round 1 stops: startup round 2, timing round 3, startup round 3 — one GPU 0 hold each, in sequence.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "stopped startup_c2 after round 1\|^ALLDONE" vllm/logs/startup_c2.log; do sleep 30; done
ROUNDS=2 bash vllm/gpu0_job.sh vllm/startup_c2_rounds.sh > vllm/logs/startup_c2_r2.log 2>&1
ROUNDS=3 PAD_ARMS=0 bash vllm/gpu0_job.sh vllm/timing_c2.sh > vllm/logs/timing_c2_r3.log 2>&1
ROUNDS=3 bash vllm/gpu0_job.sh vllm/startup_c2_rounds.sh > vllm/logs/startup_c2_r3.log 2>&1
echo ALLDONE
