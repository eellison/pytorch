#!/usr/bin/env bash
# After the churn set: the sweep set under the decode convention update (swph1, vllm/round_sweep2_line.sh): decode at ~1K and ~8K
# context, 128 measured steps per point, plus prefill / mixed / alternating; GPU 0 / CPUs 0-35, default / fullnc / vcsl, 3 runs.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q CBH3_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl" bash serve/ix/set_chain.sh
