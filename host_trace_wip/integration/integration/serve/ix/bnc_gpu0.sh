#!/usr/bin/env bash
# The breakable-graph arm (bnc) on GPU 0 / CPUs 0-35 after swph1, for the same sets: swph1 (decode 1K/8K, prefill, mixed, alternating)
# and cbh3 (churn), 3 runs each, so every line of the charts shares one GPU.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q SWPH1_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
SET=cbh3 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
