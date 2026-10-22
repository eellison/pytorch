#!/usr/bin/env bash
# Ours on the eager forward (vrs) on GPU 0 / CPUs 0-35 after the breakable arm, for swph1 and cbh3 (3 runs each, strict windows):
# the uncompiled group (vLLM default-nc, vLLM breakable, vrs) on one GPU next to the compiled group (vLLM default, vcsl).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while [ "$(grep -c 'CBH3_DONE' serve/logs/swp_timeline.log)" -lt 2 ]; do sleep 60; done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vrs" bash serve/ix/set_chain.sh
SET=cbh3 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vrs" bash serve/ix/set_chain.sh
