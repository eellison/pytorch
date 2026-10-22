#!/usr/bin/env bash
# GPU 0 / CPUs 0-35 after the cold inits: rerun runs flagged for high host load at start (renamed *hl_*): today swph1 bnc r1.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q COLD_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
echo "$(date +%T) FLAGGED_RERUNS_DONE" >> serve/logs/swp_timeline.log
