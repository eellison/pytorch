#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
OWN_BUSY_FRAC=0.25 SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
echo "$(date +%T) BNC_R3_RERUN_DONE" >> serve/logs/swp_timeline.log
