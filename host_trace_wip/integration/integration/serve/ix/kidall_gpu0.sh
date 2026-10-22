#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
OWN_BUSY_FRAC=1 SET=kidall R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 ARMS="default vcsl vcslm" bash vllm/gpuN_job.sh vllm/round_kidall_line.sh >> vllm/logs/q_kidall.log 2>&1
echo "$(date +%T) KIDALL_COMPILED_DONE" >> serve/logs/swp_timeline.log
OWN_BUSY_FRAC=1 SET=kidall R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 ARMS="fullnc bnc vrs vrsm" bash vllm/gpuN_job.sh vllm/round_kidall_line.sh >> vllm/logs/q_kidall.log 2>&1
echo "$(date +%T) KIDALL_DONE" >> serve/logs/swp_timeline.log
