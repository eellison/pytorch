#!/usr/bin/env bash
# GPU 0 / CPUs 0-35 after the max_model_len A/B (within noise: 16384 kept): the breakable (bnc) and eager-forward ours (vrs) arms of
# swph1, 3 runs each; the unsynced churn set cbh4 (all 5 arms x 3, async scheduling as shipped, wall ms/token); one cold-cache init per arm.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q MML_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc vrs" bash serve/ix/set_chain.sh
CB_NOSYNC=1 SET=cbh4 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc bnc vcsl vrs" bash serve/ix/set_chain.sh
SET=cold R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_cold_line.sh >> vllm/logs/q_cold_r1.log 2>&1
echo "$(date +%T) COLD_DONE" >> serve/logs/swp_timeline.log
