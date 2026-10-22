#!/usr/bin/env bash
# Replica of the c5h sweep set on GPU 0 (CPUs 0-35) after the churn set: host-bound vLLM rows ran slower on GPU 3 / CPUs 108-143
# than in the GPU 0 r1 set, so the sweep set is repeated on GPU 0. Bindings: the swph learn file (same build and config).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q CBH_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=swph0 ROUND=vllm/round_sweep_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl" bash serve/ix/set_chain.sh
