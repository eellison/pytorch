#!/usr/bin/env bash
# The Qwen3-8B churn set on GPU 0 (CPUs 0-35): the bindings-learning pass first (if its file is missing), then default / fullnc /
# vcs (cold learns) / vcsl (preloaded bindings), 3 runs each, all on GPU 0.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
if [ ! -s vllm/out/bindings_cbh.pkl ]; then
  echo "$(date +%T) cbh learn: waiting for gpu0.lock" >> $T
  STOCK_LINE=c5h ARMS=learn R=0 SET=cbh GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_cb_line.sh >> vllm/logs/q_cbh_learn.log 2>&1
  echo "$(date +%T) cbh learn: done ($(ls -la vllm/out/bindings_cbh.pkl 2>/dev/null | awk '{print $5}') bytes)" >> $T
fi
SET=cbh ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcs vcsl" bash serve/ix/set_chain.sh
