#!/usr/bin/env bash
# The Qwen3-8B churn set, final protocol (cbh3): untimed warm-up in every arm (one prefill per token count 1-1300, a short decode at
# every bs 1-160, three mixed steps), then 400 warm-up + 3000 measured steps per C; ours (vcsl) strict: the run fails on any trace or
# learn inside a window. GPU 0 / CPUs 0-35, default / fullnc / vcsl (headline) / vcs (cold diagnostic), 3 runs each. Then swph0.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
SET=cbh3 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl vcs" bash serve/ix/set_chain.sh
SET=swph0 ROUND=vllm/round_sweep_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl" bash serve/ix/set_chain.sh
