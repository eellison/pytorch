#!/usr/bin/env bash
# The Qwen3-8B churn set, final protocol (cbh2: untimed family warm-up of one decode, one prefill and one mixed step before the
# windows; binds counted), GPU 0 / CPUs 0-35, default / fullnc / vcsl (preloaded, headline) / vcs (cold, diagnostic), 3 runs each;
# bindings: the cbh learn pass (same build and config). Then the GPU 0 replica of the sweep set (swph0).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
SET=cbh2 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl vcs" bash serve/ix/set_chain.sh
SET=swph0 ROUND=vllm/round_sweep_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc vcsl" bash serve/ix/set_chain.sh
