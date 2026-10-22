#!/usr/bin/env bash
# GPU 0 after swph1: the max_model_len A/B (set mml, 3 runs per arm), CPUs 0-35.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q SWPH1_DONE serve/logs/swp_timeline.log; do sleep 60; done
SET=mml ROUND=vllm/round_mml_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default16 default40 fullnc16 fullnc40" bash serve/ix/set_chain.sh
