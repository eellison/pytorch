#!/usr/bin/env bash
# Runs JOBLIST (queue_gpu0.sh format) after logs/$1 says ALLDONE.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "^ALLDONE" vllm/logs/$1; do sleep 60; done
exec bash vllm/queue_gpu0.sh
