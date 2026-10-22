#!/usr/bin/env bash
# Decode parity: V and default alternated 3 times (separate processes), 65 decode tokens per bs, stamps on.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
for r in 1 2 3; do
  TAG=s6r$r DRIVE_ARGS="--stamps --decode-tokens 65 --prefill-lens 64 --mixed 0" bash bench/run_arms.sh V default
done
echo ALLDONE
