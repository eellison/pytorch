#!/usr/bin/env bash
# Phase 5: s5 timing (stage 3; V, default, eager; --stamps --stamps-fine), then decode dumps (fullnc, V) for the bitwise check.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=s5 DRIVE_ARGS="--stamps --stamps-fine --prefill-reqs 4 8" bash bench/run_arms.sh V default eager
for a in fullnc V; do
  TAG=dump DRIVE_ARGS="--prefill-lens 64 --mixed 0 --dump-decode bench/out/dump_$a.pt" bash bench/run_arms.sh $a
done
echo ALLDONE
