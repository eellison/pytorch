#!/usr/bin/env bash
# s2: stage-2 timing set, same session: V (traced metadata), V with ARMV_METADATA=eager (stage 1), default, eager.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
TAG=s2 DRIVE_ARGS=--stamps bash bench/run_arms.sh V
TAG=s2md_eager ARMV_METADATA=eager DRIVE_ARGS=--stamps bash bench/run_arms.sh V
TAG=s2 DRIVE_ARGS=--stamps bash bench/run_arms.sh default eager
