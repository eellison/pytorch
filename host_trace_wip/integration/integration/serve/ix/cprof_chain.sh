#!/usr/bin/env bash
# gb300-box GPU 0 after the startup chain (PID $1): the cProfile diag (main 10-14). Then GPU 0 goes to the 27B lane.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 20; done
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_cprof.sh > serve/logs/cprof.log 2>&1
echo "$(date +%T) CPROF_DONE (GPU 0 free for the 27B lane)" >> serve/logs/swp_timeline.log
