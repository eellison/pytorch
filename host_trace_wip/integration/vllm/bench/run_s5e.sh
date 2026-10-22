#!/usr/bin/env bash
# perf stat counters of the main thread during the torch_cpu probe, arm V vs eager (short runs).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
for a in V eager; do
  PROBE_STAT=1 PROBE_PERF=$V/bench/out/stat_$a.txt TAG=stat DRIVE_ARGS="--batch-size 64 --prefill-lens 64 --mixed 0" bash bench/run_arms.sh $a
done
echo ALLDONE
