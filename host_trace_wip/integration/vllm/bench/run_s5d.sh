#!/usr/bin/env bash
# After run_s5c.sh: perf profiles of the torch_cpu probe in arm V's and the eager arm's processes (short runs).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
until grep -q ALLDONE bench/logs/run_s5c.log; do sleep 20; done
for a in V eager; do
  PROBE_PERF=$V/bench/out/perf_$a.data TAG=perf DRIVE_ARGS="--batch-size 64 --prefill-lens 64 --mixed 0" bash bench/run_arms.sh $a
done
echo ALLDONE
