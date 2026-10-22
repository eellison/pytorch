#!/usr/bin/env bash
# The c5h sweep set on GPU 3 (CPUs 108-143) with warm GEMMs: the learn pass (this workload once, ARMV_BINDINGS_SAVE) if its file
# is missing, then default / fullnc / vcsl (preloaded), 3 runs each. The cold vcs runs already made stay as diagnostics.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
if [ ! -s vllm/out/bindings_swph.pkl ]; then
  echo "$(date +%T) swph learn: waiting for gpu3.lock" >> $T
  STOCK_LINE=c5h ARMS=learn R=0 SET=swph GPU=3 CPUS=108-143 bash vllm/gpuN_job.sh vllm/round_sweep_line.sh >> vllm/logs/q_swph_learn.log 2>&1
  echo "$(date +%T) swph learn: done ($(ls -la vllm/out/bindings_swph.pkl 2>/dev/null | awk '{print $5}') bytes)" >> $T
fi
SET=swph ROUND=vllm/round_sweep_line.sh GPU=3 CPUS=108-143 LINE=c5h ARMS="default fullnc vcsl" bash serve/ix/set_chain.sh
