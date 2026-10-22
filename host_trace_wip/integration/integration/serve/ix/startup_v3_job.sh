#!/usr/bin/env bash
# startup v3 (main 10-14): ours' minimal warm-up v3, 2 runs per arm (r4, r5), PSI-gated, one gpu0.lock hold each.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
for r in 4 5; do for arm in vcsl vrs; do
  STEP=run ARM=$arm R=$r GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
done; done
echo "$(date +%T) STARTUP_V3_DONE" >> serve/logs/swp_timeline.log
