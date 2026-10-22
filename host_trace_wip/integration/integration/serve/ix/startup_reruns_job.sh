#!/usr/bin/env bash
# startup reruns for host-flagged runs (PSI > 2% of the stream): bnc r2 -> r3, fullnc r2 -> r3 (main 10-14). One gpu0.lock hold each.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
for arm in bnc fullnc; do
  STEP=run ARM=$arm R=3 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
done
echo "$(date +%T) startup reruns (bnc r3, fullnc r3) done" >> serve/logs/swp_timeline.log
