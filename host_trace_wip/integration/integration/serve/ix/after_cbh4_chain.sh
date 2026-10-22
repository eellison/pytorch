#!/usr/bin/env bash
# gb300-box GPU 0 after cbh4 (its chain PID $1; main 10-14): churnprof (untimed), then the fair startup measurement (seeds + learnfast
# bindings for vcsl / vrs, 2 cold runs per arm), then the cProfile diag. Then GPU 0 goes to the 27B lane.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 20; done
T=serve/logs/swp_timeline.log
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_churnprof.sh > serve/logs/churnprof.log 2>&1
echo "$(date +%T) CHURNPROF_DONE" >> $T
for arm in vcsl vrs; do
  STEP=seed ARM=$arm GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
  STEP=learn ARM=$arm GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
done
echo "$(date +%T) startup seeds + learns done" >> $T
for r in 1 2; do
  for arm in default fullnc bnc vcsl vrs; do
    STEP=run ARM=$arm R=$r GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
  done
done
echo "$(date +%T) STARTUP_DONE" >> $T
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_cprof.sh > serve/logs/cprof.log 2>&1
echo "$(date +%T) CPROF_DONE (GPU 0 free for the 27B lane)" >> $T
