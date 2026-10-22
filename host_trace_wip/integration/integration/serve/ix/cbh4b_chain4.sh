#!/usr/bin/env bash
# gb300-box GPU 0 (main 10-14; replaces cbh4b_chain3.sh + churnprof3_job.sh before any cbh4b run started): cbh4b (truly unsynced churn,
# fixed harness, 7 arms incl. the new head) x 3 rounds interleaved, own-core gate 0.30 (gb300-box timed-run gate); then
# serve/out/CBH4B_DONE (Flash-Next's chain13_g0 resumes on it); churnprof3; the startup runs after CHAIN13_W1_DONE; the cProfile diag.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
CB_NOSYNC=1 OWN_BUSY_FRAC=0.30 SET=cbh4b ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc bnc vcsl vrs vcsl_h vrs_h" bash serve/ix/set_chain.sh
touch serve/out/CBH4B_DONE
echo "$(date +%T) touched serve/out/CBH4B_DONE" >> $T
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_churnprof3.sh > serve/logs/churnprof3.log 2>&1
echo "$(date +%T) CHURNPROF3_DONE" >> $T
W1=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/out/timing/CHAIN13_W1_DONE
echo "$(date +%T) waiting for $W1 before the startup runs" >> $T
while [ ! -e $W1 ]; do sleep 60; done
for r in 1 2; do
  for arm in default fullnc bnc vcsl vrs; do
    STEP=run ARM=$arm R=$r GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
  done
done
echo "$(date +%T) STARTUP_DONE" >> $T
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_cprof.sh > serve/logs/cprof.log 2>&1
echo "$(date +%T) CPROF_DONE (GPU 0 free for the 27B lane)" >> $T
