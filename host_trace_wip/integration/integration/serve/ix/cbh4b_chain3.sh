#!/usr/bin/env bash
# gb300-box GPU 0 (main 10-14; replaces cbh4b_chain2.sh while its churnprof2 job, PID $1, runs): the new-head smoke, cbh4b (truly
# unsynced churn, fixed harness; 7 arms if the head runs as-is, else 5) x 3 rounds interleaved. Then Flash-Next's chain13 takes GPU 0;
# the startup runs and the cProfile diag wait until their CHAIN13_W1_DONE exists, then use the gaps between their arms.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 10; done
T=serve/logs/swp_timeline.log
echo "$(date +%T) CHURNPROF2_DONE" >> $T
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_smoke_head.sh > serve/logs/smoke_head.log 2>&1
ARMS="default fullnc bnc vcsl vrs"
grep -q SMOKE_HEAD_OK serve/logs/smoke_head.log && ARMS="default fullnc bnc vcsl vrs vcsl_h vrs_h"
echo "$(date +%T) SMOKE_HEAD $(grep -c SMOKE_HEAD_OK serve/logs/smoke_head.log); cbh4b arms: $ARMS" >> $T
CB_NOSYNC=1 OWN_BUSY_FRAC=0.25 SET=cbh4b ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="$ARMS" bash serve/ix/set_chain.sh
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
