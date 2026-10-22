#!/usr/bin/env bash
# after cbh4b (its CBH4B_DONE line), untimed: churnprof3 (config-change vs args-only set durations over steps 100-400). One gpu0.lock hold.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q 'CBH4B_DONE' serve/logs/swp_timeline.log; do sleep 60; done
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_churnprof3.sh > serve/logs/churnprof3.log 2>&1
echo "$(date +%T) CHURNPROF3_DONE" >> serve/logs/swp_timeline.log
