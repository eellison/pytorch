#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q 'CHURNPROF3_DONE' serve/logs/swp_timeline.log; do sleep 30; done
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_churnprof4.sh > serve/logs/churnprof4.log 2>&1
echo "$(date +%T) CHURNPROF4_DONE" >> serve/logs/swp_timeline.log
