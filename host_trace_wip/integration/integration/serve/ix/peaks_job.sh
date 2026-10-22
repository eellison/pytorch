#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_peaks.sh > serve/logs/peaks.log 2>&1
echo "$(date +%T) PEAKS_DONE" >> serve/logs/swp_timeline.log
