#!/usr/bin/env bash
# vcsl r2 started (02:40) before minimal warm-up v2 landed: it is a v1 run; vcsl r3 = the v2 run (main 10-14). One gpu0.lock hold.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while pgrep -f 'drive_diag.py --arm Vc' > /dev/null; do sleep 30; done
STEP=run ARM=vcsl R=3 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_startup.sh >> serve/logs/startup_chain.log 2>&1
echo "$(date +%T) startup vcsl r3 (minimal warm-up v2) done" >> serve/logs/swp_timeline.log
