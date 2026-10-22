#!/usr/bin/env bash
# GPU 0 gap after the resumed timed sets: the untimed kernel-identity check (vllm/round_kid_line.sh), then the comparison.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q TIMED_RESUMED_DONE serve/logs/swp_timeline.log; do sleep 60; done
OWN_BUSY_FRAC=1 SET=kid R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_kid_line.sh >> vllm/logs/q_kid.log 2>&1
python3 serve/ix/kid_compare.py > serve/out/kid_compare.md 2>&1
echo "$(date +%T) KID_DONE" >> serve/logs/swp_timeline.log
