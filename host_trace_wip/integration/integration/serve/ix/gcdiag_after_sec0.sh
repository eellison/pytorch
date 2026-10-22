#!/usr/bin/env bash
# GPU 0 / CPUs 0-35: the untimed GC diagnostic right after the sec0 checks (own-core gate off; it competes for gpu0.lock only with the
# held timed queue, which resumes only on quiet cores).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
true
OWN_BUSY_FRAC=1 SET=gcdiag R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_gcdiag_line.sh >> vllm/logs/q_gcdiag.log 2>&1
echo "$(date +%T) GCDIAG1_DONE" >> serve/logs/swp_timeline.log
