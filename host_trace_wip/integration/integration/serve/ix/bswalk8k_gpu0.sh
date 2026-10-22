#!/usr/bin/env bash
# gb300-box GPU 0 right after the bnc r3 rerun (ahead of the 27B lane's kidw): the 8K bs-walk correctness checks, vcsl config then vrs.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
# queued on gpu0.lock directly behind the bnc r3 rerun
TAG=_vcsl ARM=Vc GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_bswalk8k.sh > serve/logs/bswalk8k_vcsl.log 2>&1
TAG=_vrs ARM=V GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_bswalk8k.sh > serve/logs/bswalk8k_vrs.log 2>&1
echo "$(date +%T) BSWALK8K_DONE" >> serve/logs/swp_timeline.log
