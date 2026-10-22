#!/usr/bin/env bash
# gb300-box GPU 3 (CPUs 108-143; main 10-14): the swph1 memory replicas, one gpu3.lock hold per (part, arm), untimed.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
for part in 1 2 3; do
  for arm in vrs fullnc; do
    [ -f vllm/out/t_memrep_${arm}_p$part.json ] && continue
    PART=$part ARM=$arm GPU=3 CPUS=108-143 bash vllm/gpuN_job.sh serve/repro/run_memrep.sh >> serve/logs/memrep_gpu3.log 2>&1
    echo "$(date +%T) memrep $arm p$part done on GPU 3" >> serve/logs/swp_timeline.log
  done
done
echo "$(date +%T) MEMREP_DONE" >> serve/logs/swp_timeline.log
