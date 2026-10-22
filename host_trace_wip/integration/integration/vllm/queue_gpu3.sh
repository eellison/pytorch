#!/usr/bin/env bash
# GPU 3 jobs in order, one hold each (JOBLIST "script [ENV=..];..."), as queue_gpu0.sh.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
IFS=';' read -ra JOBS <<< "${JOBLIST:?}"
for job in "${JOBS[@]}"; do
  set -- $job; name=$1; shift
  tag=$name; [ $# -gt 0 ] && tag=${name}_$(echo "$*" | tr ' =' '__')
  echo "$(date +%T) queue $tag"
  env "$@" bash vllm/gpu3_job.sh vllm/$name.sh > vllm/logs/q3_$tag.log 2>&1
  echo "$(date +%T) done $tag rc=$?"
done
echo ALLDONE
