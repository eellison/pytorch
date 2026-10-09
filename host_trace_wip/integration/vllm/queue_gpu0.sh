#!/usr/bin/env bash
# GPU 0 jobs in order, one hold each (other lanes can take the lock between them). Each entry: "script [ENV=..]".
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
JOBS=("timing_sweep_c2" "timing_r3b_c2" "timing_pad_c2" "timing_shrink_c2" "startup_c2_rounds ROUNDS=3")
[ -n "${JOBLIST:-}" ] && IFS=';' read -ra JOBS <<< "$JOBLIST"
for job in "${JOBS[@]}"; do
  set -- $job; name=$1; shift
  tag=$name; [ $# -gt 0 ] && tag=${name}_$(echo "$*" | tr ' =' '__')
  echo "$(date +%T) queue $tag"
  env "$@" bash vllm/gpu0_job.sh vllm/$name.sh > vllm/logs/q_$tag.log 2>&1
  echo "$(date +%T) done $tag rc=$?"
done
echo ALLDONE
