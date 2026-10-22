#!/usr/bin/env bash
# GPU 1 after gpu1_q3.sh: untimed every-size kernel identity for the decode curves (kidd) and churn (kidcb), every arm
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
cd $R27
G=1 CPUS=72-107
while pgrep -f "^bash gpu1_q3.sh" > /dev/null || pgrep -f "^bash gpu1_q2.sh" > /dev/null; do sleep 60; done
hold() {  # tag cmd...
  local tag=$1; shift
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=$G timeout 7200 taskset -c $CPUS "$@"
  echo "$(date +%T) end $tag rc=$?"
  exec 9>&-
}
for WL in cb d; do
  for ARM in default vcvp fullnc breakable vsvp; do
    [ -f out/kid/t_kid${WL}_${ARM}_r1.json ] && continue
    hold "kid$WL $ARM" bash -c "bash kidx_line.sh $WL $ARM > logs/kid${WL}_$ARM.log 2>&1"
  done
  python3 scripts/kid_cmp_all.py out/kid kid$WL > out/kid/KID_$WL.txt 2>&1
done
echo "$(date +%T) done"
