#!/usr/bin/env bash
# gb300-box GPU 3 (coordinator 10-14 11:2x: GPU 1 went to MiMo): the remaining untimed kid runs (kidcb, then kidd), one gpu3.lock
# hold per run (LANES.md: flock exit status checked, used memory < 20 GiB inside the hold; waits behind learnfast's diag27)
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
cd $R27
G=3 CPUS=108-143
while kill -0 742521 2>/dev/null; do sleep 30; done  # the kidcb fullnc run still on GPU 1
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
