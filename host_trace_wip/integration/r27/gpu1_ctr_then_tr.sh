#!/usr/bin/env bash
# GPU 1: the counter checks (eager, V, Vc), then the kidw tr sweep (5 arms); one hold per run (LANES.md wrapper)
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
cd $R27
G=1 CPUS=72-107
hold() {  # tag cmd...
  local tag=$1; shift
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=$G timeout 3600 taskset -c $CPUS "$@"
  echo "$(date +%T) end $tag rc=$?"
  exec 9>&-
}
for A in eager Vk V Vc; do hold ctr_$A bash run_ctr_check.sh $A; done
for ARM in default vcvp fullnc breakable vsvp; do
  [ -f out/kid/t_kidwtr_${ARM}_r1.json ] && continue
  hold "kidw tr $ARM" bash -c "bash kidw_line.sh tr $ARM > logs/kidw_tr_$ARM.log 2>&1"
done
python3 scripts/kid_cmp_all.py out/kid kidwtr > out/kid/KIDW_tr.txt 2>&1
echo "$(date +%T) done"
