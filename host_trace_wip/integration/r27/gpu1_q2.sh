#!/usr/bin/env bash
# GPU 1 (coordinator order 10-14): counter checks (eager, Vk, V, Vc), breakable GDN logit checks (tr, fi), then the rest of kidw tr
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
for C in tr fi; do for A in breakable fullnc; do hold "brk3 $C $A" bash -c "bash run_brk3.sh $C $A > logs/brk3_${C}_$A.log 2>&1"; done
  python3 scripts/logit_cmp.py out/brk3 ${C}_breakable ${C}_fullnc > out/brk3/CMP_$C.txt 2>&1; done
for ARM in default vcvp fullnc breakable vsvp; do
  [ -f out/kid/t_kidwtr_${ARM}_r1.json ] && continue
  hold "kidw tr $ARM" bash -c "bash kidw_line.sh tr $ARM > logs/kidw_tr_$ARM.log 2>&1"
done
python3 scripts/kid_cmp_all.py out/kid kidwtr > out/kid/KIDW_tr.txt 2>&1
echo "$(date +%T) done"
