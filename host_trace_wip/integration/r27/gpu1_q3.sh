#!/usr/bin/env bash
# GPU 1 after gpu1_q2.sh: the breakable+gdnbreak verification: kernel identity at all n27wf / n27wt sizes (kidw breakablegb, fi and tr)
# and the logit probe vs default-nc (fi and tr)
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
cd $R27
G=1 CPUS=72-107
while pgrep -f "^bash gpu1_q2.sh" > /dev/null; do sleep 60; done
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
for C in fi tr; do
  hold "kidw $C breakablegb" bash -c "bash kidw_line.sh $C breakablegb > logs/kidw_${C}_breakablegb.log 2>&1"
  hold "brk3 $C breakablegb" bash -c "bash run_brk3.sh $C breakablegb > logs/brk3_${C}_breakablegb.log 2>&1"
  python3 scripts/logit_cmp.py out/brk3 ${C}_breakablegb ${C}_fullnc > out/brk3/CMP_${C}_gb.txt 2>&1
done
for C in fi tr; do python3 scripts/kid_cmp_all.py out/kid kidw$C > out/kid/KIDW_$C.txt 2>&1; done
echo "$(date +%T) done"
