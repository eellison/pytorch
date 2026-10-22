#!/usr/bin/env bash
# GPU 1 after the n27d chain (PID 1859114) and the KID smoke: the every-size kernel sweep of the n27wf workload
# (kidw_line.sh fi, 5 arms) (the n27dF "full" rounds were cancelled 10-14: user keeps actual max_seq_len)
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
cd $R27
G=1 CPUS=72-107
while kill -0 1859114 2>/dev/null || pgrep -f "bash run_kidsmoke.sh" > /dev/null; do sleep 60; done
echo "$(date +%T) n27d chain, smoke and smoke done"
for ARM in default vcvp fullnc breakable vsvp; do
  [ -f out/kid/t_kidwfi_${ARM}_r1.json ] && { echo "$(date +%T) have fi $ARM"; continue; }
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start kidw fi $ARM"
  CUDA_VISIBLE_DEVICES=$G timeout 5400 taskset -c $CPUS bash kidw_line.sh fi $ARM > logs/kidw_fi_$ARM.log 2>&1
  echo "$(date +%T) end kidw fi $ARM rc=$?"
  exec 9>&-
done
python3 scripts/kid_cmp_all.py out/kid kidwfi > out/kid/KIDW_fi.txt 2>&1
echo "$(date +%T) kidw done"
