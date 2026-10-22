#!/usr/bin/env bash
# kidw_queue.sh: GPU 1 after the n27d chain (PID 1859114) and the KID smoke test: kidw_line.sh fi then tr, every arm, one hold per
# run (LANES.md: flock exit status checked, used memory < 20 GiB inside the hold); then the comparison into out/kid/KIDW.txt
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=1 CPUS=72-107
while kill -0 1859114 2>/dev/null || pgrep -f "bash run_kidsmoke.sh" > /dev/null; do sleep 60; done
echo "$(date +%T) n27d chain and smoke done"
for CFG in fi tr; do for ARM in default vcvp fullnc breakable vsvp; do
  [ -f $R27/out/kid/t_kidw${CFG}_${ARM}_r1.json ] && { echo "$(date +%T) have $CFG $ARM"; continue; }
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start kidw $CFG $ARM"
  CUDA_VISIBLE_DEVICES=$G timeout 5400 taskset -c $CPUS bash $R27/kidw_line.sh $CFG $ARM > $R27/logs/kidw_${CFG}_$ARM.log 2>&1
  echo "$(date +%T) end kidw $CFG $ARM rc=$?"
  exec 9>&-
done; done
{ echo "##### FlashInfer GDN (n27wf workload)"; python3 $R27/scripts/kid_cmp_all.py $R27/out/kid kidwfi
  echo; echo "##### Triton GDN (n27wt workload)"; python3 $R27/scripts/kid_cmp_all.py $R27/out/kid kidwtr; } > $R27/out/kid/KIDW.txt 2>&1
echo "$(date +%T) kidw done"
