#!/usr/bin/env bash
# kid27.sh: after the n27d chain (PID 1859114) exits, the kernel-identity runs on GPU 1 (LANES.md wrapper: flock exit status
# checked, used memory < 20 GiB inside the hold, one hold per run), then the comparisons into out/kid/KID.txt.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=1 CPUS=72-107
while kill -0 1859114 2>/dev/null; do sleep 60; done
echo "$(date +%T) n27d chain done"
for CFG in fi tr; do for ARM in default vcvp fullnc vsvp; do
  [ -f $R27/out/kid/$CFG/${ARM}_8x8.json ] && { echo "$(date +%T) have $CFG $ARM"; continue; }
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start kid $CFG $ARM"
  CUDA_VISIBLE_DEVICES=$G timeout 3600 taskset -c $CPUS bash $R27/kid27_line.sh $CFG $ARM > $R27/logs/kid_${CFG}_$ARM.log 2>&1
  echo "$(date +%T) end kid $CFG $ARM rc=$?"
  exec 9>&-
done; done
{ for CFG in fi tr; do
    echo "##### $CFG: vLLM default vs ours Vc+VP"; python3 $R27/scripts/kid27_cmp.py $R27/out/kid/$CFG default vcvp
    echo "##### $CFG: vLLM default-nc vs ours V+VP"; python3 $R27/scripts/kid27_cmp.py $R27/out/kid/$CFG fullnc vsvp
  done; } > $R27/out/kid/KID.txt 2>&1
echo "$(date +%T) kid done"
