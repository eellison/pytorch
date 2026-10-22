#!/usr/bin/env bash
# nsys_watch.sh TAG DIAGFILE: while the run's python is alive, if DIAGFILE has not changed for 90 s, capture
# nvidia-smi -q -d PERFORMANCE,CLOCK (GPU $LF_GPU) and a py-spy dump once into logs/nsyswatch_TAG.txt
TAG=$1 F=$2 OUT=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/logs/nsyswatch_$TAG.txt
: > $OUT
sleep 60
for i in $(seq 1 120); do
  P=$(ps -eo pid,args | awk -v t="rnsys$TAG" '$2=="python" && index($0, t) {print $1; exit}')
  [ -z "$P" ] && { [ $i -gt 5 ] && break; sleep 10; continue; }
  age=$(( $(date +%s) - $(stat -c %Y $F 2>/dev/null || date +%s) ))
  if [ $age -ge 90 ]; then
    { echo "$(date +%T) stalled ${age}s pid $P"; timeout 60 nvidia-smi -q -d PERFORMANCE,CLOCK -i ${LF_GPU:-0}; timeout 40 py-spy dump --pid $P; } >> $OUT 2>&1
    break
  fi
  sleep 10
done
