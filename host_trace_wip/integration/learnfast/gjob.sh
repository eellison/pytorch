#!/usr/bin/env bash
# gjob.sh LOG cmd...: GPU ${G:-2} (lock land/scratch/eager/gpu$G.lock, flock -x); waits for gpu_idle_ok.sh $G first (no other user's process).
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=${G:-2}; LOG=$1; shift
w=0
until $E/gpu_idle_ok.sh $G; do sleep 20; w=$((w+20)); [ $w -ge ${WAIT_S:-10800} ] && { echo "$(date +%T) gave up waiting" >> $LOG; exit 9; }; done
flock -x $E/gpu$G.lock bash -c "echo \"\$(date +%T) start gpu$G (idle wait ${w}s) load \$(cut -d' ' -f1-3 /proc/loadavg) gpu \$(nvidia-smi -i $G --query-gpu=memory.used,utilization.gpu --format=csv,noheader)\" >> $LOG; LF_GPU=$G \"\$@\" >> $LOG 2>&1; echo \"\$(date +%T) end rc=\$?\" >> $LOG" _ "$@"
