#!/usr/bin/env bash
# gpulock.sh GPU WAIT_S CMD...: LANES.md GPU wrapper rule. Take land/scratch/eager/gpu$GPU.lock (flock -x, WAIT_S per try),
# abort if flock fails or is killed. Lowest priority (coordinator 10-14): once holding, if any other lane's flock process has
# the lock file open (a waiter), release and retry after 120 s (until WAIT_S total). Inside the hold wait until GPU $GPU's
# used memory is < 20 GiB, then run CMD under timeout ${HOLD_S:-1150} (the ~20 min hold limit). Exit 75: no lock.
set -u
G=$1 W=$2; shift 2
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$G.lock
ME=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
foreign_waiters() {  # flock processes (not started from learnfast) with the lock file open: every waiter runs one
  local p fd
  for p in $(pgrep -x flock); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$ME" ] && continue
    for fd in /proc/$p/fd/*; do
      [ "$(readlink "$fd" 2>/dev/null)" = "$LOCK" ] && { echo $p; break; }
    done
  done
}
t0=$(date +%s)
while :; do
  left=$(( W - ($(date +%s) - t0) ))
  [ $left -le 0 ] && { echo "$(date +%T) gpulock: no lock within ${W}s; aborting"; exit 75; }
  exec 9>"$LOCK"
  flock -w "$left" -x 9
  rc=$?
  if [ $rc -ne 0 ]; then echo "$(date +%T) gpulock: flock on gpu$G failed rc=$rc; aborting"; exit 75; fi
  w=$(foreign_waiters | tr '\n' ' ')
  [ -z "$w" ] && break
  echo "$(date +%T) gpulock: other lanes waiting ($w); yielding"
  exec 9>&-
  sleep 120
done
echo "$(date +%T) gpulock: holding gpu$G.lock"
# SHARE=1 (coordinator 10-22, untimed work: suites, bitwise checks, warm-ups, probes): the GPU is shared with other users' processes;
# the caller guards free memory (gate_lib) and the lock still serializes our own runs per GPU; no memory wait, no foreign-user wait
while [ "${SHARE:-0}" != 1 ]; do
  used=$(timeout 60 nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "gpulock: nvidia-smi failed"; exit 75; }
  [ "$used" -lt 20480 ] && break
  echo "$(date +%T) gpulock: GPU $G ${used} MiB used, waiting for < 20 GiB"
  sleep 20
done
foreign_users() {  # other users' compute processes on GPU $G
  local pid
  for pid in $(timeout 60 nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$(id -un)" ] && [ -n "$(ps -o user= -p $pid 2>/dev/null)" ] && echo $pid
  done
}
while [ "${SHARE:-0}" != 1 ] && [ -n "$(foreign_users)" ]; do echo "$(date +%T) gpulock: another user's process on GPU $G, waiting"; sleep 60; done
[ "${SHARE:-0}" = 1 ] && echo "$(date +%T) gpulock: sharing GPU $G; foreign processes now: $(for p in $(foreign_users); do echo -n "$p($(ps -o user= -p $p 2>/dev/null | tr -d ' ')) "; done)$(timeout 60 nvidia-smi -i "$G" --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
descendants() {  # every descendant PID of $1
  local c
  for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo $c; descendants $c; done
}
# STEP_OFF=1 (GPU 0 rule): if another user's process appears during the run, stop ours (own process group only)
setsid timeout "${HOLD_S:-1150}" "$@" &
cmd=$!
if [ "${STEP_OFF:-0}" = 1 ]; then
  ( while kill -0 $cmd 2>/dev/null; do
      if [ -n "$(foreign_users)" ]; then
        # the run's whole process tree (collected before any kill: orphans re-parent), TERM, then KILL what survives 20 s;
        # a process-group TERM alone left the python running (10-18)
        tree=$(descendants $cmd)
        echo "$(date +%T) gpulock: another user appeared on GPU $G: stepping off (pids $cmd $tree)"
        kill -TERM -$cmd $cmd $tree 2>/dev/null
        sleep 20
        kill -KILL $cmd $tree 2>/dev/null
        break
      fi
      sleep 30
    done ) &
fi
wait $cmd
rc=$?
echo "$(date +%T) gpulock: rc=$rc, releasing gpu$G.lock"
exit $rc
