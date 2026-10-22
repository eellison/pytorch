#!/usr/bin/env bash
# MiMo redo_context runs (mrc_dc, then mrc_chk), after rcmc and after the MiMo lane's vc_long1 (their order 10-14: diag_dc8k,
# vc_long1, then us before v_ilv9); a diag_dc8k / vc_long1 gpu3.sh still queued or running holds us back.
cd $(dirname $0)
M=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_moe_cpp
while ! grep -q "^done" logs/rcmc.log 2>/dev/null; do sleep 30; done
until grep -q "vc_long1 rc=" $M/logs/vc_long1.log 2>/dev/null; do sleep 60; done
ahead() { ps -eo args | grep -v grep | grep -q "gpu3.sh \(diag_dc8k\|vc_long1\) "; }
RC="DIAG27_REDOCTX=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143 RUN_TIMEOUT=1120"
for job in "mrc_dc dc" "mrc_chk chk"; do
  set -- $job
  while ahead; do sleep 60; done
  exec 9>../../scratch/eager/gpu3.lock
  flock -w 14400 -x 9 || { echo "$(date +%T) $1: no lock" > logs/$1.log; exit 1; }
  echo "$(date +%T) $1 start" > logs/$1.log
  timeout 1150 env $RC ./mimorc.sh $1 $2 >> logs/$1.log 2>&1
  echo "$(date +%T) done" >> logs/$1.log
  exec 9>&-
done
