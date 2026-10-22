#!/usr/bin/env bash
# GPU 0 (talk, 10-12): after the q1p rounds, the Flash-Next controlled-step rounds (vllm/round_fn_line.sh, SET=fn1, 4 rounds,
# arms rotated) if the drive_fn smoke (serve/logs/fnsmoke.log) replayed bitwise.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q ALLDONE vllm/logs/rounds_q1p.out 2>/dev/null; do sleep 60; done
while ! grep -q "FNSMOKE_CHECKS\|fn_smoke_V rc=[1-9]" serve/logs/fnsmoke.log 2>/dev/null; do sleep 60; done
read n ok < <(grep FNSMOKE_CHECKS serve/logs/fnsmoke.log | awk '{print $2, $3}')
if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$n" = "$ok" ]; then
  echo "$(date +%T) fnsmoke $ok/$n bitwise: fn1 rounds"
  ROUNDS=4 SET=fn1 ROUND_SCRIPT=round_fn_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_fn1.out 2>&1
  echo "$(date +%T) fn1 done"
else
  echo "$(date +%T) fnsmoke not bitwise ($ok/$n): fn1 not started"
fi
