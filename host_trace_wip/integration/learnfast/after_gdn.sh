#!/usr/bin/env bash
# after_gdn.sh: wait (<= 90 min) until the 27B lane's run_gdn_ours3.sh has appeared and finished, then run the queued jobs
w=0; seen=0
while [ $w -lt 5400 ]; do
  if ps -eo args | awk '/[r]un_gdn_ours3\.sh/' | grep -q .; then seen=1; elif [ $seen = 1 ]; then break; fi
  sleep 20; w=$((w+20))
done
L=../../scratch/eager/gpu2.lock
M9="MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.12 DIAG27_OPSPAN=1 DIAG27_RESOLVE_SAMEROOT=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=2 LF_CPUS=36-71"
flock -w 7200 -x $L timeout 900 env $M9 LEARN_TIMEOUT=600 ./run27m.sh q9b 60 61 > logs/q9b.log 2>&1
flock -w 7200 -x $L timeout 1150 env SANITIZE="/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --print-limit 50 --show-backtrace device --log-file $PWD/logs/q9t_memcheck.%p.log" $M9 LEARN_TIMEOUT=1100 ./run27s.sh q9t 60 61 > logs/q9t.log 2>&1
flock -w 7200 -x $L timeout 1150 env DIAG27_CLOSURE=1 DIAG27_CLOSURE_OBJECTS=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=2 LF_CPUS=36-71 LEARN_TIMEOUT=1100 ./run27.sh dcl3 1 40 > logs/dcl3.log 2>&1
echo done > logs/after_gdn.done
