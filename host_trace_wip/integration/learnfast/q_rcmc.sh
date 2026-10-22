#!/usr/bin/env bash
# memcheck on the 9B redo_context config (rc9a's flags), learn range 60..62: q9t's crash case (section made at 60, replayed at 61)
cd $(dirname $0)
while ! grep -q "^done" logs/rc27a.log 2>/dev/null; do sleep 30; done
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_REDOCTX=1 DIAG27_OPSPAN=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
flock -w 7200 -x ../../scratch/eager/gpu3.lock timeout 1150 env SANITIZE="/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --print-limit 50 --show-backtrace device --log-file $PWD/logs/rcmc_memcheck.%p.log" $M9 LEARN_TIMEOUT=1100 ./run27s.sh rcmc 60 62 > logs/rcmc.log 2>&1
echo done >> logs/rcmc.log
