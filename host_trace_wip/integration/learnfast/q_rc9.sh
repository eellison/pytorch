#!/usr/bin/env bash
cd $(dirname $0)
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_REDOCTX=1 DIAG27_SECTIONDIAG=1 DIAG27_OPSPAN=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
flock -w 7200 -x ../../scratch/eager/gpu3.lock timeout 1000 env $M9 LEARN_TIMEOUT=900 ./run27m.sh rc9a 60 140 > logs/rc9a.log 2>&1
echo done >> logs/rc9a.log
