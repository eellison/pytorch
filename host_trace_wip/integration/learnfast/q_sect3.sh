#!/usr/bin/env bash
cd $(dirname $0)
flock -w 3600 -x ../../scratch/eager/gpu3.lock timeout 900 env ARMV_SECTIONS=1 DIAG27_SECTIONDIAG=1 DIAG27_OPSPAN=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_RESOLVE_SAMEROOT=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143 LEARN_TIMEOUT=600 ./run27m.sh q9e 60 61 > logs/q9e.log 2>&1
echo done >> logs/q9e.log
