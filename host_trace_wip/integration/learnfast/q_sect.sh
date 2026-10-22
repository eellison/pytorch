#!/usr/bin/env bash
cd $(dirname $0)
L=../../scratch/eager/gpu2.lock
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.12 DIAG27_SECTIONDIAG=1 DIAG27_OPSPAN=1 DIAG27_RESOLVE_SAMEROOT=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=2 LF_CPUS=36-71"
flock -w 7200 -x $L timeout 900 env $M9 LEARN_TIMEOUT=600 ./run27m.sh q9d 60 61 > logs/q9c.log 2>&1
echo done > logs/q_sect.done
