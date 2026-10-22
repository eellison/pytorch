#!/usr/bin/env bash
# no_sections validation on GPU 3 (sharing rule): the 8B Vc decode step that crashed (sections attribute left True:
# ARMV_SECTIONS=1 so the adapter does not flip it; the helper refuses at creation) and the 9B section case (60 -> 61)
cd $(dirname $0)
L=../../scratch/eager/gpu3.lock
flock -w 3600 -x $L timeout 1100 env ARMV_SECTIONS=1 COLD_NO_SECTIONS=1 LF_GPU=3 CPU0=108 UTIL=0.15 PRE_BINDINGS=$PWD/out/bind8_full.pkl ./cold8.sh ns8_Vc Vc out/keys8_hot.pkl 8 > logs/ns8_Vc.log 2>&1
flock -w 3600 -x $L timeout 1100 env ARMV_SECTIONS=1 DIAG27_NO_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_RESOLVE_SAMEROOT=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143 LEARN_TIMEOUT=900 ./run27m.sh ns9 60 70 > logs/ns9.log 2>&1
echo done > logs/q_ns.done
