#!/usr/bin/env bash
cd $(dirname $0)
L=../../scratch/eager/gpu3.lock
flock -w 3600 -x $L timeout 1100 env ARMV_SECTIONS=1 COLD_NO_SECTIONS=1 COLD_NO_LIB_REDO=1 LF_GPU=3 CPU0=108 UTIL=0.15 PRE_BINDINGS=$PWD/out/bind8_full.pkl bash -c "./cold8.sh nlr8_Vc Vc out/keys8_hot.pkl 8 > logs/nlr8_Vc.log 2>&1; ./cold8.sh nlr8_VR VR out/keys8_hot.pkl 8 > logs/nlr8_VR.log 2>&1"
flock -w 3600 -x $L timeout 1100 env ARMV_SECTIONS=1 DIAG27_NO_SECTIONS=1 DIAG27_NO_LIB_REDO=1 LF_GPU=3 LF_CPUS=108-143 LEARN_TIMEOUT=1000 CHECK_TIMEOUT=1000 ./run27check.sh nlr27 "2 3 5 9 17" > logs/nlr27.log 2>&1
echo done > logs/q_nlr.done
