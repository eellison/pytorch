#!/usr/bin/env bash
cd $(dirname $0)
while [ ! -f logs/q_cm.done ]; do sleep 30; done
flock -w 7200 -x ../../scratch/eager/gpu3.lock timeout 600 env ARMV_SECTIONS=1 COLD_NO_SECTIONS=1 COLD_ROOTS_CACHE=1 LF_GPU=3 CPU0=108 UTIL=0.15 PRE_BINDINGS=$PWD/out/bind8_full.pkl COLD_CAPMEMO=1 CAPMEMO_CHECK=1 ./cold8.sh cmchk2_Vc Vc out/keys8_hot.pkl 8 > logs/cmchk2_Vc.log 2>&1
echo done > logs/q_cm2.done
