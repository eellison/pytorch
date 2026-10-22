#!/usr/bin/env bash
cd $(dirname $0)
L=../../scratch/eager/gpu2.lock
flock -w 7200 -x $L timeout 1000 env ARMV_SECTIONS=1 LF_GPU=2 CPU0=36 UTIL=0.4 COLD_ROOTS_CACHE=1 COLD_ROOTS_CHECK=1 PRE_BINDINGS=$PWD/out/bind8_full.pkl bash -c "./cold8.sh rc2_chk_Vc Vc out/keys8_hot.pkl 8 > logs/rc2_chk_Vc.log 2>&1; COLD_ROOTS_CHECK=0 ./cold8.sh rc2_Vc Vc out/keys8_hot.pkl 8 > logs/rc2_Vc.log 2>&1; COLD_ROOTS_CHECK=0 ./cold8.sh rc2_VR VR out/keys8_hot.pkl 8 > logs/rc2_VR.log 2>&1; COLD_ROOTS_CACHE=0 ./cold8.sh nrc2_VR VR out/keys8_hot.pkl 8 > logs/nrc2_VR.log 2>&1"
echo done > logs/q_roots.done
