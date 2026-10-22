#!/usr/bin/env bash
# capmemo: equality (check) runs then timing runs vs the root-index baseline, same session, GPU 3 (short holds per run)
cd $(dirname $0)
L=../../scratch/eager/gpu3.lock
B="ARMV_SECTIONS=1 COLD_NO_SECTIONS=1 COLD_ROOTS_CACHE=1 LF_GPU=3 CPU0=108 UTIL=0.15 PRE_BINDINGS=$PWD/out/bind8_full.pkl"
for spec in "cmchk_Vc Vc COLD_CAPMEMO=1 CAPMEMO_CHECK=1" "cmchk_VR VR COLD_CAPMEMO=1 CAPMEMO_CHECK=1" "cm_Vc Vc COLD_CAPMEMO=1" "base_Vc Vc COLD_CAPMEMO=0" "cm_VR VR COLD_CAPMEMO=1" "base_VR VR COLD_CAPMEMO=0"; do
  set -- $spec; tag=$1 arm=$2; shift 2
  flock -w 7200 -x $L timeout 600 env $B "$@" ./cold8.sh $tag $arm out/keys8_hot.pkl 8 > logs/$tag.log 2>&1
done
echo done > logs/q_cm.done
