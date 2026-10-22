#!/usr/bin/env bash
# q_prof.sh: after after_gdn.sh finishes, profile each trace (cProfile) on both forwards with full preloaded bindings
cd $(dirname $0)
while [ ! -f logs/after_gdn.done ]; do sleep 20; done
L=../../scratch/eager/gpu2.lock
flock -w 7200 -x $L timeout 900 env LF_GPU=2 CPU0=36 UTIL=0.4 COLD_PROFILE=1 PRE_BINDINGS=$PWD/out/bind8_full.pkl bash -c "./cold8.sh cprof_Vc Vc out/keys8_hot.pkl 8 > logs/cprof_Vc.log 2>&1; ./cold8.sh cprof_VR VR out/keys8_hot.pkl 8 > logs/cprof_VR.log 2>&1"
echo done > logs/q_prof.done
