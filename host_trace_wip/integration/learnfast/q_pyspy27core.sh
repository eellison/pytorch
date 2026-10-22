#!/usr/bin/env bash
# q_pyspy27core.sh: sampling profile (py-spy record, 100 Hz, speedscope) of the core-only arm-v run on step_ms27 (decode + mixed
# warm-up, 5 steps), untimed, GPU 2 (core agent 10-18). Waits for q_census.done.
export LANE_GPU=2 LANE_CPUS=36-71
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_census.done ]; do sleep 30; done
INST=$I/build_cpp_ms12c/install EXT=$L/torch_ext_ms12c
gates; echo "$(date +%T) pyspy27core"
( for i in $(seq 1 120); do
    P=$(ps -eo pid,args | awk '$2=="python" && index($0, "rpyspy27core") {print $1; exit}')
    [ -n "$P" ] && break; sleep 2
  done
  [ -n "$P" ] && py-spy record --pid $P --rate 100 --format speedscope -o $L/out/pyspy27core.json --nonblocking > $L/logs/pyspy27core_record.log 2>&1 ) &
w=$!
env STEP_OFF=1 UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms27 HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=1500 RUN_TIMEOUT=1400 RANGE=1 \
  DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode,mixed ./gpulock.sh 2 43200 ./startup27.sh run v pyspy27core > logs/rw_vpyspy27core.log 2>&1
echo "$(date +%T) pyspy27core rc=$?"
wait $w; ls -la out/pyspy27core.json; tail -3 logs/pyspy27core_record.log
touch logs/q_pyspy27core.done
