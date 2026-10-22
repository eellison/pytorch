#!/usr/bin/env bash
# q_slocv.sh: guard_sloc memo on a core-only run (arm v: none of learnfast's wrappers), final wip32, split off, decode warm-up +
# 5 steps; then wip31 (= ms28) the same; py-spy record (100 Hz) of each. GPU 2, step-off on. Waits for q_wip32c.done.
export LANE_GPU=2 LANE_CPUS=36-71 STEP_OFF=1
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_wip32c.done ]; do sleep 30; done
INST=$I/build_cpp_ms12d/install EXT=$L/torch_ext_ms12d
for s in wip32 wip31; do
  gates; echo "$(date +%T) slocv_$s"
  ( for i in $(seq 1 600); do
      P=$(ps -eo pid,args | awk -v t="rslocv_$s" '$2=="python" && index($0, t) {print $1; exit}')
      [ -n "$P" ] && break; sleep 2
    done
    [ -n "$P" ] && py-spy record --pid $P --rate 100 --format speedscope -o $L/out/pyspy_slocv_$s.json --nonblocking > $L/logs/pyspy_slocv_$s.log 2>&1 ) &
  w=$!
  env TORCH_HOST_TRACE_SPLIT=0 UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/$s HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=1500 RUN_TIMEOUT=1400 RANGE=1 \
    DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode ./gpulock.sh 2 43200 ./startup27.sh run v slocv_$s > logs/rw_vslocv_$s.log 2>&1
  rc=$?; wait $w
  echo "$(date +%T) slocv_$s rc=$rc $(grep -h SLOC_MEMO logs/rw_vslocv_$s.log) $(grep -h 'RANGE_WARM decode' logs/rw_vslocv_$s.log | grep -o '"trace": [0-9.]*' | head -1)"
done
touch logs/q_slocv.done
