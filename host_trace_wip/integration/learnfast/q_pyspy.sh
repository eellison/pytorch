#!/usr/bin/env bash
# q_pyspy.sh: GIL contention probe for the core agent (10-18): ms27 decode warm-up + 5 steps, untimed; py-spy dump (all threads)
# every 3 s for 150 s after LLM() starts tracing -> logs/pyspy_ms27.txt. Waits for q_ms27.done.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_ms27.done ]; do sleep 60; done
INST=$I/build_cpp_ms12c/install EXT=$L/torch_ext_ms12c
OUT=$L/logs/pyspy_ms27.txt; : > $OUT
gates; echo "$(date +%T) pyspy27"
( sleep 20
  for i in $(seq 1 200); do
    P=$(ps -eo pid,args | awk '$2=="python" && index($0, "rpyspy27") {print $1; exit}')
    if [ -n "$P" ] && grep -q '"trace"\|"miss"\|redispatch' $L/out/startup27_h3t_vt_rw_sec_rpyspy27.diag.jsonl 2>/dev/null || [ -n "$P" ] && [ $i -gt 15 ]; then
      { echo "=== $(date +%T.%N) sample $i"; timeout 20 py-spy dump --pid $P; } >> $OUT 2>&1
    fi
    [ -z "$P" ] && [ $i -gt 30 ] && break
    sleep 3
  done ) &
w=$!
env UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms27 HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=900 RUN_TIMEOUT=850 RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer \
  MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode ./gpulock.sh 0 43200 ./startup27.sh run vt pyspy27 > logs/rw_vtpyspy27.log 2>&1
echo "$(date +%T) pyspy27 rc=$? samples $(grep -c '^===' $OUT)"
kill $w 2>/dev/null
touch logs/q_pyspy.done
