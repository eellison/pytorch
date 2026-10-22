#!/usr/bin/env bash
# q_gc3.sh (core agent's request): GC-timed 3-way, same harness as prof A/B (vt RANGE=1, sections, memo prototype, 50 steps, cProfile
# of _redispatch) + DIAG27_GC_TIME; order wip14, step_ms12, wip14_nomask; build_cpp_ms9; GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=2 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 DRIVE_STARTUP_STEPS=50 DIAG27_GC_TIME=1
for t in wip14 step_ms12 wip14_nomask; do
  w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  echo "$(date +%T) $t"
  HEAD_WT=$I/snap/$t DIAG27_PROFILE_REDISPATCH=$L/out/prof_gc_$t.pstats ./gpulock.sh 0 3600 ./startup27.sh run vt gc_$t > logs/rw_vtgc_$t.log 2>&1
  echo "$(date +%T) $t rc=$? $(grep GC_TIME logs/rw_vtgc_$t.log)"
done
