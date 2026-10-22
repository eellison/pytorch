#!/usr/bin/env bash
# q_profab.sh (core agent's request): REDISPATCH_OVERHEAD harness before/after: step_ms12 vs wip14, both build_cpp_ms9, GPU 0 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export UTIL=0.6 HEAD=2 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 DRIVE_STARTUP_STEPS=50
for t in step_ms12 wip14; do
  echo "$(date +%T) $t"
  HEAD_WT=$I/snap/$t DIAG27_PROFILE_REDISPATCH=$L/out/prof_redispatch_$t.pstats ./gpulock.sh 0 3600 ./startup27.sh run vt prof_$t > logs/rw_vtprof_$t.log 2>&1
  echo "$(date +%T) $t rc=$?"
done
