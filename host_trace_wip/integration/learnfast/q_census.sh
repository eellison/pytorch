#!/usr/bin/env bash
# q_census.sh: exec_census on step_ms27 (core agent 10-18), rs5-style vt run (core only), 50 stream steps, untimed, GPU 2
# (gpu2.lock, CPUs 36-71; STEP_OFF=1: stop if another user's process appears).
export LANE_GPU=2 LANE_CPUS=36-71
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
export SNAP=step_ms27 INST=$I/build_cpp_ms12c/install STEP_OFF=1
EXT=$L/torch_ext_ms12c
source <(sed -n '/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 2/g")
STEPS=50 startup census27 DIAG27_EXEC_CENSUS=1
grep -h "^EXEC_CENSUS" logs/rw_vtcensus27.log | cut -c1-400
touch logs/q_census.done
