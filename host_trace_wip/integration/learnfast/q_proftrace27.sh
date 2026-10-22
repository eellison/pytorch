#!/usr/bin/env bash
# q_proftrace27.sh: untimed cProfile of R.trace only on step_ms27, core only (arm v: no lifted redo / closure / FNRESOLVE / xvr /
# tpl keyout / sections), decode warm-up + mixed phase, 5 stream steps (core agent 10-18). GPU 0.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
INST=$I/build_cpp_ms12c/install EXT=$L/torch_ext_ms12c
gates; echo "$(date +%T) proftrace27"
env DIAG27_PROFILE_PARTS=$L/out/prof27core UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms27 HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=1500 RUN_TIMEOUT=1400 RANGE=1 \
  DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode,mixed ./gpulock.sh 0 43200 ./startup27.sh run v proftrace27 > logs/rw_vproftrace27.log 2>&1
echo "$(date +%T) proftrace27 rc=$? $(ls -la out/prof27core_* 2>&1 | tr '\n' ' ')"
cudaerr logs/rw_vproftrace27.log && echo "CUDA ERROR"
touch logs/q_proftrace27.done
