#!/usr/bin/env bash
# q_rwd.sh: key-dump diagnostic (decode phase of the range warm-up only), GPU 3, untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=${UTIL:-0.6} HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=900 RANGE=1 TPL_LAYER_KEYOUT=1 TPL_DIFF_FUNC=${TPL_DIFF_FUNC:-qwen_gdn} TPL_DIFFS=6 TPL_DUMP_KEY=${TPL_DIFF_FUNC:-qwen_gdn}
export DRIVE_RANGE_PHASES=${PHASES:-decode} DRIVE_STARTUP_STEPS=${STEPS:-10}
echo "$(date +%T) run vt dump"; ./gpulock.sh 3 3600 ./startup27.sh run vt ${R:-d1} > logs/rw_vt${R:-d1}.log 2>&1
echo "$(date +%T) rc=$?"
