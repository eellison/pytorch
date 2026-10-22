#!/usr/bin/env bash
# q_rw2.sh: vt RANGE=1 with item (a) (TPL_LAYER_KEYOUT=1 + one-time confirmation per (key, layer)), GPU 3, untimed diagnostic.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=${UTIL:-0.6} HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 TPL_LAYER_KEYOUT=1 TPL_DIFF_FUNC=qwen_gdn TPL_DIFFS=4
echo "$(date +%T) run vt RANGE=1 keyout"; ./gpulock.sh 3 3600 ./startup27.sh run vt ${R:-1} > logs/rw_vt${R:-1}.log 2>&1
echo "$(date +%T) rc=$? $(grep -c RANGE_WARM logs/rw_vt${R:-1}.log)"
touch logs/q_rw2.done
