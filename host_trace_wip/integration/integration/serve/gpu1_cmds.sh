#!/usr/bin/env bash
# Small correctness/repro scripts in one gpu1.lock hold: bash gpu1_cmds.sh TAG "script args" ["script args" ...]
# (paths relative to land/scratch/integration; launcher serve/python_vllm_cand3.sh unless LAUNCHER is set; LAUNCHER=none
# runs each as a bash script). Output:
# serve/logs/TAG.log.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
TAG=$1; shift
cd $G
export CMD_TIMEOUT=${CMD_TIMEOUT:-1800} CUDA_VISIBLE_DEVICES=1 CAND_LINE=cpp FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 L=${LAUNCHER:-serve/python_vllm_cand3.sh}
echo "$(date +%T) waiting for gpu1.lock"
flock -x $LOCK bash -c '
for c in "$@"; do echo "== $(date +%T) $c"; if [ "$L" = none ]; then timeout ${CMD_TIMEOUT:-1800} taskset -c 72-107 bash $c; else timeout ${CMD_TIMEOUT:-1800} taskset -c 72-107 bash $L $c; fi; echo "rc=$?"; done' _ "$@" > serve/logs/$TAG.log 2>&1
echo "$(date +%T) done"
