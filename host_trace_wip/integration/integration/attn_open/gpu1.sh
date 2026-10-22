#!/usr/bin/env bash
# One attn_open job on GPU 1 under gpu1.lock (correctness / acceptance): bash attn_open/gpu1.sh TAG [ENV=..] -- script args
# FLASHINFER_NO_DOWNLOAD=1, HF offline, CPUs 72-107. Waits for >= 100 GiB free on GPU 1 before taking the lock.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
TAG=$1; shift; envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
cd $G
echo "$(date +%T) queued $TAG" >> attn_open/logs/jobs.log
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 1 | tr -d , | awk '{print $1 - $2}') -ge 102400 ]; do sleep 30; done
flock -x $LOCK bash -c 'echo "$(date +%T) start '$TAG'" >> attn_open/logs/jobs.log; exec env "$@"' _ "${envs[@]}" CUDA_VISIBLE_DEVICES=1 CAND_LINE=${CAND_LINE:-pinned} FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  INTEG_ARMV=$G/attn_open/armV_open timeout ${JOB_TIMEOUT:-2400} taskset -c 72-107 bash python_vllm_cand2.sh "$@" > attn_open/logs/$TAG.log 2>&1
echo "$(date +%T) end $TAG rc=$?" >> attn_open/logs/jobs.log
