#!/usr/bin/env bash
# One r1_probe.py run on GPU 0 under gpu0.lock (correctness/diagnosis only): bash probe_gpu0.sh TAG [r1_probe args]
# FLASHINFER_NO_DOWNLOAD=1, HF offline. Waits for >= 90 GiB free on GPU 0 before taking the lock.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
TAG=$1; shift
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge 92160 ]; do sleep 30; done
echo "$(date +%T) waiting for gpu0.lock"
cd $G
CUDA_VISIBLE_DEVICES=0 CAND_LINE=cpp FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 flock -x $LOCK \
  bash -c 'echo "$(date +%T) holding gpu0.lock"; exec timeout 2400 taskset -c 0-71 bash python_vllm_cand2.sh "$@"' _ serve/r1_probe.py --out serve/out/$TAG.json "$@" > serve/logs/$TAG.log 2>&1
echo "$(date +%T) rc=$?"
