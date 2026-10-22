#!/usr/bin/env bash
# One ix_run.py job on GPU 0 under gpu0.lock: bash ix/ix_gpu0.sh <ix_run args without --gpu>. Waits for >= FREE_GIB (150)
# free on GPU 0 before taking the lock. FLASHINFER_NO_DOWNLOAD=1 and HF offline (no downloads).
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
NEED=$(( ${FREE_GIB:-150} * 1024 ))
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge $NEED ]; do sleep 30; done
echo "$(date +%T) GPU 0 free; waiting for gpu0.lock"
cd $G
FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CAND_LINE=cpp flock -x $LOCK bash -c \
  'echo "$(date +%T) holding gpu0.lock"; exec taskset -c 64-71 bash python_vllm_cand2.sh serve/ix/ix_run.py --gpu 0 "$@"' _ "$@"
echo "$(date +%T) rc=$?"
