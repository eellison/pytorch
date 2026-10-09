#!/usr/bin/env bash
# Several r1_probe.py runs in one gpu1.lock hold: [LAUNCHER=serve/python_vllm_cand3.sh] bash probe_pair_gpu1.sh "TAG1 args..." "TAG2 args..." ...
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 1 | tr -d , | awk '{print $1 - $2}') -ge 92160 ]; do sleep 30; done
echo "$(date +%T) waiting for gpu1.lock"
cd $G
export CUDA_VISIBLE_DEVICES=1 CAND_LINE=cpp FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
flock -x $LOCK bash -c '
for spec in "$@"; do
  set -- $spec; tag=$1; shift
  echo "$(date +%T) $tag start (gpu1 used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1))"
  timeout 2400 taskset -c 72-107 bash ${LAUNCHER:-python_vllm_cand2.sh} serve/r1_probe.py --out serve/out/$tag.json "$@" > serve/logs/$tag.log 2>&1
  echo "$(date +%T) $tag rc=$?"
done' _ "$@"
