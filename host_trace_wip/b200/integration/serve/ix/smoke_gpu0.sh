#!/usr/bin/env bash
# Harness smoke (GPU 0, one gpu0.lock hold): qwen38b default and ht SKUs, 1k1k, conc 1 and 8, one server per SKU,
# profiler window 20 steps per point.
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/ix
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge 153600 ]; do sleep 30; done
echo "$(date +%T) GPU 0 free; waiting for gpu0.lock"
cd $G
FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CAND_LINE=cpp flock -x $LOCK bash -c '
echo "$(date +%T) holding gpu0.lock"
for sku in qwen38b-bf16-gb300-vllm qwen38bht-bf16-gb300-vllm; do
  timeout 2400 taskset -c 64-71 bash python_vllm_cand2.sh serve/ix/ix_run.py --gpu 0 --sku $sku --scenario 1k1k --concs 1,8 --tag smoke1 --reuse-server --profile-steps 20
done
echo IX_SMOKE_DONE'
