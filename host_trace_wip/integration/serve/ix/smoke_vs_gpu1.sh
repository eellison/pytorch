#!/usr/bin/env bash
# V-stock SKU smoke on GPU 1 (gpu1.lock, correctness only): qwen38bvs 1k1k c4, reuse server, no profile.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $G
FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock bash -c \
  'echo "$(date +%T) holding gpu1.lock"; timeout 1800 taskset -c 102-107 bash python_vllm_cand2.sh serve/ix/ix_run.py --gpu 1 --sku qwen38bvs-bf16-gb300-vllm --scenario 1k1k --concs 4 --tag smoke_vs --profile-steps 20'
echo "$(date +%T) rc=$?"
