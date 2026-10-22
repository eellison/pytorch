#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
echo "GPU apps at start: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i ${GPU_IDX:-1} | tr '\n' ' ')"
CAND_LINE=c4 FLASHINFER_NO_DOWNLOAD=1 timeout 900 bash python_vllm_cand2.sh serve/repro/gdn_prefill_repro.py 2>&1 | grep -v "WARNING\|^$"
echo "gdn_repro rc=${PIPESTATUS[0]}"
