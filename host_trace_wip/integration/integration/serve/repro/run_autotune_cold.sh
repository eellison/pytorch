#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
for line in c5g c4; do echo "== $line"; CAND_LINE=$line HOSTTRACE_TORCH_DISABLE_CACHES=0 timeout 600 bash python_vllm_cand2.sh serve/repro/autotune_cold_repro.py 2>&1 | grep -v "WARNING\|^$" | tail -9; done
