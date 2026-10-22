#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
PORTS_DIR=${PORTS_DIR:-$PWD/vllm/armV_stock_fast_m} CAND_LINE=c4 timeout 900 bash python_vllm_cand2.sh serve/repro/ports_probe.py
echo "ports_probe rc=$?"
