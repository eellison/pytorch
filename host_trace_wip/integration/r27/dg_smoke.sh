#!/usr/bin/env bash
# n27dg smoke (~2 min of workload per arm): default and vcvp, 1 round, R27_DG_SMOKE workload; PSI gate relaxed for the smoke only
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_DG_SMOKE=1 PSI_GATE=100 ROUNDS=1 R27_KV_GIB=72 R27_TIMEOUT=1800 R27_ARMS="vcvp default" bash timing_gpuN4.sh round_27dg_line.sh n27dgs 2 36-71 >> logs/timing_n27dgs.out 2>&1
echo "$(date +%T) smoke done"
