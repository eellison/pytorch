#!/usr/bin/env bash
# dev-box GPU 3 (inside dev3_run.sh, which holds <path> for the whole chain): the Triton-GDN sets on
# KV pool 64 GiB: vLLM's default max_num_seqs 1024 needs one Mamba cache block per sequence (40 GiB gave 835 blocks and every vLLM arm refused to start).
# vLLM-default max_num_seqs: n27wt (cold-cache startup runs, offline learns, 3 rounds x 5 arms), then n27cbt. CPUs 108-143.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
mkdir -p out/timing logs/timing bindings
COLD=1 R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN2.sh round_27wt_line.sh n27wt 3 108-143 >> logs/timing_n27wt.out 2>&1
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN2.sh round_27cbt_line.sh n27cbt 3 108-143 >> logs/timing_n27cbt.out 2>&1
echo "dev3_chain done $(date +%T)" >> logs/timing_n27cbt.out
