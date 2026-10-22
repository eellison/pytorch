#!/usr/bin/env bash
# GPU 2 queue (10-13): after the FlashInfer-GDN checks, the eager-twice checks (2 processes per forward), then n27wf (cold runs, learns,
# 3 rounds of five arms), then n27cbf
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=../../scratch/eager
while pgrep -f "run_ficheck.sh" > /dev/null; do sleep 30; done
for fwd in c e; do flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 taskset -c 36-71 bash run_ficheck2.sh $fwd 2 >> logs/ficheck2.out 2>&1; done
COLD=1 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN2.sh round_27wf_line.sh n27wf 2 36-71 >> logs/timing_n27wf.out 2>&1
R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN2.sh round_27cbf_line.sh n27cbf 2 36-71 >> logs/timing_n27cbf.out 2>&1
