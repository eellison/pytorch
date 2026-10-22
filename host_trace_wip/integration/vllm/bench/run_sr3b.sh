#!/usr/bin/env bash
# After run_sr2.sh (SR2DONE): sr3 again (its warm-up died on a client connection reset), as sr3b.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock
cd $V
until grep -q SR2DONE bench/logs/run_sr2.log; do sleep 20; done
echo "=== sr3b V redispatch_learners $(date)"
ARMV_REDISPATCH_LEARNERS=1 TAG=sr3b PORT=18744 CUDA_VISIBLE_DEVICES=2 flock $L timeout 7000 bash bench/serve_range2.sh V
echo "rc=$? $(date)"
echo SR3BDONE
