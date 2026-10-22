#!/usr/bin/env bash
# After run_serve_range.sh (SRDONE): arm V again with bench/serve_range2.sh (serve_warm2.py: no client connection cap),
# TAG=sr2, then arm V with ARMV_REDISPATCH_LEARNERS=1 as sr3. One gpu2.lock hold each.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock
cd $V
until grep -q SRDONE bench/logs/run_serve_range.log; do sleep 20; done
echo "=== sr2 V $(date)"
TAG=sr2 PORT=18742 CUDA_VISIBLE_DEVICES=2 flock $L timeout 7000 bash bench/serve_range2.sh V
echo "rc=$? $(date)"
echo "=== sr3 V redispatch_learners $(date)"
ARMV_REDISPATCH_LEARNERS=1 TAG=sr3 PORT=18743 CUDA_VISIBLE_DEVICES=2 flock $L timeout 7000 bash bench/serve_range2.sh V
echo "rc=$? $(date)"
echo SR2DONE
