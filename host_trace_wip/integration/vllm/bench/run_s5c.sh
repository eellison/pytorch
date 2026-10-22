#!/usr/bin/env bash
# After run_s5b.sh: arm V again with drive.py's prefill warm-up skip of 4 (s5 V counted a key's learner/retrace calls).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
until grep -q ALLDONE bench/logs/run_s5b.log; do sleep 20; done
TAG=s5v2 DRIVE_ARGS="--stamps --stamps-fine --prefill-reqs 4 8" bash bench/run_arms.sh V default
echo ALLDONE
