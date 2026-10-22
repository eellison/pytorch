#!/usr/bin/env bash
# swpe1 (main 10-15): the eager decode curves (fullnc, bnc, vrs) x 3 rounds interleaved on gb300-box GPU 0, one gpu0.lock hold per run,
# own-core gate 0.30 + PSI start gate (timing_lib.trun); start only after the learnfast lane's 27B startup chain is done.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
OWN_BUSY_FRAC=0.30 SET=swpe1 ROUND=vllm/round_swpe_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="fullnc bnc vrs" bash serve/ix/set_chain.sh
touch /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/logs/gpu0_8b_set.done; echo "$(date +%T) touched learnfast/logs/gpu0_8b_set.done" >> serve/logs/swp_timeline.log
