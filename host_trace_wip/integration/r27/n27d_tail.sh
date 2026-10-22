#!/usr/bin/env bash
# after the n27d chain: rerun round-1 vsvp (the first one had 14 redispatches with sections at the default and no section counter)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "timing_gpuN2.sh round_27dv2_line.sh n27d" > /dev/null; do sleep 60; done
ROUNDS=1 R27_ARMS="vsvp" R27_TIMEOUT=7200 R27_KV_GIB=72 bash timing_gpuN2.sh round_27dv2_line.sh n27d 1 72-107 >> logs/timing_n27d.out 2>&1
ROUNDS=3 R27_ARMS="default fullnc vcvp breakable vsvp" R27_TIMEOUT=7200 R27_KV_GIB=72 bash timing_gpuN2.sh round_27dv2_line.sh n27d 1 72-107 >> logs/timing_n27d.out 2>&1
