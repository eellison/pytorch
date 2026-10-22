#!/usr/bin/env bash
# n27d with timing_gpuN3 (LANES.md wrapper rules, 10-14): the remaining rounds (existing outputs skipped), then the round-1 vsvp rerun
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_ARMS="default fullnc vcvp breakable vsvp" R27_TIMEOUT=7200 R27_KV_GIB=72 bash timing_gpuN3.sh round_27dv2_line.sh n27d 1 72-107 >> logs/timing_n27d.out 2>&1
