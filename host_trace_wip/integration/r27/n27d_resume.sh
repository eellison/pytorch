#!/usr/bin/env bash
# resume the n27d chain after the running learnv arm (started by the previous chain instance) finishes
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "R27_ARMS=learnv SET=n27d" > /dev/null; do sleep 30; done
R27_ARMS="default fullnc vcvp breakable vsvp" R27_TIMEOUT=7200 R27_KV_GIB=72 exec bash timing_gpuN2.sh round_27dv_line.sh n27d 1 72-107
