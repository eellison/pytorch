#!/usr/bin/env bash
# waits for the n27cb2 learn run to save its bindings and exit, then runs the n27cb2 churn chain with the breakable arm added
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
until [ -f bindings/n27cb2.pkl ]; do sleep 30; done
while pgrep -f "R27_ARMS=learn SET=n27cb2" > /dev/null; do sleep 30; done
R27_ARMS="default fullnc vcvp breakable" exec bash timing_gpuN.sh round_27cb_line.sh n27cb2 2 36-71
