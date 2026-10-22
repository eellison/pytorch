#!/usr/bin/env bash
# when n27w's ours-uncompiled bindings exist (they cover decode bs 1-128), reuse them for n27d and move n27d to round_27dv_line.sh
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
until [ -f bindings/n27w_v.pkl ]; do sleep 60; done
cp bindings/n27w_v.pkl bindings/n27d_v.pkl
pkill -f "timing_gpuN.sh round_27d_line.sh n27d" || true
R27_ARMS="default fullnc vcvp breakable vsvp" R27_TIMEOUT=7200 R27_KV_GIB=72 exec bash timing_gpuN2.sh round_27dv_line.sh n27d 1 72-107
