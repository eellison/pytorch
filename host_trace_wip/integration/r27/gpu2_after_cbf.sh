#!/usr/bin/env bash
# GPU 2 after n27cbf: one untimed run per ours arm of the n27wf workload (SET n27wfs, n27wf's bindings) to record our graph warm-up
# (trace seconds + learns + bindings load) for the n27wf startup table; step metrics of this set are not used
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "gpu2_fi_timing3.sh" > /dev/null; do sleep 60; done
cp bindings/n27wf.pkl bindings/n27wfs.pkl; cp bindings/n27wf_v.pkl bindings/n27wfs_v.pkl
ROUNDS=1 R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="vcvp vsvp" bash timing_gpuN3.sh round_27wf_line.sh n27wfs 2 36-71 >> logs/timing_n27wfs.out 2>&1
