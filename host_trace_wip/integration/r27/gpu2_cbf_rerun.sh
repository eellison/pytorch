#!/usr/bin/env bash
# GPU 2 after gpu2_after_cbf.sh (n27wfs): rerun the n27cbf arms whose process also ran KID profiling (CUPTI stayed attached, +3-10%;
# moved to out/timing/invalid_kid/), without KID (drive default off since 09:40); timing_gpuN3 skips the existing runs
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash gpu2_after_cbf.sh" > /dev/null; do sleep 60; done
echo "$(date +%T) n27wfs done; n27cbf reruns"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN3.sh round_27cbf_line.sh n27cbf 2 36-71 >> logs/timing_n27cbf_rerun.out 2>&1
echo "$(date +%T) n27cbf reruns done"
