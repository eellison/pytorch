#!/usr/bin/env bash
# GPU 2: churn rerun after the unsynced-window fix (verified 14:41: harness syncs 0, device syncs = the window's end barrier only;
# out/synccheck_fixed): n27cbn (default, fullnc, vcvp, vsvp x 3), then n27cbng (breakable+gdnbreak x 3)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp vsvp" bash timing_gpuN3.sh round_27cbn_line.sh n27cbn 2 36-71 >> logs/timing_n27cbn.out 2>&1
echo "$(date +%T) n27cbng"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27cbng_line.sh n27cbng 2 36-71 >> logs/timing_n27cbng.out 2>&1
echo "$(date +%T) done"
