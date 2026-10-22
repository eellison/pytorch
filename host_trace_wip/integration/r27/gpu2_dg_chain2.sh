#!/usr/bin/env bash
# GPU 2 (coordinator 10-15): the timed n27dg decode curves with 6 arms (breakable+gdnbreak added as the breakable decode arm; plain
# breakable kept as a cross-check at its padded sizes), 3 rounds, interleaved, gates before the lock; finished runs are skipped
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_KV_GIB=72 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp breakablegb" bash timing_gpuN4.sh round_27dg_line.sh n27dg 2 36-71 >> logs/timing_n27dg.out 2>&1
echo "$(date +%T) n27dg done"
