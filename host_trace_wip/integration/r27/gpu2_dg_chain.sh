#!/usr/bin/env bash
# GPU 2 (coordinator 10-15): after the n27dg smoke, the breakable decode check (dp_brk27.sh, untimed), then, once the smoke is reviewed
# (out/timing/DG_SMOKE_OK), the timed n27dg decode curves: 5 arms x 3 rounds, gates before the lock (timing_gpuN4.sh)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash dg_smoke.sh" > /dev/null; do sleep 30; done
bash dp_brk27.sh >> logs/dp_brk27.out 2>&1
while [ ! -f out/timing/DG_SMOKE_OK ]; do sleep 60; done
echo "$(date +%T) n27dg timed"
R27_KV_GIB=72 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN4.sh round_27dg_line.sh n27dg 2 36-71 >> logs/timing_n27dg.out 2>&1
echo "$(date +%T) n27dg done"
