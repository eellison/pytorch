#!/usr/bin/env bash
# GPU 2 after the n27dg rounds (coordinator 10-15): rerun, PSI-gated, the decode-curve points an arm had flagged in 2+ of its 3 rounds
# (sets n27dgf1024 / n27dgf8192, one run per arm and context, gates before the lock)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash gpu2_dg_chain2.sh" > /dev/null; do sleep 60; done
for arm in default fullnc vcvp breakable vsvp breakablegb; do
  python3 scripts/flag_points27.py n27dg $arm 2 | while read ctx bss; do
    echo "$(date +%T) flagged rerun $arm ctx $ctx: $bss"
    R27_DG_CTX=$ctx R27_DG_BS="$bss" ROUNDS=1 R27_KV_GIB=72 R27_TIMEOUT=7200 R27_ARMS=$arm bash timing_gpuN4.sh round_27dg_line.sh n27dgf$ctx 2 36-71 >> logs/timing_n27dgf.out 2>&1 < /dev/null
  done
done
echo "$(date +%T) flagged reruns done"
