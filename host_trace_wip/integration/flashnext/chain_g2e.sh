#!/usr/bin/env bash
# chain_g2e.sh: gb300-box GPU 2, timed g2w9 x3 (fullnc / gdnple / v; label "6 ATen elementwise templates differ (strided vs vectorized,
# same ops)", provisional until the logits check) then fresh decode grid rounds g2e20 x3 (coordinator 10-15 (b): all rounds on GPU 2) after chain_g2c (PID $1);
# d20 (1K + 8K, 20-point grid, 8K capped at 96), fullnc / breakable / v interleaved, round_fn_line3.sh (gate 0.30 + PSI).
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2e.pid
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep chain_g2c)" ]; do sleep 60; done
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g2e: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g2e: round rc $rc ($*)"; return 0; }
# (g2w9 dropped 10-15 11:4x: gdnple failed the logits check, out/BREAKABLE_PLE.md)
for r in 1 2 3; do run SET=g2e20 KIND=d20 R=$r; done
echo "$(date +%T) chain_g2e done"
