#!/usr/bin/env bash
# chain_g2f.sh: resume GPU 2 after the 27B lane's decode rerun (coordinator 10-15): waits for r27/out/timing/N27D_RERUN_DONE, then
# w round 6 afresh (t_g2w_fullnc_r6 from before the pause moved to out/timing/aside/ so the round's arms run together), w round 7
# (replaces round 5, whose v arm is PSI-flagged: full 12.5% of its window), then the fresh decode rounds g2e20 x3.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2f.pid
while [ ! -f /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/out/timing/N27D_RERUN_DONE ]; do sleep 60; done
echo "$(date +%T) chain_g2f: N27D_RERUN_DONE seen"
mkdir -p $D/out/timing/aside
for f in $D/out/timing/t_g2w_*_r6.json; do [ -f "$f" ] && mv "$f" $D/out/timing/aside/; done
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g2f: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g2f: round rc $rc ($*)"; return 0; }
for r in 6 7; do run SET=g2w KIND=w R=$r ARMS="fullnc breakable v"; done
for r in 1 2 3; do run SET=g2e20 KIND=d20 R=$r; done
echo "$(date +%T) chain_g2f done"
