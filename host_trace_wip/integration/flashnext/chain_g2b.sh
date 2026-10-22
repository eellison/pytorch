#!/usr/bin/env bash
# chain_g2b.sh: gb300-box GPU 2 after the g2chk kernel check (round PID $1, started by chain_g2fix.sh): kid_compare, then timed
# g2w9 (the 9 breakable-INVALID sizes, fullnc / gdnbreak / v) x3 and the w rerun g2w r4-r6 (fullnc / breakable / v), all with
# round_fn_line3.sh (gate 0.30 + PSI avg60 < 2% at start + PSI over the window, flagged > 2%), DRIVE_FWD_SYNC=0.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2b.pid
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep round_fn_line)" ]; do sleep 30; done
python3 $D/scripts/kid_compare.py $D/out/timing/t_g2chk_v_r1.json $D/out/timing/t_g2chk_fullnc_r1.json $D/out/timing/t_g2chk_breakable_r1.json $D/out/timing/t_g2chk_gdnbreak_r1.json > $D/logs/timing/kid_g2chk.txt 2>&1
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g2b: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g2b: round rc $rc ($*)"; return 0; }
for r in 1 2 3; do run SET=g2w9 KIND=w9 R=$r ARMS="fullnc gdnbreak v"; done
for r in 4 5 6; do run SET=g2w KIND=w R=$r ARMS="fullnc breakable v"; done
echo "$(date +%T) chain_g2b done"
