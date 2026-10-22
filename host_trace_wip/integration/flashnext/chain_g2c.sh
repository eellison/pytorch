#!/usr/bin/env bash
# chain_g2c.sh: gb300-box GPU 2 (CPUs 36-71, gpu2.lock per arm, round_fn_line3.sh: gate 0.30 + PSI gate + PSI_WINDOW):
# (a) untimed kernel check g2chk9: breakable+gdnbreak+plebreak (arm gdnple) vs ours at the 9 sizes; the timed rounds run only if
#     kid_compare finds no DIFF there; (b) timed g2w9 x3 (fullnc / gdnple / v); (c) the w rerun g2w r4-r6 (fullnc / breakable / v).
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2c.pid
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g2c: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g2c: round rc $rc ($*)"; return 0; }
run SET=g2chk9 KIND=idchk9 R=1 DRIVE_REQUIRE_WARM=0 OWN_BUSY_FRAC=1.0 ARMS="gdnple v"
python3 $D/scripts/kid_compare.py $D/out/timing/t_g2chk9_v_r1.json $D/out/timing/t_g2chk9_gdnple_r1.json > $D/logs/timing/kid_g2chk9.txt 2>&1
grep SUMMARY $D/logs/timing/kid_g2chk9.txt
if grep -q "DIFF" $D/logs/timing/kid_g2chk9.txt || ! grep -q SUMMARY $D/logs/timing/kid_g2chk9.txt; then
  echo "$(date +%T) chain_g2c: gdnple still differs from ours (or no record): no timed w9; continuing with the w rerun"
else
  for r in 1 2 3; do run SET=g2w9 KIND=w9 R=$r ARMS="fullnc gdnple v"; done
fi
for r in 4 5 6; do run SET=g2w KIND=w R=$r ARMS="fullnc breakable v"; done
echo "$(date +%T) chain_g2c done"
