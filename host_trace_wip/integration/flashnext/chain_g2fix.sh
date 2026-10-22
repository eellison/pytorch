#!/usr/bin/env bash
# chain_g2fix.sh: gb300-box GPU 2 (the w set's GPU), CPUs 36-71, gpu2.lock per arm run after the gates (round_fn_line2.sh):
# (1) untimed kernel check g2chk (DRIVE_KID=1): the 9 breakable-INVALID prefill / mixed sizes and decode bs 8 / 9 / 31 / 32 / 33 /
#     64 / 128 (1K; 8 / 9 at 8K), arms fullnc / breakable / gdnbreak / v: does breakable+gdnbreak match ours' kernels at the 9 sizes,
#     and why is breakable's decode faster at unpadded capture sizes?
# (2) timed w9 (the 9 sizes), 3 rounds, arms fullnc / gdnbreak / v interleaved, DRIVE_FWD_SYNC=0 as w, set g2w9 = gb300-box GPU 2.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2fix.pid
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g2fix: $*"; env "$@" bash $D/round_fn_line2.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g2fix: round rc $rc ($*)"; return 0; }
run SET=g2chk KIND=idchk R=1 DRIVE_REQUIRE_WARM=0 OWN_BUSY_FRAC=1.0 ARMS="fullnc breakable gdnbreak v"
python3 $D/scripts/kid_compare.py $D/out/timing/t_g2chk_v_r1.json $D/out/timing/t_g2chk_fullnc_r1.json $D/out/timing/t_g2chk_breakable_r1.json $D/out/timing/t_g2chk_gdnbreak_r1.json > $D/logs/timing/kid_g2chk.txt 2>&1
for r in 1 2 3; do run SET=g2w9 KIND=w9 R=$r ARMS="fullnc gdnbreak v"; done
echo "$(date +%T) chain_g2fix done"
