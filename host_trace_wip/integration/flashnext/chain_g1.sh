#!/usr/bin/env bash
# chain_g1.sh: Flash-Next finals on gb300-box GPU 1 (coordinator 10-15: GPU 1 freed by the MiMo lane), CPUs 72-107 (GPU 1's timing range,
# as the MiMo lane), gpu1.lock per arm after the gates, round_fn_line3.sh (gate 0.30, PSI gate + PSI_WINDOW), line step334_ss + build_cpp15.
# Each set on one GPU: w as g1w r1-r3 (fullnc / breakable / v, DRIVE_FWD_SYNC=0), then decode g1e20 r1-r3 (20-point grid, 8K capped at 96).
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g1.pid
export HT_GPU=1 HT_CPUS=72-107 FN_FI_TAG=f1 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g1: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g1: round rc $rc ($*)"; return 0; }
for r in 1 2 3; do run SET=g1w KIND=w R=$r ARMS="fullnc breakable v"; done
for r in 1 2 3; do run SET=g1e20 KIND=d20 R=$r ARMS="fullnc breakable v"; done
echo "$(date +%T) chain_g1 done"
