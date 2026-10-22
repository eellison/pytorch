#!/usr/bin/env bash
# chain_g0e.sh: gb300-box GPU 0, fresh decode grid rounds (coordinator 10-15): d20 x3 (set g0e20; 1K + 8K, the 20-point grid, 8K capped
# at 96), arms fullnc / breakable / v interleaved, round_fn_line3.sh (gate 0.30, PSI), CPUs 0-35, gpu0.lock per arm. Replaces the
# gate-timeout-flagged g0d r1 in the decode table (g0d r1's full curve stays as a labelled supplementary file).
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g0e.pid
export HT_GPU=0 HT_CPUS=0-35 FN_FI_TAG=f0 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock FN_YIELD_PATTERN=
run() { echo "$(date +%T) chain_g0e: $*"; env "$@" bash $D/round_fn_line3.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain_g0e: round rc $rc ($*)"; return 0; }
for r in 1 2 3; do run SET=g0e20 KIND=d20 R=$r; done
echo "$(date +%T) chain_g0e done"
