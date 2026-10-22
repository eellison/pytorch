#!/usr/bin/env bash
# chain_g2dur.sh: untimed per-kernel GPU durations (KID kernel_us / gpu_span_us), breakable vs fullnc, at the idchk points (1K bs 8 / 9 /
# 31 / 32 / 33 / 64 / 128, 8K bs 8 / 9, and the 9 prefill / mixed sizes) on GPU 2 between chain_g2c's arms (gpu2.lock per arm).
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g2dur.pid
export HT_GPU=2 HT_CPUS=36-71 FN_FI_TAG=f2 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock FN_YIELD_PATTERN=
env SET=g2dur KIND=idchk R=1 DRIVE_REQUIRE_WARM=0 OWN_BUSY_FRAC=1.0 ARMS="fullnc breakable" bash $D/round_fn_line3.sh
echo "$(date +%T) chain_g2dur done"
