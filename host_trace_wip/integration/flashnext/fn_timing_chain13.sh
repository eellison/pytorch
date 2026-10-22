#!/usr/bin/env bash
# fn_timing_chain13.sh: gb300-box GPU 0 FALLBACK for the Flash-Next timing (coordinator 10-14, while dev-box's SSH is closed;
# launch only on the coordinator's go). Same harness / build as fn_timing_chain.sh (step334_ss + build_cpp15, drive_sweep27
# snapshot, 8K curve capped at bs 96, DRIVE_KID=0, arms fullnc / breakable / v interleaved per round), sets tagged g0* so its
# rows are labelled gb300-box GPU 0 and never paired with dev-box's f3* rows. gpu0.lock is held per arm run (FN_LOCK; GPU_LANES.md).
# CPUs 0-35 (GPU 0's timing range, LANES.md). Order (coordinator 10-14): d r1 (full curve), w r1, d20 r2 / r3, w r2 / r3.
# Kernel identity: the dev-box f3id record (same build); no gb300-box idall. Detach: setsid nohup bash fn_timing_chain13.sh > logs/chain13.out 2>&1 &
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain13.pid
export HT_GPU=0 HT_CPUS=${HT_CPUS:-0-35} FN_FI_TAG=f0 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
export FN_YIELD_PATTERN= CMD_TIMEOUT=
run() { echo "$(date +%T) chain13: $*"; env "$@" bash $D/round_fn_line.sh; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) chain13: round rc $rc ($*)"; return 0; }
run SET=g0d KIND=d R=1
run SET=g0w KIND=w R=1
# the 8B lane's startup runs wait for this file, then may take gpu0.lock between our arms (coordinator 10-14)
touch $D/out/timing/CHAIN13_W1_DONE; echo "$(date +%T) chain13: w r1 done, CHAIN13_W1_DONE touched"
for r in 2 3; do run SET=g0d20 KIND=d20 R=$r; done
for r in 2 3; do run SET=g0w KIND=w R=$r; done
echo "$(date +%T) chain13 done"
