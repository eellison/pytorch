#!/usr/bin/env bash
# fn_timing_chain.sh: Flash-Next eager-forward timing on dev-box GPU 1 (coordinator / user 10-14), started by dev3_run.sh
# (which holds gpu1.lock, sets HT_GPU=1 / CUDA_VISIBLE_DEVICES=1, CPUs 36-71). Order (coordinator): kernel identity at every
# timed size (idall, untimed), then decode (full curve r1, 20-point grid r2 / r3), then churn, then prefill / mixed. Arms
# fullnc, breakable, v (fast-learned cuBLAS bindings; learnv fallback). Round scripts skip outputs that exist, so a rerun
# resumes. Sets (one per workload, dev-box-unique tags): f3id, f3d, f3d20, f3cb, f3w.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
export HT_GPU=${HT_GPU:-1} HT_CPUS=${HT_CPUS:-36-71} FN_FI_TAG=f3  # one FlashInfer autotune file for every set and arm (fullnc of f3id primes it)
export OWN_BUSY_FRAC=${OWN_BUSY_FRAC:-0.20}  # dev-box gate 0.20 (shared-host floor ~0.15; coordinator 10-14)
export FN_YIELD_PATTERN=${FN_YIELD_PATTERN:-dev3_run.sh 1 [^ ]*cbh4}  # the 8B lane's queued cbh4 (coordinator: yield at arm boundaries)
run() { echo "$(date +%T) chain: $*"; env "$@" bash $D/round_fn_line.sh; local rc=$?; [ $rc = 75 ] && { echo "$(date +%T) chain: yielded to cbh4; stopping (resumable)"; exit 75; }; [ $rc = 0 ] || echo "$(date +%T) chain: round rc $rc ($*)"; return 0; }
run SET=f3id KIND=idall R=1 DRIVE_REQUIRE_WARM=0 OWN_BUSY_FRAC=1.0 ARMS="fullnc breakable v"  # untimed: no own-core gate
python3 $D/scripts/kid_compare.py $D/out/timing/t_f3id_v_r1.json $D/out/timing/t_f3id_fullnc_r1.json $D/out/timing/t_f3id_breakable_r1.json \
  > $D/logs/timing/kid_f3id.txt 2>&1
# idmem (untimed memset bytes vs bs, 3 arms) is deferred until the core memset-binding fix lands (coordinator 10-14), then run as
# its verification: run SET=f3idm KIND=idmem R=1 DRIVE_REQUIRE_WARM=0 OWN_BUSY_FRAC=1.0 ARMS="fullnc breakable v"
run SET=f3d KIND=d R=1  # timed rows carry no kernel record (DRIVE_KID=0: CUPTI slows the rest of a profiled process)
for r in 2 3; do run SET=f3d20 KIND=d20 R=$r; done
for r in 1 2 3; do run SET=f3cb KIND=cb R=$r; done
for r in 1 2 3; do run SET=f3w KIND=w R=$r; done
echo "$(date +%T) chain done"
