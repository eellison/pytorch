#!/usr/bin/env bash
# cbsmoke13.sh: untimed churn smoke on gb300-box GPU 3 (gpu3.sh, one job per arm): the cb path's unsynced window with the fixed
# driver must show 0 harness synchronizes (UNSYNCED_SYNCS assert) and its Kineto CUDA-runtime sync counts (CB_SYNCPROF lines).
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $F/logs/mon/cbsmoke13.pid
for arm in fullnc breakable v; do
  CMD_TIMEOUT=1200 bash $F/gpu3.sh fn_cbs_$arm env SET=cbs2 KIND=cbsmoke R=1 ARMS=$arm FN_SKIP_IDLE=1 OWN_BUSY_FRAC=1.0 \
    DRIVE_REQUIRE_WARM=0 HT_GPU=3 HT_CPUS=108-143 FN_FI_TAG=mset FN_TIMEOUT=1100 FN_YIELD_PATTERN= bash $F/round_fn_line2.sh
done
grep -h "CB_SYNCPROF\|unsynced\|RuntimeError" $F/logs/timing/t_cbs2_*_r1.log | cut -c1-400
