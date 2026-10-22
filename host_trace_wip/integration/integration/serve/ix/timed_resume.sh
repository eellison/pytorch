#!/usr/bin/env bash
# GPU 0 / CPUs 0-35 (coordinator 10-14 03:25, option b): resume the timed sets once cores 0-35 sit at or below ~25% busy (9 of 36;
# the observed quiet floor) for two samples 10 min apart; the per-run own-core gate at the same 25% (OWN_BUSY_FRAC=0.25). Each run
# keeps foreign_busy_frac. Queue: swph1 vcsl r3, swph1 bnc r1 (flagged rerun), swph1 vrs r1-3 (both helpers). cbh4 waits.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
export OWN_BUSY_FRAC=0.25
quiet=0
while [ $quiet -lt 2 ]; do
  b=$(python3 vllm/cpubusy.py 10 0 35 | cut -d' ' -f1)
  if python3 -c "import sys; sys.exit(not ($b / 36 < 0.25))"; then quiet=$((quiet + 1)); else quiet=0; fi
  echo "$(date +%T) hold25: cores 0-35 busy $b of 36 (quiet samples $quiet/2)" >> serve/logs/hold_poll.log
  [ $quiet -lt 2 ] && sleep 600
done
echo "$(date +%T) RESUME timed sets (cores 0-35 <= 25% busy)" >> $T
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vcsl" bash serve/ix/set_chain.sh
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
for r in 1 2 3; do
  [ -f vllm/out/t_swph1_vrs_r$r.json ] && mv vllm/out/t_swph1_vrs_r$r.json vllm/out/t_swph1sec_vrs_r$r.json && mv vllm/logs/t_swph1_vrs_r$r.log vllm/logs/t_swph1sec_vrs_r$r.log
done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vrs" bash serve/ix/set_chain.sh
echo "$(date +%T) TIMED_RESUMED_DONE" >> $T
