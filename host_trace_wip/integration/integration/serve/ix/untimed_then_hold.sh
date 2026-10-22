#!/usr/bin/env bash
# GPU 0 / CPUs 0-35 (coordinator 10-14 00:10, option a): untimed work now (no own-core gate: OWN_BUSY_FRAC=1): the GC verification
# (gcv), one cold-cache init per arm (wall times coarse under load; per-phase split kept), the sec0 checks (q8vcs_c5h_sec0,
# bs-walk_sec0). Then hold the timed sets until cores 0-35 stay under 10% busy for 10 min (polled every 10 min), and resume:
# swph1 vcsl r3, swph1 bnc r1 (flagged rerun), swph1 vrs r1-3 (both helpers). cbh4 waits for the coordinator.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
export OWN_BUSY_FRAC=1
SET=gcv R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_gcv_line.sh >> vllm/logs/q_gcv.log 2>&1
echo "$(date +%T) GCV_DONE" >> $T
SET=cold R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_cold_line.sh >> vllm/logs/q_cold_r1.log 2>&1
echo "$(date +%T) COLD_DONE" >> $T
TAG=_c5h_sec0 VC_LINE=c5h GPU_IDX=0 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_q8vcs_check.sh > serve/logs/q8vcs_c5h_sec0.log 2>&1
TAG=_c5h_bswalk_sec0 VC_LINE=c5h GPU_IDX=0 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_q8vcs_bswalk.sh > serve/logs/q8vcs_c5h_bswalk_sec0.log 2>&1
echo "$(date +%T) CHECKS_SEC0_DONE" >> $T
unset OWN_BUSY_FRAC
# hold: two consecutive quiet samples 10 min apart (each a 10 s mean) on cores 0-35
quiet=0
while [ $quiet -lt 2 ]; do
  b=$(python3 vllm/cpubusy.py 10 0 35 | cut -d' ' -f1)
  if python3 -c "import sys; sys.exit(not ($b / 36 < 0.10))"; then quiet=$((quiet + 1)); else quiet=0; fi
  echo "$(date +%T) hold: cores 0-35 busy $b of 36 (quiet samples $quiet/2)" >> serve/logs/hold_poll.log
  [ $quiet -lt 2 ] && sleep 600
done
echo "$(date +%T) RESUME timed sets (cores 0-35 quiet)" >> $T
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vcsl" bash serve/ix/set_chain.sh
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="bnc" bash serve/ix/set_chain.sh
for r in 1 2 3; do
  [ -f vllm/out/t_swph1_vrs_r$r.json ] && mv vllm/out/t_swph1_vrs_r$r.json vllm/out/t_swph1sec_vrs_r$r.json && mv vllm/logs/t_swph1_vrs_r$r.log vllm/logs/t_swph1sec_vrs_r$r.log
done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vrs" bash serve/ix/set_chain.sh
echo "$(date +%T) TIMED_RESUMED_DONE" >> $T
