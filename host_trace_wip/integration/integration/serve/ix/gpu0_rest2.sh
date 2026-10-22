#!/usr/bin/env bash
# GPU 0 / CPUs 0-35, after the swph1 bnc + vrs runs: swph1 vcsl again with the untimed warm-up + gc.freeze() (the earlier vcsl runs kept
# as t_swph1nf_vcsl_r*), one untimed GC-log verification run, the unsynced churn set cbh4 (5 arms x 3), the cold-cache inits.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
while [ "$(grep -c 'SWPH1_DONE' $T)" -lt 2 ]; do sleep 60; done
for r in 1 2 3; do
  [ -f vllm/out/t_swph1_vcsl_r$r.json ] && mv vllm/out/t_swph1_vcsl_r$r.json vllm/out/t_swph1nf_vcsl_r$r.json && mv vllm/logs/t_swph1_vcsl_r$r.log vllm/logs/t_swph1nf_vcsl_r$r.log
done
echo "$(date +%T) swph1 vcsl r1-3 (no freeze) kept as t_swph1nf_vcsl_r*; rerunning with warm-up + gc.freeze" >> $T
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vcsl" bash serve/ix/set_chain.sh
SET=gcv R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_gcv_line.sh >> vllm/logs/q_gcv.log 2>&1
echo "$(date +%T) GCV_DONE" >> $T
CB_NOSYNC=1 SET=cbh4 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc bnc vcsl vrs" bash serve/ix/set_chain.sh
SET=cold R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh vllm/round_cold_line.sh >> vllm/logs/q_cold_r1.log 2>&1
echo "$(date +%T) COLD_DONE" >> $T
