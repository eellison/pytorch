#!/usr/bin/env bash
# Trim kidall (coordinator 10-14 06:30): after the vcsl arm ends, stop the running round (before vcslm), then run fullnc / bnc / vrs;
# then the timed queue (the same flow as before) and the cbh4 launch decision.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
J=$(cut -d' ' -f3 /tmp/kidall_job.pid)
killtree() { local p=$1; for c in $(ps -eo pid,ppid | awk -v p=$p '$2==p{print $1}'); do killtree $c; done; kill $p 2>/dev/null; }
until grep -q "end t_kidall_vcsl_r1" vllm/logs/timing_idle.log 2>/dev/null || grep -q "end t_kidall_vcsl_r1" vllm/logs/q_kidall.log 2>/dev/null; do sleep 15; done
sleep 2
killtree $J
sleep 10
echo "$(date +%T) kidall: vcslm / vrsm dropped (coordinator); running fullnc / bnc / vrs" >> serve/logs/swp_timeline.log
OWN_BUSY_FRAC=1 SET=kidall R=1 STOCK_LINE=c5h GPU=0 CPUS=0-35 ARMS="fullnc bnc vrs" bash vllm/gpuN_job.sh vllm/round_kidall_line.sh >> vllm/logs/q_kidall.log 2>&1
python3 serve/ix/kid_every_size.py kidall > serve/out/kidall_compare.md 2>&1
echo "$(date +%T) KIDALL_DONE" >> serve/logs/swp_timeline.log
