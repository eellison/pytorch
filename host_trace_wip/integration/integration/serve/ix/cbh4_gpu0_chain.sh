#!/usr/bin/env bash
# gb300-box GPU 0 after the diag3b chain (PID $1; main 10-14, option b): the setlog run (untimed, about 6 min), then cbh4: the unsynced
# churn set (async scheduling as shipped, no syncs in the windows), 5 arms x 3 rounds interleaved, same bindings as cbh3
# (bindings_cbh4.pkl -> bindings_cbh.pkl), per-step intervals split by size changed / same (ns_split). GPU 0 goes to 27B after.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 5; done
GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_setlog.sh > serve/logs/setlog_vcsl.log 2>&1
echo "$(date +%T) SETLOG_DONE" >> serve/logs/swp_timeline.log
CB_NOSYNC=1 OWN_BUSY_FRAC=0.25 SET=cbh4 ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default fullnc bnc vcsl vrs" bash serve/ix/set_chain.sh
