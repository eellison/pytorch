#!/usr/bin/env bash
# Phase A timing (GPU 0, one gpu0.lock hold for the whole job): Qwen3-8B under `vllm serve`, vllm bench serve random
# dataset with the InferenceX fixed-seq client settings (see STATUS.md), 1k1k then 8k1k, concurrency 1..64, arms
# default (vLLM shipped) and V (enforce_eager + arm V; HTTP range warm-up before the sweep). One server per arm.
# Fixed KV pool for both arms (--kv-cache-memory-bytes 120 GiB), util 0.3 for vLLM's startup check.
# gpu0.lock does not keep foreign users off GPU 0 (round t1 failed: a foreign benchmark took 130-230 GB at startup):
# before each arm, wait (<= 20 min) until GPU 0 has no compute process and <= 5% util over 10 samples, else skip the arm;
# logs/${TAG}_gpu0_monitor.log samples util and processes every 5 s during the job.
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
export TAG=${TAG:-t2} GPU=0 UTIL=0.3 SERVER_EXTRA="--kv-cache-memory-bytes 128849018880"
C=${CONCS:-1,2,4,8,16,32,64}
echo "$(date +%T) waiting for gpu0.lock"
exec flock -x $LOCK bash -c '
S=$0 C=$1
echo "$(date +%T) holding gpu0.lock; load $(cut -d" " -f1-3 /proc/loadavg)"
idle() {
  local t0=$(date +%s) apps busy u
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l); busy=0
    for i in $(seq 10); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); [ "$u" -gt 5 ] && busy=1; sleep 1; done
    [ "$apps" = 0 ] && [ $busy = 0 ] && { echo "idle ok after $(( $(date +%s) - t0 )) s"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "idle CONTENDED: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0 | tr "\n" " ")"; return 1; }
    sleep 30
  done
}
while :; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0 | tr "\n" " ")"; sleep 5; done > $S/logs/${TAG}_gpu0_monitor.log 2>&1 &
MON=$!
idle && timeout 5400 bash $S/session.sh default sweep:1k1k:922:922:0.1111:$C sweep:8k1k:7373:922:0.1111:$C
idle && timeout 7200 bash $S/session.sh V warm ctr:warm sweep:1k1k:922:922:0.1111:$C sweep:8k1k:7373:922:0.1111:$C ctr:end summary:end
kill $MON
echo PHASEA_GPU0_DONE' $SV $C
