#!/usr/bin/env bash
# Provisional InferenceX-method sweep (pre-core-done: learn-cost follow-ups, attention fixes, entry binding pending), GPU 0,
# one gpu0.lock hold per point so other lanes interleave. Fresh server per point (ix_run.py), InferenceX's client,
# profiler window (20 steps) after each point, V-stock gets the HTTP range warm-up after install (startup).
# Before each hold: >= 150 GiB free on GPU 0; inside: idle check (<= 20 min, no other compute process, <= 5% util),
# recorded per point in logs/${TAG}_idle.log (a contended point still runs, flagged).
#   setsid nohup bash serve/ix/overnight.sh TAG > serve/logs/TAG.log 2>&1 &
TAG=${1:-ov1}
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
POINTS=${POINTS:-"1k1k:1 1k1k:4 1k1k:16 1k1k:64 8k1k:1 8k1k:4 8k1k:16"}
SKUS=${SKUS:-"qwen38bp-bf16-gb300-vllm qwen38bvs-bf16-gb300-vllm"}
cd $G
for pt in $POINTS; do
  sc=${pt%%:*} c=${pt##*:}
  for sku in $SKUS; do
    [ -f serve/out/ix/$TAG/${sku}_${sc}_c${c}_point.json ] && { echo "$(date +%T) skip $sku $sc c$c (done)"; continue; }
    until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge 153600 ]; do sleep 60; done
    warm=; case $sku in *vs-*|*ht-*) warm=--warm ;; esac
    echo "$(date +%T) $sku $sc c$c waiting for gpu0.lock"
    FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 flock -x $LOCK bash -c '
      t0=$(date +%s)
      while :; do
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l); busy=0
        for i in $(seq 10); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); [ "$u" -gt 5 ] && busy=1; sleep 1; done
        [ "$apps" = 0 ] && [ $busy = 0 ] && { v="idle ok after $(( $(date +%s) - t0 )) s"; break; }
        [ $(( $(date +%s) - t0 )) -gt 1200 ] && { v="CONTENDED: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0 | tr "\n" " ")"; break; }
        sleep 30
      done
      echo "$(date +%T) $2 $3 c$4: $v load $(cut -d" " -f1-3 /proc/loadavg)" >> serve/logs/$1_idle.log
      timeout 2700 taskset -c 64-71 bash python_vllm_cand2.sh serve/ix/ix_run.py --gpu 0 --sku $2 --scenario $3 --concs $4 --tag $1 --profile-steps 20 $5
      echo "$(date +%T) $2 $3 c$4 rc=$?"' _ $TAG $sku $sc $c "$warm"
  done
done
echo OVERNIGHT_DONE
