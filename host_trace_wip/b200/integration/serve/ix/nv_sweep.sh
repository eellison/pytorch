#!/usr/bin/env bash
# NVFP4 InferenceX-method sweep (user priority 10-12): Qwen3.8-27B (default / default-nc / V-stock) then Qwen3.8-Flash-Next
# (default = the recipe's no-compile override / V-stock), on candidate 4, SKUs in vllm/ix_master_attnlc.yaml. 1k1k + 8k1k at
# conc 1/4/16/64; repeat 1 of every point first, then repeats 2 and 3 of conc 4 and 64. Fresh server per point (ix_run.py).
# GPU 0 only, one gpu0.lock hold per point. Before the hold: >= FREE_MIB free on GPU 0. In the hold: idle check (no compute
# process, <= 5% util over 10 s) for up to 10 min; if GPU 0 stays busy the hold is released and the point waits again
# (no contended points). Per-point idle record: serve/logs/${TAG}_idle.log.
#   setsid nohup bash serve/ix/nv_sweep.sh TAG > serve/logs/TAG.log 2>&1 &
TAG=${1:-nv1}
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
MASTER=$G/vllm/ix_master_attnlc.yaml
M27=qwen3.827bp4-fp4-gb300-vllm,qwen3.827bnc4-fp4-gb300-vllm,qwen3.827bvs4-fp4-gb300-vllm
MFN=qwen3.8flashnextp4-fp4-gb300-vllm,qwen3.8flashnextvs4-fp4-gb300-vllm
ALL=1k1k:1,8k1k:1,1k1k:4,8k1k:4,1k1k:16,8k1k:16,1k1k:64,8k1k:64
REP=1k1k:4,8k1k:4,1k1k:64,8k1k:64
PLAN=${PLAN:-"r1|$M27|$ALL r1|$MFN|$ALL r2|$M27|$REP r2|$MFN|$REP r3|$M27|$REP r3|$MFN|$REP"}
cd $G
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}'; }
for blk in $PLAN; do
  rep=${blk%%|*} rest=${blk#*|} skus=${rest%%|*} pts=${rest#*|}
  skus=${skus//,/ } pts=${pts//,/ }
  for pt in $pts; do
    sc=${pt%%:*} c=${pt##*:}
    for sku in $skus; do
      t=${TAG}_$rep
      case $sku in *flashnext*) need=250000 ;; *) need=150000 ;; esac
      while :; do
        [ -f serve/out/ix/$t/${sku}_${sc}_c${c}_point.json ] && { echo "$(date +%T) $t $sku $sc c$c done"; break; }
        until [ $(free_mib) -ge $need ]; do sleep 60; done
        warm=; case $sku in *vs4-*|*vs-*|*ht-*) warm=--warm ;; esac
        echo "$(date +%T) $t $sku $sc c$c waiting for gpu0.lock"
        FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 flock -x $LOCK bash -c '
          t0=$(date +%s)
          while :; do
            apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l); busy=0
            for i in $(seq 10); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); [ "$u" -gt 5 ] && busy=1; sleep 1; done
            [ "$apps" = 0 ] && [ $busy = 0 ] && break
            [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "$(date +%T) $2 $3 c$4: busy, hold released: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0 | tr "\n" " ")" >> serve/logs/${1%_r*}_idle.log; exit 0; }
            sleep 20
          done
          echo "$(date +%T) $1 $2 $3 c$4: idle ok after $(( $(date +%s) - t0 )) s load $(cut -d" " -f1-3 /proc/loadavg)" >> serve/logs/${1%_r*}_idle.log
          timeout 5400 taskset -c 64-71 bash python_vllm_cand2.sh serve/ix/ix_run.py --master '$MASTER' --gpu 0 --sku $2 --scenario $3 --concs $4 --tag $1 --profile-steps 20 $5
          rc=$?; echo "$(date +%T) $1 $2 $3 c$4 rc=$rc"' _ $t $sku $sc $c "$warm"
        [ -f serve/out/ix/$t/${sku}_${sc}_c${c}_point.json ] || { [ -n "$(grep -s "$t $sku $sc c$c rc=" serve/logs/$TAG.log)" ] && { echo "$(date +%T) $t $sku $sc c$c FAILED (no point json), next"; break; }; }
        sleep 30
      done
    done
  done
done
echo NV_SWEEP_DONE
