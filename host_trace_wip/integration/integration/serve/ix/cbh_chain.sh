#!/usr/bin/env bash
# GPU 0 after the c5h sweeps: continuous-batching churn (vllm/round_cb_line.sh), set cbh, one gpu0.lock hold per arm-run, 3 runs per arm.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
while ! grep -q SWPH_DONE $T; do sleep 30; done
ALL=(default fullnc vcs)
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_cbh_${a}_r$r.json ] && continue
      echo "$(date +%T) cbh r$r $a: waiting for gpu0.lock" >> $T
      STOCK_LINE=c5h ARMS=$a R=$r SET=cbh bash vllm/gpu0_job2.sh vllm/round_cb_line.sh >> vllm/logs/q_cbh_r$r.log 2>&1
      echo "$(date +%T) cbh r$r $a: done ($( [ -f vllm/out/t_cbh_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_cbh_*_r[123].json 2>/dev/null | wc -l) -ge 9 ] && break
  sleep 120
done
echo "$(date +%T) CBH_DONE $(ls vllm/out/t_cbh_*_r[123].json 2>/dev/null | wc -l)/9" | tee -a $T
