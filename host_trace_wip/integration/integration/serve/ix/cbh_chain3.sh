#!/usr/bin/env bash
# GPU 3 (CPUs 108-143), in parallel with the GPU 0 sweeps: continuous-batching churn (vllm/round_cb_line.sh), set cbh, one gpu0.lock hold per arm-run, 3 runs per arm.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
ALL=(default fullnc vcs)
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_cbh_${a}_r$r.json ] && continue
      echo "$(date +%T) cbh r$r $a: waiting for gpu0.lock" >> $T
      STOCK_LINE=c5h ARMS=$a R=$r SET=cbh GPU=3 CPUS=108-143 bash vllm/gpuN_job.sh vllm/round_cb_line.sh >> vllm/logs/q_cbh_r$r.log 2>&1
      echo "$(date +%T) cbh r$r $a: done ($( [ -f vllm/out/t_cbh_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_cbh_*_r[123].json 2>/dev/null | wc -l) -ge 9 ] && break
  sleep 120
done
echo "$(date +%T) CBH_DONE $(ls vllm/out/t_cbh_*_r[123].json 2>/dev/null | wc -l)/9" | tee -a $T
