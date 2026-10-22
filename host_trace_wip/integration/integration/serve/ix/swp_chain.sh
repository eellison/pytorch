#!/usr/bin/env bash
# GPU 0: the slide sweeps (vllm/round_sweep_line.sh), one gpu0.lock hold per arm-run, arm order rotated per round, 3 runs per arm.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
ALL=(default fullnc vcs)
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_swp_${a}_r$r.json ] && continue
      echo "$(date +%T) swp r$r $a: waiting for gpu0.lock" >> $T
      STOCK_LINE=c5g ARMS=$a R=$r SET=swp bash vllm/gpu0_job.sh vllm/round_sweep_line.sh >> vllm/logs/q_swp_r$r.log 2>&1
      echo "$(date +%T) swp r$r $a: done ($( [ -f vllm/out/t_swp_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_swp_*_r[123].json 2>/dev/null | wc -l) -ge 9 ] && break
  sleep 120
done
echo "$(date +%T) SWP_DONE $(ls vllm/out/t_swp_*_r[123].json 2>/dev/null | wc -l)/9" | tee -a $T
