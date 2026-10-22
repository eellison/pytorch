#!/usr/bin/env bash
# GPU 0: the slide sweeps on candidate 5 green (c5h: snap/step324 + build_cpp12), set swph, one gpu0.lock hold per arm-run, rotated, 3 runs per arm.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
ALL=(default fullnc vcs)
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_swph_${a}_r$r.json ] && continue
      echo "$(date +%T) swph r$r $a: waiting for gpu0.lock" >> $T
      STOCK_LINE=c5h ARMS=$a R=$r SET=swph bash vllm/gpu0_job2.sh vllm/round_sweep_line.sh >> vllm/logs/q_swph_r$r.log 2>&1
      echo "$(date +%T) swph r$r $a: done ($( [ -f vllm/out/t_swph_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_swph_*_r[123].json 2>/dev/null | wc -l) -ge 9 ] && break
  sleep 120
done
echo "$(date +%T) SWPH_DONE $(ls vllm/out/t_swph_*_r[123].json 2>/dev/null | wc -l)/9" | tee -a $T
