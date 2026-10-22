#!/usr/bin/env bash
# GPU 0: the capture-size study (vllm/round_capsz_line.sh), one gpu0.lock hold per run, arm order rotated per round, 3 runs per arm
# (fnc2304 once: it documents vLLM's downgrade). Timeline: serve/logs/capsz_timeline.log.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/capsz_timeline.log
ALL=(dc2304 bnc2304 dc4096 bnc4096)
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    [ $r = 1 ] && order+=(fnc2304)
    for a in "${order[@]}"; do
      [ -f vllm/out/t_capsz_${a}_r$r.json ] && continue
      echo "$(date +%T) capsz r$r $a: waiting for gpu0.lock" >> $T
      STOCK_LINE=c5g ARMS=$a R=$r SET=capsz bash vllm/gpu0_job.sh vllm/round_capsz_line.sh >> vllm/logs/q_capsz_r$r.log 2>&1
      echo "$(date +%T) capsz r$r $a: done ($( [ -f vllm/out/t_capsz_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_capsz_*_r[123].json 2>/dev/null | wc -l) -ge 13 ] && break
  sleep 120
done
echo "$(date +%T) CAPSZ_DONE $(ls vllm/out/t_capsz_*_r[123].json 2>/dev/null | wc -l)/13" | tee -a $T
