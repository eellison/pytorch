#!/usr/bin/env bash
# One comparison set on ONE GPU (10-13 multi-GPU rule: all arms of a set on the same device), one lock hold per arm-run, arm order
# rotated per round, until 3 runs per arm:  SET=swph ROUND=vllm/round_sweep_line.sh GPU=3 CPUS=108-143 LINE=c5h ARMS="default fullnc vcs" bash serve/ix/set_chain.sh
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${SET:?} ${ROUND:?} ${GPU:?} ${CPUS:?} ${LINE:?}
T=serve/logs/swp_timeline.log
ALL=(${ARMS:-default fullnc vcs})
N=$(( 3 * ${#ALL[@]} ))
for pass in 1 2 3; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_${SET}_${a}_r$r.json ] && continue
      echo "$(date +%T) $SET r$r $a: waiting for gpu$GPU.lock" >> $T
      STOCK_LINE=$LINE ARMS=$a R=$r SET=$SET GPU=$GPU CPUS=$CPUS bash vllm/gpuN_job.sh $ROUND >> vllm/logs/q_${SET}_r$r.log 2>&1
      echo "$(date +%T) $SET r$r $a: done on GPU $GPU ($( [ -f vllm/out/t_${SET}_${a}_r$r.json ] && echo measured || echo skipped))" >> $T
    done
  done
  [ $(ls vllm/out/t_${SET}_*_r[123].json 2>/dev/null | wc -l) -ge $N ] && break
  sleep 120
done
echo "$(date +%T) ${SET^^}_DONE on GPU $GPU $(ls vllm/out/t_${SET}_*_r[123].json 2>/dev/null | wc -l)/$N" | tee -a $T
