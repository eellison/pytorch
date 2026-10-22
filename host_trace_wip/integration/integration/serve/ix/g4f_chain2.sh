#!/usr/bin/env bash
# GPU 0: the full-step set g4f on candidate 5 gated (CAND_LINE=c5g), ONE gpu0.lock hold per run (vllm/gpu0_job.sh around a one-arm
# round), so other lanes' runs interleave; arm order rotated per round; refilled until 3 runs per arm. Timeline: serve/logs/g4f_timeline.log.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q "done on" serve/logs/g4fwarm.outer 2>/dev/null; do sleep 30; done
ARMS_ALL=(default fullnc vreg vc vcs vrs)
for pass in 1 2 3 4 5; do
  for r in 1 2 3; do
    k=$(( (r - 1) % ${#ARMS_ALL[@]} )); order=("${ARMS_ALL[@]:$k}" "${ARMS_ALL[@]:0:$k}")
    for a in "${order[@]}"; do
      [ -f vllm/out/t_g4f_${a}_r$r.json ] && continue
      echo "$(date +%T) g4f r$r $a: waiting for gpu0.lock" >> serve/logs/g4f_timeline.log
      STOCK_LINE=c5g ARMS=$a R=$r SET=g4f bash vllm/gpu0_job.sh vllm/round_g4f_line.sh >> vllm/logs/q_g4f_r$r.log 2>&1
      echo "$(date +%T) g4f r$r $a: done ($( [ -f vllm/out/t_g4f_${a}_r$r.json ] && echo measured || echo skipped))" >> serve/logs/g4f_timeline.log
    done
  done
  [ $(ls vllm/out/t_g4f_*_r[123].json 2>/dev/null | wc -l) -ge 18 ] && break
  sleep 120
done
echo "$(date +%T) G4F_DONE $(ls vllm/out/t_g4f_*_r[123].json 2>/dev/null | wc -l)/18" | tee -a serve/logs/g4f_timeline.log
