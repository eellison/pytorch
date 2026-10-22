#!/usr/bin/env bash
# After swp_chain: runs made before the fixed rows at the alternating KV were added (no ix decode_37) are kept as t_swp0_* and
# rerun with the current round script, one gpu0.lock hold each.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
while ! grep -q SWP_DONE $T; do sleep 30; done
for f in vllm/out/t_swp_*_r[123].json; do
  python3 -c "import json,sys; sys.exit(0 if 'decode_37' in json.load(open('$f'))['ix'] else 1)" && continue
  b=$(basename $f .json); a=$(echo $b | cut -d_ -f3); r=${b##*_r}
  mv $f vllm/out/${b/t_swp_/t_swp0_}.json; mv vllm/logs/$b.log vllm/logs/${b/t_swp_/t_swp0_}.log 2>/dev/null
  echo "$(date +%T) swp r$r $a: rerun (fixed rows at alt KV); old kept as ${b/t_swp_/t_swp0_}" >> $T
  STOCK_LINE=c5g ARMS=$a R=$r SET=swp bash vllm/gpu0_job.sh vllm/round_sweep_line.sh >> vllm/logs/q_swp_fix.log 2>&1
  echo "$(date +%T) swp r$r $a: done ($( [ -f vllm/out/$b.json ] && echo measured || echo skipped))" >> $T
done
echo "$(date +%T) SWP_FIX_DONE" >> $T
