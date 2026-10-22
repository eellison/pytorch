#!/usr/bin/env bash
# after the n27cbn + n27cbng churn rerun (gpu2_cbn.sh) exits and gpu2.lock is released: start Flash-Next's w set on GPU 2 (coordinator 10-14)
while pgrep -f "^bash gpu2_cbn.sh" > /dev/null; do sleep 30; done
while lsof /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock 2>/dev/null | grep -q .; do
  if lsof /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock 2>/dev/null | awk 'NR>1{print $2}' | xargs -r ps -o args= -p | grep -q "r27"; then sleep 30; else break; fi
done
touch /data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/logs/mon/g2.go
echo "$(date +%T) touched g2.go"
