#!/usr/bin/env bash
# q_split4.sh: after q_split3: decode bs 1 / 32, split off / on, walls + launch shim + pre-launch breakdown (vLLM host vs ours).
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while pgrep -f "q_split3.sh" >/dev/null; do sleep 30; done
export DRIVE_SPLIT_BS=1,32 DRIVE_SPLIT_ONLY_DECODE=1 DRIVE_RANGE_PHASES=decode DRIVE_SPLIT_STEPS=32
sed -e 's/splitab2/split4/g' q_splitab2.sh > q_split4_inner.sh && chmod +x q_split4_inner.sh && ./q_split4_inner.sh
