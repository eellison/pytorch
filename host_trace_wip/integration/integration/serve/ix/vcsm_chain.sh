#!/usr/bin/env bash
# GPU 0 after the capture-size study: g4f arm vcsm (vcs with ARMV_DECODE_MAX_SEQ=model), 3 runs, one gpu0.lock hold per run.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q CAPSZ_DONE serve/logs/capsz_timeline.log 2>/dev/null; do sleep 60; done
for r in 1 2 3; do
  [ -f vllm/out/t_g4f_vcsm_r$r.json ] && continue
  STOCK_LINE=c5g ARMS=vcsm R=$r SET=g4f bash vllm/gpu0_job.sh vllm/round_g4f_line.sh >> vllm/logs/q_g4f_vcsm.log 2>&1
  echo "$(date +%T) g4f r$r vcsm: done" >> serve/logs/g4f_timeline.log
done
