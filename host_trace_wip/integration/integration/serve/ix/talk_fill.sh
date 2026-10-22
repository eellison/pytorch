#!/usr/bin/env bash
# Talk timing on GPU 0 under another user's on/off job: rerun the g4s rounds (trun skips existing tags; contended runs are skipped,
# never measured) until all 9 runs exist or 14:15; then the sound startup set (2 rounds) until complete or 15:10.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
have() { ls vllm/out/t_g4s_{default,fullnc,vstock}_r{1,2,3}.json 2>/dev/null | wc -l; }
while pgrep -f "rounds_gpu0.sh" > /dev/null; do sleep 30; done
while [ $(have) -lt 9 ] && [ $(date +%H%M) -lt 1415 ]; do
  echo "$(date +%T) g4s have $(have)/9: rounds again"
  ROUNDS=3 SET=g4s ROUND_SCRIPT=round_g4s_line bash vllm/rounds_gpu0.sh >> vllm/logs/rounds_g4s.out 2>&1
  [ $(have) -lt 9 ] && sleep 120
done
echo "$(date +%T) g4s final $(have)/9"
T=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/talk
shave() { ls $T/out/s2_{default,fullnc,vstock}_r{1,2}.json 2>/dev/null | wc -l; }
while [ $(shave) -lt 6 ] && [ $(date +%H%M) -lt 1510 ]; do
  echo "$(date +%T) startup have $(shave)/6"
  ROUNDS="1 2" bash vllm/gpu0_job.sh $T/startup_round_s.sh >> serve/logs/startup_s.log 2>&1
  [ $(shave) -lt 6 ] && sleep 120
done
echo "$(date +%T) startup final $(shave)/6"
echo TALK_FILL_DONE
