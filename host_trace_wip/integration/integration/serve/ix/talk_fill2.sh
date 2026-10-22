#!/usr/bin/env bash
# GPU 0 (coordinator 13:00): V-registered arm (ARMS=vreg, SET=g4s: reuses g4s's default / default-nc columns) until 3 runs or 15:10,
# then the sound startup set until 15:10. Contended runs are skipped (trun idle check), never measured.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
have() { ls vllm/out/t_g4s_vreg_r{1,2,3}.json 2>/dev/null | wc -l; }
while [ $(have) -lt 3 ] && [ $(date +%H%M) -lt 1510 ]; do
  echo "$(date +%T) vreg have $(have)/3"
  ARMS=vreg ROUNDS=3 SET=g4s ROUND_SCRIPT=round_g4r_line GATE_MAX=600 bash vllm/rounds_gpu0.sh >> vllm/logs/rounds_g4s_vreg.out 2>&1
  [ $(have) -lt 3 ] && sleep 60
done
echo "$(date +%T) vreg final $(have)/3"
T=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/talk
shave() { ls $T/out/s2_{default,fullnc,vstock}_r{1,2}.json 2>/dev/null | wc -l; }
while [ $(shave) -lt 6 ] && [ $(date +%H%M) -lt 1510 ]; do
  echo "$(date +%T) startup have $(shave)/6"
  ROUNDS="1 2" bash vllm/gpu0_job.sh $T/startup_round_s.sh >> serve/logs/startup_s.log 2>&1
  [ $(shave) -lt 6 ] && sleep 60
done
echo "$(date +%T) startup final $(shave)/6"
echo TALK_FILL2_DONE
