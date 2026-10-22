#!/usr/bin/env bash
# GPU 0: the Vc arm (vllm/round_g4vc_line.sh ARMS=vc into SET=g4s, reusing g4s's default / default-nc / V-registered columns) until
# 3 runs; contended runs are skipped by trun's idle check and refilled.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
have() { ls vllm/out/t_g4s_vc_r{1,2,3}.json 2>/dev/null | wc -l; }
while [ $(have) -lt 3 ]; do
  echo "$(date +%T) vc have $(have)/3"
  ARMS=vc ROUNDS=3 SET=g4s ROUND_SCRIPT=round_g4vc_line GATE_MAX=1200 bash vllm/rounds_gpu0.sh >> vllm/logs/rounds_g4s_vc.out 2>&1
  [ $(have) -lt 3 ] && sleep 120
done
echo "$(date +%T) VC_FILL_DONE $(have)/3"
