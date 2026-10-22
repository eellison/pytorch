#!/usr/bin/env bash
# Talk chain (10-12, sound config only):
#  A) q1s on GPU 2 (vllm/rounds_gpu2.sh, one gpu2.lock hold, 3 rounds) once q35s (Qwen3.5-9B ARMV_SOUND --check) is all bitwise;
#  B) talk/startup_round_s.sh on GPU 0 (one gpu0_job hold) after the g4s rounds.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
(
  while ! grep -q "^Q35S_CHECKS\|q35s rc=[1-9]" serve/logs/q35s.log 2>/dev/null; do sleep 30; done
  read n ok < <(grep "^Q35S_CHECKS" serve/logs/q35s.log | awk '{print $2, $3}')
  if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$n" = "$ok" ]; then
    echo "$(date +%T) q35s $ok/$n bitwise: q1s on GPU 2"
    SET=q1s ROUND_SCRIPT=round_q35s_line ROUNDS=3 bash vllm/rounds_gpu2.sh > vllm/logs/rounds_q1s.out 2>&1
    echo "$(date +%T) q1s done"
  else
    echo "$(date +%T) q35s not bitwise ($ok/$n): q1s not started"
  fi
) &
(
  while ! grep -q ALLDONE vllm/logs/rounds_g4s.out 2>/dev/null; do sleep 30; done
  echo "$(date +%T) g4s done: startup_s on GPU 0"
  ROUNDS="1 2" bash vllm/gpu0_job.sh /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/talk/startup_round_s.sh > serve/logs/startup_s.log 2>&1
  echo "$(date +%T) startup_s done rc=$?"
) &
wait
echo TALK_CHAIN_DONE
