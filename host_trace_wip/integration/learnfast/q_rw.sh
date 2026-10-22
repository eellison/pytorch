#!/usr/bin/env bash
# q_rw.sh: after the template gate, on GPU 3 (untimed, diagnostic): seed + learn the eager-V arm's bindings on the head (Triton GDN,
# max_num_seqs 128), then one vt RANGE=1 run (the range warm-up's traces / redispatches / breakdown).
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
while kill -0 ${GATE_PID:-0} 2>/dev/null; do sleep 30; done
export HEAD=1 LF_GPU=3 LF_CPUS=108-143 LF_CPU0=108 HOLD_S=1150
for st in "seed v" "learn v"; do
  echo "$(date +%T) $st"; ./gpulock.sh 3 3600 ./startup27.sh $st > logs/rw_${st// /_}.log 2>&1; rc=$?
  echo "$(date +%T) $st rc=$rc $(tail -2 logs/rw_${st// /_}.log | tr '\n' ' ')"
  [ $rc -eq 0 ] || exit 1
done
echo "$(date +%T) run vt RANGE=1"; RANGE=1 ./gpulock.sh 3 3600 ./startup27.sh run vt 0 > logs/rw_vt0.log 2>&1
echo "$(date +%T) run vt rc=$? $(grep -c RANGE_WARM logs/rw_vt0.log)"
touch logs/q_rw.done
