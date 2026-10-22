#!/usr/bin/env bash
# Like rounds_gpu0.sh on GPU 2 (user rule 10-12: GPUs 2/3 when idle), but ONE gpu2.lock hold for the whole set so the arms' ratios
# come from one GPU: waits for gpu_idle_ok.sh 2, takes the lock, rechecks before each round (another user appears: stop), CPUs 108-125.
#   SET=q1s ROUND_SCRIPT=round_q35s_line ROUNDS=3 setsid nohup bash vllm/rounds_gpu2.sh > vllm/logs/rounds_q1s.out 2>&1 &
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
SET=${SET:?} RS=${ROUND_SCRIPT:?} N=${ROUNDS:-3}
until bash $E/gpu_idle_ok.sh 2 > /dev/null 2>&1; do sleep 60; done
exec 9>$E/gpu2.lock
flock -x 9
echo "$(date +%T) holding gpu2.lock"
for r in $(seq ${ROUND_START:-1} $N); do
  bash $E/gpu_idle_ok.sh 2 > /dev/null 2>&1 || { echo "$(date +%T) gpu2 taken by another user: stop"; break; }
  echo "$(date +%T) round $r load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a vllm/logs/rounds_$SET.log
  SET=$SET R=$r HT_GPU=2 CUDA_VISIBLE_DEVICES=2 taskset -c 108-125 bash vllm/$RS.sh > vllm/logs/q_${SET}_r$r.log 2>&1
  echo "$(date +%T) round $r done rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a vllm/logs/rounds_$SET.log
done
echo ALLDONE
