#!/usr/bin/env bash
# rounds_gpuN.sh (G=1|2|3): like rounds_gpu0.sh on GPU $G (user rule 10-12: GPUs 2/3 when idle), but ONE gpu$G.lock hold for the whole set so the arms' ratios
# come from one GPU: waits for gpu_idle_ok.sh $G, takes the lock, rechecks before each round (another user appears: stop), CPUs 108-125.
#   SET=q1s ROUND_SCRIPT=round_q35s_line ROUNDS=3 setsid nohup bash vllm/rounds_gpu$G.sh > vllm/logs/rounds_q1s.out 2>&1 &
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
SET=${SET:?} RS=${ROUND_SCRIPT:?} N=${ROUNDS:-3} G=${G:?}
case $G in 1) CPUS=72-107 ;; 2) CPUS=108-125 ;; 3) CPUS=126-143 ;; esac
until bash $E/gpu_idle_ok.sh $G > /dev/null 2>&1; do sleep 60; done
exec 9>$E/gpu$G.lock
flock -x 9
echo "$(date +%T) holding gpu$G.lock"
for r in $(seq ${ROUND_START:-1} $N); do
  bash $E/gpu_idle_ok.sh $G > /dev/null 2>&1 || { echo "$(date +%T) gpu$G taken by another user: stop"; break; }
  echo "$(date +%T) round $r load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a vllm/logs/rounds_$SET.log
  SET=$SET R=$r HT_GPU=$G CUDA_VISIBLE_DEVICES=$G taskset -c $CPUS bash vllm/$RS.sh > vllm/logs/q_${SET}_r$r.log 2>&1
  echo "$(date +%T) round $r done rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a vllm/logs/rounds_$SET.log
done
echo ALLDONE
