#!/usr/bin/env bash
# Low-noise timing set on GPU 0 (CPUs 0-71 via gpu0_job.sh): ROUNDS (default 6) rounds of ROUND_SCRIPT, one gpu0.lock hold per round
# (other lanes can take the lock between rounds). Each round starts only after load_gate (outside the lock: load1 < LOAD_MAX and
# busy cores on 0-71 < BUSY_MAX); the gate verdict and load per round go to vllm/logs/rounds_<SET>.log, per run to timing_idle.log.
# A gate timeout still runs the round, tagged in the log, so a set always finishes; c2tables reports its rounds' gate verdicts.
#   SET=g1 ROUND_SCRIPT=round_stock_c2 setsid nohup bash vllm/rounds_gpu0.sh > vllm/logs/rounds_g1.out 2>&1 < /dev/null &
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
source vllm/timing_lib.sh
SET=${SET:?} RS=${ROUND_SCRIPT:?} N=${ROUNDS:-6}
for r in $(seq ${ROUND_START:-1} $N); do
  v=$(load_gate | tail -1)
  echo "$(date +%T) round $r $v" | tee -a vllm/logs/rounds_$SET.log
  SET=$SET R=$r bash vllm/gpu0_job.sh vllm/$RS.sh > vllm/logs/q_${SET}_r$r.log 2>&1
  echo "$(date +%T) round $r done rc=$? load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a vllm/logs/rounds_$SET.log
done
echo ALLDONE
