#!/usr/bin/env bash
# timing_gpuN4.sh: timing_gpuN3.sh plus the start gates (coordinator 10-15), run before taking the GPU lock: own pinned cores' mean busy
# fraction over 10 s < ${OWN_BUSY_FRAC:-0.30} (waits up to 30 min, then flags) and host memory pressure /proc/pressure/memory full avg60
# < ${PSI_GATE:-2}% (waits up to 60 min, then flags); both values per run in logs/timing_gates.log and out/timing/<tag>.gates.
# timing_gpuN3.sh ROUND_SCRIPT SET GPU CPUS: timing_gpuN2.sh with LANES.md's wrapper rules (10-14): the GPU lock is taken on an fd and
# flock's exit status checked (the job never starts without the lock; abort if flock fails or is killed), and inside the hold the job
# waits for the GPU's used memory to be < 20 GiB. Offline learns for our arms first, optional cold-cache startup runs (COLD=1), then
# ${ROUNDS:-3} rounds, every arm of the comparison on the SAME GPU, arms rotated per round, one hold per arm-run; then the table.
RS=$1 SET=$2 G=$3 CPUS=$4
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
arms=(${R27_ARMS:-default fullnc vcvp})
CB=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/cpubusy.py
gates() {  # tag: own-core and PSI gates before the lock
  local tag=$1 lo=${CPUS%-*} hi=${CPUS#*-} n lw=0 pw=0 own p60 flag=""
  n=$((hi - lo + 1))
  while own=$(python3 $CB 10 $lo $hi | cut -d' ' -f1); python3 -c "import sys; sys.exit(not ($own / $n >= ${OWN_BUSY_FRAC:-0.30}))" && [ $lw -lt 1800 ]; do sleep 20; lw=$((lw + 30)); done
  [ $lw -ge 1800 ] && flag="$flag OWN_CORES_GATE_TIMEOUT"
  p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
  while python3 -c "import sys; sys.exit(not ($p60 >= ${PSI_GATE:-2}))" && [ $pw -lt 3600 ]; do sleep 30; pw=$((pw + 30)); p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory); done
  [ $pw -ge 3600 ] && flag="$flag PSI_GATE_TIMEOUT"
  local frac=$(python3 -c "print(round($own / $n, 3))")
  echo "$(date +%T) GATES $tag own_busy $frac (waited ${lw}s) psi_full_avg60 $p60 (waited ${pw}s)$flag" | tee -a $R27/logs/timing_gates.log
  echo "{\"own_busy\": $frac, \"own_wait_s\": $lw, \"psi_full_avg60\": $p60, \"psi_wait_s\": $pw, \"flag\": \"${flag# }\"}" > $R27/out/timing/$tag.gates
}
run() {  # arm round
  gates t_${SET}_${1}_r$2
  echo "$(date +%T) $SET round $2 $1 waiting for an empty GPU $G and gpu$G.lock"
  until [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) $SET: flock on gpu$G.lock failed (rc $?); aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do
    echo "$(date +%T) $SET holding gpu$G.lock, waiting for GPU $G used memory < 20 GiB"; sleep 30
  done
  env R27_ARMS=$1 SET=$SET R=$2 HT_GPU=$G R27_KV_GIB=${R27_KV_GIB:-40} taskset -c $CPUS bash $R27/$RS
  exec 9>&-
}
if [ -n "${COLD:-}" ]; then
  for a in "${arms[@]}"; do [ -f $R27/out/timing/t_${SET}_${a}cold_r0.json ] || run ${a}cold 0; done
fi
[[ " ${arms[*]} " == *" vcvp "* ]] && [ ! -f $R27/bindings/$SET.pkl ] && run learn 0
[[ " ${arms[*]} " == *" vsvp "* ]] && [ ! -f $R27/bindings/${SET}_v.pkl ] && run learnv 0
for r in $(seq 1 ${ROUNDS:-3}); do
  k=$(( (r - 1) % ${#arms[@]} )); rot=("${arms[@]:$k}" "${arms[@]:0:$k}")
  for a in "${rot[@]}"; do run $a $r; done
  echo "$(date +%T) $SET round $r done"
done
{ echo "# $SET on GB300 GPU $G (all arms on this GPU), candidate c5h (int6 snap/step324 + build_cpp12)"; python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing $SET; } | tee $R27/out/timing/TABLE_$SET.txt
