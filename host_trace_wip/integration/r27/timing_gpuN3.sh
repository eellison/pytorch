#!/usr/bin/env bash
# timing_gpuN3.sh ROUND_SCRIPT SET GPU CPUS: timing_gpuN2.sh with LANES.md's wrapper rules (10-14): the GPU lock is taken on an fd and
# flock's exit status checked (the job never starts without the lock; abort if flock fails or is killed), and inside the hold the job
# waits for the GPU's used memory to be < 20 GiB. Offline learns for our arms first, optional cold-cache startup runs (COLD=1), then
# ${ROUNDS:-3} rounds, every arm of the comparison on the SAME GPU, arms rotated per round, one hold per arm-run; then the table.
RS=$1 SET=$2 G=$3 CPUS=$4
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
arms=(${R27_ARMS:-default fullnc vcvp})
run() {  # arm round
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
