#!/usr/bin/env bash
# timing_gpuN.sh ROUND_SCRIPT SET GPU CPUS: ${ROUNDS:-3} rounds of an r27 round script (round_27w_line.sh = sweeps, round_27cb_line.sh =
# churn) with every arm of the comparison on the SAME GPU (user rule 10-13), arms rotated per round, one gpuN.lock hold per arm-run, each
# waiting for that GPU to have no compute process and idle-gated inside (timing_lib idle_check on HT_GPU); then the table.
RS=$1 SET=$2 G=$3 CPUS=$4
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
arms=(${R27_ARMS:-default fullnc vcvp})
if [ ! -f $R27/bindings/$SET.pkl ]; then  # the offline cuBLAS learning run (untimed), once, before any timed vcvp run
  echo "$(date +%T) $SET learn waiting for an empty GPU $G and gpu$G.lock"
  until [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  flock -x $E/gpu$G.lock env R27_ARMS=learn SET=$SET R=0 HT_GPU=$G R27_KV_GIB=${R27_KV_GIB:-40} taskset -c $CPUS bash $R27/$RS
fi
for r in $(seq 1 ${ROUNDS:-3}); do
  k=$(( (r - 1) % ${#arms[@]} )); rot=("${arms[@]:$k}" "${arms[@]:0:$k}")
  for a in "${rot[@]}"; do
    echo "$(date +%T) $SET round $r $a waiting for an empty GPU $G and gpu$G.lock"
    until [ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
    flock -x $E/gpu$G.lock env R27_ARMS=$a SET=$SET R=$r HT_GPU=$G R27_KV_GIB=${R27_KV_GIB:-40} taskset -c $CPUS bash $R27/$RS
  done
  echo "$(date +%T) $SET round $r done"
done
{ echo "# $SET on GB300 GPU $G (all arms on this GPU), candidate c5h (int6 snap/step324 + build_cpp12)"; python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing $SET; } | tee $R27/out/timing/TABLE_$SET.txt
