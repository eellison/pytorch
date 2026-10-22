#!/usr/bin/env bash
# The r27 slide sweeps (round_27w_line.sh) on GPU 0: ${ROUNDS:-3} rounds, arms rotated per round, one gpu0.lock hold per arm-run, each
# waiting for a GPU 0 with no compute process, idle-gated inside; then the table (timing_table.py, headline = the faster vLLM config overall).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
arms=(${R27_ARMS:-default fullnc vcvp})
for r in $(seq 1 ${ROUNDS:-3}); do
  k=$(( (r - 1) % ${#arms[@]} )); rot=("${arms[@]:$k}" "${arms[@]:0:$k}")
  for a in "${rot[@]}"; do
    echo "$(date +%T) round $r $a waiting for an empty GPU 0 and gpu0.lock"
    until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
    flock -x $E/gpu0.lock env R27_ARMS=$a SET=${SET:-n27w} R=$r HT_GPU=0 R27_KV_GIB=40 taskset -c 0-71 bash $R27/round_27w_line.sh
  done
  echo "$(date +%T) round $r done"
done
python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing ${SET:-n27w} | tee $R27/out/timing/TABLE_${SET:-n27w}.txt
