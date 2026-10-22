#!/usr/bin/env bash
# 3 rounds of round_27s_line.sh (decode only; diagnosis config gdn_prefill_backend=triton) on GPU 0, CPUs 0-71, one gpu0.lock
# hold per round, each run idle-gated by timing_lib's idle_check; then the table.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
for r in 1 2 3; do
  echo "$(date +%T) round $r waiting for gpu0.lock"
  flock -x $E/gpu0.lock env R27_ARMS="${R27_ARMS:-vc}" SET=n27t R=$r HT_GPU=0 R27_KV_GIB=40 taskset -c 0-71 bash $R27/round_27s_line.sh
  echo "$(date +%T) round $r done"
done
python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing | tee $R27/out/timing/TABLE_n27t_vc.txt
