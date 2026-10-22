#!/usr/bin/env bash
# 3 rounds of round_27s_line.sh (decode only; diagnosis config gdn_prefill_backend=triton) on GPU 0, CPUs 0-71, one gpu0.lock
# hold per round, each run idle-gated by timing_lib's idle_check; then the table.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
for r in 1 2 3; do
  echo "$(date +%T) round $r waiting for gpu0.lock"
  until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done  # no other process on GPU 0
  flock -x $E/gpu0.lock env R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/symti/wt314 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/symti/install15 FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/build_c5s TORCH_EXTENSIONS_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/torch_ext_c5s R27_ARMS="${R27_ARMS:-default fullnc vstock vc vcvp}" R27_W_EXTRA="--async-decode 1 8 64" SET=n27s R=$r HT_GPU=0 R27_KV_GIB=40 taskset -c 0-71 bash $R27/round_27s_line.sh
  echo "$(date +%T) round $r done"
done
python3 $R27/b200_bundle/tools/timing_table.py $R27/out/timing n27s | tee $R27/out/timing/TABLE_n27s.txt
