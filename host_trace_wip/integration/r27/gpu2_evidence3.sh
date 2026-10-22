#!/usr/bin/env bash
# sentinel-filled output test (stock and PR #5335), 3 fresh processes each, after gpu2_evidence2
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
E=../../scratch/eager
while pgrep -f gpu2_evidence2.sh > /dev/null; do sleep 30; done
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$PWD/torch_ext_c12
for fence in 0 1; do for k in 1 2 3; do
  echo "== outfill fence=$fence proc $k" >> logs/fi_evidence.out
  flock -x $E/gpu2.lock env CUDA_VISIBLE_DEVICES=2 FI_FENCE=$fence taskset -c 36-71 timeout 600 bash python_r27.sh scripts/gdn_outfill.py 2>&1 | grep -E "^1x|^4x" >> logs/fi_evidence.out
done; done
echo "gpu2_evidence3 done $(date +%T)" >> logs/fi_evidence.out
