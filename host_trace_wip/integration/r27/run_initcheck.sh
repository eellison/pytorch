#!/usr/bin/env bash
# compute-sanitizer initcheck on the stock FlashInfer GDN prefill (scripts/gdn_rep2.py), 2 fresh processes (the first attempt passed
# --track-unused-memory with a value it does not take, so the tool never ran)
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
CS=/usr/local/cuda-13.0/bin/compute-sanitizer
for k in 1 2; do
  timeout 2400 $CS --tool initcheck --print-limit 50 bash $R/python_r27.sh $R/scripts/gdn_rep2.py > $R/logs/fiev_initcheck_$k.log 2>&1
  echo "initcheck $k rc=$? uninitialized-read reports: $(grep -c 'Uninitialized __global__ memory read' $R/logs/fiev_initcheck_$k.log)"
  grep -E "^1x|^4x|ERROR SUMMARY" $R/logs/fiev_initcheck_$k.log | sed 's/digest -*[0-9]*//'
done
