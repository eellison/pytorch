#!/usr/bin/env bash
# FlashInfer GDN evidence (10-13): (1) compute-sanitizer initcheck on the stock GDN prefill (gdn_rep2.py), 2 fresh processes, and racecheck
# at 1x512; (2) NaN / run-to-run agreement over 8 fresh processes, stock vs fenced (flashinfer#5334); (3) the strict 27B check with the
# fenced kernel in eager and ours, both forwards, 2 processes each. Logs: logs/fiev_*.log; summary in out/ficheck/UNINIT_EVIDENCE.md.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
CS=/usr/local/cuda-13.0/bin/compute-sanitizer
for k in 1 2; do
  timeout 1800 $CS --tool initcheck --track-unused-memory no --print-limit 50 bash $R/python_r27.sh $R/scripts/gdn_rep2.py > $R/logs/fiev_initcheck_$k.log 2>&1
  echo "initcheck $k rc=$? errors: $(grep -c 'Uninitialized __global__ memory read' $R/logs/fiev_initcheck_$k.log)"; grep -E "^1x|^4x|ERROR SUMMARY" $R/logs/fiev_initcheck_$k.log
done
GDN_SHAPES=1x512 timeout 1800 $CS --tool racecheck --print-limit 50 bash $R/python_r27.sh $R/scripts/gdn_rep2.py > $R/logs/fiev_racecheck.log 2>&1
echo "racecheck rc=$?"; grep -E "^1x|RACECHECK SUMMARY|ERROR SUMMARY|hazard" $R/logs/fiev_racecheck.log | head -5
for fence in 0 1; do for k in $(seq 1 8); do
  echo "fence=$fence proc $k: $(FI_FENCE=$fence timeout 600 bash $R/python_r27.sh $R/scripts/gdn_rep2.py 2>&1 | grep -E '^1x|^4x' | tr '\n' ' ')"
done; done
