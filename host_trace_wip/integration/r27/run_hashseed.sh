#!/usr/bin/env bash
# Is the per-process NaN set of FlashInfer's GDN prefill a function of Python's hash seed (code generation order in the per-process CuTe DSL
# compile)? gdn_rep2.py (stock) with PYTHONHASHSEED 0-5, each seed in two fresh processes.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
for s in 0 1 2 3 4 5; do for k in a b; do
  echo "hashseed $s run $k: $(PYTHONHASHSEED=$s timeout 600 bash $R/python_r27.sh $R/scripts/gdn_rep2.py 2>&1 | grep -E '^1x|^4x' | sed 's/digest -*[0-9]*//; s/ runs-agree True//' | awk '{split($4,a,"/"); printf "%s %s ", $1, (a[1]=="0" ? "ok" : "NaN")}' )"
done; done
