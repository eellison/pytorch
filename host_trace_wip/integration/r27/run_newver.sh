#!/usr/bin/env bash
# FlashInfer GDN prefill NaN / run-to-run agreement with newer versions (isolated dirs under r27/fi_new): 8 fresh processes per combination
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
F=$R/fi_new
for combo in "fi071+dsl462:$F/fi071:" "fi0618+dsl480::$F/dsl480" "fi071+dsl480:$F/fi071:$F/dsl480"; do
  name=${combo%%:*}; rest=${combo#*:}; fidir=${rest%%:*}; dsldir=${rest#*:}
  for k in $(seq 1 8); do
    out=$(FI_DIR=$fidir FI_DSL_DIR=$dsldir timeout 900 bash $R/python_r27.sh $R/scripts/gdn_newver.py 2>&1)
    [ $k = 1 ] && echo "$name: $(echo "$out" | grep -m1 '^versions')" && echo "$out" | grep -E "Error|error" | head -3
    echo "$name proc $k: $(echo "$out" | grep -E '^1x|^4x' | awk '{split($4,a,"/"); printf "%s %s%s ", $1, (a[1]=="0" ? "ok" : "NaN"), ($9=="True"||$10=="True" ? "" : "!") }')"
  done
done
