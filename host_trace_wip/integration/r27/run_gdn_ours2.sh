#!/usr/bin/env bash
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
for k in 1 2 3 4 5 6; do echo "== proc $k"; timeout 600 bash $R/python_r27.sh $R/scripts/gdn_ours2.py 2>&1 | grep -E "^1x|^4x|declines|Error" | head -8; done
