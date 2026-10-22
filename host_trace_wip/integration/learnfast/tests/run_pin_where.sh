#!/usr/bin/env bash
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export LF_GPU=3 LF_CPUS=108-143 LF_WT=${WT:-$S/snap/step_ms11} LF_INSTALL=$S/build_cpp_ms8/install TORCH_EXTENSIONS_DIR=$L/torch_ext_ms8 HOLD_S=600
export PROV_PIN_ATEN=1 DIAG27_FILE=$L/out/pin_where.diag.jsonl
$L/gpulock.sh 3 3600 $L/py.sh $L/diag27.py $L/tests/test_pin_where.py -v
