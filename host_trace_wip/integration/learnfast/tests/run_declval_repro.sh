#!/usr/bin/env bash
# run_declval_repro.sh: the repro on step_ms19 / build_cpp_ms12a, core alone then with TPL_DECLVAL=1 (GPU 3, short hold)
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export LF_GPU=3 LF_CPUS=108-143 LF_WT=$S/snap/step_ms19 LF_INSTALL=$S/build_cpp_ms12a/install TORCH_EXTENSIONS_DIR=$L/torch_ext_ms19 HOLD_S=600
$L/gpulock.sh 3 43200 bash -c "$L/py.sh $L/tests/test_declval_repro.py -v; TPL_DECLVAL=1 $L/py.sh $L/tests/test_declval_repro.py -v"
