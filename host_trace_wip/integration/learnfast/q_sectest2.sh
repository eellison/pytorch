#!/usr/bin/env bash
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
export LF_GPU=3 LF_CPUS=108-143 LF_INSTALL=$S/build_cpp_ms8/install TORCH_EXTENSIONS_DIR=$L/torch_ext_ms8 HOLD_S=600 HOSTTRACE_TORCH_DISABLE_CACHES=0
LF_WT=$L/sec_snap $L/gpulock.sh 3 3600 $L/py.sh $L/sec_snap/test/test_cuda_host_trace_replay.py -v -k section 2>&1 | grep -E "FAIL|Ran |^OK|^FAILED|Error:|AssertionError" | cut -c1-250
echo "== desync repro on step_ms10"
LF_WT=$S/snap/step_ms10 $L/gpulock.sh 3 3600 $L/py.sh $L/sec_snap/test/test_section_flag_desync.py -v 2>&1 | grep -E "before flip|test_|Ran |^OK|^FAILED|Error|line " | cut -c1-300
