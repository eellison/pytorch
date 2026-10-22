#!/usr/bin/env bash
# Vc feasibility: the compiled forward under the entry raised "Cannot call numel() on tensor with symbolic sizes" through
# the C++ entry; the Python line (no C++ entry) gives the Python stack.
source $(dirname $0)/timing_lib.sh
trun c2_Vc_trap2 CAND_LINE=cpp ARMV_HT_SET=torch.cuda._host_trace.cpp_entry=0 ARMV_BOUND=0 HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm Vc --check --batch-size 1 --prefill-lens 64 --mixed 0 --out vllm/out/c2_Vc_trap2.json
echo ALLDONE
