#!/usr/bin/env bash
# the trtllm_cpp lane's trace tests (Qwen3-8B context/decode, R1 class, guard-region sweep) on c5 + r27/build_c5, stock vLLM cache
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
FI_HT_CPP_ROOT=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27 FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/cache \
  bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/python_r27.sh \
  /data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_cpp/tests/run_gpu_tests.py test_trtllm_cpp_trace.py
