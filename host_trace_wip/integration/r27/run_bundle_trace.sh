source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/bundle_selftest_env.sh
export FI_HT_BUILD=$WORK/fi_ht_build FLASHINFER_NO_DOWNLOAD=1 FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/cache
B=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/b200_bundle
FI_HT_CPP_ROOT=$WORK/overlay bash $B/bin/python_27b.sh -u $B/trtllm_cpp/tests/test_trtllm_cpp_trace.py -v
