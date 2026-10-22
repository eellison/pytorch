# local self-test of b200_bundle on this GB300 box: our c5 python environment underneath the bundle's scripts
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=$P/land/int6/snap/step310 HOSTTRACE_INSTALL=$P/land/int6/build_cpp10/install HOSTTRACE_PREFIX=$P/land/int6/build_cpp10/install
export PYTHONPATH="$P/hosttrace_build/site_land:$P/land/scratch/vllm/site:$P/land/scratch/sglang/site"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export PATH="$P/land/scratch/vllm/site/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.0
export FI_SITE_PKG=$P/land/scratch/sglang/site/flashinfer WORK=$P/land/core/r27/bundle_selftest
export FLASHINFER_WORKSPACE_BASE=$P/land/core/r27/fi_ws HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODEL_27B=/data/eellison/models/Qwen3.8-27B-NVFP4
