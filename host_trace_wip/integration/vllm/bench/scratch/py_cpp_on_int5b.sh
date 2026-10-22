#!/usr/bin/env bash
# python_int5b.sh with HOSTTRACE_WT=src_sglpre (arm F prefill-harvest overlay (copy of src_sgl1) on install_land_int5b_frozen).
#   CUDA_VISIBLE_DEVICES=1 bash land/scratch/sglang/python_sgl1.sh <script> [args]
set -euo pipefail
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/src_vllmcpp HOSTTRACE_INSTALL=$P/hosttrace_build/install_land_int5b_frozen
export PYTHONPATH="$P/hosttrace_build/site_land:$S/site:$S/src/sglang/python"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$S/armF/ext}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache/tvm-ffi}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
exec python "$@"
