#!/usr/bin/env bash
# python_sgl1.sh with land/scratch/vllm/site (vLLM v0.29.0 source build + vLLM-only deps) ahead of sglang/site
# (flashinfer 0.6.18 and shared pure-Python deps, read only). Torch: install_land_int5b_frozen + src_sgl1 overlay.
#   CUDA_VISIBLE_DEVICES=2 bash land/scratch/vllm/python_vllm.sh <script> [args]
set -euo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=$S/src_sgl1 HOSTTRACE_INSTALL=$P/hosttrace_build/install_land_int5b_frozen
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:$S/site${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
# vllm _C references cublas without a DT_NEEDED entry; the pip torch preloads it globally, the source build does not
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$V/cache/ext}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec python "$@"
