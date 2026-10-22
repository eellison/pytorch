#!/usr/bin/env bash
# python_vllm_cpp.sh with the trtllm fork (sglang/fork/VLLM_NOTES.md): $S/fork's flashinfer (traced trtllm-gen decode/context)
# ahead of $S/site, vLLM's own cubins, no download, a separate FlashInfer JIT / TVM-FFI cache. Pair with ARMV_TRTLLM_FORK=1.
# python_vllm_v.sh on install_sgl_cpp (C++ bound entry; read only) with the src_vllmcpp overlay (arm V-cpp).
# python_sgl1.sh with land/scratch/vllm/site (vLLM v0.29.0 source build + vLLM-only deps) ahead of sglang/site
# (flashinfer 0.6.18 and shared pure-Python deps, read only). Torch: install_land_int5b_frozen + src_sgl1 overlay.
#   CUDA_VISIBLE_DEVICES=2 bash land/scratch/vllm/python_vllm.sh <script> [args]
set -euo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=$V/src_vllmcpp HOSTTRACE_PREFIX=$P/hosttrace_build/install_sgl_cpp HOSTTRACE_INSTALL=$P/hosttrace_build/install_sgl_cpp
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:$S/fork:$S/site${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
# vllm _C references cublas without a DT_NEEDED entry; the pip torch preloads it globally, the source build does not
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$V/cache/ext_cpp}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache_fork}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache_fork/tvm-ffi}
export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-$V/cache/.cache/flashinfer/cubins} FLASHINFER_NO_DOWNLOAD=1
export MAX_JOBS=${MAX_JOBS:-16}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec python "$@"
