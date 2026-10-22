#!/usr/bin/env bash
# arm V on the even int6 line: HOSTTRACE_WT=int6/snap/step178 on int6/build_cpp6/install (C++ entry default). Env from python_vllm_cpp.sh 2026-10-05.
#   CUDA_VISIBLE_DEVICES=1 bash python_vllm_int6_even.sh <script> [args]
set -euo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
G=$P/land/scratch/integration
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${INT6_WT:-$P/land/int6/snap/step178} HOSTTRACE_INSTALL=$P/land/int6/build_cpp6/install HOSTTRACE_PREFIX=$P/land/int6/build_cpp6/install
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:$S/site${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
# vllm _C references cublas without a DT_NEEDED entry; the pip torch preloads it globally, the source build does not
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$G/vllm/ext_even}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec python "$@"
