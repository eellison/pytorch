#!/usr/bin/env bash
# python_fn.sh <script> [args]: the Flash-Next lane's line. Torch: int6 snap/step331 on build_cpp14/install (candidate 5 + D1
# + D-M1; FN_STEP / FN_INSTALL override), read only. PYTHONPATH as the MiMo lane assembles it: vLLM 0.29 site, the
# trtllm_moe_cpp lane's FlashInfer overlay (attention + fused-MoE C++ launcher hooks, read only; FN_MOE_OVERLAY overrides),
# SGLang site (FlashInfer 0.6.18), the deepgemm_cpp lane's overlay (top-level deep_gemm). Extensions read only: attention
# deepgemm_cpp/ext/trtllm_attn_d1, MoE trtllm_moe_cpp/build_d1 (FN_MOE_BUILD overrides), DeepGEMM deepgemm_cpp/build_d1.
# Writable state in this lane only: FlashInfer workspace fi_ws/ (copied from trtllm_moe_cpp/fi_ws 10-13), vLLM / Triton /
# Inductor / torch-extension caches under cache/; cubin dir cubins/ (symlink tree of trtllm_moe_cpp/cubins 10-13, new fetches land here only).
# FLASHINFER_NO_DOWNLOAD=1 unless FN_ALLOW_DOWNLOAD=1 (downloads recorded in STATUS.md).
# vLLM defaults: VLLM_USE_DEEP_GEMM is left to vLLM unless set by the caller.
set -euo pipefail
P=/data/eellison/src/pytorch/agent_space/paramgraph
L=$P/land
F=$L/core/flashnext
M=$L/core/trtllm_moe_cpp
DG=$L/core/deepgemm_cpp
V=$L/scratch/vllm
S=$L/scratch/sglang
I=$L/int6
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${FN_WT:-$I/snap/${FN_STEP:-step331}} HOSTTRACE_INSTALL=${FN_INSTALL:-$I/build_cpp14/install}
export HOSTTRACE_PREFIX=$HOSTTRACE_INSTALL
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:${FN_MOE_OVERLAY:-$M/overlay}:$S/site:$DG/overlay${FN_EXTRA_PYTHONPATH:+:$FN_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$F/cache/torch_ext}
export FI_HT_BUILD=${FI_HT_BUILD:-$DG/ext/trtllm_attn_d1} FI_HT_CPP_ROOT=$L/core/trtllm_cpp
# the traced MoE extension the overlay loads (flashinfer/trtllm_moe_cpp.py ext() reads FI_HTM_MOE_BUILD and checks its fi_htm.version
# marker): build_fix (check_routing_logits compared by sym_size, 10-15). Until 10-21 this line exported FI_HT_MOE_BUILD, which nothing
# reads, so our arm loaded the overlay's default build/ (the 10-13 pre-fix build). FN_MOE_BUILD overrides.
export FI_HTM_MOE_BUILD=${FN_MOE_BUILD:-$M/build_fix} FI_HT_MOE_BUILD=${FN_MOE_BUILD:-$M/build_fix}
export DG_HT_BUILD=$DG/build_d1 DG_JIT_CACHE_DIR=$F/cache/deep_gemm
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$F/fi_ws} FLASHINFER_CUBIN_DIR=${FN_CUBIN_DIR:-$F/cubins}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
if [ "${FN_ALLOW_DOWNLOAD:-0}" = 1 ]; then unset FLASHINFER_NO_DOWNLOAD; else export FLASHINFER_NO_DOWNLOAD=1; fi
export MAX_JOBS=${MAX_JOBS:-8} FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-10.3a}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$F/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$F/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$F/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export INTEG_ARMV=${INTEG_ARMV:-$F/armV}
exec python "$@"
