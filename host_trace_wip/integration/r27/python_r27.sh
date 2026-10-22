#!/usr/bin/env bash
# python_r27.sh <script> [args]: CAND_LINE=c5 (int6 snap/step310 on build_cpp10/install: mm_fp4 stack 97b_c3 + R1_ii + 99)
# with the TRT-LLM C++ launcher overlay (land/core/trtllm_cpp/overlay) ahead of the FlashInfer site, the traced extension
# built against build_cpp10 (r27/build_c5), and the serve lane's observed FlashInfer workspace (serve/fi_ws_ht: mm_fp4's
# CuTe DSL compiled under observation, the R1 trtllm-gen cubins). Adapter: r27/armV_r27 (INTEG_ARMV).
set -euo pipefail
P=/data/eellison/src/pytorch/agent_space/paramgraph
G=$P/land/scratch/integration
T=$P/land/core/trtllm_cpp
R=$P/land/core/r27
V=$P/land/scratch/vllm
S=$P/land/scratch/sglang
I=$P/land/int6
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${R27_WT:-$I/snap/step310} HOSTTRACE_INSTALL=${R27_INSTALL:-$I/build_cpp10/install} HOSTTRACE_PREFIX=${R27_INSTALL:-$I/build_cpp10/install}
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:${R27_OVERLAY:-$R/overlay}:$S/site${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$R/torch_ext}
export FI_HT_BUILD=${FI_HT_BUILD:-$R/build_c5}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$R/fi_ws} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
export FLASHINFER_NO_DOWNLOAD=1 MAX_JOBS=${MAX_JOBS:-16}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export INTEG_ARMV=${INTEG_ARMV:-$R/armV_r27}
exec python "$@"
