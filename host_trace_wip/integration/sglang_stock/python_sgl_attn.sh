#!/usr/bin/env bash
# SGLang (land/scratch/sglang/src/sglang cc012ab, stock) on the current core: the vLLM lane's attn line
# (integration/attnline/py on land/core/pinned/c8x/install) with the FlashInfer trtllm-gen fork attnline/fork (F1) ahead of
# sglang/site, the fork's own JIT caches and the shared cubins, no downloads. SGL_LINE=pinned: land/core/pinned/py instead.
#   CUDA_VISIBLE_DEVICES=1 bash python_sgl_attn.sh <script> [args]
set -euo pipefail
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
# SGL_LINE=rd2cpp: the redispatch2 lane's candidate-4 tree with RD2 sections (land/core/redispatch2/c7_e = int6 step272 +
# RD2) on its build (build_rd2/install), attention through FlashInfer's own trtllm-gen C++ launcher traced
# (land/core/trtllm_cpp: overlay ahead of the site, extension build_rd2/fi_ht_trtllm.so built against that install;
# the adapter imports flashinfer.trtllm_cpp with INTEG_ATTN=cpp). No fork on that line.
INST=$P/land/core/pinned/c8x/install
FORK_DIR=$G/attnline/fork; [ "${SGL_FORK:-1}" = 1 ] || FORK_DIR=""
case ${SGL_LINE:-attn} in
  attn) WT=$G/attnline/py ;;
  pinned) WT=$P/land/core/pinned/py ;;
  rd2cpp) WT=${RD2_WT:-$P/land/core/redispatch2/c7_e}; INST=$P/land/core/redispatch2/build_rd2/install
          FORK_DIR=$P/land/core/trtllm_cpp/overlay; export FI_HT_BUILD=$P/land/core/trtllm_cpp/build_rd2 INTEG_ATTN=${INTEG_ATTN:-cpp} ;;
  *) echo "SGL_LINE=$SGL_LINE" >&2; exit 2 ;;
esac
export HOSTTRACE_WT=$WT HOSTTRACE_INSTALL=$INST HOSTTRACE_PREFIX=$INST
export PYTHONPATH="$P/hosttrace_build/site_land${FORK_DIR:+:$FORK_DIR}:$S/site:$S/src/sglang/python${SGL_EXTRA_PYTHONPATH:+:$SGL_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$G/sglang_stock/ext_${SGL_LINE:-attn}}
export TMPDIR=${SGL_TMPDIR:-$G/sglang_stock/tmp}
if [ "${SGL_LINE:-attn}" = rd2cpp ]; then  # the stock FlashInfer modules (the overlay's jit/ is the site's)
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache/tvm-ffi}
  export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-$S/cache/.cache/flashinfer/cubins}
elif [ -n "$FORK_DIR" ]; then
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache_fork} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache_fork/tvm-ffi}
  export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-$S/cache/.cache/flashinfer/cubins}
else
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache/tvm-ffi}
fi
export FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
exec python "$@"
