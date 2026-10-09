#!/usr/bin/env bash
# arm V on int6 step288 (latest even top: FZ12 + FZ12b + 98m + R1_i) on build_cpp9 (read only; no scratch tree). CAND_LINE=cpp (default): candidate2/src_cpp on int6/build_cpp7/install (C++ entry);
# CAND_LINE=py: candidate2/src_py on install_land_int5b_frozen. CAND_FORK=1 puts the FlashInfer trtllm fork ($S/fork) ahead
# of $S/site with its own caches, as scratch/vllm/python_vllm_cpp_fork.sh (the open variant: ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1).
# INTEG_ARMV=vllm/armV_cand2 (the scratch adapter copy) for the drivers in vllm/bench.
#   CUDA_VISIBLE_DEVICES=1 CAND_LINE=cpp bash python_vllm_cand2.sh <script> [args]
set -euo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
I=$P/land/int6
G=$P/land/scratch/integration
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
LINE=${CAND_LINE:-cpp} FORK=${CAND_FORK:-0}
case $LINE in
  cpp) export HOSTTRACE_WT=$I/snap/step288 HOSTTRACE_INSTALL=$I/build_cpp9/install HOSTTRACE_PREFIX=$I/build_cpp9/install ;;
  py) export HOSTTRACE_WT=$I/candidate2/src_py HOSTTRACE_INSTALL=$P/hosttrace_build/install_land_int5b_frozen HOSTTRACE_PREFIX=$P/hosttrace_build/install_land_int5b_frozen ;;
  *) echo "CAND_LINE=$LINE" >&2; exit 2 ;;
esac
SITE="$V/site:$S/site"
[ "$FORK" = 1 ] && SITE="$V/site:$S/fork:$S/site"
export PYTHONPATH="$P/hosttrace_build/site_land:$SITE${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
# vllm _C references cublas without a DT_NEEDED entry; the pip torch preloads it globally, the source build does not
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$G/vllm/../serve/ext_s288_$LINE}
if [ "$FORK" = 1 ]; then
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache_fork} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache_fork/tvm-ffi}
  export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-$V/cache/.cache/flashinfer/cubins} FLASHINFER_NO_DOWNLOAD=1 MAX_JOBS=${MAX_JOBS:-16}
else
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
fi
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export INTEG_ARMV=${INTEG_ARMV:-$G/vllm/armV_cand2}
exec python "$@"
