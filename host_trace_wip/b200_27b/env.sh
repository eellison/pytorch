# Source this (bin/* scripts do). Every value can be overridden from the environment.
export BUNDLE=${BUNDLE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export WORK=${WORK:-$BUNDLE/work}                                  # caches, builds, logs, results
export MODEL_27B=${MODEL_27B:-Inferact/Qwen3.8-27B-NVFP4}         # HF id or a local snapshot directory
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$WORK/fi_ws}   # FlashInfer JIT modules + cubins
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$WORK/tvm-ffi}
export FI_HT_BUILD=${FI_HT_BUILD:-$WORK/fi_ht_build}              # the traced trtllm-gen extension (bin/build_ext.sh)
export FI_HT_CPP_ROOT=${FI_HT_CPP_ROOT:-$BUNDLE/trtllm_cpp}
export OVERLAY=${OVERLAY:-$WORK/overlay}                          # bin/make_overlay.sh
export VLLM_PORTS_DIR=${VLLM_PORTS_DIR:-$BUNDLE/vllm_ports} SGL_OPS_TRACE_DIR=${SGL_OPS_TRACE_DIR:-$BUNDLE/vllm_ports}
export INTEG_ARMV=${INTEG_ARMV:-$BUNDLE/armV_r27}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$WORK/vllm_cache} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$WORK/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$WORK/inductor} TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$WORK/torch_ext}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$WORK"
