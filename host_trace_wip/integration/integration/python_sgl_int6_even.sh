#!/usr/bin/env bash
# SGLang on the int6 even line: HOSTTRACE_WT=land/int6/snap/step178 (C++ line top; cpp_entry default ON) on
# land/int6/build_cpp6/install (= int6/python_int6_build6.sh with the step178 tree and an extensions dir of our own).
#   CUDA_VISIBLE_DEVICES=1 bash land/scratch/integration/python_sgl_int6_even.sh <script> [args]
# INT6_WT=<dir> overrides the tree (e.g. a scratch overlay under land/scratch/integration/).
set -euo pipefail
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${INT6_WT:-$I/snap/step178} HOSTTRACE_INSTALL=$I/build_cpp6/install HOSTTRACE_PREFIX=$I/build_cpp6/install
export PYTHONPATH="$P/hosttrace_build/site_land:$S/site:$S/src/sglang/python${INT6_EXTRA_PYTHONPATH:+:$INT6_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$X/ext_even}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache/tvm-ffi}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
exec python "$@"
