#!/usr/bin/env bash
# SGLang on the int6 candidate 2 cpp top (src_cpp = frozen step188) on build_cpp7
# (= python_sgl_int6_even.sh with the candidate 2 tree/install). Own extension dir and TMPDIR (fresh quack JIT cache, 97).
set -euo pipefail
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${CAND2_WT:-$I/candidate2/src_cpp} HOSTTRACE_INSTALL=$I/build_cpp7/install HOSTTRACE_PREFIX=$I/build_cpp7/install 
export PYTHONPATH="$P/hosttrace_build/site_land:$S/site:$S/src/sglang/python${INT6_EXTRA_PYTHONPATH:+:$INT6_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$X/ext_c2_even}
export TMPDIR=${C2_TMPDIR:-$X/tmp_c2_even}
mkdir -p "$TMPDIR"
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$S/cache}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$S/cache/tvm-ffi}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
exec python "$@"
