#!/usr/bin/env bash
# py.sh <script> [args]: c5h (int6 snap/step324 Python on build_cpp12/install), read-only; GPU 3, CPUs 108-143.
set -euo pipefail
P=/data/eellison/src/pytorch/agent_space/paramgraph
I=$P/land/int6
V=$P/land/scratch/vllm
S=$P/land/scratch/sglang
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
export HOSTTRACE_WT=${LF_WT:-$I/snap/step324} HOSTTRACE_INSTALL=${LF_INSTALL:-$I/build_cpp12/install} HOSTTRACE_PREFIX=${LF_INSTALL:-$I/build_cpp12/install}
export PYTHONPATH="$P/hosttrace_build/site_land:$V/site:$S/site${LF_EXTRA_PYTHONPATH:+:$LF_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export CUDA_VISIBLE_DEVICES=${LF_GPU:-2}
export PYTHONDONTWRITEBYTECODE=1
exec taskset -c ${LF_CPUS:-36-71} python "$@"
