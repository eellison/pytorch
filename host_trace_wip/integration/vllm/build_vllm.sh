#!/usr/bin/env bash
# vLLM v0.29.0 (src/vllm) against install_land_int5b_frozen torch, into land/scratch/vllm/site (--no-deps).
#   setsid nohup bash land/scratch/vllm/build_vllm.sh > land/scratch/vllm/logs/build.log 2>&1 &
set -uo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
H=/data/eellison/src/pytorch/agent_space/paramgraph/hosttrace_build
cd $V/src/vllm
mkdir -p $V/build/tmp $V/build/deps
export TMPDIR=$V/build/tmp
export CCACHE_DISABLE=1 SCCACHE_DISABLE=1
export CC=/opt/rh/gcc-toolset-13/root/usr/bin/cc CXX=/opt/rh/gcc-toolset-13/root/usr/bin/c++ CUDAHOSTCXX=/opt/rh/gcc-toolset-13/root/usr/bin/c++
export CMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
# 10.0 -> vLLM's CUDA 13 path builds the Blackwell kernels as 10.0f (family: sm_100 and sm_103), generic ones as sm_100
export TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=16 CMAKE_BUILD_TYPE=Release
export FETCHCONTENT_BASE_DIR=$V/build/deps
export CMAKE_ARGS="-DTorch_DIR=$H/install_land_int5b_frozen/torch/share/cmake/Torch"
export CARGO_HOME=$V/build/cargo RUSTUP_HOME=$V/build/rustup
export PIP_CACHE_DIR=$V/cache/pip
echo "build start $(date -Is)"
CUDA_VISIBLE_DEVICES= taskset -c 72-107 nice -n 10 bash $V/python_vllm.sh -m pip install --target $V/site --no-deps --no-build-isolation -v .
rc=$?
echo "build end $(date -Is) rc $rc"
