"""The flags of FlashInfer's own fmha_gen build (its JIT build.ninja under the vLLM lane's cache), shared by the
traced extension (build.py), the DWARF probe (gen_fields.py) and the drift check (tests/test_drift.py)."""

import os
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAND = os.path.dirname(os.path.dirname(ROOT))
SITE_DATA = os.path.join(LAND, "scratch/sglang/site/flashinfer/data")
CUBIN_SUB = "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/"
CUBIN_DIR = os.path.join(LAND, "scratch/vllm/cache/.cache/flashinfer/cubins")
METAINFO_HASH = "892b5a755dde041ec97eacd1ba2156643977cc8e34d882a108a7e45096c0fae4"
STOCK_BUILD = os.path.join(LAND, "scratch/vllm/cache/.cache/flashinfer/0.6.18/103a/cached_ops/fmha_gen")
TVM_FFI_INCLUDE = "/data/eellison/envs/pytorch-dev/lib/python3.12/site-packages/tvm_ffi/include"
CUDA_HOME = "/usr/local/cuda-13.0"
NVCC = os.path.join(CUDA_HOME, "bin/nvcc")
PATCHED = os.path.join(ROOT, "src")


def include_flags(patched_first: bool = False) -> list[str]:
    inc = []
    if patched_first:
        inc += [f"-I{PATCHED}/include", f"-I{PATCHED}/csrc"]
    inc += [
        f"-I{CUBIN_DIR}/{CUBIN_SUB}include",
        f"-I{SITE_DATA}/cccl/cub",
        f"-I{SITE_DATA}/cccl/libcudacxx/include",
        f"-I{SITE_DATA}/cccl/thrust",
        "-isystem", sysconfig.get_paths()["include"],
        "-isystem", f"{CUDA_HOME}/include",
        "-isystem", TVM_FFI_INCLUDE,
        "-isystem", f"{SITE_DATA}/include",
        "-isystem", f"{SITE_DATA}/csrc",
        "-isystem", f"{SITE_DATA}/cutlass/include",
        "-isystem", f"{SITE_DATA}/cutlass/tools/util/include",
        "-isystem", f"{SITE_DATA}/spdlog/include",
    ]
    if patched_first:
        # the patched headers' relative includes of unpatched ones ("../../exception.h") resolve to the stock files
        inc.append(f"-I{SITE_DATA}/include/flashinfer/trtllm/fmha")
    return inc


DEFINES = [
    "-DPy_LIMITED_API=0x03090000",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
    "-DNDEBUG",
    f'-DTLLM_GEN_FMHA_CUBIN_PATH="{CUBIN_SUB}"',
    f'-DTLLM_GEN_FMHA_METAINFO_HASH="{METAINFO_HASH}"',
]

CUDA_ONLY = [
    "--compiler-options=-fPIC",
    "--expt-relaxed-constexpr",
    "-static-global-template-stub=false",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-std=c++17",
    "-use_fast_math",
    "-O3",
]


def stock_cuda_flags(patched_first: bool = False) -> list[str]:
    """fmha_gen's cuda_cflags (Py_LIMITED_API and the includes in its order)."""
    return [*DEFINES, *include_flags(patched_first), *CUDA_ONLY]
