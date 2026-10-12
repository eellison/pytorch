"""The flags of FlashInfer's own fmha_gen build (its JIT build.ninja), shared by the
traced extension (build.py), the DWARF probe (gen_fields.py) and the drift check (tests/test_drift.py)."""

import os
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _flashinfer_dir() -> str:
    # the installed FlashInfer package (not the overlay): FI_SITE_PKG, else found on sys.path without importing it
    if os.environ.get("FI_SITE_PKG"):
        return os.environ["FI_SITE_PKG"]
    import importlib.util

    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("set FI_SITE_PKG to the installed flashinfer package directory")
    return os.path.realpath(list(spec.submodule_search_locations)[0])


def _arch() -> str:
    # FlashInfer's JIT arch directory and nvcc arch: "100a" on B200, "103a" on GB300 (FI_ARCH overrides)
    if os.environ.get("FI_ARCH"):
        return os.environ["FI_ARCH"]
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}a"


def _tvm_ffi_include() -> str:
    if os.environ.get("TVM_FFI_INCLUDE"):
        return os.environ["TVM_FFI_INCLUDE"]
    import importlib.util

    spec = importlib.util.find_spec("tvm_ffi")
    return os.path.join(list(spec.submodule_search_locations)[0], "include")


SITE_DATA = os.path.join(os.path.realpath(os.path.join(_flashinfer_dir(), "data")))
# FlashInfer 0.6.18's trtllm-gen FMHA artifact directory (flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA) and the meta
# hash its fmha_gen JIT build passes (computed from that directory's checksums.txt; arch independent)
CUBIN_SUB = os.environ.get("FI_FMHA_CUBIN_SUB", "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/")
METAINFO_HASH = os.environ.get("FI_FMHA_METAINFO_HASH", "892b5a755dde041ec97eacd1ba2156643977cc8e34d882a108a7e45096c0fae4")
WORKSPACE = os.environ.get("FLASHINFER_WORKSPACE_BASE", os.path.expanduser("~"))
CUBIN_DIR = os.environ.get("FLASHINFER_CUBIN_DIR", os.path.join(WORKSPACE, ".cache/flashinfer/cubins"))
ARCH = _arch()
# FlashInfer's own fmha_gen JIT build (its build.ninja holds the flags below); the drift test compares against it
STOCK_BUILD = os.environ.get("FI_STOCK_FMHA_BUILD", os.path.join(WORKSPACE, f".cache/flashinfer/0.6.18/{ARCH}/cached_ops/fmha_gen"))
TVM_FFI_INCLUDE = _tvm_ffi_include()
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC = os.path.join(CUDA_HOME, "bin/nvcc")
PATCHED = os.path.join(ROOT, "src")
PY_INCLUDE = sysconfig.get_paths()["include"]


def include_flags(patched_first: bool = False) -> list[str]:
    inc = []
    if patched_first:
        inc += [f"-I{PATCHED}/include", f"-I{PATCHED}/csrc"]
    inc += [
        f"-I{CUBIN_DIR}/{CUBIN_SUB}include",
        f"-I{SITE_DATA}/cccl/cub",
        f"-I{SITE_DATA}/cccl/libcudacxx/include",
        f"-I{SITE_DATA}/cccl/thrust",
        "-isystem", PY_INCLUDE,
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
    f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
    "-std=c++17",
    "-use_fast_math",
    "-O3",
]


def stock_cuda_flags(patched_first: bool = False) -> list[str]:
    """fmha_gen's cuda_cflags (Py_LIMITED_API and the includes in its order)."""
    return [*DEFINES, *include_flags(patched_first), *CUDA_ONLY]
