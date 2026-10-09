"""Builds the traced extension land/core/trtllm_cpp/build/fi_ht_trtllm.so: ht_traced.cu (the patched launcher with
FI_HT_TRACED, nvcc with fmha_gen's flags) and ht_bind.cpp (its pybind module), linked against the torch install.

    python ht/build.py [--only traced|bind]   (rebuilds what is older than its sources)
"""

import argparse
import glob
import os
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_flags as bf  # noqa: E402

# FI_HT_BUILD: another build directory (an extension linked against another torch install, e.g. redispatch2's)
OUT = os.environ.get("FI_HT_BUILD", os.path.join(bf.ROOT, "build"))
SO = os.path.join(OUT, "fi_ht_trtllm.so")


def torch_root() -> str:
    """The torch C++ install (headers, libs): the pinned build the launcher runs on."""
    return os.path.join(os.environ.get("HOSTTRACE_INSTALL", os.path.join(bf.LAND, "core/pinned/c8x/install")), "torch")


def torch_includes() -> list[str]:
    t = torch_root()
    return ["-isystem", f"{t}/include", "-isystem", f"{t}/include/torch/csrc/api/include"]


def newer(target: str, sources: list[str]) -> bool:
    if not os.path.exists(target):
        return True
    t = os.path.getmtime(target)
    return any(os.path.getmtime(s) > t for s in sources)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd[:3]), "...", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["traced", "bind"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    patched = glob.glob(f"{bf.PATCHED}/**/*.*", recursive=True)
    shim = [os.path.join(HERE, f) for f in ("fi_ht.h", "ht_types.h", "ht_trace.h", "ht_entry.h", "kernel_params_fields.h")]
    traced_o = os.path.join(OUT, "ht_traced.o")
    bind_o = os.path.join(OUT, "ht_bind.o")
    if a.only != "bind" and newer(traced_o, [os.path.join(HERE, "ht_traced.cu"), *patched, *shim]):
        # torch's headers need C++20; the stock code here only defines the types the traced code shares
        flags = [{"-std=c++17": "-std=c++20"}.get(f, f) for f in bf.stock_cuda_flags() if f != "-DPy_LIMITED_API=0x03090000"]
        run([bf.NVCC, "-c", os.path.join(HERE, "ht_traced.cu"), "-o", traced_o, f"-I{HERE}", *flags, *torch_includes(),
             # the patched headers' relative includes of unpatched ones ("../../exception.h") resolve to the stock files
             f"-I{bf.SITE_DATA}/include/flashinfer/trtllm/fmha",
             "-Xcompiler", "-Wno-class-memaccess", "--diag-suppress=20011,20012,20014"])
    if a.only != "traced" and newer(bind_o, [os.path.join(HERE, "ht_bind.cpp"), *shim]):
        py = sysconfig.get_paths()["include"]
        run(["c++", "-c", os.path.join(HERE, "ht_bind.cpp"), "-o", bind_o, "-fPIC", "-O2", "-std=c++20",
             "-D_GLIBCXX_USE_CXX11_ABI=1", "-DTORCH_EXTENSION_NAME=fi_ht_trtllm", f"-I{HERE}", "-isystem", py,
             "-isystem", bf.TVM_FFI_INCLUDE, "-isystem", f"{bf.CUDA_HOME}/include", *torch_includes()])
    if a.only is None and newer(SO, [traced_o, bind_o]):
        t = torch_root()
        tvm = os.path.join(os.path.dirname(bf.TVM_FFI_INCLUDE), "lib")
        # linked aside and renamed: a process importing the extension meanwhile sees the old file or the new one
        run(["c++", "-shared", traced_o, bind_o, "-o", SO + ".tmp", f"-L{t}/lib", "-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu",
             "-ltorch_python", f"-L{bf.CUDA_HOME}/lib64", f"-L{bf.CUDA_HOME}/lib64/stubs", "-lcudart", "-lcuda",
             f"-L{tvm}", "-ltvm_ffi", f"-Wl,-rpath,{t}/lib", f"-Wl,-rpath,{tvm}"])
        os.replace(SO + ".tmp", SO)
    print(SO)


if __name__ == "__main__":
    main()
