"""Disk cache for CuTe DSL compiled kernel artifacts.

Persists compiled GPU binaries (CUBIN/PTX via TVM FFI) across process
boundaries so that subprocess compilation workers can produce artifacts
that the parent process loads without recompilation.

Cache keys include the module's PyCodeCache path (content-addressed),
compile-time configuration, runtime shapes/dtypes, GPU architecture,
and CUDA toolkit version to prevent stale artifact reuse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import struct
from pathlib import Path
from types import ModuleType
from typing import Any


log = logging.getLogger(__name__)

# the parent's cudagraph_host_trace, for compile workers
HOST_TRACE_ENV = "TORCHINDUCTOR_HOST_TRACE_CUTE_WORKER"


def host_trace_cute() -> ModuleType | None:
    """torch.cuda._host_trace_cute, armed, under Inductor's cudagraph_host_trace
    (in a compile worker, the parent's). Host tracing describes a CuTe function
    by its compile's host program, which export_to_c then writes beside the object."""
    from torch._inductor import config

    if not (config.triton.cudagraph_host_trace or os.environ.get(HOST_TRACE_ENV) == "1"):
        return None
    import cutlass  # noqa: F401  install() observes a loaded cutlass

    from torch.cuda import _host_trace_cute

    _host_trace_cute.install()
    return _host_trace_cute


def _fix_elf_dup_text_flags(data: bytes) -> bytes:
    """Harmonize flags on duplicate .text ELF sections.

    The CUTLASS MLIR compiler emits .o files with two .text sections:
    one for code (ALLOC|EXECINSTR) and one for writable data
    (WRITE|ALLOC). LLVM's MCJIT mishandles this in multi-process
    workloads, causing non-deterministic SIGSEGV. Fix: set duplicate
    .text sections' flags to match the first (ALLOC|EXECINSTR = 0x6).

    See https://github.com/NVIDIA/cutlass/issues/3162
    """
    if len(data) < 64 or data[4] != 2 or data[5] != 1:
        return data
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]
    if not e_shoff or not e_shnum or e_shstrndx >= e_shnum:
        return data
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from("<Q", data, shstr_hdr + 24)[0]
    text_secs: list[tuple[int, int]] = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        ni = struct.unpack_from("<I", data, sh)[0]
        ns = shstr_off + ni
        if ns + 6 <= len(data) and data[ns : ns + 6] == b".text\x00":
            text_secs.append((i, sh))
    if len(text_secs) <= 1:
        return data
    r = bytearray(data)
    for _, sh in text_secs[1:]:
        struct.pack_into("<Q", r, sh + 8, 0x6)
    return bytes(r)


def _cache_dir() -> Path:
    cache_root = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch_inductor"),
    )
    return Path(cache_root) / "cutedsl_compile_cache"


def _make_disk_key(
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> str:
    import torch

    arch = device_capability
    if arch is None and torch.cuda.is_available():
        arch = torch.cuda.get_device_capability(device_index)
    cuda_version = getattr(torch.version, "cuda", None)

    full_key = (module_path, config_key, runtime_key, arch, cuda_version)
    return hashlib.sha256(pickle.dumps(full_key)).hexdigest()


def disk_cache_get(
    mem_cache: dict[Any, Any],
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> Any | None:
    """Look up a compiled CuTe DSL function, checking memory then disk."""
    mem_key = (runtime_key, device_index)
    if mem_key in mem_cache:
        return mem_cache[mem_key]

    h = _make_disk_key(
        module_path, config_key, runtime_key, device_index, device_capability
    )
    obj_path = _cache_dir() / f"{h}.o"
    htc = host_trace_cute()
    # under host trace, an object exported without its host function is a
    # miss: the recompile exports both
    if obj_path.exists() and (htc is None or Path(f"{obj_path}{htc.SIDECAR}").exists()):
        try:
            raw = obj_path.read_bytes()
            patched = _fix_elf_dup_text_flags(raw)
            if patched is not raw:
                obj_path.write_bytes(patched)

            import cutlass.cute as cute

            m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
            fn = m.func
            mem_cache[mem_key] = fn
            return fn
        except Exception:
            log.debug("Failed to load cached artifact %s", obj_path, exc_info=True)
    return None


def disk_cache_set(
    mem_cache: dict[Any, Any],
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    compiled_fn: Any,
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> None:
    """Store a compiled CuTe DSL function to memory and disk."""
    mem_key = (runtime_key, device_index)
    mem_cache[mem_key] = compiled_fn

    h = _make_disk_key(
        module_path, config_key, runtime_key, device_index, device_capability
    )
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    obj_path = d / f"{h}.o"
    import tempfile

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(d), suffix=".o.tmp")
        os.close(fd)
        compiled_fn.export_to_c(object_file_path=tmp_path, function_name="func")
        with open(tmp_path, "rb") as f:
            patched = _fix_elf_dup_text_flags(f.read())
        with open(tmp_path, "wb") as f:
            f.write(patched)
        from torch.cuda._host_trace_cute import SIDECAR

        # host trace's export writes the host function beside the object
        if os.path.exists(tmp_path + SIDECAR):
            os.replace(tmp_path + SIDECAR, f"{obj_path}{SIDECAR}")
        os.replace(tmp_path, str(obj_path))
    except (AttributeError, RuntimeError, TypeError):
        log.debug(
            "export_to_c not available for %s, skipping disk persistence",
            type(compiled_fn).__name__,
            exc_info=True,
        )
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
