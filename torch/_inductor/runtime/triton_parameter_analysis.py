"""Selected user Triton record types and host compilation controls."""

from __future__ import annotations

import os
from dataclasses import dataclass


ANALYSIS_VERSION = 2


@dataclass(frozen=True)
class PointerFormal:
    name: str
    ttir_index: int
    required_alignment: int
    read: bool
    written: bool


@dataclass(frozen=True)
class KernelAnalysis:
    compiled_hash: str
    ttir_sha256: str
    supported: bool
    reason: str
    pointer_formals: tuple[PointerFormal, ...]
    operations: tuple[str, ...]
    analyzer_version: int


@dataclass(frozen=True)
class SelectedUserKernelFacts:
    source: str
    signature: tuple
    constants: tuple
    attributes: tuple
    arguments: tuple
    cubin_sha256: str
    analysis: KernelAnalysis


def user_kernel_controls_supported() -> bool:
    from triton import knobs
    from triton.backends.nvidia.compiler import CUDABackend

    for name in ("launch_enter_hook", "launch_exit_hook", "kernel_load_start_hook",
                 "kernel_load_end_hook", "kernel_unload_hook", "jit_cache_hook",
                 "jit_post_compile_hook", "add_stages_inspection_hook"):
        hook = getattr(knobs.runtime, name)
        if hook is not None and not (isinstance(hook, knobs.HookChain) and not hook.calls):
            return False
    return not (knobs.runtime.interpret or knobs.runtime.override_arch
                or knobs.compilation.override or knobs.compilation.instrumentation_mode
                or knobs.compilation.listener is not None or knobs.compilation.store_binary_only
                or CUDABackend.instrumentation is not None
                or any(os.environ.get(name) for name in ("LLVM_PASS_PLUGIN_PATH", "TRITON_PLUGIN_PATHS")))
