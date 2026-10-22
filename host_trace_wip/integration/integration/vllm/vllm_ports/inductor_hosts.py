# The Inductor seams for a host trace around a compiled forward (arm Vc), one implementation for the vLLM lanes (serve, r27).
# Inductor's generated wrappers call C++ helpers from torch._C._dynamo.guards (assert_size_stride[_grouped], assert_alignment,
# copy_if_misaligned, _empty_strided_cuda/_cpu, _reinterpret_tensor) and launch their Triton kernels through CachingAutotuner's own
# launcher, which the core's JITFunction hook never sees. install() routes both to the core's Inductor host-trace seams
# (torch._inductor.cudagraph_host_trace: tape allocations, alignment as a guard, _InductorKernel recording the tuned launcher's
# launch, an untuned autotuner a retried decline), except the size/stride asserts, which the core skips and which become guards
# here (each size equal to its expected value; each stride where the size is not 1), and assert_alignment, a guard on the
# traced storage offset. Outside a trace every seam calls the original. Installed before the engine compiles (or loads)
# anything: the wrappers bind the helpers at module load.
# (The serve lane's first version reimplemented the launch recording; replaced 10-12 by the core's, as the r27 lane uses.)
from __future__ import annotations

from typing import Any

import torch


def _check(tr: Any, t: Any, size: Any, stride: Any, what: Any) -> None:
    # the C++ check_size_stride: ranks equal; no elements passes; a size, or where the size exceeds 1 a stride, mismatch fails
    if t.dim() != len(size):
        raise tr.decline(f"Inductor assert_size_stride {what}: rank {t.dim()} vs {len(size)}")
    if bool(t.numel() == 0):
        return
    for i, (n, st) in enumerate(zip(size, stride)):
        if not bool(t.size(i) == n):
            raise tr.decline(f"Inductor assert_size_stride {what}: size {i}")
        if bool(t.size(i) > 1) and not bool(t.stride(i) == st):
            raise tr.decline(f"Inductor assert_size_stride {what}: stride {i}")


def _assert_ss(tr: Any, t: Any, size: Any, stride: Any, what: Any = "", *rest: Any) -> None:
    _check(tr, t, size, stride, what)


def _assert_ss_grouped(tr: Any, ts: Any, sizes: Any, strides: Any, what: Any = "", *rest: Any) -> None:
    for t, size, stride in zip(ts, sizes, strides):
        _check(tr, t, size, stride, what)


def _assert_alignment(tr: Any, t: Any, alignment: int, what: Any = "", *rest: Any) -> None:
    if not bool(t.storage_offset() * t.element_size() % alignment == 0):
        raise tr.decline(f"Inductor assert_alignment {what}: not {alignment}-byte aligned")


def install() -> list[str]:
    import torch._C._dynamo.guards as G
    import torch._inductor.cudagraph_host_trace as cht
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch.cuda._host_trace_tape import current_trace

    done = []
    orig_run = CachingAutotuner.run
    if not getattr(orig_run, "_armv_seam", False):
        def _run(self: Any, *args: Any, stream: Any, **kwargs: Any) -> Any:
            if current_trace() is None or type(self) is not CachingAutotuner:
                return orig_run(self, *args, stream=stream, **kwargs)
            return cht._InductorKernel(self).run(*args, stream=stream, **kwargs)
        _run._armv_seam = True  # type: ignore[attr-defined]
        CachingAutotuner.run = _run  # type: ignore[method-assign]
        done.append("CachingAutotuner.run")
    seams = dict(cht._GUARDS_SEAMS, assert_size_stride=_assert_ss, assert_size_stride_grouped=_assert_ss_grouped, assert_alignment=_assert_alignment)
    for name, traced in seams.items():
        orig = getattr(G, name, None)
        if orig is not None and not getattr(orig, "_armv_seam", False):
            f = cht._traced_or(orig, traced)
            f._armv_seam = True
            setattr(G, name, f)
            done.append(name)
    return done
