"""Offline memory planning for CUDA graphs (prototype).

:class:`PlannedGraph` captures a function twice:

1. Record: every tensor allocation of the capture goes to a recording allocator
   (a passthrough :class:`torch.cuda.MemPool`), which hands out distinct,
   never-reused addresses from reserved, unmapped virtual address space (captured
   kernels do not run) and logs the allocation and free sequence. This graph is
   discarded.
2. Plan: each allocation gets an offset in one arena such that allocations that
   may be in use on the GPU at the same time never overlap.
3. Place: the function is captured again; its k-th allocation is placed at
   ``arena + offset[k]``. Frees are no-ops, since the plan already accounts for
   them.

This module only uses public APIs (``MemPool.from_py_allocator(...,
passthrough=True)`` and :func:`torch.cuda.graph`): other planners can be written
the same way outside of PyTorch.

Sharing rule: allocation ``j`` may reuse the memory of allocation ``i`` if ``i``
was freed before ``j`` was allocated (in host order) and both were allocated on
the same stream, so stream order separates all uses of ``i`` from all uses of
``j``. As in eager mode, a tensor used on another stream must not be freed
before that stream's work has been joined back.
"""

from __future__ import annotations

import ctypes
import dataclasses
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


__all__: list[str] = []


# cudaMalloc's alignment guarantee, so placed tensors keep every alignment property
# (e.g. what cuBLASLt picks kernels by) that code can assume of a fresh allocation
_ALIGNMENT = 256


class _AddressSpace:
    """Bump allocator over reserved, unmapped device virtual address space."""

    _CHUNK = 1 << 40

    def __init__(self) -> None:
        from torch.cuda._utils import _get_cuda_library

        if torch.version.hip:
            raise RuntimeError("PlannedGraph is not supported on ROCm yet")
        self._lib = _get_cuda_library()
        self._lib.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_ulonglong,
        ]
        self._lib.cuMemAddressFree.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        self._reserved: list[tuple[int, int]] = []
        self._top = 0
        self._end = 0

    def alloc(self, size: int) -> int:
        from torch.cuda._utils import _check_cuda

        size = (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        if self._top + size > self._end:
            nbytes = max(self._CHUNK, size)
            base = ctypes.c_uint64(0)
            _check_cuda(
                self._lib.cuMemAddressReserve(ctypes.byref(base), nbytes, 0, 0, 0)
            )
            self._reserved.append((base.value, nbytes))
            self._top, self._end = base.value, base.value + nbytes
        ptr = self._top
        self._top += size
        return ptr

    def release(self) -> None:
        for base, nbytes in self._reserved:
            self._lib.cuMemAddressFree(base, nbytes)
        self._reserved.clear()
        self._top = self._end = 0


@dataclasses.dataclass
class _Allocation:
    size: int
    stream: int
    alloc_time: int
    free_time: int | None = None  # None: live at the end of the capture
    offset: int = -1


class _Recorder:
    def __init__(self) -> None:
        self.address_space = _AddressSpace()
        self.allocations: list[_Allocation] = []
        self._live: dict[int, int] = {}
        self._time = 0

    def alloc(self, size: int, device: int, stream: int) -> int:
        ptr = self.address_space.alloc(size)
        self._live[ptr] = len(self.allocations)
        self.allocations.append(_Allocation(size, stream, self._time))
        self._time += 1
        return ptr

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        index = self._live.pop(ptr, None)
        if index is not None:
            self.allocations[index].free_time = self._time
            self._time += 1


class _Placer:
    def __init__(self, allocations: list[_Allocation], base: int) -> None:
        self.allocations = allocations
        self.base = base
        self.count = 0

    def alloc(self, size: int, device: int, stream: int) -> int:
        if (
            self.count >= len(self.allocations)
            or self.allocations[self.count].size != size
        ):
            raise RuntimeError(
                "PlannedGraph: the second capture made a different allocation "
                f"than the first (allocation {self.count}, size {size}); the "
                "captured function must allocate deterministically"
            )
        ptr = self.base + self.allocations[self.count].offset
        self.count += 1
        return ptr

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        pass


def _may_overlap(a: _Allocation, b: _Allocation) -> bool:
    """Whether a and b may be in use on the GPU at the same time."""
    if a.stream != b.stream:
        return True
    first, second = (a, b) if a.alloc_time < b.alloc_time else (b, a)
    return first.free_time is None or first.free_time > second.alloc_time


def _plan(allocations: list[_Allocation], alignment: int = _ALIGNMENT) -> int:
    """Assigns offsets, largest first, each at the lowest offset that does not
    overlap an already placed allocation it may be live with. Returns the arena
    size."""
    placed: list[_Allocation] = []
    for a in sorted(allocations, key=lambda a: (-a.size, a.alloc_time)):
        busy = sorted(
            (b.offset, b.offset + b.size) for b in placed if _may_overlap(a, b)
        )
        offset = 0
        for lo, hi in busy:
            if offset + a.size <= lo:
                break
            offset = max(offset, (hi + alignment - 1) // alignment * alignment)
        a.offset = offset
        placed.append(a)
    return max((a.offset + a.size for a in allocations), default=0)


def _peak_live_bytes(allocations: list[_Allocation]) -> int:
    """Largest total size live at once in host order: a lower bound on the arena
    size when every allocation is on one stream."""
    deltas = []
    for a in allocations:
        deltas.append((a.alloc_time, a.size))
        if a.free_time is not None:
            deltas.append((a.free_time, -a.size))
    live = peak = 0
    for _, delta in sorted(deltas):
        live += delta
        peak = max(peak, live)
    return peak


class PlannedGraph:
    r"""A CUDA graph of ``fn`` whose temporary memory is planned offline.

    Warm up ``fn`` eagerly (as for :class:`torch.cuda.CUDAGraph`) before calling
    :meth:`capture`. ``fn`` must allocate deterministically, and must drop
    references to tensors from a previous call when called again (for example,
    ``optimizer.zero_grad(set_to_none=True)``), since it is captured twice.

    Example::

        >>> # xdoctest: +SKIP("requires CUDA")
        >>> g = torch.cuda._planned_graph.PlannedGraph(train_step)
        >>> g.capture()
        >>> for _ in range(steps):
        ...     loss = g.replay()

    Attributes:
        arena_bytes: size of the planned arena.
        peak_live_bytes: largest total size of the capture's allocations live at
            once (a lower bound for single-stream captures).
        num_allocations: number of allocations made by one capture of ``fn``.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: Any = None
        self.arena: torch.Tensor | None = None
        self.arena_bytes = 0
        self.peak_live_bytes = 0
        self.num_allocations = 0
        self._pool: torch.cuda.MemPool | None = None

    def capture(self, *args: Any, **kwargs: Any) -> Any:
        r"""Captures ``fn(*args, **kwargs)`` and returns its outputs, which
        :meth:`replay` updates in place."""
        recorder = _Recorder()
        try:
            record_pool = torch.cuda.MemPool.from_py_allocator(
                recorder.alloc, recorder.free, passthrough=True
            )
            record_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(record_graph, pool=record_pool.id):
                outputs = self.fn(*args, **kwargs)
            del outputs, record_graph, record_pool
            torch.cuda.synchronize()
        finally:
            recorder.address_space.release()

        allocations = recorder.allocations
        self.num_allocations = len(allocations)
        self.peak_live_bytes = _peak_live_bytes(allocations)
        self.arena_bytes = _plan(allocations)
        self.arena = torch.empty(
            max(self.arena_bytes, 1), dtype=torch.uint8, device="cuda"
        )

        placer = _Placer(allocations, self.arena.data_ptr())
        self._pool = torch.cuda.MemPool.from_py_allocator(
            placer.alloc, placer.free, passthrough=True
        )
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self._pool.id):
            self.outputs = self.fn(*args, **kwargs)
        if placer.count != len(allocations):
            raise RuntimeError(
                "PlannedGraph: the second capture made fewer allocations "
                f"({placer.count}) than the first ({len(allocations)})"
            )
        return self.outputs

    def replay(self) -> Any:
        r"""Replays the graph and returns the outputs of :meth:`capture`."""
        if self.graph is None:
            raise RuntimeError("PlannedGraph.replay() called before capture()")
        self.graph.replay()
        return self.outputs
