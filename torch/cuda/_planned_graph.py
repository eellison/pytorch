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
was freed before ``j`` was allocated (in host order) and every captured node
that may use ``i`` happens before every captured node that may use ``j``. The
recorder notes the capture frontier (``cuStreamGetCaptureInfo``) of the
allocating stream at each allocation, and at each free the frontiers of the
allocation's stream and of every stream recorded for it with
:meth:`torch.Tensor.record_stream`. ``i`` and ``j`` may then share memory if
each node of ``i``'s free frontier is an ancestor (or equal to) a node of
``j``'s allocation frontier, since all work issued after ``j``'s allocation
depends on that frontier. This allows sharing across streams along
synchronization edges. As in eager mode, a tensor used on another stream must
either be recorded with ``record_stream`` or not be freed before that stream's
work has been joined back.
"""

from __future__ import annotations

import ctypes
import dataclasses
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


__all__: list[str] = []


class _Driver:
    """The few CUDA driver calls the planner needs, through ctypes."""

    def __init__(self) -> None:
        from torch.cuda._utils import _get_cuda_library

        if torch.version.hip:
            raise RuntimeError("PlannedGraph is not supported on ROCm yet")
        lib = _get_cuda_library()
        u64 = ctypes.c_uint64
        lib.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(u64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            u64,
            ctypes.c_ulonglong,
        ]
        lib.cuMemAddressFree.argtypes = [u64, ctypes.c_size_t]
        self.lib = lib
        capture_info_v3 = getattr(lib, "cuStreamGetCaptureInfo_v3", None)
        self._capture_info_v3 = capture_info_v3 is not None
        self._capture_info = capture_info_v3 or lib.cuStreamGetCaptureInfo_v2
        edges_v2 = getattr(lib, "cuGraphGetEdges_v2", None)
        self._edges_v2 = edges_v2 is not None
        self._edges = edges_v2 or lib.cuGraphGetEdges

    def capture_frontier(self, stream: int) -> list[int] | None:
        """The nodes that work issued next on ``stream`` would depend on, or
        None if ``stream`` is not being captured."""
        status = ctypes.c_int(0)
        capture_id = ctypes.c_uint64(0)
        graph = ctypes.c_void_p()
        deps = ctypes.POINTER(ctypes.c_void_p)()
        n = ctypes.c_size_t(0)
        args = [
            ctypes.c_void_p(stream),
            ctypes.byref(status),
            ctypes.byref(capture_id),
            ctypes.byref(graph),
            ctypes.byref(deps),
        ]
        if self._capture_info_v3:
            # edge data must be requested, or edges that carry it make the
            # query fail as lossy
            args.append(ctypes.byref(ctypes.c_void_p()))
        args.append(ctypes.byref(n))
        rc = self._capture_info(*args)
        if rc != 0:
            from torch.cuda._utils import _check_cuda

            _check_cuda(rc)
        if status.value != 1:  # CU_STREAM_CAPTURE_STATUS_ACTIVE
            return None
        return [deps[i] or 0 for i in range(n.value)]

    def graph_topology(self, graph: int) -> tuple[list[int], list[tuple[int, int]]]:
        from torch.cuda._utils import _check_cuda

        g = ctypes.c_void_p(graph)
        n = ctypes.c_size_t(0)
        _check_cuda(self.lib.cuGraphGetNodes(g, None, ctypes.byref(n)))
        nodes = (ctypes.c_void_p * max(n.value, 1))()
        _check_cuda(self.lib.cuGraphGetNodes(g, nodes, ctypes.byref(n)))
        m = ctypes.c_size_t(0)
        extra: list[Any] = [None] if self._edges_v2 else []
        _check_cuda(self._edges(g, None, None, *extra, ctypes.byref(m)))
        src = (ctypes.c_void_p * max(m.value, 1))()
        dst = (ctypes.c_void_p * max(m.value, 1))()
        if self._edges_v2:
            # CUgraphEdgeData is 8 bytes; edges that carry data (e.g.
            # programmatic dependencies) make the query fail without it
            extra = [(ctypes.c_uint64 * max(m.value, 1))()]
        if m.value:  # (non-null arrays with a count of 0 are rejected)
            _check_cuda(self._edges(g, src, dst, *extra, ctypes.byref(m)))
        return (
            [nodes[i] or 0 for i in range(n.value)],
            [(src[i] or 0, dst[i] or 0) for i in range(m.value)],
        )


def _ancestors_or_self(
    nodes: list[int], edges: list[tuple[int, int]]
) -> dict[int, int]:
    """For each node, a bitset (int over node indices) of itself and its
    ancestors."""
    index = {h: i for i, h in enumerate(nodes)}
    parents: list[list[int]] = [[] for _ in nodes]
    children: list[list[int]] = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for a, b in edges:
        parents[index[b]].append(index[a])
        children[index[a]].append(index[b])
        indegree[index[b]] += 1
    ready = [i for i, d in enumerate(indegree) if d == 0]
    masks = [0] * len(nodes)
    while ready:  # Kahn: a node is visited after all of its parents
        i = ready.pop()
        m = 1 << i
        for p in parents[i]:
            m |= masks[p]
        masks[i] = m
        for c in children[i]:
            indegree[c] -= 1
            if indegree[c] == 0:
                ready.append(c)
    return {h: masks[i] for h, i in index.items()}


# cudaMalloc's alignment guarantee, so placed tensors keep every alignment property
# (e.g. what cuBLASLt picks kernels by) that code can assume of a fresh allocation
_ALIGNMENT = 256


class _AddressSpace:
    """Bump allocator over reserved, unmapped device virtual address space."""

    _CHUNK = 1 << 40

    def __init__(self, driver: _Driver) -> None:
        self._lib = driver.lib
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

    def release(self, retire: bool) -> None:
        """Frees the reservations, or with ``retire`` keeps them reserved so
        that their addresses are never handed out again."""
        for base, nbytes in self._reserved:
            if retire:
                _retired_ranges.append((base, nbytes))
            else:
                self._lib.cuMemAddressFree(base, nbytes)
        self._reserved.clear()
        self._top = self._end = 0


# Address ranges that stay reserved for the life of the process, so that no
# later allocation can reuse them (see PlannedGraph(vmm_arena=True)).
_retired_ranges: list[tuple[int, int]] = []


class _VMMArena:
    """Physical memory mapped at its own reserved virtual address range."""

    class _Location(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class _AllocFlags(ctypes.Structure):
        _fields_ = [
            ("compressionType", ctypes.c_ubyte),
            ("gpuDirectRDMACapable", ctypes.c_ubyte),
            ("usage", ctypes.c_ushort),
            ("reserved", ctypes.c_ubyte * 4),
        ]

    class _Prop(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("requestedHandleTypes", ctypes.c_int),
            ("location", ctypes.c_byte * 8),  # _Location
            ("win32HandleMetaData", ctypes.c_void_p),
            ("allocFlags", ctypes.c_byte * 8),  # _AllocFlags
        ]

    class _AccessDesc(ctypes.Structure):
        _fields_ = [("location", ctypes.c_byte * 8), ("flags", ctypes.c_int)]

    def __init__(
        self, driver: _Driver, nbytes: int, device: int, handle_types: int
    ) -> None:
        from torch.cuda._utils import _check_cuda

        lib = driver.lib
        self._lib = lib
        location = self._Location(1, device)  # CU_MEM_LOCATION_TYPE_DEVICE
        prop = self._Prop()
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = handle_types
        ctypes.memmove(prop.location, ctypes.byref(location), 8)
        granularity = ctypes.c_size_t(0)
        _check_cuda(
            lib.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                1,  # RECOMMENDED
            )
        )
        g = granularity.value
        self.size = max((nbytes + g - 1) // g * g, g)
        self.handle = ctypes.c_uint64(0)
        _check_cuda(
            lib.cuMemCreate(
                ctypes.byref(self.handle),
                ctypes.c_size_t(self.size),
                ctypes.byref(prop),
                ctypes.c_ulonglong(0),
            )
        )
        base = ctypes.c_uint64(0)
        _check_cuda(lib.cuMemAddressReserve(ctypes.byref(base), self.size, g, 0, 0))
        self.ptr = base.value
        _check_cuda(
            lib.cuMemMap(
                ctypes.c_uint64(self.ptr),
                ctypes.c_size_t(self.size),
                ctypes.c_size_t(0),
                self.handle,
                ctypes.c_ulonglong(0),
            )
        )
        access = self._AccessDesc()
        ctypes.memmove(access.location, ctypes.byref(location), 8)
        access.flags = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check_cuda(
            lib.cuMemSetAccess(
                ctypes.c_uint64(self.ptr),
                ctypes.c_size_t(self.size),
                ctypes.byref(access),
                ctypes.c_size_t(1),
            )
        )

    def release(self) -> None:
        """Unmaps and frees the memory; the address range stays reserved."""
        from torch.cuda._utils import _check_cuda

        if self.ptr:
            _check_cuda(
                self._lib.cuMemUnmap(
                    ctypes.c_uint64(self.ptr), ctypes.c_size_t(self.size)
                )
            )
            _check_cuda(self._lib.cuMemRelease(self.handle))
            _retired_ranges.append((self.ptr, self.size))
            self.ptr = 0


@dataclasses.dataclass
class _Allocation:
    size: int
    stream: int
    alloc_time: int
    free_time: int | None = None  # None: live at the end of the capture
    # capture frontiers (graph node handles); None if a stream was not captured
    alloc_frontier: list[int] | None = None
    free_frontier: list[int] | None = None
    used_streams: set[int] = dataclasses.field(default_factory=set)
    # bitsets over graph nodes, filled in from the topology after capture
    alloc_closure: int = 0
    free_mask: int | None = None
    offset: int = -1


class _Recorder:
    def __init__(self, driver: _Driver) -> None:
        self.driver = driver
        self.address_space = _AddressSpace(driver)
        self.allocations: list[_Allocation] = []
        self._live: dict[int, int] = {}
        self._time = 0

    def alloc(self, size: int, device: int, stream: int) -> int:
        ptr = self.address_space.alloc(size)
        self._live[ptr] = len(self.allocations)
        self.allocations.append(
            _Allocation(
                size,
                stream,
                self._time,
                alloc_frontier=self.driver.capture_frontier(stream),
            )
        )
        self._time += 1
        return ptr

    def record_stream(self, ptr: int, stream: int) -> None:
        index = self._live.get(ptr)
        if index is not None:
            self.allocations[index].used_streams.add(stream)

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        index = self._live.pop(ptr, None)
        if index is None:
            return
        a = self.allocations[index]
        a.free_time = self._time
        self._time += 1
        frontier: list[int] = []
        for s in {a.stream} | a.used_streams:
            f = self.driver.capture_frontier(s)
            if f is None:  # freed outside of the capture
                return
            frontier.extend(f)
        a.free_frontier = frontier

    def resolve(self, nodes: list[int], edges: list[tuple[int, int]]) -> None:
        """Turns the recorded frontiers into bitsets over the captured graph."""
        closure = _ancestors_or_self(nodes, edges)
        index = {h: i for i, h in enumerate(nodes)}
        for a in self.allocations:
            for h in a.alloc_frontier or ():
                a.alloc_closure |= closure.get(h, 0)
            if (
                a.free_time is not None
                and a.free_frontier is not None
                and all(h in index for h in a.free_frontier)
            ):
                a.free_mask = 0
                for h in a.free_frontier:
                    a.free_mask |= 1 << index[h]


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
    first, second = (a, b) if a.alloc_time < b.alloc_time else (b, a)
    if first.free_time is None or first.free_time > second.alloc_time:
        return True
    if first.free_mask is None:  # no frontier information: same stream only
        return first.stream != second.stream
    # every node of first's free frontier must precede second's allocation
    return (first.free_mask & ~second.alloc_closure) != 0


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

    With ``vmm_arena=True``, the arena is physical memory mapped at its own
    virtual address range (exportable with ``handle_types``, a
    ``CUmemAllocationHandleType`` mask, POSIX file descriptors by default), and
    neither its addresses nor those used by the recording capture are ever
    reused after :meth:`release` or a new :meth:`capture`. Use it when
    collectives in ``fn`` register their buffers by address (for example NCCL
    with ``NCCL_GRAPH_REGISTER``), so that a stale registration can never
    refer to new memory.

    Attributes:
        arena_bytes: size of the planned arena.
        peak_live_bytes: largest total size of the capture's allocations live at
            once (a lower bound for single-stream captures).
        num_allocations: number of allocations made by one capture of ``fn``.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        vmm_arena: bool = False,
        handle_types: int = 1,
    ) -> None:
        self.fn = fn
        self.vmm_arena = vmm_arena
        self.handle_types = handle_types
        self._vmm: _VMMArena | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: Any = None
        self.arena: torch.Tensor | None = None
        self.arena_bytes = 0
        self.peak_live_bytes = 0
        self.num_allocations = 0
        self._pool: torch.cuda.MemPool | None = None

    def capture(self, *args: Any, **kwargs: Any) -> Any:
        r"""Captures ``fn(*args, **kwargs)`` and returns its outputs, which
        :meth:`replay` updates in place. Capturing again re-plans: the previous
        graph and arena are released first."""
        if self.graph is not None:
            self.release()
        driver = _Driver()
        recorder = _Recorder(driver)
        try:
            record_pool = torch.cuda.MemPool.from_py_allocator(
                recorder.alloc,
                recorder.free,
                record_stream_fn=recorder.record_stream,
                passthrough=True,
            )
            record_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(record_graph, pool=record_pool):
                outputs = self.fn(*args, **kwargs)
            recorder.resolve(
                *recorder.driver.graph_topology(record_graph.raw_cuda_graph())
            )
            del outputs, record_graph, record_pool
            torch.cuda.synchronize()
        finally:
            recorder.address_space.release(retire=self.vmm_arena)

        allocations = recorder.allocations
        self.num_allocations = len(allocations)
        self.peak_live_bytes = _peak_live_bytes(allocations)
        self.arena_bytes = _plan(allocations)
        if self.vmm_arena:
            self._vmm = _VMMArena(
                driver,
                self.arena_bytes,
                torch.cuda.current_device(),
                self.handle_types,
            )
            base = self._vmm.ptr
        else:
            self.arena = torch.empty(
                max(self.arena_bytes, 1), dtype=torch.uint8, device="cuda"
            )
            base = self.arena.data_ptr()

        placer = _Placer(allocations, base)
        self._pool = torch.cuda.MemPool.from_py_allocator(
            placer.alloc, placer.free, passthrough=True
        )
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self._pool):
            self.outputs = self.fn(*args, **kwargs)
        if placer.count != len(allocations):
            raise RuntimeError(
                "PlannedGraph: the second capture made fewer allocations "
                f"({placer.count}) than the first ({len(allocations)})"
            )
        return self.outputs

    def release(self) -> None:
        r"""Destroys the graph and frees the arena. The outputs of
        :meth:`capture` must not be used afterwards."""
        self.graph = None
        self.outputs = None
        self._pool = None
        torch.cuda.synchronize()
        if self._vmm is not None:
            self._vmm.release()
            self._vmm = None
        self.arena = None

    def replay(self) -> Any:
        r"""Replays the graph and returns the outputs of :meth:`capture`."""
        if self.graph is None:
            raise RuntimeError("PlannedGraph.replay() called before capture()")
        self.graph.replay()
        return self.outputs
