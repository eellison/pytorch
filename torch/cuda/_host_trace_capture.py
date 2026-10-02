"""Host tracing (private): a lowered tape captured once into CUDA graphs
at the traced call, one per run of launches, and verified byte for byte.

The capture does not run the host again. It launches the tape's kernels
itself, over the caller's addresses or buffers it allocates, each with the
parameter bytes the compiled program yields at the traced call
(launch_images, the same bytes a replay patches). First the compiled program at the traced call must agree
with the trace: every row equals the value the program computed while it was
built, and every launch slot, launch dimension and allocation extent equals its
value in the trace (for a pointer, its displacement from its root, since the
trace's addresses are placeholders), as must every eager call's argument and
predicted output rows. Then each launch must add exactly one kernel node to
its run's capturing stream, each graph must hold only its run's nodes, and
each node must hold its launch's function, grid, block, shared bytes and
parameter bytes. Anything else declines.

No eager call runs at the capture. An eager output's address, known only
once its op runs at a replay, is a placeholder under a non-canonical top
(like the trace's), so a node the replay left unpatched faults instead of
reading stale memory; a run reading an eager output is verified with that
placeholder, the same bytes launch_images yields for it, and the replay
patches the slot with the op's real address before the run's graph goes.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import heapq
import logging
import struct
import warnings
from dataclasses import dataclass, replace
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch._logging._internal import warning_once
from torch.cuda._host_trace import Declined, declined
from torch.cuda._host_trace_lower_tape import (
    LoweredMemcpy,
    LoweredMemset,
    LoweredView,
    PredictedOutput,
    ScalarSlot,
)
from torch.cuda._host_trace_tape import (
    _ALLOC_ALIGNMENT,
    _ALLOC_SHIFT,
    _EAGER_TAG,
    _gc_hold,
    _hint,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.cuda._host_trace_launch import KernelLaunch
    from torch.cuda._host_trace_lower_tape import (
        LoweredLaunch,
        LoweredTape,
        PointerSlot,
    )
    from torch.cuda._host_trace_opaque import OpaqueBinding


log = logging.getLogger(__name__)
# the test suites raise a trace value that disagrees with the live one
raise_trace_disagreements = False

_POINTER = struct.Struct("<Q")
_POINTERS = {4: struct.Struct("<I"), 8: _POINTER}  # a 4-byte one is the low half
_SCALARS = {1: struct.Struct("<b"), 2: struct.Struct("<h"), 4: struct.Struct("<i"), 8: struct.Struct("<q")}


@dataclass(frozen=True)
class CapturedLaunch:
    node: int  # its kernel, memset or memcpy node in the graph, a raw CUgraphNode handle
    launch: LoweredLaunch | LoweredMemset | LoweredMemcpy


@dataclass(frozen=True)
class CapturedSegment:
    graph: torch.cuda.CUDAGraph  # instantiated and uploaded; keeps its graph
    launches: tuple[CapturedLaunch, ...]
    # an RNG segment's generator and the addresses of its capture's seed and
    # offset, which its RNG kernels read
    rng: tuple[torch.Generator, int, int] | None


@dataclass(frozen=True)
class CapturedTape:
    lowered: LoweredTape
    segments: tuple[CapturedSegment, ...]  # one per run of launches, in order

    @property
    def launches(self) -> tuple[CapturedLaunch, ...]:
        return tuple(c for s in self.segments for c in s.launches)


class SegmentFailed(Declined):
    """A segment's capture, verify or instantiation failed: at the launches of
    tape sequence numbers `seqs` (the one it failed at, else all of the
    segment's); the ops that made them can run eagerly instead."""

    def __init__(self, msg: str, seqs: tuple[int, ...]) -> None:
        super().__init__(msg)
        self.seqs = seqs


def launch_slots(
    launch: LoweredLaunch, values: Sequence[int], bases: Sequence[int]
) -> tuple[int, ...]:
    """Each parameter slot's value at a call: `values` are the compiled
    program's rows there, `bases` each allocation's then each eager output's
    base address."""
    return tuple(
        values[slot.row] if isinstance(slot, ScalarSlot) else slot_address(slot, values, bases)
        for slot in launch.slots
    )


def pack_params(
    launch: KernelLaunch, slots: Sequence[int], pointers: Sequence[bool]
) -> list[bytearray]:
    """The parameters of a launch with `fields`: each field's slot packed at
    its offset, then each descriptor encoded from its slots."""
    packed = [bytearray(image) for image in launch.images] or [bytearray(n) for _, n in launch.layout]
    for v, pointer, (param, at, width) in zip(slots, pointers, launch.fields):  # type: ignore[arg-type]
        if pointer:
            _POINTERS[width].pack_into(packed[param], at, v & (1 << 8 * width) - 1)
        else:
            _SCALARS[width].pack_into(packed[param], at, v)
    for d in launch.descriptors:
        packed[d.param][:] = d.encode(slots)
    return packed


def pack_cpu_scalars(launch: KernelLaunch, images: Sequence[bytearray]) -> None:
    """Packs the launch's CPU scalar members, read from their tensors now."""
    for param, at, cls, source in launch.cpu_scalars:
        value = torch._C._cuda_hostTraceCpuScalarBytes(source, cls)
        images[param][at : at + len(value)] = value


def launch_images(launch: LoweredLaunch, slots: Sequence[int]) -> tuple[bytes, ...]:
    """Each parameter's bytes at a call, from launch_slots."""
    t = launch.launch
    if t.fields is not None:
        pointers = [not isinstance(slot, ScalarSlot) for slot in launch.slots]
        packed = pack_params(t, slots, pointers)
        pack_cpu_scalars(t, packed)
        return tuple(map(bytes, packed))
    return tuple(
        (_SCALARS[size] if isinstance(slot, ScalarSlot) else _POINTER).pack(v)
        for v, slot, (_, size) in zip(slots, launch.slots, t.layout)
    )


# per launch attribute read or launched by id: its CUlaunchAttributeValue
# field, the field's members if a struct, the enum its value converts to
_ATTRS = {
    "CLUSTER_DIMENSION": ("clusterDim", ("x", "y", "z"), None),
    "PREFERRED_CLUSTER_DIMENSION": ("preferredClusterDim", ("x", "y", "z"), None),
    "CLUSTER_SCHEDULING_POLICY_PREFERENCE": ("clusterSchedulingPolicyPreference", None, "CUclusterSchedulingPolicy"),
    "COOPERATIVE": ("cooperative", None, None),
    "PRIORITY": ("priority", None, None),
    "MEM_SYNC_DOMAIN": ("memSyncDomain", None, "CUlaunchMemSyncDomain"),
    "MEM_SYNC_DOMAIN_MAP": ("memSyncDomainMap", ("default_", "remote"), None),
    "PREFERRED_SHARED_MEMORY_CARVEOUT": ("sharedMemCarveout", None, None),
    "PROGRAMMATIC_STREAM_SERIALIZATION": ("programmaticStreamSerializationAllowed", None, None),
}


def launch_attributes(attributes: Sequence[tuple[Any, Any]]) -> list[Any]:
    """(CUlaunchAttributeID, value) pairs as cuLaunchKernelEx's attributes; a
    raw value (bytes) is the CUlaunchAttributeValue's bytes."""
    from cuda.bindings import driver

    out = []
    for key, v in attributes:
        a = driver.CUlaunchAttribute()
        a.id = key
        if isinstance(v, bytes):
            a.value.pad = v
            out.append(a)
            continue
        field, subs, enum = _ATTRS[key.name.removeprefix("CU_LAUNCH_ATTRIBUTE_")]
        if subs:
            sub = getattr(a.value, field)
            for f, x in zip(subs, v):
                setattr(sub, f, x)
        else:
            setattr(a.value, field, getattr(driver, enum)(v) if enum else v)
        out.append(a)
    return out


def reproducible(key: Any) -> bool:
    """Whether a launch reproduces the launch attribute: all but a
    device-updatable node's, whose kernel is handed a device-side handle to
    its node in the exec it was instantiated in."""
    return key.name != "CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE"


def node_attributes(node: int) -> dict[Any, Any]:
    """Every launch attribute the driver reports for a kernel node, by
    CUlaunchAttributeID, as launch_attributes takes it: one of _ATTRS
    decoded, another (an access policy window, an ID this module does not
    know) as its raw CUlaunchAttributeValue bytes."""
    from cuda.bindings import driver

    out = {}
    for key in driver.CUlaunchAttributeID:
        err, v = driver.cuGraphKernelNodeGetAttribute(node, key)
        if err == driver.CUresult.CUDA_SUCCESS:
            out[key] = attribute_value(key, v)
    return out


def attribute_value(key: Any, v: Any) -> Any:
    """A CUlaunchAttributeValue of the CUlaunchAttributeID as
    launch_attributes takes it: one of _ATTRS decoded, another as its raw
    bytes."""
    name = key.name.removeprefix("CU_LAUNCH_ATTRIBUTE_")
    if name not in _ATTRS:
        return bytes(v.pad)
    field, subs, _ = _ATTRS[name]
    sub = getattr(v, field)  # a view into v, which must outlive it
    return tuple(getattr(sub, s) for s in subs) if subs else int(sub)


@functools.cache
def plain_attributes(device: int) -> dict[Any, Any]:
    """The attributes of a kernel node captured from a launch that sets none.
    A node's others are those its launch set: an explicit cluster of (1, 1,
    1) is one, and a kernel built for clusters traps without it."""
    with _disable_current_modes():  # a trace's
        anchor = torch.empty(1, device=device)
        (fill,) = capture_kernel_nodes(lambda _: anchor.fill_(1))
    return dict(fill.attributes)


def explicit_attributes(node: KernelNode) -> tuple[tuple[Any, Any], ...]:
    """The attributes a launch sets to reproduce the node, but the
    programmatic edge into it."""
    from cuda.bindings import driver

    plain = plain_attributes(torch.cuda.current_device())
    pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    return tuple((k, v) for k, v in node.attributes if k != pdl and v != plain.get(k))


@dataclass(frozen=True)
class KernelNode:
    """A kernel node of a capture, as the driver holds it."""

    name: str
    function: int  # the CUfunction
    layout: tuple[tuple[int, int], ...]  # (offset, size) of each parameter
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int
    # (CUlaunchAttributeID, value) of each attribute the driver reports, by
    # id; PROGRAMMATIC_STREAM_SERIALIZATION is 1 when the edge into the node
    # is programmatic
    attributes: tuple[tuple[Any, Any], ...]
    images: tuple[bytes, ...]  # each parameter's bytes

    def attribute(self, name: str) -> Any:
        from cuda.bindings import driver

        key = getattr(driver.CUlaunchAttributeID, "CU_LAUNCH_ATTRIBUTE_" + name)
        return dict(self.attributes).get(key)


@dataclass(frozen=True)
class MemsetNode:
    dst: int
    value: int
    element_size: int
    width: int
    height: int
    pitch: int


@dataclass(frozen=True)
class MemcpyNode:
    """A 1D device-to-device memcpy node."""

    dst: int
    src: int
    nbytes: int


def graph_node_handles(raw: int) -> list[int]:
    from cuda.bindings import runtime

    _, count = _check_cuda_bindings(runtime.cudaGraphGetNodes(raw, numNodes=0))
    held, _ = _check_cuda_bindings(runtime.cudaGraphGetNodes(raw, numNodes=count))
    return [int(n) for n in held]


# each kernel's parameter layout, by handle and name (a handle an unloaded
# module freed can come back as another kernel's)
_LAYOUTS: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}


def graph_nodes(raw: int, memcpy: bool = False) -> list[KernelNode | MemsetNode | MemcpyNode]:
    """The nodes of a captured graph in stream order, from its edges: a fork
    and join (cuDNN's grouped and FFT paths launch on side streams) in a
    topological order, ties in node order, as serializing loses only eager's
    concurrency. A node of another kind (a memcpy's but for `memcpy` and a 1D
    device-to-device one), or a programmatic edge into anything but a chain's
    kernel's programmatic port raises ValueError."""
    from cuda.bindings import driver

    handles = graph_node_handles(raw)
    *_, n_edges = _check_cuda_bindings(driver.cuGraphGetEdges(raw, numEdges=0))
    frm, to, data, _ = _check_cuda_bindings(driver.cuGraphGetEdges(raw, numEdges=n_edges))
    position = {n: i for i, n in enumerate(handles)}
    succ: dict[int, list[int]] = {n: [] for n in handles}
    preds: dict[int, list[Any]] = {n: [] for n in handles}
    for a, b, e in zip(map(int, frm), map(int, to), data):
        succ[a].append(b)
        preds[b].append(e)
    pdl = driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
    forks = any(len(v) > 1 for v in (*succ.values(), *preds.values()))
    if forks and any(e.type == pdl for e in data):
        raise ValueError("a programmatic edge in a fork or join")
    waiting = {n: len(preds[n]) for n in handles}
    ready = [position[n] for n in handles if not waiting[n]]
    chain = []
    while ready:
        chain.append(handles[heapq.heappop(ready)])
        for b in succ[chain[-1]]:
            waiting[b] -= 1
            if not waiting[b]:
                heapq.heappush(ready, position[b])
    pred = {n: preds[n][0] for n in chain if preds[n]}
    kernel_t = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
    ids = driver.CUlaunchAttributeID
    out: list[KernelNode | MemsetNode | MemcpyNode] = []
    for node in chain:
        kind = _check_cuda_bindings(driver.cuGraphNodeGetType(node))
        edge = pred.get(node)
        programmatic = edge is not None and edge.type == pdl
        if programmatic and (kind != kernel_t or edge.from_port != driver.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC):
            raise ValueError(f"a programmatic edge from port {edge.from_port} into a {kind.name}")
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET:
            p = _check_cuda_bindings(driver.cuGraphMemsetNodeGetParams(node))
            out.append(MemsetNode(int(p.dst), p.value, p.elementSize, p.width, p.height, p.pitch))
            continue
        if memcpy and kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY:
            out.append(_memcpy_node(node))
            continue
        if kind != kernel_t:
            raise ValueError(f"the call adds a {kind.name} node")
        p = _check_cuda_bindings(driver.cuGraphKernelNodeGetParams(node))
        name = _check_cuda_bindings(driver.cuFuncGetName(p.func)).decode()
        layout = _LAYOUTS.get((int(p.func), name))
        if layout is None:
            params: list[tuple[int, int]] = []
            while True:
                err, offset, size = driver.cuFuncGetParamInfo(p.func, len(params))
                if err != driver.CUresult.CUDA_SUCCESS:
                    break
                params.append((offset, size))
            layout = _LAYOUTS[int(p.func), name] = tuple(params)
        if p.kernelParams:
            args = (ctypes.c_void_p * len(layout)).from_address(int(p.kernelParams))
            images = tuple(ctypes.string_at(a, n) for a, (_, n) in zip(args, layout))
        else:
            # CU_LAUNCH_PARAM_BUFFER_POINTER (1), _BUFFER_SIZE (2), _END (0)
            extra = (ctypes.c_void_p * 5).from_address(int(p.extra))
            found = dict(zip(extra[0:4:2], extra[1:4:2]))
            buf, size = found.get(1), found.get(2)
            end = max((o + n for o, n in layout), default=0)
            if not buf or not size or ctypes.c_size_t.from_address(size).value < end:
                raise ValueError("a kernel node's extra parameters are not one buffer")
            images = tuple(ctypes.string_at(buf + o, n) for o, n in layout)
        attributes = node_attributes(node)
        attributes[ids.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION] = int(programmatic)
        out.append(
            KernelNode(
                name,
                int(p.func),
                layout,
                (p.gridDimX, p.gridDimY, p.gridDimZ),
                (p.blockDimX, p.blockDimY, p.blockDimZ),
                p.sharedMemBytes,
                tuple(sorted(attributes.items(), key=lambda a: int(a[0]))),
                images,
            )
        )
    return out


def _memcpy_node(node: int) -> MemcpyNode:
    from cuda.bindings import driver

    p = _check_cuda_bindings(driver.cuGraphMemcpyNodeGetParams(node))
    device = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    plain = (p.srcMemoryType, p.dstMemoryType, p.Height, p.Depth, p.srcXInBytes, p.dstXInBytes, p.srcY, p.dstY, p.srcZ, p.dstZ)
    if plain != (device, device, 1, 1, 0, 0, 0, 0, 0, 0):
        raise ValueError("a memcpy node other than a 1D device-to-device one")
    return MemcpyNode(int(p.dstDevice), int(p.srcDevice), p.WidthInBytes)


def capture_kernel_nodes(
    fn: Callable[[torch.cuda.Stream], Any],
    stream: torch.cuda.Stream | None = None,
    mode: str = "thread_local",
    memcpy: bool = False,
) -> list[KernelNode | MemsetNode | MemcpyNode]:
    """fn(stream) captured on `stream` (by default a new side stream no
    capture holds), also the current stream, and never replayed: the graph's
    nodes, graph_nodes."""
    from cuda.bindings import runtime

    if stream is None:
        stream = torch.cuda.Stream()
        # the pool's streams wrap around, onto streams other captures hold
        idle = runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone
        while _check_cuda_bindings(runtime.cudaStreamIsCapturing(stream.cuda_stream)) != idle:
            stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with _gc_hold, torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode=mode)
        try:
            fn(stream)
        finally:
            graph.capture_end()
    return graph_nodes(graph.raw_cuda_graph(), memcpy)


def slot_address(slot: PointerSlot, values: Sequence[int], bases: Sequence[int]) -> int:
    if slot.base is not None:
        return bases[slot.base] + values[slot.displacement]
    if slot.address is None:
        raise AssertionError(f"host_trace: pointer slot {slot} has no address")
    return values[slot.address]


def memset_extent(m: LoweredMemset, values: Sequence[int]) -> tuple[int, int, int]:
    return values[m.width], values[m.height], values[m.pitch]


def _trace_disagrees(msg: str) -> NoReturn:
    # host tracing's bug, not the user's: the call falls back to eager
    if raise_trace_disagreements:
        raise AssertionError(f"host_trace: {msg}")
    warning_once(log, "host_trace: %s; the call runs without host tracing", msg)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "host_trace_trace_disagreement",
            "encoding": "string",
        },
        payload_fn=lambda: msg,
    )
    raise declined(msg)


def _check_trace_values(lowered: LoweredTape, values: Sequence[int]) -> None:
    if list(values) != list(lowered.program.values):
        _trace_disagrees("the compiled program disagrees with its rows at the traced call")
    tape = lowered.tape
    roots = {("argument", rec.position): rec.root for rec in tape.inputs}
    roots |= {("allocation", k): rec.root for k, rec in enumerate(tape.allocs)}
    roots |= {("eager", j): root for j, root in enumerate(lowered.eager_roots)}

    def check(row: int, traced: int, what: str) -> None:
        if values[row] != traced:
            _trace_disagrees(f"{what} is {values[row]} at the traced call; the trace has {traced}")

    def check_view(v: LoweredView | PredictedOutput, t: Any, what: str) -> None:
        for d, (row, s) in enumerate(zip(v.sizes, t.shape)):
            check(row, _hint(s), f"{what} size {d}")
        for d, (row, s) in enumerate(zip(v.strides, t._sym_strides)):
            check(row, _hint(s), f"{what} stride {d}")
        check(v.offset, _hint(t._sym_offset), f"{what} storage offset")

    for step in lowered.steps:
        if isinstance(step, range):
            continue
        call = step.call
        traced = pytree.tree_leaves((call.args, call.kwargs))
        for i, (leaf, v) in enumerate(zip(step.leaves, traced)):
            if isinstance(leaf, ScalarSlot):
                check(leaf.row, _hint(v), f"{call.name} argument {i}")
            elif isinstance(leaf, LoweredView):
                check_view(leaf, v, f"{call.name} argument {i}")
        for i, (p, t) in enumerate(zip(step.outputs, call.outputs)):
            if isinstance(p, PredictedOutput):
                check_view(p, t, f"{call.name} output {i}")
        for axis, row in enumerate(step.grid or ()):
            check(row, _hint(call.target[2][axis]), f"{call.name} grid axis {axis}")

    for alloc, rec in zip(lowered.allocations, tape.allocs):
        for d, (row, v) in enumerate(zip(alloc.sizes, rec.sizes)):
            check(row, _hint(v), f"{rec.name} size {d}")
        for d, (row, v) in enumerate(zip(alloc.strides, rec.strides)):
            check(row, _hint(v), f"{rec.name} stride {d}")
    for lo in lowered.launches:
        name = lo.launch.name
        if isinstance(lo, LoweredMemset):
            traced = (lo.launch.width, lo.launch.height, lo.launch.pitch)
            for row, v in zip((lo.width, lo.height, lo.pitch), traced):
                check(row, _hint(v), f"memset {name} extent")
        elif isinstance(lo, LoweredMemcpy):
            check(lo.nbytes, _hint(lo.launch.nbytes), f"memcpy {name} bytes")
        else:
            dims = zip((*lo.grid, *lo.block, lo.smem), (*lo.launch.grid, *lo.launch.block, lo.launch.smem))
            for axis, (row, v) in enumerate(dims):
                check(row, _hint(v), f"kernel {name} launch dimension {axis}")
        for i, (slot, v) in enumerate(zip(lo.slots, lo.launch.slots)):
            if isinstance(slot, ScalarSlot):
                check(slot.row, _hint(v), f"kernel {name} slot {i}")
            else:
                offset = _hint(v) - _hint(roots[slot.root].sym)
                check(
                    slot.displacement, offset, f"kernel {name} slot {i} offset"
                )


def _frontier(stream: int) -> list[int]:
    from cuda.bindings import runtime

    _, _, _, nodes, edges, _ = _check_cuda_bindings(
        runtime.cudaStreamGetCaptureInfo(stream)
    )
    if any(edge.type != 0 for edge in edges):
        raise declined("a capture dependency is not a default edge")
    return [int(n) for n in nodes]


def _launch(
    launches: Sequence[LoweredLaunch | LoweredMemset | LoweredMemcpy],
    values: Sequence[int],
    bases: Sequence[int],
    stream: int,
    copy_at: tuple[int, int],
    nodes: list[int],
) -> tuple[list[int], list[LoweredLaunch | LoweredMemset | LoweredMemcpy], tuple[torch.Generator, int, int] | None]:
    """Launch the kernels, memsets and memcpys into the capture on `stream`; each
    one's node (into `nodes`, so a failure is at the launch after them), the
    launches as captured: an RNG kernel's images hold
    the capture's generator state, and that state as CapturedSegment.rng. A
    memcpy copies between `copy_at` (dst, src; _copy_operands)."""
    from cuda.bindings import driver, runtime

    captured: list[LoweredLaunch | LoweredMemset | LoweredMemcpy] = []
    philox = (0, 0, 0)
    rng = None
    for lo in launches:
        captured.append(lo)
        if isinstance(lo, LoweredMemset):
            # the driver takes the placeholder address: the replay patches it
            m, (width, height, pitch) = lo.launch, memset_extent(lo, values)
            dst = slot_address(lo.slots[0], values, bases)
            memset = getattr(driver, f"cuMemsetD2D{8 * m.element_size}Async")
            _check_cuda_bindings(memset(dst, pitch, m.value, width, height, stream))
            after = _frontier(stream)
            if len(after) != 1 or after[0] in nodes:
                raise declined(f"memset {m.name} did not add one node to the capture")
            nodes.append(after[0])
            continue
        if isinstance(lo, LoweredMemcpy):
            d2d = runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            _check_cuda_bindings(runtime.cudaMemcpyAsync(*copy_at, values[lo.nbytes], d2d, stream))
            after = _frontier(stream)
            if len(after) != 1 or after[0] in nodes:
                raise declined(f"memcpy {lo.launch.name} did not add one node to the capture")
            nodes.append(after[0])
            continue
        t = lo.launch
        if t.rng_increment:
            gen = t.generator or torch.cuda.default_generators[torch.cuda.current_device()]
            seed, offset, intragraph = gen.philox_state(t.rng_increment)
            philox = (seed.data_ptr(), offset.data_ptr(), int(intragraph))
            if rng is not None and (rng[0] is not gen or rng[1:] != philox[:2]):
                raise declined("a run's RNG kernels draw from two generators")
            rng = (gen, *philox[:2])
        # the node holds these bytes, which a native replay's first patch compares with
        if t.rng or t.cpu_scalars:
            images = [bytearray(image) for image in t.images]
            for param, at, kind, delta in t.rng:
                v = philox[("philox_seed", "philox_offset", "philox").index(kind)] + delta
                _POINTER.pack_into(images[param], at, v)
            pack_cpu_scalars(t, images)
            lo = captured[-1] = replace(lo, launch=replace(t, images=tuple(map(bytes, images))))
        images = launch_images(lo, launch_slots(lo, values, bases))
        storage = [ctypes.create_string_buffer(image, len(image)) for image in images]
        arguments = (ctypes.c_void_p * len(storage))(*map(ctypes.addressof, storage))
        ids = driver.CUlaunchAttributeID
        attributes = list(t.attributes)
        if t.programmatic:
            attributes.append((ids.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, 1))
        config = driver.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = (values[row] for row in lo.grid)
        config.blockDimX, config.blockDimY, config.blockDimZ = (values[row] for row in lo.block)
        config.sharedMemBytes = values[lo.smem]
        config.hStream = stream
        config.attrs = launch_attributes(attributes)
        config.numAttrs = len(attributes)
        _check_cuda_bindings(driver.cuLaunchKernelEx(config, t.function, ctypes.addressof(arguments), 0))
        after = _frontier(stream)
        if len(after) != 1 or after[0] in nodes:
            raise declined(
                f"kernel {t.name} did not add one node to the capture"
            )
        nodes.append(after[0])
    return nodes, captured, rng


def _verify(
    graph: int,
    nodes: Sequence[int],
    launches: Sequence[LoweredLaunch | LoweredMemset | LoweredMemcpy],
    values: Sequence[int],
    bases: Sequence[int],
    copy_at: tuple[int, int],
) -> None:
    """The graph holds exactly `nodes`, the i-th launch's kernel, memset or
    memcpy node."""
    from cuda.bindings import driver

    held = graph_node_handles(graph)
    if len(held) != len(nodes) or set(held) != set(nodes):
        raise declined("the capture holds nodes the tape did not launch")
    for node, lo in zip(nodes, launches):
        if isinstance(lo, LoweredMemset):
            m, (width, height, pitch) = lo.launch, memset_extent(lo, values)
            kind = _check_cuda_bindings(driver.cuGraphNodeGetType(node))
            if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET:
                raise declined(f"memset {m.name} captured a {kind.name} node")
            p = _check_cuda_bindings(driver.cuGraphMemsetNodeGetParams(node))
            dst = slot_address(lo.slots[0], values, bases)
            got = (int(p.dst), p.value, p.elementSize, p.width, p.height)
            if got != (dst, m.value, m.element_size, width, height) or (
                height > 1 and p.pitch != pitch
            ):
                raise declined(f"the captured node of memset {m.name} has other parameters")
            continue
        if isinstance(lo, LoweredMemcpy):
            kind = _check_cuda_bindings(driver.cuGraphNodeGetType(node))
            if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY:
                raise declined(f"memcpy {lo.launch.name} captured a {kind.name} node")
            try:
                if _memcpy_node(node) != MemcpyNode(*copy_at, values[lo.nbytes]):
                    raise ValueError("other parameters")
            except ValueError:
                raise declined(f"the captured node of memcpy {lo.launch.name} has other parameters") from None
            continue
        t = lo.launch
        where = f"the captured node of kernel {t.name}"
        p = _check_cuda_bindings(driver.cuGraphKernelNodeGetParams(node))
        got = (
            int(p.func),
            (p.gridDimX, p.gridDimY, p.gridDimZ),
            (p.blockDimX, p.blockDimY, p.blockDimZ),
            p.sharedMemBytes,
        )
        grid, block = (tuple(values[row] for row in rows) for rows in (lo.grid, lo.block))
        if got != (t.function, grid, block, values[lo.smem]):
            raise declined(f"{where} has another launch config")
        plain = plain_attributes(torch.cuda.current_device())
        attributes = {k: v for k, v in node_attributes(node).items() if v != plain.get(k)}
        if attributes != dict(t.attributes):
            raise declined(f"{where} has launch attributes {attributes}; the launch {dict(t.attributes)}")
        if not p.kernelParams or p.extra:
            raise declined(f"{where} has no parameter array")
        params = int(p.kernelParams)
        held_params = (ctypes.c_void_p * len(t.layout)).from_address(params)
        images = launch_images(lo, launch_slots(lo, values, bases))
        for i, (image, pointer) in enumerate(zip(images, held_params)):
            if ctypes.string_at(pointer, len(image)) != image:
                raise declined(f"{where} holds other bytes in slot {i}")


def _copy_operands(device: torch.device, nbytes: int) -> torch.Tensor:
    """What a captured memcpy node copies between: a memcpy node keeps its
    capture's kind of memory (cudaMalloc'd or cuMemMap'd, as expandable
    segments are), and a replay can only set it to operands of that kind, so
    the capture copies within an allocation of the caching allocator's. The
    capture runs no node, and the first replay sets every memcpy."""
    return torch.empty(2 * nbytes, dtype=torch.uint8, device=device)


def _segment(
    launches: Sequence[LoweredLaunch | LoweredMemset | LoweredMemcpy],
    values: Sequence[int],
    bases: Sequence[int],
    stream: torch.cuda.Stream,
) -> CapturedSegment:
    from cuda.bindings import runtime

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    # a replay advances each generator the graph draws from, as the default
    generators = (lo.launch.generator for lo in launches if not isinstance(lo, (LoweredMemset, LoweredMemcpy)))
    for gen in {id(g): g for g in generators if g is not None}.values():
        graph.register_generator_state(gen)
    # the largest memcpy's span, so both operands lie inside the allocation
    sizes = [values[lo.nbytes] for lo in launches if isinstance(lo, LoweredMemcpy)]
    span = -(-max([1, *sizes]) // _ALLOC_ALIGNMENT) * _ALLOC_ALIGNMENT
    copies = _copy_operands(stream.device, span) if sizes else None
    copy_at = (0, 0) if copies is None else (copies.data_ptr(), copies.data_ptr() + span)
    # relaxed, unlike the trace's thread-local capture (a detector there):
    # the tape, not the capture, decides what is replayed, and a thread-local
    # capture is invalidated by other threads' allocations
    nodes: list[int] = []
    with _gc_hold, torch.cuda.stream(stream), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The CUDA Graph is empty")
        graph.capture_begin(capture_error_mode="relaxed")
        try:
            nodes, captured, rng = _launch(launches, values, bases, stream.cuda_stream, copy_at, nodes)
        except BaseException as e:
            with contextlib.suppress(Exception):
                graph.capture_end()
            _segment_failed(e, launches[len(nodes) : len(nodes) + 1])
        graph.capture_end()
    try:
        _verify(graph.raw_cuda_graph(), nodes, captured, values, bases, copy_at)
        graph.instantiate()
        # upload before any node is patched (an update of a never-uploaded exec
        # can leave the node on a slow path for good)
        exec_ = graph.raw_cuda_graph_exec()
        _check_cuda_bindings(runtime.cudaGraphUpload(exec_, stream.cuda_stream))
    except BaseException as e:
        _segment_failed(e, launches)
    return CapturedSegment(
        graph, tuple(CapturedLaunch(n, lo) for n, lo in zip(nodes, captured)), rng
    )


def _segment_failed(e: BaseException, at: Sequence[LoweredLaunch | LoweredMemset | LoweredMemcpy]) -> NoReturn:
    """A segment's failure at launches `at` as a SegmentFailed: the call runs
    their ops eagerly, the rest replays. An unexpected error raises as itself
    in the test suites, as an interrupt or an OOM (the call's) does."""
    if not isinstance(e, Exception) or isinstance(e, torch.OutOfMemoryError) or (torch.cuda._host_trace.raise_unexpected and not isinstance(e, Declined)):
        raise e
    names = ", ".join(lo.launch.name for lo in at[:3]) + (f" and {len(at) - 3} more" if len(at) > 3 else "")
    if isinstance(e, Declined):
        why = str(e).removeprefix("host_trace: ").removesuffix(" (declined)")
    else:
        why = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    raise SegmentFailed(str(declined(f"the capture of {names} failed: {why}")), tuple(lo.seq for lo in at)) from e


def capture_tape(lowered: LoweredTape, addresses: Sequence[int]) -> CapturedTape:
    """The lowered tape's runs of launches captured into CUDA graphs at the
    traced call (`lowered.tape.args`), verified, instantiated and uploaded.
    Their kernels point at `addresses`, one per allocation, and at a
    placeholder per eager output. The capture runs no kernel and allocates
    nothing, so its graphs' pools stay empty."""
    tape = lowered.tape
    values = lowered.evaluate(tape.args)
    if values is None:
        raise declined("the traced call misses its own tape")
    _check_trace_values(lowered, values)
    with torch.cuda.device(tape.device):
        if any(a % _ALLOC_ALIGNMENT for a in addresses):
            raise declined(f"an allocation is not {_ALLOC_ALIGNMENT}-byte aligned")
        eager = range(1, len(lowered.eager_roots) + 1)
        bases = [*addresses, *(_EAGER_TAG | (j << _ALLOC_SHIFT) for j in eager)]
        stream = torch.cuda.Stream(tape.device)
        segments = tuple(
            _segment(lowered.launches[r.start : r.stop], values, bases, stream)
            for r in lowered.steps
            if isinstance(r, range)
        )
        stream.synchronize()
    return CapturedTape(lowered, segments)


def _capture_piece(graph: int, dependencies: list[Any], binding: OpaqueBinding) -> list[tuple[int, int]]:
    """The binding's nodes captured into the graph after `dependencies`, on
    default edges into its first; each one's (kind, node), 0 a kernel's and 1
    a memset's. Its slots hold don't-care bytes, a replay's patch fills them."""
    from cuda.bindings import driver

    from torch.cuda._host_trace_opaque import OpaqueMemset

    pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    stream = torch.cuda.Stream().cuda_stream
    relaxed = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
    _check_cuda_bindings(driver.cuStreamBeginCaptureToGraph(stream, graph, dependencies, None, len(dependencies), relaxed))
    nodes: list[tuple[int, int]] = []
    try:
        for n in binding.nodes:
            if isinstance(n, OpaqueMemset):
                memset = getattr(driver, f"cuMemsetD2D{8 * n.element_size}Async")
                _check_cuda_bindings(memset(_EAGER_TAG, n.pitch, n.value, n.width, n.height, stream))
            else:
                images = n.images or tuple(bytes(size) for _, size in n.params)
                storage = [ctypes.create_string_buffer(image, len(image)) for image in images]
                arguments = (ctypes.c_void_p * len(storage))(*map(ctypes.addressof, storage))
                attributes = [(k, v) for k, v in n.attributes if nodes or k != pdl]
                config = driver.CUlaunchConfig()
                config.gridDimX, config.gridDimY, config.gridDimZ = n.grid
                config.blockDimX, config.blockDimY, config.blockDimZ = n.block
                config.sharedMemBytes = n.smem
                config.hStream = stream
                config.attrs = launch_attributes(attributes)
                config.numAttrs = len(attributes)
                _check_cuda_bindings(driver.cuLaunchKernelEx(config, n.function, ctypes.addressof(arguments), 0))
            after = _frontier(stream)
            if len(after) != 1:
                raise declined(f"a piece's node {len(nodes)} did not add one node to the capture")
            nodes.append((int(isinstance(n, OpaqueMemset)), after[0]))
    finally:
        _check_cuda_bindings(driver.cuStreamEndCapture(stream))
    return nodes


def _neighbors(get: Callable[..., Any], node: Any) -> list[Any]:
    count = _check_cuda_bindings(get(node, 0))[-1]
    return list(_check_cuda_bindings(get(node, count))[0][:count])


def instantiate_form(
    graph: int,
    attributes: Sequence[tuple[int, dict[Any, Any]]],
    pieces: Sequence[tuple[Sequence[int], OpaqueBinding]],
) -> tuple[int, int, list[list[tuple[int, int]]]]:
    """An exec, uploaded on the current stream, of a clone of the graph with
    each (kernel node, {CUlaunchAttributeID: value}) at those launch
    attributes and each piece (a chain of nodes, binding) the binding's nodes
    in the chain's place, on default edges: a programmatic edge into or out of
    the chain is dropped. The clone, and each piece's nodes as _capture_piece."""
    from cuda.bindings import driver

    clone = _check_cuda_bindings(driver.cuGraphClone(graph))
    try:
        replaced = [n for nodes, _ in pieces for n in nodes]
        found = {n: _check_cuda_bindings(driver.cuGraphNodeFindInClone(n, clone)) for n in (*(n for n, _ in attributes), *replaced)}
        for node, values in attributes:
            for a in launch_attributes(tuple(values.items())):
                v = driver.CUkernelNodeAttrValue()
                v.pad = a.value.pad
                _check_cuda_bindings(driver.cuGraphKernelNodeSetAttribute(found[node], a.id, v))
        placed = []
        for nodes, binding in pieces:
            into = _neighbors(driver.cuGraphNodeGetDependencies, found[nodes[0]])
            out = _neighbors(driver.cuGraphNodeGetDependentNodes, found[nodes[-1]])
            for n in nodes:
                _check_cuda_bindings(driver.cuGraphDestroyNode(found[n]))
            placed.append(_capture_piece(clone, into, binding))
            if out:
                last = driver.CUgraphNode(init_value=placed[-1][-1][1])  # a list takes no int
                _check_cuda_bindings(driver.cuGraphAddDependencies(clone, [last] * len(out), out, None, len(out)))
        flags = driver.CUgraphInstantiate_flags
        exec_ = _check_cuda_bindings(
            driver.cuGraphInstantiate(
                clone,
                int(flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                | int(flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY),
            )
        )
    except BaseException:
        _check_cuda_bindings(driver.cuGraphDestroy(clone))
        raise
    _check_cuda_bindings(driver.cuGraphUpload(exec_, torch.cuda.current_stream().cuda_stream))
    return int(exec_), int(clone), [[(kind, int(node)) for kind, node in p] for p in placed]
