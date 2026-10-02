"""Host tracing (private): the seam for opaque library calls (cuBLAS, cuDNN).

An eager call an OpaqueProvider accepts at trace time becomes an OpaqueCall.
At a replay its key (OpaqueKey: the operands' dtypes, sizes, strides and
alignments and the scalar arguments) is evaluated from the tape's rows and
bound: the provider returns the kernels and memsets the call launches for
exactly that key (OpaqueBinding), or None. On a miss the call runs eagerly,
as any eager call, and the provider may then learn the key from that run. A
binding's slots say which bytes of a node are an operand's or a scratch
buffer's address. The global cuBLAS state eager read at the traced call
(library_state) is in the key, and the call's eager runs and its learning
run under it: a call keeps the math mode it was traced under, whatever the
state at a replay.

A call whose key binds at trace time (bind_at_trace) is the binding itself
(record_binding): each kernel a KernelLaunch whose `fields` place its slots
in its parameter images, each memset a Memset, its fresh outputs and scratch
buffers allocations of the trace. The key is not guarded: the call is a
KeyedSite, a launch table from key to its binding's node parameters, which a
replay looks up per call. A key's binding is added to the table once: one
of the site's topology runs on the site's nodes, one of another topology on
an arm of the site, nodes added beside the site's own in its graph, of which
the key's row enables one. Each scratch buffer takes the key's bytes when a
replay allocates it, as eager's call would. A replay skips a variant whose
OpaqueCall now binds and traces again.

A binding that consumes philox offsets (an RNG call) is a KeyedSite of RNG
bindings of its topology: a replay packs each RNG kernel's generator state,
its graph's captured seed and offset and the offsets the graph's earlier RNG
calls take at their keys, and advances the generator by what the graph's
calls take, as eager's calls would (HostTraceVariant::patch_and_replay).

Operands, in this order everywhere (key, slots, learn): the tensor leaves of
pytree.tree_leaves((args, kwargs)) in leaf order (a tensor passed twice
appears twice), then the call's fresh outputs in return order; an in-place
or out= return is its argument, not another operand. An operand's address is
its data_ptr(), storage offset included.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch._ops import OpOverload
    from torch.cuda._host_trace_tape import _Trace


@dataclass(frozen=True)
class OpaqueKey:
    """What the library's kernel choice may read; a binding holds for exactly
    the calls with its key."""

    op: OpOverload
    dtypes: tuple[torch.dtype, ...]  # per operand
    sizes: tuple[tuple[int, ...], ...]  # per operand
    strides: tuple[tuple[int, ...], ...]  # per operand, in elements
    # per operand: address % 256 (a fresh output's is its storage offset's
    # bytes, the allocator's blocks being 256-byte aligned)
    align: tuple[int, ...]
    scalars: tuple  # the non-tensor leaves, in leaf order, SymInts as ints
    device: int  # the operands' device: a binding's functions are loaded in its context
    state: tuple  # library_state() at the traced call


_STATE = (
    (lambda: torch._C._get_fp32_precision_getter("cuda", "matmul"), lambda v: torch._C._set_fp32_precision_setter("cuda", "matmul", v)),
    (torch._C._get_cublas_allow_fp16_reduced_precision_reduction, lambda v: torch._C._set_cublas_allow_fp16_reduced_precision_reduction(*v)),
    (torch._C._get_cublas_allow_bf16_reduced_precision_reduction, lambda v: torch._C._set_cublas_allow_bf16_reduced_precision_reduction(*v)),
    (torch._C._get_cublas_allow_fp16_accumulation, torch._C._set_cublas_allow_fp16_accumulation),
    (torch._C._get_sm_carveout_experimental, torch._C._set_sm_carveout_experimental),
    (torch._C._get_blas_preferred_backend, torch._C._set_blas_preferred_backend),
    (lambda: torch._C._get_fp32_precision_getter("cuda", "conv"), lambda v: torch._C._set_fp32_precision_setter("cuda", "conv", v)),
    (torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled),
    (torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark),
    (torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic),
    (
        lambda: (torch._C._get_deterministic_algorithms(), torch._C._get_deterministic_algorithms_warn_only()),
        lambda v: torch._C._set_deterministic_algorithms(v[0], warn_only=v[1]),
    ),
)


def library_state() -> tuple:
    """The global state cuBLAS and cuDNN calls read: the fp32 math mode
    (allow_tf32 and float32_matmul_precision set it), the fp16 and bf16
    reduced-precision reductions, fp16 accumulation, the SM carveout, the
    preferred library; and cuDNN's conv fp32 precision, enabled, benchmark
    and deterministic flags, and deterministic algorithms."""
    return tuple(get() for get, _ in _STATE)


@contextlib.contextmanager
def library_state_as(state: tuple) -> Iterator[None]:
    """Runs under `state`, a library_state(), and restores the prior one."""
    prior = library_state()
    changed = [(set_, v, p) for (_, set_), v, p in zip(_STATE, state, prior) if v != p]
    try:
        for set_, v, _ in changed:
            set_(v)
        yield
    finally:
        for set_, _, p in changed:
            set_(p)


@dataclass(frozen=True)
class Slot:
    """Bytes of a node a replay writes per call: an address plus `delta`."""

    param: int  # 0 for a memset's dst
    offset: int  # into the parameter, of 8 bytes
    # "operand" or "scratch"; or the generator state an RNG kernel reads,
    # fixed per capture: "philox_seed" and "philox_offset", the addresses of
    # the capture's seed and offset, and "philox", its intragraph offset at
    # the call
    kind: str
    index: int  # operand i or scratch buffer j
    delta: int


@dataclass(frozen=True)
class OpaqueKernel:
    function: int  # CUfunction handle
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int  # dynamic shared memory bytes
    # sorted (CUlaunchAttributeID, value) pairs: those the launch set, as
    # explicit_attributes has them, and PROGRAMMATIC_STREAM_SERIALIZATION 1
    # for a programmatic edge into it
    attributes: tuple
    params: tuple[tuple[int, int], ...]  # (offset, size) per parameter
    images: tuple[bytes, ...]  # each parameter's bytes; slot bytes are don't-care
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class OpaqueMemset:
    dst: Slot
    value: int
    element_size: int  # 1, 2 or 4
    width: int  # elements
    height: int
    pitch: int  # bytes


@dataclass(frozen=True)
class OpaqueBinding:
    nodes: tuple[OpaqueKernel | OpaqueMemset, ...]  # one chain, in stream order
    scratch: tuple[int, ...]  # bytes per scratch buffer
    rng_increment: int = 0  # philox offsets the call consumes
    # what keeps the library's kernels loaded (a cuDNN plan), held by the
    # binding and each launch recorded from it
    owner: Any = field(default=None, compare=False, repr=False)

    @property
    def topology(self) -> tuple:
        """What a graph exec fixes at instantiation: a kernel's launch
        attributes and programmatic edge, a memset's element size, height
        and, over several rows, width and pitch. One variant per topology."""
        nodes: list[tuple] = []
        for n in self.nodes:
            if isinstance(n, OpaqueMemset):
                nodes.append(("memset", n.element_size, n.height, *((n.width, n.pitch) if n.height > 1 else (0, 0))))
                continue
            nodes.append(("kernel", n.attributes))
        return tuple(nodes)

    @property
    def used_scratch(self) -> set[int]:
        slots = (s for n in self.nodes for s in (n.slots if isinstance(n, OpaqueKernel) else (n.dst,)))
        return {s.index for s in slots if s.kind == "scratch"}

    @property
    def rng(self) -> bool:
        """Whether the call draws from the generator: a replay packs its
        kernels' philox slots per call, in the graph's node order."""
        kernels = (n for n in self.nodes if isinstance(n, OpaqueKernel))
        return bool(self.rng_increment) or any(s.kind.startswith("philox") for n in kernels for s in n.slots)


@dataclass(frozen=True)
class KeyedSite:
    """A call bound at trace time: its nodes' parameters are looked up per
    call by the call's key."""

    op: OpOverload
    provider: OpaqueProvider
    operands: tuple[Any, ...]  # traced tensors, in operand order
    scalars: tuple  # the non-tensor leaves
    scratch: dict[int, tuple[Any, int]]  # buffer j: (its allocation, traced bytes)
    nodes: tuple[Any, ...]  # each binding node's KernelLaunch or Memset
    topology: tuple
    rng: bool
    state: tuple  # library_state() at the traced call
    # the call's pytree spec and its tensor leaves' positions: a key's call
    # is its op on its operands and scalars (_host_trace_replay's _learn)
    call: tuple[Any, frozenset[int]] | None = None

    def fits(self, binding: OpaqueBinding) -> bool:
        """Whether the site can run the binding: on its nodes or an arm's,
        with its buffers at the key's bytes. An RNG site's rows are RNG
        bindings on its own nodes, whose philox offsets a replay takes in
        node order."""
        if binding.rng != self.rng or (self.rng and binding.topology != self.topology):
            return False
        return recordable(binding) and binding.used_scratch <= self.scratch.keys()


class OpaqueProvider(Protocol):
    def accepts(self, op: OpOverload, args: tuple, kwargs: dict) -> str | None:
        """At trace time, over traced tensors: why the provider declines the
        call, or None to take it. It may read the op, dtypes, ranks, devices
        and constant leaves, never sizes, strides or SymInt values: a branch
        on those records a guard."""
        ...

    def bind(self, key: OpaqueKey) -> OpaqueBinding | None:
        """A lookup, with no GPU work."""
        ...

    def refusal(self, key: OpaqueKey) -> str | None:
        """Why the provider will never bind the key, or None if a later call
        may: a lookup, with no GPU work."""
        ...

    def learn(
        self, key: OpaqueKey, args: tuple, kwargs: dict, operands: list[Any]
    ) -> OpaqueBinding | None:
        """At a replay whose bind missed, right after the call ran eagerly on
        the current stream: `operands` are its real tensors in operand order,
        the fresh outputs holding its result. The provider caches a refusal
        itself."""
        ...


def recordable(binding: OpaqueBinding) -> bool:
    """Whether a trace can record the binding's nodes as its launches."""
    from torch.cuda._host_trace_capture import reproducible

    kernels = [n for n in binding.nodes if isinstance(n, OpaqueKernel)]
    return all(reproducible(a) for k in kernels for a, _ in k.attributes)


def trace_key(op: OpOverload, args: tuple, kwargs: dict, fresh: list) -> tuple[OpaqueKey, list[Any]]:
    """A call's key at the trace's hints, and the values it reads."""
    from torch.cuda._host_trace_launch import _probe_address
    from torch.cuda._host_trace_tape import _hint

    leaves = pytree.tree_leaves((args, kwargs))
    traced = [t for t in leaves if isinstance(t, torch.Tensor)]
    sizes = [tuple(t.shape) for t in (*traced, *fresh)]
    strides = [tuple(t._sym_strides) for t in traced] + [o.stride() for o in fresh]
    # an allocation's base is 256-byte aligned
    align = [_probe_address(t) % 256 for t in traced]
    align += [o.storage_offset() * o.element_size() % 256 for o in fresh]
    scalars = [v for v in leaves if not isinstance(v, torch.Tensor)]
    key = OpaqueKey(
        op,
        tuple(t.dtype for t in (*traced, *fresh)),
        tuple(tuple(map(_hint, s)) for s in sizes),
        tuple(tuple(map(_hint, s)) for s in strides),
        tuple(map(_hint, align)),
        tuple(map(_hint, scalars)),
        next(t.device for t in (*traced, *fresh)).index,
        library_state(),
    )
    return key, [*sum(sizes, ()), *sum(strides, ()), *align, *scalars]


def bind_at_trace(
    provider: OpaqueProvider, op: OpOverload, args: tuple, kwargs: dict, fresh: list
) -> tuple[OpaqueBinding | None, list[Any], str | None]:
    """The binding of an accepted call at the trace's hints, if the trace can
    record it, the values to guard at their hints: the fresh outputs'
    symbolic storage offsets (0); and why the key never binds, if it doesn't
    (the key is guarded: the call is a plain eager step, which no replay
    learns from). `fresh` are the fake kernel's fresh outputs, which the
    binding's allocations replace."""
    from torch.cuda._host_trace_tape import _hint

    key, all_values = trace_key(op, args, kwargs, fresh)
    binding = provider.bind(key)
    refusal = provider.refusal(key)
    if binding is not None and not recordable(binding):
        refusal = "a kernel node of its binding is device-updatable"
    if refusal is not None:
        # a plain eager step at this key: the variant does not learn it
        why = f"{op} at sizes {key.sizes}: {refusal}"
        return None, [v for v in all_values if isinstance(v, torch.SymInt)], why
    if binding is None:
        return None, [], None
    # a fresh output is an allocation: storage offset 0
    offsets = [o.storage_offset() for o in fresh]
    if any(_hint(o) for o in offsets):
        return None, [], None
    return binding, [o for o in offsets if isinstance(o, torch.SymInt)], None


def record_binding(
    tr: _Trace,
    op: OpOverload,
    provider: OpaqueProvider,
    binding: OpaqueBinding,
    operands: list[Any],
    scalars: list[Any],
    call: tuple[Any, frozenset[int]] | None = None,
) -> KeyedSite:
    """A bound call's nodes as the trace's launches and memsets, its scratch
    buffers as allocations; `operands` are traced tensors."""
    from cuda.bindings import driver

    from torch.cuda._host_trace_launch import KernelLaunch
    from torch.cuda._host_trace_tape import Memset

    pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION

    empty = torch.ops.aten.empty.memory_format
    kw = {"dtype": torch.uint8, "device": tr.device}
    # eager's own buffers, at eager's sizes; one no node reads has 0 bytes. A
    # binding without one has one of 0 bytes, which another key's may use.
    used = binding.used_scratch
    sizes = [n if j in used else 0 for j, n in enumerate(binding.scratch)] or [0]
    scratch = {j: tr.allocate(empty, ([n],), kw) for j, n in enumerate(sizes)}
    buffers = {"operand": operands, "scratch": scratch}
    roots = tuple({id(t._root): t._root for t in (*operands, *scratch.values())}.values())

    def address(s: Slot) -> Any:
        return buffers[s.kind][s.index].data_ptr() + s.delta

    nodes: list[Any] = []
    # the call's philox offsets are taken at its first kernel
    first = next((n for n in binding.nodes if isinstance(n, OpaqueKernel)), None)
    for i, n in enumerate(binding.nodes):
        what = f"{op} node {i}"
        if isinstance(n, OpaqueMemset):
            extent = (n.width, n.height, n.pitch)
            nodes.append(Memset(what, (address(n.dst),), roots, n.value, n.element_size, *extent))
            tr.record_launch(nodes[-1])
            continue
        placed = [s for s in n.slots if not s.kind.startswith("philox")]
        philox = tuple((s.param, s.offset, s.kind, s.delta) for s in n.slots if s.kind.startswith("philox"))
        slots = tuple(map(address, placed))
        nodes.append(
            KernelLaunch(
                what,
                n.function,
                None,
                n.params,
                n.grid,
                n.block,
                n.smem,
                slots,
                roots,
                provider if binding.owner is None else (provider, binding.owner),
                tuple((s.param, s.offset, 8) for s in placed),
                attributes=tuple((k, v) for k, v in n.attributes if k != pdl),
                pointers=frozenset(range(len(slots))),
                images=n.images,
                programmatic=bool(dict(n.attributes).get(pdl)),
                rng=philox,
                rng_increment=binding.rng_increment if n is first else 0,
                generator=tr.generator,
            )
        )
        tr.record_launch(nodes[-1])
    sized = {j: (t, sizes[j]) for j, t in scratch.items()}
    return KeyedSite(op, provider, tuple(operands), tuple(scalars), sized, tuple(nodes), binding.topology, binding.rng, library_state(), call)
