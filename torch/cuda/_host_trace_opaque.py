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
import functools
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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
    # per operand on a storage another operand is on: (operand, the operand
    # lowest on the storage, the byte distance up from its address)
    alias: tuple[tuple[int, int, int], ...] = ()


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
    # (parameter, byte, length) runs of don't-care bytes outside the slots:
    # padding and host state of the call, which a restore does not compare
    loose: tuple[tuple[int, int, int], ...] = ()


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

    @functools.cached_property
    def topology(self) -> tuple:
        """What a graph exec fixes at instantiation: a kernel's launch
        attributes, programmatic edge and compile-time cluster, a memset's
        element size, height and, over several rows, width and pitch. One
        variant per topology. Cached: a key's binding is read at each site
        that takes the key, and a cluster is a driver query per kernel."""
        from torch.cuda._host_trace_launch import compiled_cluster

        nodes: list[tuple] = []
        for n in self.nodes:
            if isinstance(n, OpaqueMemset):
                nodes.append(("memset", n.element_size, n.height, *((n.width, n.pitch) if n.height > 1 else (0, 0))))
                continue
            nodes.append(("kernel", n.attributes, *compiled_cluster(n.function)))
        return tuple(nodes)

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
    # the key the binding its nodes hold is the provider's for
    key: OpaqueKey | None = field(default=None, compare=False)

    def fits(self, binding: OpaqueBinding) -> bool:
        """Whether the site can run the binding: on its nodes or an arm's,
        with its buffers at the key's bytes. An RNG site's rows are RNG
        bindings on its own nodes, whose philox offsets a replay takes in
        node order."""
        if binding.rng != self.rng or (self.rng and binding.topology != self.topology):
            return False
        return recordable(binding) and set(range(len(binding.scratch))) <= self.scratch.keys()


class OpaqueProvider(Protocol):
    # the ops whose written operands may share a storage with other operands:
    # their bindings hold for exactly their keys' OpaqueKey.alias
    aliased: frozenset[OpOverload]

    def accepts(self, op: OpOverload, args: tuple, kwargs: dict) -> str | None:
        """At trace time, over traced tensors: why the provider declines the
        call ("" for an op outside its set, which names no reason), or None
        to take it. It may read the op, dtypes, ranks, devices
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

    def refuses_layout(self, key: OpaqueKey) -> bool:
        """Whether the key's refusal holds at every size of its layout (its
        op's kernels refuse, not its sizes): a trace guards none of the key's
        sizes for the plain eager step. A lookup."""
        ...

    def operands(self, key: OpaqueKey, inputs: int) -> dict[int, torch.Tensor] | None:
        """For a key learned without its call: the inputs the provider lends
        (by position, a buffer too big to allocate again), or None to learn
        it only at a call: a lookup."""
        ...

    def learn(
        self, key: OpaqueKey, args: tuple, kwargs: dict, operands: list[Any]
    ) -> OpaqueBinding | None:
        """At a replay whose bind missed, right after the call ran eagerly on
        the current stream: `operands` are its real tensors in operand order,
        the fresh outputs holding its result. The provider caches a refusal
        itself."""
        ...


def takes_generator(op: OpOverload) -> bool:
    """Whether op draws from a generator by its schema: it takes a Generator,
    or an overload of its packet does (rand_like's default overload draws
    from the default one its .generator overload takes). One that draws
    otherwise shows at its harvest: it takes philox offsets."""
    packet = op.overloadpacket
    for overload in (op, *(getattr(packet, name) for name in packet.overloads())):
        for a in overload._schema.arguments:
            t = a.type.getElementType() if isinstance(a.type, torch.OptionalType) else a.type
            if isinstance(t, torch._C._GeneratorType):
                return True
    return False


def recordable(binding: OpaqueBinding) -> bool:
    """Whether a trace can record the binding's nodes as its launches."""
    from torch.cuda._host_trace_capture import reproducible

    kernels = [n for n in binding.nodes if isinstance(n, OpaqueKernel)]
    return all(reproducible(a) for k in kernels for a, _ in k.attributes)


def fill_uniform(t: torch.Tensor, low: float, high: float, generator: torch.Generator) -> None:
    # float8 has no uniform_: its values are float32's draws, rounded
    if t.dtype.itemsize == 1:
        t.copy_(torch.empty(t.shape, device=t.device).uniform_(low, high, generator=generator))
    else:
        t.uniform_(low, high, generator=generator)


def span_bytes(sizes: tuple[int, ...], strides: tuple[int, ...], itemsize: int) -> int:
    """A strided tensor's bytes from its first element through its last (0 when empty)."""
    return (1 + sum((n - 1) * s for n, s in zip(sizes, strides))) * itemsize if all(sizes) else 0


def aliasing(operands: Iterable[Any]) -> tuple[tuple[tuple[int, int], ...], list[Any]]:
    """Per traced operand on a storage an earlier operand is on: (operand, the
    storage's first operand), and its byte distance up from that one's
    address, symbolic. OpaqueKey.alias takes each storage's lowest operand at
    a call, from these distances there (_host_trace_lower_tape._opaque_key)."""
    from torch.cuda._host_trace_tape import _TracedTensor

    on: dict[int, list[tuple[int, Any]]] = {}
    for i, t in enumerate(operands):
        if isinstance(t, _TracedTensor):
            on.setdefault(id(t._root), []).append((i, t._sym_offset * t.element_size()))
    pairs, distances = [], []
    for (lead, low), *rest in on.values():
        for i, at in rest:
            pairs.append((i, lead))
            distances.append(at - low)
    return tuple(pairs), distances


def _key_values(op: OpOverload, args: tuple, kwargs: dict, fresh: list) -> tuple[list[Any], list[Any], tuple[tuple[int, int], ...]]:
    """A call's operands (its tensor leaves, then the fresh outputs), the
    values its key reads, symbolic, in _opaque_key's order (sizes, strides,
    address % 256, the scalar leaves, the alias distances), and its alias
    pairs (aliasing)."""
    from torch.cuda._host_trace_launch import _probe_address

    leaves = pytree.tree_leaves((args, kwargs))
    traced = [t for t in leaves if isinstance(t, torch.Tensor)]
    sizes = [tuple(t.shape) for t in (*traced, *fresh)]
    strides = [tuple(t._sym_strides) for t in traced] + [o.stride() for o in fresh]
    # an allocation's base is 256-byte aligned
    align = [_probe_address(t) % 256 for t in traced]
    align += [o.storage_offset() * o.element_size() % 256 for o in fresh]
    scalars = [v for v in leaves if not isinstance(v, torch.Tensor)]
    pairs, distances = aliasing(traced)
    return [*traced, *fresh], [*sum(sizes, ()), *sum(strides, ()), *align, *scalars, *distances], pairs


def trace_key(op: OpOverload, args: tuple, kwargs: dict, fresh: list, values: Callable[[Any], Any]) -> tuple[OpaqueKey, list[Any]]:
    """A call's key at the traced call, its values evaluated by `values` (the
    trace's CallValues) as a replay evaluates a bound site's rows; and the
    values it reads, symbolic (_key_values)."""
    from torch.cuda._host_trace_lower_tape import _opaque_key

    operands, symbolic, pairs = _key_values(op, args, kwargs, fresh)
    ranks = tuple(t.dim() for t in operands)
    head = 2 * sum(ranks) + len(operands)  # sizes, strides, align
    scalars = symbolic[head : len(symbolic) - len(pairs)]
    ints = [*symbolic[:head], *(v for v in scalars if isinstance(v, torch.SymInt)), *symbolic[len(symbolic) - len(pairs) :]]
    # a SymInt scalar takes its int from `ints`, as _opaque_key reads a row
    scalars = [values(v) if isinstance(v, (torch.SymFloat, torch.SymBool)) else v for v in scalars]
    device = operands[0].device.index
    key = _opaque_key(op, tuple(t.dtype for t in operands), ranks, tuple(scalars), device, library_state(), map(values, ints), pairs)
    return key, symbolic


def key_guards(call: Any, key: OpaqueKey) -> list[Any] | None:
    """That the opaque call `call`'s record is at `key`, as conditions in
    the tape's symbols (sympy): the values bind_at_trace guards at a refused
    key, and the alias distances. None where its key has other values."""
    import sympy

    from torch.cuda._host_trace_tape import _sym_expr

    leaves = pytree.tree_leaves((call.args, call.kwargs))
    fresh = [o for o in call.outputs if all(o is not t for t in leaves)]
    operands, values, pairs = _key_values(call.target, call.args, call.kwargs, fresh)
    values = values[: len(values) - len(pairs)]
    # the same operands share a storage as at the key, each distance up from its lowest there
    groups: dict[int, set[int]] = {}
    for i, lead in pairs:
        groups.setdefault(lead, {lead}).add(i)
    keyed: dict[int, set[int]] = {}
    for i, lead, _ in key.alias:
        keyed.setdefault(lead, {lead}).add(i)
    if sorted(map(sorted, groups.values())) != sorted(map(sorted, keyed.values())):
        return None
    at = [*sum(key.sizes, ()), *sum(key.strides, ()), *key.align, *key.scalars]
    if len(values) != len(at):
        return None
    offset = [t._sym_offset * t.element_size() for t in operands]
    guards = [sympy.Eq(_sym_expr(v), a) for v, a in zip(values, at) if isinstance(v, torch.SymInt)]
    return guards + [sympy.Eq(_sym_expr(offset[i] - offset[lead]), d) for i, lead, d in key.alias]


def bind_at_trace(
    provider: OpaqueProvider, op: OpOverload, args: tuple, kwargs: dict, fresh: list, values: Callable[[Any], Any]
) -> tuple[OpaqueBinding | None, list[Any], str | None]:
    """The binding of an accepted call at the traced call's key, if the
    trace can record it; the values to guard (each Eq to its value at the
    call): a refused key's (the call is a plain eager step, which no replay
    learns from); and why the key never binds, if it doesn't. The key is
    otherwise no guard: a bound call is a KeyedSite, its key's rows, the
    alias distances too, evaluated per call. A fresh output is an
    allocation: one at a storage offset other than 0 (a guard) does not
    bind. `fresh` are the fake kernel's fresh outputs, which the binding's
    allocations replace."""
    key, all_values = trace_key(op, args, kwargs, fresh, values)
    binding = provider.bind(key)
    refusal = provider.refusal(key)
    if binding is not None and not recordable(binding):
        refusal = "a kernel node of its binding is device-updatable"
    if refusal is not None:
        # a plain eager step at this key, or at every size of its layout: the
        # variant does not learn it (the alias distances where its provider keys them)
        why = f"{op} at sizes {key.sizes}: {refusal}"
        n = len(all_values) - len(key.alias)
        if provider.refuses_layout(key):
            guarded = all_values[n:] if op in provider.aliased else []
        else:
            guarded = all_values if op in provider.aliased else all_values[:n]
        return None, [v for v in guarded if isinstance(v, torch.SymInt)], why
    if binding is None or any(bool(o.storage_offset() != 0) for o in fresh):
        return None, [], None
    return binding, [], None


def record_binding(
    tr: _Trace,
    op: OpOverload,
    provider: OpaqueProvider,
    binding: OpaqueBinding,
    operands: list[Any],
    scalars: list[Any],
    call: tuple[Any, frozenset[int]] | None = None,
    key: OpaqueKey | None = None,
) -> KeyedSite:
    """A bound call's nodes as the trace's launches and memsets, its scratch
    buffers as allocations; `operands` are traced tensors."""
    from cuda.bindings import driver

    from torch.cuda._host_trace_launch import KernelLaunch
    from torch.cuda._host_trace_tape import Memset

    pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION

    empty = torch.ops.aten.empty.memory_format
    kw = {"dtype": torch.uint8, "device": tr.device}
    # eager's own buffers, at eager's sizes, read or not. A binding without one
    # has one of 0 bytes, which another key's may use.
    sizes = list(binding.scratch) or [0]
    scratch = {j: tr.allocate(empty, ([n],), kw) for j, n in enumerate(sizes)}
    buffers = {"operand": operands, "scratch": scratch}
    roots = tuple({id(t._root): t._root for t in (*operands, *scratch.values())}.values())

    def address(s: Slot) -> Any:
        # as the site's operand rows have it at every call: its root plus its offset (an empty one's too)
        t = buffers[s.kind][s.index]
        if torch.cuda._host_trace.bound_addresses_unguarded:
            return t._root.sym + t._sym_offset * t.element_size() + s.delta
        return t.data_ptr() + s.delta

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
    return KeyedSite(op, provider, tuple(operands), tuple(scalars), sized, tuple(nodes), binding.topology, binding.rng, library_state(), call, key)
