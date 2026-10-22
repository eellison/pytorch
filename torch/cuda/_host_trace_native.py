"""Host tracing (private): a variant's replay in C++, as
torch._C._HostTraceVariant.

The object is built once per variant from a VariantSpec of tuples of ints,
bytes, the segments' graphs and each eager step's callback (flatten_variant),
so the C++ never reads the lowering's dataclasses. A call runs entirely in
C++: evaluate the compiled program into rows; per step, allocate as _host_trace_memory
plans (from the caching allocator, in eager order or with a run buffer),
patch the kernel and memset nodes whose parameters differ from what their
exec holds and replay the run's graph, or call an eager step's operator
through the boxed dispatcher and check its outputs (a step it cannot express,
a Triton launch, calls its Python callback); drop what the plan drops; and
build the outputs. HostTraceReplay's base, torch._C._HostTraceEntry, calls it
without entering Python on a hit.

A launch is one record kind of three, the shape of a KernelLaunch, a
Memset and a Memcpy as lowered:
  (0, node, function, block rows, smem row, grid rows, per-parameter template
   images, fields ((param, offset, width, is pointer, row, base, delta),
   ...), descriptors ((param, first field, dtype, box, swizzle, fill, Triton
   element, library bits, L2 promotion), ...), philox fields ((param, offset, kind,
   delta), ...), philox increment, CPU scalars)
  (1, node, dst (row, base, delta), value, element size, width row, height
   row, pitch row)
  (2, node, dst (row, base, delta), src (row, base, delta), bytes row,
   cudaMemcpyKind, the pinned host argument's position or -1)
A field, dst or src with base -1 is its row's value plus delta; otherwise the
base's address plus the row's and delta. A 4-byte pointer field is the
address's low half; a 0-byte field only feeds a descriptor. A philox
field's kind indexes _PHILOX: its segment's captured seed or offset address,
or the call's intragraph offset.

A keyed site's nodes are records whose parameters its table selects by the
call's key, one hash lookup of the site's key rows. The table starts with
the traced key, the records' own; HostTraceReplay adds a key's binding as a
row once (binding_row), a kernel's with constant grid, a memset's with
constant shape.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.cuda._host_trace import declined
from torch.cuda._host_trace_lower_tape import (
    LoweredMemcpy,
    LoweredMemset,
    LoweredView,
    PredictedOutput,
    ScalarSlot,
)
from torch.cuda._host_trace_tape import HOST_SEED_OFFSET, MEMCPY_KINDS


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.cuda._host_trace_capture import CapturedTape
    from torch.cuda._host_trace_lower_tape import LoweredEagerCall, LoweredKeyedSite, LoweredLaunch
    from torch.cuda._host_trace_memory import MemoryPlan
    from torch.cuda._host_trace_opaque import OpaqueBinding, Slot


_RESULT_KINDS = {"none": 0, "tensor": 1, "tuple": 2, "list": 3}
_PHILOX = ("philox_seed", "philox_offset", "philox")
# _HostTraceVariant.call's outcomes
MISS, MISALIGNED, CAPTURE, HIT, MEMORY_KIND = range(5)


class VariantSpec(NamedTuple):
    """_HostTraceVariant's constructor argument. A base is allocation k or
    eager output len(allocations) + j; a base reference is (is argument,
    index)."""

    program: torch._C._HostTraceProgram
    valid: int  # the row that validates a call
    base_count: int
    segments: tuple  # per segment (graph, launch count, generator, seed, offset), an RNG one's generator
    launches: tuple
    allocations: tuple
    views: tuple
    steps: tuple
    memory: tuple  # the memory plan, per step and after the last
    outputs: tuple
    result_kind: int
    device: int
    disagreement: type[AssertionError]
    replay_hooks: tuple  # the global replay (start, end) hooks
    traced: tuple  # (rows, bases) at the traced call, which the nodes hold
    # per keyed site (key rows, record indices, scratch allocations), then per
    # selector ((), record indices, (), predicate row)
    sites: tuple
    argument_pairs: tuple  # Tape.argument_pairs
    # "held": (the buffer's bytes row, its (allocation, offset row,
    # tensor) triples), else ()
    planned: tuple = ()


def flatten_variant(
    captured: CapturedTape,
    traced: tuple[Sequence[int], Sequence[int]],
    memory: MemoryPlan,
    eager_step: Callable[[LoweredEagerCall], Callable[..., tuple]],
    disagreement: type[AssertionError],
) -> VariantSpec:
    """The variant's VariantSpec."""
    lowered = captured.lowered
    n_alloc = len(lowered.allocations)
    base_count = n_alloc + len(lowered.eager_roots)

    def ref(r: tuple[str, int]) -> tuple[bool, int]:
        kind, k = r
        return (kind == "argument", n_alloc + k if kind == "eager" else k)

    views: list[tuple] = []
    view_index: dict[int, int] = {}

    def view(v: LoweredView) -> int:
        if id(v) not in view_index:
            view_index[id(v)] = len(views)
            views.append((*ref(v.base), v.sizes, v.strides, v.offset, v.dtype))
        return view_index[id(v)]

    launches = [(row[0], c.node, *row[1:]) for c in captured.launches for row in [launch_row(c.launch)]]
    allocations = tuple(
        (a.sizes, a.strides, a.dtype, a.nbytes, a.arena) for a in lowered.allocations
    )
    steps: list[Any] = []
    segment = 0
    for i, step in enumerate(lowered.steps):
        if isinstance(step, range):
            steps.append(segment)
            segment += 1
            continue
        leaves = []
        for leaf in step.leaves:
            if isinstance(leaf, LoweredView):
                leaves.append((0, view(leaf)))
            elif isinstance(leaf, ScalarSlot):
                leaves.append((1, leaf.row))
            else:
                leaves.append((2, leaf))
        # an opaque call runs in Python, which binds its key and learns
        boxed = None if i in lowered.opaque else _boxed(step)
        steps.append((eager_step(step), tuple(leaves), boxed))
    plan = tuple((m.tensors, m.temporaries, m.drops, m.order, m.arguments) for m in memory.steps)
    outputs: list[tuple] = []
    for out in lowered.outputs:
        if isinstance(out, ScalarSlot):
            outputs.append((0, out.row))
        elif isinstance(out, LoweredView):
            outputs.append((1, view(out)))
        elif out[0] == "output":
            outputs.append((3, out[1]))
        else:
            outputs.append((2, *ref(out)))
    device = lowered.tape.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    graphs = torch.cuda.graphs
    return VariantSpec(
        lowered.compiled,
        lowered.valid,
        base_count,
        tuple((s.graph, len(s.launches), *(s.rng or (None, 0, 0))) for s in captured.segments),
        tuple(launches),
        allocations,
        tuple(views),
        tuple(steps),
        plan,
        tuple(outputs),
        _RESULT_KINDS[lowered.tape.result_kind],
        index,
        disagreement,
        (graphs._global_replay_start_hooks, graphs._global_replay_end_hooks),
        (tuple(traced[0]), tuple(traced[1])),
        (
            *((site.rows, site.nodes, tuple(site.scratch[j][1] for j in site.native_scratch())) for site in lowered.sites),
            *(((), s.nodes, (), s.predicate) for s in lowered.selectors),
        ),
        lowered.tape.argument_pairs,
        (memory.buffer, memory.planned) if memory.buffer >= 0 else (),
    )


def launch_row(lo: LoweredLaunch | LoweredMemset | LoweredMemcpy) -> tuple:
    """A launch as its record (the module docstring) without the node."""
    if isinstance(lo, LoweredMemset):
        m = lo.launch
        return (1, _source(lo.slots[0]), m.value, m.element_size, lo.width, lo.height, lo.pitch)
    if isinstance(lo, LoweredMemcpy):
        kind = lo.launch.kind
        host = -1 if kind == "D2D" else lo.slots[kind == "H2D"].root[1]
        return (2, *map(_source, lo.slots), lo.nbytes, MEMCPY_KINDS[kind], host)
    t = lo.launch
    fields = []
    for i, slot in enumerate(lo.slots):
        if t.fields is None:
            param, offset, width = i, 0, t.layout[i][1]
        elif i < len(t.fields):
            param, offset, width = t.fields[i]
        else:
            param, offset, width = 0, 0, 0
        fields.append((param, offset, width, not isinstance(slot, ScalarSlot), *_source(slot)))
    images = t.images or tuple(bytes(size) for _, size in t.layout)
    descriptors = tuple((d.param, d.first, d.dtype, d.box, d.swizzle, d.fill, d.triton_elem, d.edits, d.l2) for d in t.descriptors)
    philox = tuple((param, at, _PHILOX.index(kind), delta) for param, at, kind, delta in t.rng)
    return (0, t.function, lo.block, lo.smem, lo.grid, images, tuple(fields), descriptors, philox, t.rng_increment, t.cpu_scalars)


def _source(slot: Any) -> tuple[int, int, int]:
    """A slot's (row, base, delta)."""
    if isinstance(slot, ScalarSlot):
        return (slot.row, -1, 0)
    if slot.base is not None:
        return (slot.displacement, slot.base, 0)
    if slot.address is None:
        raise AssertionError(f"host_trace: pointer slot {slot} has no address")
    return (slot.address, -1, 0)


def binding_row(site: LoweredKeyedSite, binding: OpaqueBinding) -> tuple[tuple, list[int]]:
    """The binding's nodes as a site's table row, for add_row: records without
    the node, a kernel's also without descriptors and CPU scalars; and the
    bytes of each of the site's scratch buffers."""
    from torch.cuda._host_trace_opaque import OpaqueKernel, OpaqueMemset

    def source(s: Slot) -> tuple[int, int, int]:
        row, base = site.operands[s.index] if s.kind == "operand" else site.scratch[s.index]
        return (row, base, s.delta)

    nodes: list[tuple] = []
    # the call's philox offsets are taken at its first kernel
    first = next((n for n in binding.nodes if isinstance(n, OpaqueKernel)), None)
    for n in binding.nodes:
        if isinstance(n, OpaqueMemset):
            nodes.append((1, source(n.dst), n.value, n.element_size, n.width, n.height, n.pitch))
            continue
        images = n.images or tuple(bytes(size) for _, size in n.params)
        fields = tuple((s.param, s.offset, 8, True, *source(s)) for s in n.slots if s.kind not in _PHILOX)
        philox = tuple((s.param, s.offset, _PHILOX.index(s.kind), s.delta) for s in n.slots if s.kind in _PHILOX)
        increment = binding.rng_increment if n is first else 0
        nodes.append((0, n.function, n.block, n.smem, n.grid, images, fields, philox, increment))
    return tuple(nodes), [binding.scratch[j] if j < len(binding.scratch) else 0 for j in site.native_scratch()]


def binding_template(binding: OpaqueBinding) -> tuple | None:
    """The binding's nodes for a native binding table (HostTraceBindingTable.add): binding_row's, with each pointer
    field's source as (is scratch, index, delta) rather than a site's (row, base). None for a binding a table does not
    hold: one that draws from the generator, or that a trace could not record."""
    from torch.cuda._host_trace_opaque import OpaqueKernel, OpaqueMemset, recordable

    if binding.rng or not recordable(binding):
        return None

    def source(s: Slot) -> tuple[bool, int, int]:
        return (s.kind == "scratch", s.index, s.delta)

    nodes: list[tuple] = []
    for n in binding.nodes:
        if isinstance(n, OpaqueMemset):
            nodes.append((1, source(n.dst), n.value, n.element_size, n.width, n.height, n.pitch))
            continue
        if not isinstance(n, OpaqueKernel) or any(s.kind not in ("operand", "scratch") for s in n.slots):
            return None
        images = n.images or tuple(bytes(size) for _, size in n.params)
        nodes.append((0, n.function, n.block, n.smem, n.grid, images, tuple((s.param, s.offset, *source(s)) for s in n.slots)))
    return tuple(nodes)


# topologies (OpaqueBinding.topology, KeyedSite.topology) as ids: a table's binding fills a site's row of arm 0 when
# its topology's id is the site's
_TOPOLOGIES: dict[tuple, int] = {}


def topology_id(topology: tuple) -> int:
    return _TOPOLOGIES.setdefault(topology, len(_TOPOLOGIES))


class NativeBindings:
    """A provider's held bindings as native tables, one per class of keyed site (op, dtypes, ranks, constant
    scalars, device, library state), keyed by a site key in the provider's normal form: the C++ entry fills a site's
    row at a key it lacks from its table (HostTraceVariant.fill_from_table), with no Python on that call.

    `parts(k)` takes a key of the provider's bindings (its own normal form) to (op, dtypes, sizes, strides, align,
    scalars, device, state) as bind() compares them, or None for a key no table holds; `normal_op` takes a site's op
    to the op its keys' parts have; `align_class` says whether its
    align is _host_trace_harvest._class of the address % 256 (else the value itself), and `size1_strides(op)` whether
    its strides are 0 where the size is 1. A table key is the sizes, strides and align, flattened, then the values of
    the site's symbolic scalars: equal for two keys exactly when bind() takes them to the same binding."""

    def __init__(
        self, parts: Callable[[Any], tuple | None], normal_op: Callable[[Any], Any], align_class: bool, size1_strides: Callable[[Any], bool]
    ) -> None:
        self.parts = parts
        self.normal_op = normal_op
        self.align_class = align_class
        self.size1_strides = size1_strides
        self.tables: dict[tuple, Any] = {}  # site class -> torch._C._HostTraceBindingTable
        self.held: dict[Any, tuple[tuple, Any]] = {}  # binding key -> (its parts, the binding: keeps its kernels loaded)

    @staticmethod
    def _class(site: LoweredKeyedSite) -> tuple:
        slots = tuple(isinstance(v, (torch.SymInt, ScalarSlot)) for v in site.site.scalars)
        constants = tuple(None if slot else v for slot, v in zip(slots, site.site.scalars))
        return (site.site.op, site.dtypes, site.ranks, slots, constants, site.device, site.site.state)

    def _key(self, cls: tuple, parts: tuple) -> list[int] | None:
        op, dtypes, ranks, slots, constants, device, state = cls
        p_op, p_dtypes, sizes, strides, align, scalars, p_device, p_state = parts
        if (p_op, p_dtypes, p_device, p_state) != (self.normal_op(op), dtypes, device, state) or tuple(map(len, sizes)) != ranks:
            return None
        if len(scalars) != len(slots) or any(not slot and v != c for slot, v, c in zip(slots, scalars, constants)):
            return None
        values = [v for slot, v in zip(slots, scalars) if slot]
        if not all(type(v) is int for v in values):
            return None
        return [*(n for z in sizes for n in z), *(st for t in strides for st in t), *align, *values]

    def add(self, k: Any, binding: Any) -> None:
        """A binding the provider holds at its key k (bind's normal form)."""
        parts = self.parts(k)
        nodes = None if parts is None else binding_template(binding)
        if nodes is None:
            return
        self.held[k] = (parts, binding)
        for cls, table in self.tables.items():
            self._add(table, cls, parts, nodes, binding)

    def _add(self, table: Any, cls: tuple, parts: tuple, nodes: tuple, binding: Any) -> None:
        key = self._key(cls, parts)
        if key is not None:
            try:
                topology = topology_id(binding.topology)
            except TypeError:  # an unhashable launch attribute: no table holds it
                return
            table.add(key, nodes, list(binding.scratch), topology)

    def link(self, native: Any, i: int, site: LoweredKeyedSite, planned_scratch: bool) -> None:
        """Links the variant's keyed site i to its class's table (HostTraceVariant.set_site_table)."""
        if site.site.rng or site.alias or not hasattr(native, "set_site_table"):
            return
        try:
            cls = self._class(site)
            hash(cls)
            topology = topology_id(site.site.topology)
        except TypeError:  # an unhashable constant or attribute
            return
        table = self.tables.get(cls)
        if table is None:
            table = self.tables[cls] = torch._C._HostTraceBindingTable()
            for parts, binding in self.held.values():
                nodes = binding_template(binding)
                if nodes is not None:
                    self._add(table, cls, parts, nodes, binding)
        ranks = site.ranks
        operands = sum(ranks)
        codes = [0] * operands + [2 if self.size1_strides(site.site.op) else 0] * operands + [1 if self.align_class else 0] * len(ranks)
        companions = list(range(operands)) + list(range(operands)) + [0] * len(ranks)
        slots = sum(isinstance(v, (torch.SymInt, ScalarSlot)) for v in site.site.scalars)
        codes += [0] * slots
        companions += [0] * slots
        if len(codes) != len(site.rows):
            return
        if not self.size1_strides(site.site.op):
            companions = [0] * len(codes)
        scratch = [(j, *site.scratch[j], site.site.scratch[j][1] if planned_scratch else -1) for j in site.site.scratch if j in site.scratch]
        native.set_site_table(i, table, codes, companions, list(site.operands), scratch, site.native_scratch(), topology)


class _Leaf:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


def _boxed(step: LoweredEagerCall) -> tuple | None:
    """The step's call for the boxed dispatcher: (schema name, overload name,
    per schema argument (0, constant) | (1, leaf) | (2, ((is leaf, leaf or
    constant), ...)) | (3,) its default, per returned tensor (True, root,
    sizes, strides, offset, dtype) | (False, the leaf it is), the operator,
    the library_state() it runs under or None, its graphsafe RNG generator
    state or None, whether its CPU outputs go to the device
    (seed_offset_on_device)); None for a target without a schema. The C++
    declines what else it cannot express."""
    target = step.call.target
    if not isinstance(target, torch._ops.OpOverload):
        return None
    markers = [
        _Leaf(i) if isinstance(v, (LoweredView, ScalarSlot)) else v
        for i, v in enumerate(step.leaves)
    ]
    if step.flat:
        call_args, call_kwargs = markers, {}
    else:
        call_args, call_kwargs = pytree.tree_unflatten(markers, step.spec)
    schema = target._schema
    names = {a.name for a in schema.arguments}
    if any(k not in names for k in call_kwargs):
        return None

    def item(v: Any) -> tuple:
        return (True, v.index) if isinstance(v, _Leaf) else (False, v)

    args: list[tuple] = []
    for i, a in enumerate(schema.arguments):
        if not a.kwarg_only and i < len(call_args):
            v = call_args[i]
        elif a.name in call_kwargs:
            v = call_kwargs[a.name]
        else:
            args.append((3,))
            continue
        if isinstance(v, _Leaf):
            args.append((1, v.index))
        elif isinstance(v, (list, tuple)) and any(isinstance(x, _Leaf) for x in v):
            args.append((2, tuple(item(x) for x in v)))
        else:
            args.append((0, v))
    if len(call_args) > sum(not a.kwarg_only for a in schema.arguments):
        return None
    outputs = tuple(
        (True, p.root, p.sizes, p.strides, p.offset, p.dtype)
        if isinstance(p, PredictedOutput)
        else (False, p)
        for p in step.outputs
    )
    return (schema.name, schema.overload_name, tuple(args), outputs, target, step.call.state or None, step.call.generator, target in HOST_SEED_OFFSET)


def split_cuts(lowered: Any) -> list[tuple[int, list[int]]]:
    """Per run (native segment) of the lowered tape, the launch offsets a
    sub-exec may start at (the executor-level split, torch._C._host_trace_split):
    no programmatic edge into the launch, and not inside a keyed site's or a
    selector's node chain or an unjoined fork (_run_starts)."""
    from torch.cuda._host_trace_memory import _run_starts

    inside = set(_run_starts(lowered))
    for selector in lowered.selectors:
        if selector.nodes:
            inside.update(range(min(selector.nodes) + 1, max(selector.nodes) + 1))
    out = []
    for r in (s for s in lowered.steps if isinstance(s, range)):
        cuts = [c - r.start for c in range(r.start + 1, r.stop) if c not in inside and not getattr(lowered.launches[c].launch, "programmatic", False)]
        out.append((len(out), cuts))
    return out


def native_variant(spec: VariantSpec) -> torch._C._HostTraceVariant:
    """The variant's native object. A malformed spec is the lowering's bug:
    the suites raise it (raise_unexpected), a user's call declines, noted; a
    driver without the optional entries it needs (TMA descriptors, kernel node
    updates) declines."""
    try:
        return torch._C._HostTraceVariant(spec)
    except NotImplementedError as e:
        raise declined(str(e)) from e
    except (TypeError, ValueError, IndexError, RuntimeError) as e:
        if torch.cuda._host_trace.raise_unexpected:
            raise AssertionError(f"host_trace: the native variant rejects: {e}") from e
        raise declined(f"the native variant rejects the lowering ({e})") from e
