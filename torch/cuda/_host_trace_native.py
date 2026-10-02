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
   ...), descriptors ((param, first field, dtype, box, swizzle), ...),
   philox fields ((param, offset, kind, delta), ...), philox increment)
  (1, node, dst (row, base, delta), value, element size, width row, height
   row, pitch row)
  (2, node, dst (row, base, delta), src (row, base, delta), bytes row)
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
from torch.cuda._host_trace_tape import HOST_SEED_OFFSET


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.cuda._host_trace_capture import CapturedTape
    from torch.cuda._host_trace_lower_tape import LoweredEagerCall, LoweredKeyedSite, LoweredLaunch
    from torch.cuda._host_trace_memory import MemoryPlan
    from torch.cuda._host_trace_opaque import OpaqueBinding, Slot


_RESULT_KINDS = {"none": 0, "tensor": 1, "tuple": 2, "list": 3}
_PHILOX = ("philox_seed", "philox_offset", "philox")
# _HostTraceVariant.call's outcomes
MISS, MISALIGNED, CAPTURE, HIT = range(4)


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
        (a.sizes, a.strides, a.dtype, a.nbytes) for a in lowered.allocations
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
            *((site.rows, site.nodes, tuple(site.scratch[j][1] for j in sorted(site.scratch))) for site in lowered.sites),
            *(((), s.nodes, (), s.predicate) for s in lowered.selectors),
        ),
        lowered.tape.argument_pairs,
    )


def launch_row(lo: LoweredLaunch | LoweredMemset | LoweredMemcpy) -> tuple:
    """A launch as rows: (0, function, block, smem, grid, images, fields,
    descriptors, philox fields, philox increment, CPU scalars) for a kernel, (1, dst,
    value, element size, width, height, pitch) for a memset, (2, dst, src,
    bytes) for a memcpy."""
    if isinstance(lo, LoweredMemset):
        m = lo.launch
        return (1, _source(lo.slots[0]), m.value, m.element_size, lo.width, lo.height, lo.pitch)
    if isinstance(lo, LoweredMemcpy):
        return (2, *map(_source, lo.slots), lo.nbytes)
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
    descriptors = tuple((d.param, d.first, d.dtype, d.box, d.swizzle) for d in t.descriptors)
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
    """The binding's nodes as a site's table row, for add_row:
    (0, function, block, smem, grid, images, fields, philox fields, philox
    increment) per kernel and (1, dst, value, element size, width, height,
    pitch) per memset; and the bytes of each of the site's scratch buffers."""
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
    used = binding.used_scratch
    return tuple(nodes), [binding.scratch[j] if j in used else 0 for j in sorted(site.scratch)]


class _Leaf:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


def _boxed(step: LoweredEagerCall) -> tuple | None:
    """The step's call for the boxed dispatcher: (schema name, overload name,
    per schema argument (0, constant) | (1, leaf) | (2, ((is leaf, leaf or
    constant), ...)) | (3,) its default, per returned tensor (True, root,
    sizes, strides, offset, dtype) | (False, the leaf it is), the operator,
    the library_state() it runs under or None); None for a target without a
    schema. The C++ declines what else it cannot
    express."""
    target = step.call.target
    # a graphsafe RNG step swaps in its generator's state around the call, which the boxed call cannot
    if not isinstance(target, torch._ops.OpOverload) or step.call.generator is not None:
        return None
    # its outputs go to the device in Python (seed_offset_on_device)
    if target in HOST_SEED_OFFSET:
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
    return (schema.name, schema.overload_name, tuple(args), outputs, target, step.call.state or None)


def native_variant(spec: VariantSpec) -> torch._C._HostTraceVariant:
    """The variant's native object. A malformed spec is the lowering's bug,
    raised as one rather than declined; a driver without the optional entries
    it needs (TMA descriptors, kernel node updates) declines."""
    try:
        return torch._C._HostTraceVariant(spec)
    except NotImplementedError as e:
        raise declined(str(e)) from e
    except (TypeError, ValueError, IndexError, RuntimeError) as e:
        raise AssertionError(f"host_trace: the native variant rejects: {e}") from e
