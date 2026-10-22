"""Host tracing (private): a Tape lowered onto one integer program, compiled
once.

Every value a replay needs is a row of that program over the call's inputs:
the tape's guards and each input source's declared sign,
each allocation's sizes, strides and storage bytes, and each launch's grid
and parameter slots. A pointer slot is a root plus a byte displacement. An
argument's root is its storage base, read from the call (its data pointer
less its storage offset), so the slot's full address is a row too. An
allocation's root is the address the replay allocates, which no row holds:
its symbol stays out of the program, and a guard over it declines.
An eager call's fresh output is a root of the same kind, its address known
only once the op has run at a replay.

The records form `steps`: maximal runs of launches (one graph each) between
the eager calls, which a replay runs itself. An eager call's tensor
arguments are views of their roots, its SymInt arguments rows, and each
fresh output a prediction (its fake kernel's sizes, strides and offset) the
replay checks the op's actual output against. An opaque call, an eager call
a provider accepted, also has its key's rows (LoweredOpaqueCall).

One evaluation of the compiled program at a call (`LoweredTape.evaluate`)
is the pure validation: it either misses or yields every value the replay
patches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import functools
import math

import sympy

import torch
from torch.cuda._host_trace import Declined, declined
from torch.cuda._host_trace_launch import compiled_cluster, KernelLaunch
from torch.cuda._host_trace_ir import _record_key, Env, Node
from torch.cuda._host_trace_lower import IRLowering, Lowering
from torch.cuda._host_trace_opaque import aliasing, OpaqueKey
from torch.cuda._host_trace_program import (
    compile_program,
    IntegerProgram,
    OutOfDomain,
    Status,
)
from torch.cuda._host_trace_tape import (
    _IntOutputRec,
    _sym_expr,
    _sym_key,
    _TracedTensor,
    EagerCall,
    Memcpy,
    Memset,
    OpaqueCall,
)
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch._ops import OpOverload
    from torch.cuda._host_trace_opaque import KeyedSite, OpaqueProvider
    from torch.cuda._host_trace_tape import _AllocRec, _Root, Tape


# a root or an output: ("argument", position), ("allocation", k), ("eager",
# j) for eager output j or, for an output that is an earlier output as the
# host returned it, ("output", k)
Ref = tuple[str, int]


@dataclass(frozen=True)
class ScalarSlot:
    row: int


@dataclass(frozen=True)
class PointerSlot:
    root: Ref
    displacement: int  # row: bytes from the root's storage base
    address: int | None  # row: the full address, for an argument's root
    # otherwise the root's index in a replay's bases: allocation k, then
    # eager output j at len(allocations) + j
    base: int | None


@dataclass(frozen=True)
class LoweredLaunch:
    seq: int
    launch: KernelLaunch  # its function and layout
    slots: tuple[ScalarSlot | PointerSlot, ...]  # in the launch's slot order
    grid: tuple[int, int, int]  # rows
    block: tuple[int, int, int]  # rows
    smem: int  # row


@dataclass(frozen=True)
class LoweredMemset:
    seq: int
    launch: Memset
    slots: tuple[PointerSlot]  # the destination
    width: int  # the row of the width in elements
    height: int  # the row of the height
    pitch: int  # the row of the pitch in bytes


@dataclass(frozen=True)
class LoweredMemcpy:
    seq: int
    launch: Memcpy
    slots: tuple[PointerSlot, PointerSlot]  # the destination, the source
    nbytes: int  # row


@dataclass(frozen=True)
class LoweredAllocation:
    seq: int
    dtype: torch.dtype
    sizes: tuple[int, ...]  # rows
    strides: tuple[int, ...]  # rows
    nbytes: int  # row: the storage bytes, as computeStorageNbytes
    arena: bool = False  # a planned arena (memory.relocate); never a keyed site's scratch


@dataclass(frozen=True)
class LoweredView:
    """A tensor over an argument's, allocation's or eager output's storage."""

    base: Ref
    sizes: tuple[int, ...]  # rows
    strides: tuple[int, ...]  # rows
    offset: int  # row: the storage offset
    # set when the view reinterprets its base's storage (view_as_complex)
    dtype: torch.dtype | None = None


@dataclass(frozen=True)
class PredictedOutput:
    """An eager call's fresh output: eager output `root`'s storage."""

    root: int
    sizes: tuple[int, ...]  # rows
    strides: tuple[int, ...]  # rows
    offset: int  # row
    dtype: torch.dtype


@dataclass(frozen=True)
class LoweredEagerCall:
    seq: int
    call: EagerCall
    spec: pytree.TreeSpec  # of (args, kwargs)
    flat: bool  # the args are the leaves, and there are no kwargs
    # its leaves: a LoweredView per tensor, a ScalarSlot per SymInt, constants
    leaves: tuple[Any, ...]
    # per returned tensor, a prediction, or the index of the leaf it is
    outputs: tuple[PredictedOutput | int, ...]
    grid: tuple[int, int, int] | None  # rows, for a Triton target


def _opaque_key(
    op: OpOverload, dtypes: tuple, ranks: tuple[int, ...], scalars: tuple, device: int, state: tuple, ints: Iterable[int], alias: tuple
) -> OpaqueKey:
    # ints: per operand its sizes, then per operand its strides, then per
    # operand its address % 256, then a value per symbolic scalar, then per
    # pair of `alias` (aliasing's: an operand and its storage's first) its distance at the call
    it = iter(ints)
    sizes = tuple(tuple(next(it) for _ in range(r)) for r in ranks)
    strides = tuple(tuple(next(it) for _ in range(r)) for r in ranks)
    align = tuple(next(it) for _ in ranks)
    scalars = tuple(next(it) if isinstance(v, (torch.SymInt, ScalarSlot)) else v for v in scalars)
    # each storage's operands at the call, up from its lowest as aliasing() has them
    groups: dict[int, dict[int, int]] = {}
    for (i, lead), d in zip(alias, it, strict=True):
        groups.setdefault(lead, {lead: 0})[i] = d
    pairs = []
    for at in groups.values():
        low, lead = min((d, i) for i, d in at.items())
        pairs += [(i, lead, d - low) for i, d in at.items() if i != lead]
    return OpaqueKey(op, dtypes, sizes, strides, align, scalars, device, state, tuple(sorted(pairs)))


@dataclass(frozen=True)
class LoweredOpaqueCall:
    """An OpaqueCall's key as rows (_host_trace_opaque)."""

    op: OpOverload
    provider: OpaqueProvider
    dtypes: tuple[torch.dtype, ...]
    ranks: tuple[int, ...]
    rows: tuple[int, ...]  # in _opaque_key's order
    scalars: tuple[Any, ...]  # a ScalarSlot per SymInt, constants
    device: int
    state: tuple
    alias: tuple = ()  # aliasing's pairs; their distances' rows end `rows`

    def key(self, values: Sequence[int]) -> OpaqueKey:
        ints = (values[r] for r in self.rows)
        return _opaque_key(self.op, self.dtypes, self.ranks, self.scalars, self.device, self.state, ints, self.alias)


@dataclass(frozen=True)
class LoweredKeyedSite:
    """A KeyedSite's key as rows, and its operands' and scratch buffers'
    addresses as (row, base): a base's address plus the row's value, or with
    base -1 the row's value."""

    site: KeyedSite
    dtypes: tuple[torch.dtype, ...]
    ranks: tuple[int, ...]
    rows: tuple[int, ...]  # in _opaque_key's order
    operands: tuple[tuple[int, int], ...]
    scratch: dict[int, tuple[int, int]]  # buffer j's
    nodes: tuple[int, ...]  # indices in `launches`
    device: int
    alias: tuple = ()  # aliasing's pairs; their distances' rows end `rows`
    arena_scratch: frozenset[int] = frozenset()  # buffers in an arena at their traced bytes

    def native_scratch(self) -> list[int]:
        """The scratch buffers native add_row sizes per key."""
        return [j for j in sorted(self.scratch) if j not in self.arena_scratch]

    def key(self, ints: Sequence[int]) -> OpaqueKey:
        return _opaque_key(self.site.op, self.dtypes, self.ranks, self.site.scalars, self.device, self.site.state, ints, self.alias)


@dataclass(frozen=True)
class LoweredSelector:
    """A top-level op's own guards (OpRec.guards), which select the nodes its
    launches run: its trace's, or a folded trace's (_host_trace_replay)."""

    op: int  # in Tape.ops
    predicate: int  # row
    nodes: tuple[int, ...]  # indices in `launches`
    rows: tuple[int, ...] = ()  # per own guard (OpRec.guards), its row: which fails at a call (retrace_causes)


@dataclass(frozen=True)
class LoweredSection:
    """A redispatch entry in its op's place (a graph piece): per temporary
    (its base past the tape's, its bytes' row), and the op's allocations it
    does not make."""

    temporaries: tuple[tuple[int, int], ...]
    drops: tuple[int, ...]


@dataclass(frozen=True)
class StreamOrder:
    """Where a tape's launches ran: on the trace's stream, or on side streams
    forked from it in the trace's capture and joined back (_Trace._mark)."""

    stream: tuple[int, ...]  # per launch (LoweredTape.launches), its logical stream, 0 the trace's
    # per launch, the launches on other streams it directly follows (those of earlier steps left out)
    after: tuple[tuple[int, ...], ...]
    # per launch, the seqs it may run between: on the trace's stream its own; on a side stream, from its latest
    # ancestor on the trace's stream (-1: none) to its earliest descendant there (inf: none, joined at the end)
    span: tuple[tuple[float, float], ...]
    priorities: tuple[int, ...]  # per logical stream


def _stream_order(tape: Tape, launches: Sequence[LoweredLaunch | LoweredMemset | LoweredMemcpy]) -> StreamOrder:
    """The tape's StreamOrder. A step boundary (an eager call) while a side
    stream's work is not joined declines: the run before it would end with
    the fork still open."""
    records = [(seq, rec) for seq, rec in tape.launches if not (isinstance(rec, EagerCall) and rec.host)]
    streams, after = tape.streams, tape.after
    children: dict[int, list[int]] = {}
    for seq, preds in after.items():
        for p in preds:
            children.setdefault(p, []).append(seq)
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    prev: dict[int, float] = {}
    for seq, _ in records:
        k = streams.get(seq, 0)
        first[seq] = prev[k] = max([seq if k == 0 else prev.get(k, -1), *(first[p] for p in after.get(seq, ()) if p in first)])
    prev = {}
    for seq, _ in reversed(records):
        k = streams.get(seq, 0)
        last[seq] = prev[k] = min([seq if k == 0 else prev.get(k, math.inf), *(last[c] for c in children.get(seq, ()) if c in last)])
    pending = -math.inf
    for seq, rec in records:
        if streams.get(seq, 0) and isinstance(rec, EagerCall):
            raise declined(f"{rec.name} on a side stream runs eagerly (its key does not bind); an eager step there is not supported", side_stream=True)
        if streams.get(seq, 0):
            pending = max(pending, last[seq])
        elif isinstance(rec, EagerCall) and pending > seq:
            raise declined(f"{rec.name} runs eagerly while a side stream's work is not joined; a step boundary inside a fork is not supported", side_stream=True)
    index = {lo.seq: j for j, lo in enumerate(launches)}
    return StreamOrder(
        tuple(streams.get(lo.seq, 0) for lo in launches),
        tuple(tuple(sorted(index[p] for p in after.get(lo.seq, ()) if p in index)) for lo in launches),
        tuple((first[lo.seq], last[lo.seq]) for lo in launches),
        (0, *tape.priorities),
    )


def _stream_layout(tape: Tape) -> list[tuple[int, tuple[int, ...]]] | None:
    """Per record, its logical stream and the records (by position) it follows on other streams."""
    if not tape.streams:
        return None
    position = {seq: i for i, (seq, _) in enumerate(tape.launches)}
    return [(tape.streams.get(seq, 0), tuple(position[p] for p in tape.after.get(seq, ()))) for seq, _ in tape.launches]


@dataclass(frozen=True)
class LoweredTape:
    tape: Tape
    program: IntegerProgram  # complete: no row is added after compilation
    compiled: torch._C._HostTraceProgram
    valid: int  # row: the guards and every allocation's requirement hold
    allocations: tuple[LoweredAllocation, ...]
    launches: tuple[LoweredLaunch | LoweredMemset | LoweredMemcpy, ...]
    # a range of `launches` (one graph) or an eager call, in program order
    steps: tuple[range | LoweredEagerCall, ...]
    eager_roots: tuple[_Root, ...]  # eager output j's root
    # a Ref for the very object (an argument, an earlier output, a whole
    # allocation or eager output), otherwise a view; an int output's row
    outputs: tuple[Ref | LoweredView | ScalarSlot, ...]
    opaque: dict[int, LoweredOpaqueCall]  # by index in `steps`
    sites: tuple[LoweredKeyedSite, ...]
    selectors: tuple[LoweredSelector, ...]
    # the program's lowering, which a fold extends
    lowering: _TapeLowering
    streams: StreamOrder | None = None  # None: every launch on the trace's stream

    def span(self, j: int) -> tuple[float, float]:
        """The seqs launch j may run between in its run (StreamOrder.span): a
        side stream's launch from its fork to its join, or the run's ends."""
        if self.streams is None:
            return self.launches[j].seq, self.launches[j].seq
        start, stop = self._run_of[j]
        first, last = self.streams.span[j]
        return max(first, self.launches[start].seq), min(last, self.launches[stop - 1].seq)

    @functools.cached_property
    def _run_of(self) -> dict[int, tuple[int, int]]:
        return {j: (r.start, r.stop) for r in self.steps if isinstance(r, range) for j in r}

    def evaluate(self, args: Sequence[Any]) -> tuple[int, ...] | None:
        """Every row's value at the call `args`, or None when the call misses:
        an input of another kind than the trace's, a leaf outside int64 (a
        pointer at or above 2**63), a failing status, or a guard or
        requirement that does not hold. The argument contract (arity,
        constants, dtypes, ranks, devices) is checked before this."""
        result = self.compiled.evaluate_inputs(args)
        if result is None:
            return None
        status, values = result
        if status != Status.SUCCESS or values[self.valid] != 1:
            return None
        if any(values[s.predicate] != 1 for s in self.selectors):
            return None
        return values


class _TapeLowering:
    def __init__(self, tape: Tape, direct: bool = True) -> None:
        self.tape = tape
        self.program = IntegerProgram(tape.args)
        self.roots: dict[int, Ref] = {}
        sources: dict[Any, tuple | int] = {}
        # an IR trace lowers from its nodes unless `direct` is off, which
        # lowers their sympy export (the oracle's reference)
        env = tape.shape_env
        self.ir = direct and isinstance(env, Env)
        self.key = _sym_key if self.ir else _sym_expr
        self.lowering = IRLowering(self.program, sources, env.ctx) if self.ir else Lowering(self.program, sources)
        self._leaves: list[tuple[torch.SymInt, tuple | int]] = []
        # per fold or redispatch entry its (selector site, guards in the tape's
        # symbols: Env records in its context where it lowers IR (IRRename), else sympy)
        self.entry_guards: list[tuple[int, tuple[Any, ...]]] = []
        # selectors (native site indices) whose redispatch refused at some call
        self.refused: set[int] = set()
        # per op index (a library body's), why its redo is refused or None: its records' scalar reads, then its
        # confirmation at the trace call (_host_trace_redispatch._library_redo_refusal)
        self.library_redo: dict[int, str | None] = {}
        # the library-body ops confirmed by their own confirmation, by redo template key (_reads): (op, its read-walk roots)
        self.confirmed: dict[Any, list[tuple[int, list[Any]]]] = {}
        # the ops whose redo_context read RedoCall.rank: their redos are their own (no template, dispatch or shared confirmation)
        self.rank_read: set[int] = set()
        # library-body checks, once per lowering: the tape's tensor-layout symbols (_tensor_symbols) and the rows' values at
        # the trace call (_trace_values)
        self.tensor_symbols: set[str] | None = None
        self.trace_values: tuple[list[int], Any] | None = None  # (values, the last row they cover)
        # redo_context (_host_trace_redispatch._reads_lifted): the lifted arguments' symbols, and per op whether it reads one
        self.lifted_symbols: set[str] | None = None
        self.reads_lifted: dict[int, bool] = {}
        # the section temporaries' bases the native variant holds past the tape's (_add_entries counts them)
        self.sections = 0
        # per section temporary the native places in its run's buffer (_host_trace_memory.section_seqs), (its run's
        # step, the seq it is made at, the seq of its last use)
        self.placed_sections: dict[int, tuple[int, int, int]] = {}
        # per step, (its plan StepMemory, its events' sorted keys) for _host_trace_memory.section_places
        self.section_keys: dict[int, tuple[Any, list]] = {}
        # check_section's: per lowering, _first_seqs (keyed by the allocation and launch counts); per plan, _liveness;
        # per step, the run's own lifetimes (_lifetimes) with the plan they are of
        self.first_seqs: tuple[tuple[int, int], list[int]] | None = None
        self.liveness: tuple[Any, Any] | None = None
        self.run_lifetimes: dict[int, tuple[Any, dict]] = {}
        # _for_instance's: per static-prefix argument record (by id), its layout pins and their conjunction's row
        self.root_pins: dict[int, list] = {}
        self.root_pin_rows: dict[int, int | None] = {}
        # redo templates (_host_trace_redispatch): an entry by what its op's redo reads, rows of this program; and the
        # entries instantiated from them and redone in full
        self.templates: dict[Any, Any] = {}
        # per op (index in Tape.ops), what its redo reads (_reads): a function of the variant's tape, made once
        self.reads: dict[int, Any] = {}
        # per static prefix length, its layouts' symbols; per (template, an op's read roots), that op's template or None
        self.declared: dict[int, frozenset] = {}
        self.instance_templates: dict[tuple, Any] = {}
        self.instances = 0
        self.redos = 0

        def source(v: Any, leaf: tuple | int) -> None:
            k = self.key(v)
            if isinstance(k, sympy.Symbol) or (isinstance(k, Node) and k.op == "sym"):
                sources[k] = leaf
                self._leaves.append((v, leaf))

        for rec in tape.inputs:
            i = rec.position
            self.roots[id(rec.root)] = ("argument", i)
            for d, (size, stride) in enumerate(zip(rec.sizes, rec.strides)):
                source(size, ("size", i, d))
                source(stride, ("stride", i, d))
            source(rec.offset, ("storage_offset", i))
            offset_bytes = self.emit(
                "multiply",
                self.emit("storage_offset", i),
                self.emit("constant", -rec.dtype.itemsize),
            )
            base = self.emit("add", self.emit("pointer", i), offset_bytes)
            source(rec.root.sym, base)
        for rec in tape.int_inputs:
            source(rec.sym, ("boxed", rec.position))
        self.dtypes = {id(rec.root): rec.dtype for rec in (*tape.inputs, *tape.allocs)}
        for k, rec in enumerate(tape.allocs):
            self.roots[id(rec.root)] = ("allocation", k)
        # eager output j, as its call returned it
        self.eager: list[_TracedTensor] = []
        for _, rec in tape.launches:
            for t in rec.outputs if isinstance(rec, EagerCall) else ():
                if t._root.kind == "eager" and id(t._root) not in self.roots:
                    self.roots[id(t._root)] = ("eager", len(self.eager))
                    self.dtypes[id(t._root)] = t.dtype
                    self.eager.append(t)
        self.requirements: list[int] = []

    @functools.cached_property
    def symbolic(self) -> Lowering:
        """The lowering of sympy expressions over the same program: a twin's
        key guards (Tape.twin_guards), in sympy."""
        if not self.ir:
            return self.lowering
        return Lowering(self.program, {_sym_expr(v): leaf for v, leaf in self._leaves})

    @functools.cached_property
    def addresses(self) -> frozenset:
        """The arguments' address symbols, as a conjunction takes them (made
        once: a redispatch's entries and instances each take them)."""
        return frozenset(self.key(rec.root.sym) for rec in self.tape.inputs)

    @functools.cached_property
    def first_inputs(self) -> dict[int, Any]:
        """Per root (by id), its first argument, as _redo's root() takes it."""
        return {id(rec.root): rec for rec in reversed(self.tape.inputs)}

    def emit(self, op: str, *operands: int) -> int:
        try:
            return self.program.emit(op, *operands)
        except OutOfDomain as e:
            raise declined(str(e)) from e

    def row(self, v: Any) -> int:
        if isinstance(v, sympy.Basic):
            return self.symbolic.lower(v)
        return self.lowering.lower(self.key(v))

    def allocation(self, rec: _AllocRec) -> LoweredAllocation:
        sizes = tuple(self.row(s) for s in rec.sizes)
        strides = tuple(self.row(s) for s in rec.strides)
        one, zero = self.emit("constant", 1), self.emit("constant", 0)
        # an empty allocation has no bytes whatever its strides; otherwise
        # every stride is nonnegative, as at::empty_strided requires
        empty, nonnegative, extent = zero, one, one
        for size, stride in zip(sizes, strides):
            empty = self.emit("select", self.emit("eq", size, zero), one, empty)
            stride_ok = self.emit("ge", stride, zero)
            nonnegative = self.emit("and", nonnegative, stride_ok)
            last = self.emit("add", size, self.emit("constant", -1))
            extent = self.emit("add", extent, self.emit("multiply", last, stride))
        itemsize = self.emit("constant", rec.dtype.itemsize)
        total = self.emit("multiply", extent, itemsize)
        nbytes = self.emit("select", empty, zero, total)
        self.requirements.append(self.emit("select", empty, one, nonnegative))
        return LoweredAllocation(rec.seq, rec.dtype, sizes, strides, nbytes)

    def pointer(self, value: Any, launch: KernelLaunch | Memset | Memcpy) -> PointerSlot:
        symbolic = isinstance(value, sympy.Basic) or not self.ir
        lowering, key = (self.symbolic, _sym_expr) if symbolic else (self.lowering, _sym_key)
        expr = key(value)
        found = []
        for root in launch.roots:
            displacement = lowering.split(expr, key(root.sym))
            if displacement is not None:
                found.append((root, displacement))
        if len(found) != 1:
            raise declined(
                f"the pointer {expr} of Triton kernel {launch.name} is not one root plus an offset"
            )
        root, displacement = found[0]
        if root.host and not isinstance(launch, Memcpy):
            raise declined(f"{launch.name} points at a pinned host argument")
        ref = self.roots[id(root)]
        address = lowering.lower(expr) if ref[0] == "argument" else None
        base = self.base(ref)
        return PointerSlot(ref, lowering.lower(displacement), address, base)

    def base(self, ref: Ref) -> int | None:
        kind, k = ref
        if kind == "argument":
            return None
        # a section's temporary: its base past the tape's
        return k if kind in ("allocation", "section") else len(self.tape.allocs) + k

    def view(self, t: _TracedTensor) -> LoweredView:
        sizes = tuple(self.row(s) for s in t.shape)
        strides = tuple(self.row(s) for s in t._sym_strides)
        dtype = t.dtype if t.dtype != self.dtypes[id(t._root)] else None
        return LoweredView(
            self.roots[id(t._root)], sizes, strides, self.row(t._sym_offset), dtype
        )

    def eager_call(self, seq: int, call: EagerCall) -> LoweredEagerCall:
        traced, spec = pytree.tree_flatten((call.args, call.kwargs))
        leaves: list[Any] = []
        # a tensor passed twice (add(y, 1, out=y)) is one view object, which
        # a replay builds once, so the op sees one tensor as eager's did
        views: dict[int, LoweredView] = {}
        for v in traced:
            if isinstance(v, _TracedTensor):
                if id(v) not in views:
                    views[id(v)] = self.view(v)
                leaves.append(views[id(v)])
            elif isinstance(v, torch.SymInt):
                leaves.append(ScalarSlot(self.row(v)))
            else:
                leaves.append(v)
        outputs: list[PredictedOutput | int] = []
        for t in call.outputs:
            # an output the call was passed (an in-place or out= op's) is that argument, even one over an
            # earlier eager step's output; only a tensor the call made is a prediction of its root
            at = next((i for i, v in enumerate(traced) if v is t), None)
            if at is not None:
                outputs.append(at)
                continue
            if t._root.kind != "eager":
                raise AssertionError(f"host_trace: {call.name} returned a tensor it neither made nor was passed")
            v = self.view(t)
            outputs.append(
                PredictedOutput(
                    self.roots[id(t._root)][1], v.sizes, v.strides, v.offset, t.dtype
                )
            )
        grid = None
        if isinstance(call.target, tuple) and call.target[0] == "triton":
            x, y, z = (self.row(g) for g in call.target[2])
            grid = (x, y, z)
        flat = spec == pytree.tree_structure((tuple(traced), {}))
        return LoweredEagerCall(
            seq, call, spec, flat, tuple(leaves), tuple(outputs), grid
        )

    def opaque(self, call: OpaqueCall, eager: LoweredEagerCall) -> LoweredOpaqueCall:
        traced = pytree.tree_leaves((call.args, call.kwargs))
        operands: list[tuple[_TracedTensor, LoweredView | PredictedOutput]] = [
            (t, v) for t, v in zip(traced, eager.leaves) if isinstance(v, LoweredView)
        ]
        # an out= return is its argument, even one an eager call made
        operands += [
            (t, p)
            for t, p in zip(call.outputs, eager.outputs)
            if isinstance(p, PredictedOutput) and not any(t is v for v in traced)
        ]
        mask = self.emit("constant", 255)
        sizes = [r for _, v in operands for r in v.sizes]
        strides = [r for _, v in operands for r in v.strides]
        align = [self.emit("bitand", self.address(t, v.offset), mask) for t, v in operands]
        scalars = tuple(v for v in eager.leaves if not isinstance(v, LoweredView))
        slots = [v.row for v in scalars if isinstance(v, ScalarSlot)]
        alias, distances = aliasing([t for t, _ in operands])
        return LoweredOpaqueCall(
            call.target,
            call.provider,
            tuple(t.dtype for t, _ in operands),
            tuple(len(v.sizes) for _, v in operands),
            (*sizes, *strides, *align, *slots, *map(self.row, distances)),
            scalars,
            self.tape.device.index,
            call.state,
            alias,
        )

    def address(self, t: _TracedTensor, offset: int) -> int:
        """The row of t's address, less its root's base unless the root is an
        argument's: an allocation's or eager output's base is 256-byte aligned."""
        address = self.emit("multiply", offset, self.emit("constant", t.dtype.itemsize))
        if t._root.kind == "argument":
            address = self.emit("add", self.row(t._root.sym), address)
        return address

    def keyed_site(self, site: KeyedSite, index: dict[int, int]) -> LoweredKeyedSite:
        zero, mask = self.emit("constant", 0), self.emit("constant", 255)
        sizes: list[int] = []
        strides: list[int] = []
        align: list[int] = []
        operands: list[tuple[int, int]] = []
        for t in site.operands:
            v = self.view(t)
            address = self.address(t, v.offset)
            ref = self.roots[id(t._root)]
            operands.append((address, -1 if ref[0] == "argument" else self.base(ref)))  # type: ignore[arg-type]
            sizes += v.sizes
            strides += v.strides
            align.append(self.emit("bitand", address, mask))
        if any(isinstance(v, (torch.SymFloat, torch.SymBool)) for v in site.scalars):
            raise declined(f"{site.op} has a symbolic float or bool argument")
        scalars = [self.row(v) for v in site.scalars if isinstance(v, torch.SymInt)]
        scratch = {j: (zero, self.base(self.roots[id(t._root)])) for j, (t, _) in site.scratch.items()}
        alias, distances = aliasing(site.operands)
        return LoweredKeyedSite(
            site,
            tuple(t.dtype for t in site.operands),
            tuple(t.dim() for t in site.operands),
            (*sizes, *strides, *align, *scalars, *map(self.row, distances)),
            tuple(operands),
            scratch,  # type: ignore[arg-type]
            tuple(index[id(n)] for n in site.nodes),
            self.tape.device.index,
            alias,
        )

    def launch(self, seq: int, launch: Any) -> LoweredLaunch | LoweredMemset | LoweredMemcpy:
        if isinstance(launch, Memcpy):
            if isinstance(launch.nbytes, int) and launch.nbytes < 1:
                raise declined(f"{launch.name}: a memcpy of {launch.nbytes} bytes")
            slots = (self.pointer(launch.slots[0], launch), self.pointer(launch.slots[1], launch))
            return LoweredMemcpy(seq, launch, slots, self.row(launch.nbytes))
        if isinstance(launch, Memset):
            # a memset node's height, and its width and pitch over several
            # rows, are fixed at instantiation; a width of 0 is invalid
            height, pitch, width = launch.height, launch.pitch, launch.width
            fixed = isinstance(pitch, int) and isinstance(width, int)
            if not isinstance(height, int) or (height > 1 and not fixed):
                raise declined(f"{launch.name}: a memset of height {height}, width {width}, pitch {pitch}")
            if isinstance(width, int) and width < 1:
                raise declined(f"{launch.name}: a memset of width {width}")
            dst = self.pointer(launch.slots[0], launch)
            rows = (self.row(v) for v in (launch.width, launch.height, launch.pitch))
            return LoweredMemset(seq, launch, (dst,), *rows)
        pointers = launch.pointers
        if pointers is None:
            pointers = {a.slot for a in launch.abi.args if a.is_pointer}  # type: ignore[union-attr]
        slots = tuple(
            self.pointer(v, launch) if slot in pointers and not (type(v) is int and v == 0) else ScalarSlot(self.row(v))
            for slot, v in enumerate(launch.slots)
        )
        gx, gy, gz = (self.row(g) for g in launch.grid)
        bx, by, bz = (self.row(b) for b in launch.block)
        return LoweredLaunch(seq, launch, slots, (gx, gy, gz), (bx, by, bz), self.row(launch.smem))

    def outputs(self) -> tuple[Ref | LoweredView | ScalarSlot, ...]:
        tape, refs = self.tape, []

        def layout(sizes: Any, strides: Any, offset: Any) -> tuple:
            return tuple(map(self.key, sizes)), tuple(map(self.key, strides)), self.key(offset)

        # each allocation's and eager output's whole tensor
        whole = {id(rec.root): layout(rec.sizes, rec.strides, 0) for rec in tape.allocs}
        for t in self.eager:
            whole[id(t._root)] = layout(t.shape, t._sym_strides, t._sym_offset)
        for rec in tape.outputs:
            if isinstance(rec, _IntOutputRec):
                refs.append(ScalarSlot(self.row(rec.value)))
                continue
            if rec.identity is not None:
                refs.append(rec.identity)
                continue
            base = self.roots[id(rec.root)]
            dtype = rec.dtype if rec.dtype != self.dtypes[id(rec.root)] else None
            if dtype is None and whole.get(id(rec.root)) == layout(rec.sizes, rec.strides, rec.offset):
                refs.append(base)
                continue
            sizes = tuple(self.row(s) for s in rec.sizes)
            strides = tuple(self.row(s) for s in rec.strides)
            refs.append(LoweredView(base, sizes, strides, self.row(rec.offset), dtype))
        return tuple(refs)


def _check_records(checks: Sequence[torch.SymBool], ir: bool) -> list[Any]:
    # Tape.checks as guards: IR records (node, written form), else sympy
    return [(c.node.node, c.node.written) for c in checks] if ir else [c.node.expr for c in checks]


def lower_guards(tape: Tape, *, direct: bool = True) -> tuple[torch._C._HostTraceProgram, int]:
    """The tape's guards, the graph's and every op's, compiled with the row
    that is 1 at the calls they hold for: an all-eager tape's, which those
    calls run eagerly with no trace. Declined as lower_tape is."""
    lo = _TapeLowering(tape, direct)
    addresses = lo.addresses
    guards = tape.shape_env.records[: tape.guard_count] if lo.ir else tape.guards
    valid = lo.lowering.predicate(guards, addresses)
    compiled = compile_program(lo.program)
    lo.program.inputs = ()
    return compiled, valid


def lower_tape(tape: Tape, *, direct: bool = True) -> LoweredTape:
    """The tape's guards, allocations, launches and outputs as rows of one
    program, compiled; Declined for what the program cannot express. An IR
    trace's lowers from its nodes, or with `direct` off from their sympy
    export."""
    # a call a provider accepted binds at a replay: not eager
    calls = [rec.name for _, rec in tape.launches if isinstance(rec, EagerCall) and not isinstance(rec, OpaqueCall)]
    if calls and len(calls) == len(tape.launches):
        e = declined(
            f"no traced launches: every operation runs eagerly ({', '.join(calls)})"
        )
        e.uncaptured = True
        raise e
    lo = _TapeLowering(tape, direct)
    addresses = lo.addresses
    guards = tape.shape_env.records[: tape.guard_count] if lo.ir else tape.guards
    graph = [guards[i] for i in tape.graph]
    valid = lo.lowering.predicate(graph, addresses)
    allocations = tuple(lo.allocation(rec) for rec in tape.allocs)
    if tape.checks:
        lo.requirements.append(lo.lowering.conjunction(_check_records(tape.checks, lo.ir), addresses))
    launches: list[LoweredLaunch | LoweredMemset | LoweredMemcpy] = []
    # host steps read no device memory: all run first, not splitting a run
    host = [(seq, rec) for seq, rec in tape.launches if isinstance(rec, EagerCall) and rec.host]
    steps: list[range | LoweredEagerCall] = [lo.eager_call(seq, rec) for seq, rec in host]
    opaque: dict[int, LoweredOpaqueCall] = {}
    start = 0
    index: dict[int, int] = {}
    for seq, rec in tape.launches:
        if isinstance(rec, EagerCall) and rec.host:
            continue
        if not isinstance(rec, EagerCall):
            index[id(rec)] = len(launches)
            launches.append(lo.launch(seq, rec))
            continue
        if start < len(launches):
            steps.append(range(start, len(launches)))
        steps.append(lo.eager_call(seq, rec))
        if isinstance(rec, OpaqueCall):
            opaque[len(steps) - 1] = lo.opaque(rec, steps[-1])
        start = len(launches)
    if start < len(launches):
        steps.append(range(start, len(launches)))
    outputs = lo.outputs()
    sites = tuple(lo.keyed_site(site, index) for site in tape.sites)
    selectors = []
    # a keyed site's nodes are the site's, patched per key from its table: an op that holds the site (a library body's
    # GEMM) selects only its other launches, so each record has one owner
    keyed = {id(n) for site in tape.sites for n in site.nodes}
    for k, op in enumerate(tape.ops):
        if k in tape.twin_guards:
            # a refused key's guards (key_guards), in sympy
            symbolic = any(isinstance(g, sympy.Basic) for g in tape.twin_guards[k])
            low, key = (lo.symbolic, _sym_expr) if symbolic else (lo.lowering, lo.key)
            with low.total() as domains:
                predicate = low.conjunction(list(tape.twin_guards[k]), frozenset(key(rec.root.sym) for rec in tape.inputs))
            for ok in domains:
                predicate = lo.emit("and", predicate, ok)
            nodes = tuple(index[id(rec)] for _, rec in tape.launches[op.launches.start : op.launches.stop] if id(rec) in index and id(rec) not in keyed)
            selectors.append(LoweredSelector(k, predicate, nodes))
        elif op.guards:
            n = len(lo.lowering.guard_rows)
            predicate = lo.lowering.conjunction([guards[i] for i in op.guards], addresses)
            nodes = tuple(index[id(rec)] for _, rec in tape.launches[op.launches.start : op.launches.stop] if id(rec) in index and id(rec) not in keyed)
            selectors.append(LoweredSelector(k, predicate, nodes, tuple(lo.lowering.guard_rows[n:])))
    # a memcpy node's bytes lie in its allocations' own (an argument's or eager
    # output's are the tensors eager's copy_ reads and writes)
    zero = lo.emit("constant", 0)
    for m in launches:
        for slot in m.slots if isinstance(m, LoweredMemcpy) else ():
            if slot.root[0] == "allocation":
                end = lo.emit("add", slot.displacement, m.nbytes)
                fits = lo.emit("and", lo.emit("ge", slot.displacement, zero), lo.emit("le", end, allocations[slot.root[1]].nbytes))
                if lo.program.values[fits] != 1:
                    raise declined(f"{m.launch.name}: a memcpy past the end of allocation {slot.root[1]}")
                lo.requirements.append(fits)
    for row in lo.requirements:
        valid = lo.emit("and", valid, row)
    if lo.program.values[valid] != 1:
        raise AssertionError("host_trace: an allocation fails its requirements")
    compiled = compile_program(lo.program)
    # complete: the program keeps none of the traced call's tensors
    lo.program.inputs = ()
    eager_roots = tuple(t._root for t in lo.eager)
    return LoweredTape(
        tape,
        lo.program,
        compiled,
        valid,
        allocations,
        tuple(launches),
        tuple(steps),
        eager_roots,
        outputs,
        opaque,
        sites,
        tuple(selectors),
        lo,
        _stream_order(tape, launches) if tape.streams else None,
    )


class FoldRefused(Exception):
    """Why a trace does not fold into a variant's tape; `program`, compiled,
    when a lowering declined after appending rows the native program must
    have too."""

    def __init__(self, msg: str, program: torch._C._HostTraceProgram | None = None) -> None:
        super().__init__(msg)
        self.program = program
        # redispatch's: the op (in Tape.ops) whose dispatch refused
        self.op: int | None = None


class HintsFail(FoldRefused):
    """A trace's formula renamed into an IR variant's context fails at the
    variant's hints (a divisor 0 there): the sympy rename takes it."""


class IRRename:
    """A trace's values in an IR variant's context, as _sym_key values: its
    nodes rebuilt with each symbol the variant's node `mapping` names for it
    (Ctx.transfer). FoldRefused(`refusal`) on a symbol `mapping` lacks;
    HintsFail on a formula whose arithmetic fails at the variant's hints."""

    def __init__(self, variant: Env, trace: Env, refusal: str) -> None:
        self.variant, self.trace, self.refusal = variant, trace, refusal
        self.mapping: dict[str, Node] = {}
        self._memo: dict[int, Node] = {}

    def node(self, n: Node) -> Node:
        try:
            return self.variant.ctx.transfer(n, self.trace.ctx, self.mapping, self._memo)
        except KeyError as e:
            raise FoldRefused(self.refusal) from e
        except (ArithmeticError, ValueError) as e:
            raise HintsFail(f"a formula fails at the variant's hints: {e}") from e

    def __call__(self, v: Any) -> Any:
        k = _sym_key(v)
        if not isinstance(k, Node):
            return k
        n = self.node(k)
        return n.args[0] if n.op == "const" else n

    def record(self, record: tuple[Node, tuple | None]) -> tuple[Node, tuple | None]:
        """An Env record (a guard, its written form) of the trace's."""
        g, w = record
        return self.node(g), w and (w[0], self.node(w[1]), self.node(w[2]))


def fold(
    lowered: LoweredTape, tape: Tape, args: Sequence[Any], values: Sequence[int], unselected: Sequence[int]
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...]]]]:
    try:
        return _fold(lowered, tape, args, values, unselected, lowered.lowering.ir)
    except HintsFail:
        return _fold(lowered, tape, args, values, unselected, False)


def _fold(
    lowered: LoweredTape, tape: Tape, args: Sequence[Any], values: Sequence[int], unselected: Sequence[int], ir: bool
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...]]]]:
    """The trace `tape` of the call `args`, which the variant's graph guards
    hold (at rows `values`) but not selectors `unselected` (native site
    indices), as entries of those selectors: per selector (site, predicate
    row, its op's launches), rows appended to the variant's program, and the
    program compiled. An entry's predicate is its op's guards and the trace's
    graph guards the variant lacks, in the variant's symbols (the trace's
    renamed by role: an argument's metadata, an int argument, allocation k,
    eager output j). Sound for any mix of entries: every op returns the same
    metadata in both tapes, so a view or launch of one op reads the same
    formulas whichever entry another took. FoldRefused where the tapes differ
    otherwise: the op sequence, an op's outputs or launch topology (the
    graph's nodes, their attributes and roots, which the memory plan read),
    an eager call, the allocations, the outputs."""
    old, lo = lowered.tape, lowered.lowering
    if lowered.opaque:
        raise FoldRefused("the variant learns")
    ops = list(zip(tape.ops, old.ops))
    if len(tape.ops) != len(old.ops) or any(
        a.func != b.func or a.kind != b.kind or a.launches != b.launches or a.allocs != b.allocs for a, b in ops
    ):
        raise FoldRefused("another op sequence")
    counts = [(len(t.inputs), len(t.int_inputs), len(t.allocs), len(t.launches), len(t.sites)) for t in (tape, old)]
    if counts[0] != counts[1] or not set(tape.argument_pairs) <= set(old.argument_pairs):
        raise FoldRefused("another tape")
    if _stream_layout(tape) != _stream_layout(old):
        raise FoldRefused("another stream layout")
    eager: list[_TracedTensor] = []
    for _, rec in tape.launches:
        for t in rec.outputs if isinstance(rec, EagerCall) else ():
            if t._root.kind == "eager" and all(t._root is not e._root for e in eager):
                eager.append(t)
    if len(eager) != len(lo.eager):
        raise FoldRefused("other eager outputs")
    roles: dict[int, Ref] = {id(rec.root): ("argument", rec.position) for rec in tape.inputs}
    roles |= {id(rec.root): ("allocation", k) for k, rec in enumerate(tape.allocs)}
    roles |= {id(t._root): ("eager", j) for j, t in enumerate(eager)}
    roots = [*(rec.root for rec in old.inputs), *(rec.root for rec in old.allocs), *(t._root for t in lo.eager)]
    by_role = {lo.roots[id(r)]: r for r in roots}

    pairs = [(a.sym, b.sym) for a, b in zip(tape.int_inputs, old.int_inputs)]
    for a, b in zip(tape.inputs, old.inputs):
        pairs += [(a.root.sym, b.root.sym), (a.offset, b.offset), *zip(a.sizes, b.sizes), *zip(a.strides, b.strides)]
    pairs += [(a.q, b.q) for a, b in zip(tape.allocs, old.allocs)]
    pairs += [(a._root.sym, b._root.sym) for a, b in zip(eager, lo.eager)]
    refusal = "a symbol of no argument, allocation or eager output"
    # an IR variant's entry takes the trace's nodes renamed in the variant's context
    if ir and not isinstance(tape.shape_env, Env):
        raise FoldRefused("a sympy trace of an IR variant")
    key = lo.key
    mapping: dict[Any, Any] = {}
    if ir:
        renamer = IRRename(old.shape_env, tape.shape_env, refusal)
        mapping, rename = renamer.mapping, renamer
        symbols = old.shape_env.ctx.symbols
        for a, b in pairs:
            a, b = _sym_key(a), _sym_key(b)
            if isinstance(a, Node) and a.op == "sym":
                mapping.setdefault(a.args[0], old.shape_env.ctx.const(b) if isinstance(b, int) else b)
            elif isinstance(a, Node) and isinstance(b, Node) and len(fa := a.free_symbols) == 1 == len(fb := b.free_symbols):
                mapping.setdefault(next(iter(fa)), symbols[next(iter(fb))])
    else:
        for a, b in pairs:
            a, b = _sym_expr(a), _sym_expr(b)
            if isinstance(a, sympy.Symbol):
                mapping.setdefault(a, b)
            elif len(a.free_symbols) == 1 == len(b.free_symbols):
                mapping.setdefault(next(iter(a.free_symbols)), next(iter(b.free_symbols)))

        # xreplace rebuilds every node above a replaced symbol, even one replaced by itself
        moved = {a: b for a, b in mapping.items() if a != b}

        def rename(v: Any) -> Any:
            # simultaneous: the two traces' symbols may share names
            e = _sym_expr(v)
            if not e.free_symbols <= mapping.keys():
                raise FoldRefused(refusal)
            e = e.xreplace(moved)
            return int(e) if e.is_Integer else e

    def canon(v: Any, new: bool) -> Any:
        f = rename if new else key
        if isinstance(v, _TracedTensor):
            role = (roles if new else lo.roots).get(id(v._root))
            return (role, tuple(map(f, v.shape)), tuple(map(f, v._sym_strides)), f(v._sym_offset), v.dtype)
        return f(v) if isinstance(v, torch.SymInt) else v

    if any(rename(a) != key(b) for a, b in pairs):
        raise FoldRefused("arguments of other roles")
    for a, b in ops:
        if len(a.outputs) != len(b.outputs) or any(canon(x, True) != canon(y, False) for x, y in zip(a.outputs, b.outputs)):
            raise FoldRefused(f"{a.func} returned other metadata")
    for a, b in zip(tape.allocs, old.allocs):
        layout = (a.dtype, tuple(map(rename, a.sizes)), tuple(map(rename, a.strides)))
        if layout != (b.dtype, tuple(map(key, b.sizes)), tuple(map(key, b.strides))):
            raise FoldRefused("other allocations")
    if tape.result_kind != old.result_kind or len(tape.outputs) != len(old.outputs):
        raise FoldRefused("other outputs")
    for a, b in zip(tape.outputs, old.outputs):
        if isinstance(a, _IntOutputRec) or isinstance(b, _IntOutputRec):
            same = type(a) is type(b) and rename(a.value) == key(b.value)  # type: ignore[union-attr]
        else:
            new = (roles.get(id(a.root)), *map(tuple, (map(rename, a.sizes), map(rename, a.strides))), rename(a.offset))
            same = new == (lo.roots.get(id(b.root)), *map(tuple, (map(key, b.sizes), map(key, b.strides))), key(b.offset))
            same = same and (a.dtype, a.identity) == (b.dtype, b.identity)
        if not same:
            raise FoldRefused("other outputs")
    for a, b in ops:
        records = zip(tape.launches[a.launches.start : a.launches.stop], old.launches[b.launches.start : b.launches.stop])
        if a.kind == "eager":
            (_, x), (_, y) = next(records)
            # a Triton or CuTe DSL launch: its function, then its grid and
            # options or streams, compared as its arguments are
            split = [(c.target[:2], c.target[2:]) if isinstance(c.target, tuple) else (c.target, ()) for c in (x, y)]
            new_leaves, new_spec = pytree.tree_flatten((split[0][1], x.args, x.kwargs))
            old_leaves, old_spec = pytree.tree_flatten((split[1][1], y.args, y.kwargs))
            if (
                split[0][0] != split[1][0]
                or x.generator is not y.generator
                or new_spec != old_spec
                or any(canon(p, True) != canon(q, False) for p, q in zip(new_leaves, old_leaves))
            ):
                raise FoldRefused(f"another eager call of {a.func}")
        elif a.kind == "traced":
            for (_, x), (_, y) in records:
                same = type(x) is type(y) and {roles.get(id(r)) for r in x.roots} == {lo.roots.get(id(r)) for r in y.roots}
                if same and isinstance(x, KernelLaunch):
                    same = (x.attributes, x.programmatic) == (y.attributes, y.programmatic)
                    same = same and (x.function == y.function or compiled_cluster(x.function) == compiled_cluster(y.function))
                elif same and isinstance(x, Memset):
                    same = x.element_size == y.element_size and rename(x.height) == key(y.height)
                elif same and isinstance(x, Memcpy):
                    same = x.kind == y.kind
                if not same:
                    raise FoldRefused(f"another launch topology in {a.func}")

    # an IR variant's guards are Env records, as its own
    graph = {_record_key(*old.shape_env.records[i]) for i in old.graph} if ir else {old.guards[i] for i in old.graph}

    def guard(i: int) -> Any:
        return renamer.record(tape.shape_env.records[i]) if ir else rename(tape.guards[i])

    def trivial(r: Any) -> bool:
        return r[0].op == "true" or _record_key(*r) in graph if ir else r is sympy.true or r in graph

    extra: list[Any] = []
    for i in tape.graph:
        if not trivial(r := guard(i)) and r not in extra:
            extra.append(r)
    # the trace's checks of its call's values (Tape.checks), which an entry's calls meet as its guards
    for c in _check_records(tape.checks, ir):
        if not trivial(r := renamer.record(c) if ir else rename(c)) and r not in extra:
            extra.append(r)
    selected = []
    descriptors = getattr(torch._C, "_host_trace_entry_descriptors_enabled", bool)()
    keyed, old_keyed = ({id(n) for site in t.sites for n in site.nodes} for t in (tape, old))
    for i in unselected:
        selector = lowered.selectors[i - len(lowered.sites)]
        op = tape.ops[selector.op]
        records = [rec for _, rec in tape.launches[op.launches.start : op.launches.stop] if not isinstance(rec, EagerCall)]
        # its keyed sites' nodes stay theirs (lower_tape's selectors): where they lie among the op's launches must agree
        mine = [rec for _, rec in old.launches[old.ops[selector.op].launches.start : old.ops[selector.op].launches.stop] if not isinstance(rec, EagerCall)]
        if [id(rec) in keyed for rec in records] != [id(rec) in old_keyed for rec in mine]:
            raise FoldRefused(f"another keyed-site layout in {op.func}")
        records = [rec for rec in records if id(rec) not in keyed]
        launches = []
        for n, rec in zip(selector.nodes, records, strict=True):
            if isinstance(rec, KernelLaunch) and (rec.rng or rec.rng_increment or (rec.descriptors and not descriptors) or rec.cpu_scalars):
                raise FoldRefused(f"an RNG, TMA or CPU scalar launch in {op.func}")
            if any(roles.get(id(r)) not in by_role for r in rec.roots):
                raise FoldRefused(f"a launch in {op.func} of a root of no role")
            owned = tuple(by_role[roles[id(r)]] for r in rec.roots)
            if isinstance(rec, Memcpy):
                raise FoldRefused(f"a memcpy in {op.func}")
            if isinstance(rec, Memset):
                extents = (rename(rec.width), rename(rec.height), rename(rec.pitch))
                rec = replace(rec, slots=(rename(rec.slots[0]),), roots=owned, width=extents[0], height=extents[1], pitch=extents[2])
            else:
                geometry = (tuple(map(rename, rec.grid)), tuple(map(rename, rec.block)), rename(rec.smem))
                rec = replace(rec, slots=tuple(map(rename, rec.slots)), roots=owned, grid=geometry[0], block=geometry[1], smem=geometry[2])
            launches.append((lowered.launches[n].seq, rec))
        selected.append((i, [guard(g) for g in op.guards] + extra, launches, None))
    return lower_entries(lowered, args, values, selected)


def lower_entries(
    lowered: LoweredTape, args: Sequence[Any], values: Sequence[int], selected: list[tuple[int, list[Any], list[tuple[int, Any]], Any]]
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...], LoweredSection | None]]]:
    """Per (selector site, guards, (seq, launch) pairs, a redo's Section or
    None) in the variant's symbols, the entry (lower_entry), its rows appended
    to the variant's program at the call `args` (rows `values`), and the
    program compiled; FoldRefused where a lowering declines."""
    lo = lowered.lowering
    # every row evaluated at the call, where each new row is too
    program = lo.program
    program.inputs, program.values = tuple(args), list(values)
    entries = []
    base = len(lowered.allocations) + len(lowered.eager_roots) + lo.sections
    try:
        for i, guards, launches, section in selected:
            entry, base = lower_entry(lowered, i, guards, launches, section, base)
            entries.append(entry)
        compiled = compile_program(program)
        lo.entry_guards += [(i, tuple(guards)) for i, guards, *_ in selected]
    except Declined as e:
        raise FoldRefused(f"a lowering declined: {e}", compile_program(program)) from e
    finally:
        program.inputs = ()
    return compiled, entries


def lower_entry(
    lowered: LoweredTape, i: int, guards: list[Any], launches: list[tuple[int, Any]], section: Any, base: int
) -> tuple[tuple[int, int, tuple[LoweredLaunch | LoweredMemset | LoweredMemcpy, ...], LoweredSection | None], int]:
    """Selector site i's entry (site, predicate row, lowered launches,
    LoweredSection or None) from its guards and (seq, launch) pairs in the
    variant's symbols, with the program's inputs set to the call; and the next
    free base: a section's temporaries take the bases from `base`."""
    lo = lowered.lowering
    # sympy guards in an IR variant: its rename failed at the hints (HintsFail)
    symbolic = any(isinstance(g, sympy.Basic) for g in guards)
    low, key = (lo.symbolic, _sym_expr) if symbolic else (lo.lowering, lo.key)
    addresses = frozenset(key(rec.root.sym) for rec in lowered.tape.inputs) if symbolic else lo.addresses
    temporaries = []
    # the other entries' rows are evaluated at this one's calls
    with low.total() as domains:
        predicate = low.conjunction(guards, addresses)
        for t in section.temporaries if section else ():
            lo.roots[id(t.root)] = ("section", base)
            temporaries.append((base, lo.allocation(t).nbytes))
            # its requirement (nonnegative strides) is the entry's
            predicate = lo.emit("and", predicate, lo.requirements.pop())
            base += 1
        try:
            rows = tuple(lo.launch(seq, rec) for seq, rec in launches)
        finally:
            for t in section.temporaries if section else ():
                del lo.roots[id(t.root)]
        # a section's memcpy copies at least a byte, inside its allocations (as lower_tape requires of the tape's)
        nbytes = {**{k: a.nbytes for k, a in enumerate(lowered.allocations)}, **dict(temporaries)}
        zero, one = lo.emit("constant", 0), lo.emit("constant", 1)
        for m in rows if section else ():
            if isinstance(m, LoweredMemcpy):
                predicate = lo.emit("and", predicate, lo.emit("ge", m.nbytes, one))
                for slot in m.slots:
                    if slot.root[0] in ("allocation", "section"):
                        end = lo.emit("add", slot.displacement, m.nbytes)
                        fits = lo.emit("and", lo.emit("ge", slot.displacement, zero), lo.emit("le", end, nbytes[slot.base]))
                        predicate = lo.emit("and", predicate, fits)
    for ok in domains:
        predicate = lo.emit("and", predicate, ok)
    return (i, predicate, rows, section and LoweredSection(tuple(temporaries), section.drops)), base
