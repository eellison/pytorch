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

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.cuda._host_trace import declined
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower import Lowering
from torch.cuda._host_trace_opaque import OpaqueKey
from torch.cuda._host_trace_program import (
    compile_program,
    IntegerProgram,
    OutOfDomain,
    Status,
)
from torch.cuda._host_trace_tape import (
    _IntOutputRec,
    _sym_expr,
    _TracedTensor,
    EagerCall,
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
    launch: KernelLaunch  # its function, layout, block and shared bytes
    slots: tuple[ScalarSlot | PointerSlot, ...]  # in the launch's slot order
    grid: tuple[int, int, int]  # rows


@dataclass(frozen=True)
class LoweredMemset:
    seq: int
    launch: Memset
    slots: tuple[PointerSlot]  # the destination
    width: int  # the row of the width in elements
    height: int  # the row of the height
    pitch: int  # the row of the pitch in bytes


@dataclass(frozen=True)
class LoweredAllocation:
    seq: int
    dtype: torch.dtype
    sizes: tuple[int, ...]  # rows
    strides: tuple[int, ...]  # rows
    nbytes: int  # row: the storage bytes, as computeStorageNbytes


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
    op: OpOverload, dtypes: tuple, ranks: tuple[int, ...], scalars: tuple, device: int, ints: Iterable[int]
) -> OpaqueKey:
    # ints: per operand its sizes, then per operand its strides, then per
    # operand its address % 256, then a value per symbolic scalar
    it = iter(ints)
    sizes = tuple(tuple(next(it) for _ in range(r)) for r in ranks)
    strides = tuple(tuple(next(it) for _ in range(r)) for r in ranks)
    align = tuple(next(it) for _ in ranks)
    scalars = tuple(next(it) if isinstance(v, (torch.SymInt, ScalarSlot)) else v for v in scalars)
    return OpaqueKey(op, dtypes, sizes, strides, align, scalars, device)


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

    def key(self, values: Sequence[int]) -> OpaqueKey:
        ints = (values[r] for r in self.rows)
        return _opaque_key(self.op, self.dtypes, self.ranks, self.scalars, self.device, ints)


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

    def key(self, ints: Sequence[int]) -> OpaqueKey:
        return _opaque_key(self.site.op, self.dtypes, self.ranks, self.site.scalars, self.device, ints)


@dataclass(frozen=True)
class LoweredTape:
    tape: Tape
    program: IntegerProgram  # complete: no row is added after compilation
    compiled: torch._C._HostTraceProgram
    valid: int  # row: the guards and every allocation's requirement hold
    allocations: tuple[LoweredAllocation, ...]
    launches: tuple[LoweredLaunch | LoweredMemset, ...]
    # a range of `launches` (one graph) or an eager call, in program order
    steps: tuple[range | LoweredEagerCall, ...]
    eager_roots: tuple[_Root, ...]  # eager output j's root
    # a Ref for the very object (an argument, an earlier output, a whole
    # allocation or eager output), otherwise a view; an int output's row
    outputs: tuple[Ref | LoweredView | ScalarSlot, ...]
    opaque: dict[int, LoweredOpaqueCall]  # by index in `steps`
    sites: tuple[LoweredKeyedSite, ...]

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
        return values


class _TapeLowering:
    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.program = IntegerProgram(tape.args)
        self.roots: dict[int, Ref] = {}
        sources: dict[sympy.Symbol, tuple | int] = {}

        def source(v: Any, leaf: tuple | int) -> None:
            if isinstance(v, torch.SymInt) and isinstance(v.node.expr, sympy.Symbol):
                sources[v.node.expr] = leaf

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
        self.lowering = Lowering(self.program, sources)
        self.requirements: list[int] = []

    def emit(self, op: str, *operands: int) -> int:
        try:
            return self.program.emit(op, *operands)
        except OutOfDomain as e:
            raise declined(str(e)) from e

    def row(self, v: Any) -> int:
        return self.lowering.lower(_sym_expr(v))

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

    def pointer(self, value: Any, launch: KernelLaunch | Memset) -> PointerSlot:
        expr = _sym_expr(value)
        found = []
        for root in launch.roots:
            root_expr = _sym_expr(root.sym)
            displacement = expr - root_expr
            own = root_expr.free_symbols
            if own & expr.free_symbols and not own & displacement.free_symbols:
                found.append((root, displacement))
        if len(found) != 1:
            raise declined(
                f"the pointer {expr} of Triton kernel {launch.name} is not one root plus an offset"
            )
        root, displacement = found[0]
        ref = self.roots[id(root)]
        address = self.lowering.lower(expr) if ref[0] == "argument" else None
        base = self.base(ref)
        return PointerSlot(ref, self.lowering.lower(displacement), address, base)

    def base(self, ref: Ref) -> int | None:
        kind, k = ref
        if kind == "argument":
            return None
        return k if kind == "allocation" else len(self.tape.allocs) + k

    def view(self, t: _TracedTensor, what: str) -> LoweredView:
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
        for i, v in enumerate(traced):
            if isinstance(v, _TracedTensor):
                if id(v) not in views:
                    views[id(v)] = self.view(v, f"argument {i} of {call.name}")
                leaves.append(views[id(v)])
            elif isinstance(v, torch.SymInt):
                leaves.append(ScalarSlot(self.row(v)))
            else:
                leaves.append(v)
        outputs: list[PredictedOutput | int] = []
        for t in call.outputs:
            if t._root.kind != "eager":
                outputs.append(next(i for i, v in enumerate(traced) if v is t))
                continue
            v = self.view(t, f"an output of {call.name}")
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
        return LoweredOpaqueCall(
            call.target,
            call.provider,
            tuple(t.dtype for t, _ in operands),
            tuple(len(v.sizes) for _, v in operands),
            (*sizes, *strides, *align, *slots),
            scalars,
            self.tape.device.index,
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
            v = self.view(t, f"an operand of {site.op}")
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
        return LoweredKeyedSite(
            site,
            tuple(t.dtype for t in site.operands),
            tuple(t.dim() for t in site.operands),
            (*sizes, *strides, *align, *scalars),
            tuple(operands),
            scratch,  # type: ignore[arg-type]
            tuple(index[id(n)] for n in site.nodes),
            self.tape.device.index,
        )

    def launch(self, seq: int, launch: Any) -> LoweredLaunch | LoweredMemset:
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
            self.pointer(v, launch) if slot in pointers else ScalarSlot(self.row(v))
            for slot, v in enumerate(launch.slots)
        )
        x, y, z = (self.row(g) for g in launch.grid)
        return LoweredLaunch(seq, launch, slots, (x, y, z))

    def outputs(self) -> tuple[Ref | LoweredView | ScalarSlot, ...]:
        tape, refs = self.tape, []

        def layout(sizes: Any, strides: Any, offset: Any) -> tuple:
            return tuple(map(_sym_expr, sizes)), tuple(map(_sym_expr, strides)), _sym_expr(offset)

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


def lower_tape(tape: Tape) -> LoweredTape:
    """The tape's guards, allocations, launches and outputs as rows of one
    program, compiled; Declined for what the program cannot express."""
    # a call a provider accepted binds at a replay: not eager
    calls = [rec.name for _, rec in tape.launches if isinstance(rec, EagerCall) and not isinstance(rec, OpaqueCall)]
    if calls and len(calls) == len(tape.launches):
        e = declined(
            f"no traced launches: every operation runs eagerly ({', '.join(calls)})"
        )
        e.uncaptured = True
        raise e
    lo = _TapeLowering(tape)
    addresses = frozenset(rec.root.sym.node.expr for rec in tape.inputs)
    valid = lo.lowering.predicate(tape.guards, addresses)
    allocations = tuple(lo.allocation(rec) for rec in tape.allocs)
    launches: list[LoweredLaunch | LoweredMemset] = []
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
    )
