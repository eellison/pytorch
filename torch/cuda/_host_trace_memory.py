"""Host tracing (private): a replay's allocations, all from the caching
allocator on the current stream, in one of two modes (HostTraceReplay's
`memory`; "auto" picks one per variant).

A replay runs the tape's steps in order (a graph per run of launches, or an
eager call). An allocation is a temporary if only launches of the run it is
allocated in use it; every other allocation is a tensor, as eager's is: an
output allocation (one an output is or views) lives while the caller holds
it; one an eager call reads or writes, or that a later step uses, is dropped
after its last step, once that step is queued. An eager call's fresh outputs
are the op's own tensors, dropped after their last step too.

"eager" makes eager's requests: a run's
allocations are made in tape order, and a temporary is freed right after
the last launch that uses it, so a later temporary or tensor of the run can
take its bytes as in eager. A run's graph is one chain in tape order, so a
block is reused only by a node after the freed one's last node, as eager's
stream orders it. The
frees come before the run's graph is queued, while eager's come after its
kernels are: the native commit (HostTraceVariant::allocate) holds the
device's allocator lock from the run's first allocation to its launch, so
no other thread takes a freed temporary's bytes ahead of the graph, and
checks that no cached segment was released meanwhile (its own allocation's
OOM retry). Otherwise, or when replay hooks run Python before the launch, it
places the run's temporaries as "run_buffer" does. The price
against eager: a tensor is dropped once the step of its last use is queued,
not right after that use; split_runs splits a run where that holds a tensor
across the replay's peak, or a later peak when the sizes vary. An argument no output is or views is dropped
after its last step too when the caller hands its references over
(HostTraceReplay.call_boxed, Inductor's boxed call); otherwise the caller
holds it anyway. split_runs splits where that holds a freed argument
(HostTraceReplay's `freed_arguments`) across the peak too. Where it splits
is HostTraceReplay's `splits` (SPLITS): "walk" splits wherever
that lowers the replay's peak or, when the sizes vary, a later one; "coarse"
then undoes the replay's peak's splits that each lower it by little, a
graph launch fewer each; "peak" (the default) splits only at the replay's peak.

"auto" (HostTraceReplay's and Inductor's default) picks one of the two
per variant, by auto_memory.

"planned" places each run's temporaries (but a keyed site's scratch buffer)
at offsets in one buffer, the run's arena, that are rows of the variant's
program: a replay makes one allocation per run for them, of the arena's
size at the call, and patches nothing when the arena keeps its address.
Offsets are sums of the temporaries' padded sizes (no max or min), placed
best fit at the trace over free holes: a temporary takes the smallest hole
whose size is at least its own for every value of the symbols, proved
coefficient-wise (_Forms.le), else one that is at the trace, which then
becomes a `valid` condition (a call where it fails misses). A size that is
not a polynomial in the symbols and their floors falls back to size
classes (Arena). Tensors are made as "run_buffer" makes them.

"run_buffer" makes fewer allocator calls and holds more memory: each
allocation is made at the start of its step, and a run's temporaries are
placed first fit in one buffer freed once the run's graph is queued; two
share bytes only if their lifetimes (allocation to last use) do not
overlap. A tensor cannot reuse a temporary's bytes, and each distinct
buffer size is its own block, which fragments the allocator's pool when
the shapes vary.

"held" makes no allocator call for what does not outlive the replay:
every allocation but the outputs and keyed sites' scratch buffers is at an
offset in one buffer (_layout), the offsets and its size rows of the
program. The buffer is held across replays, one per device and stream
(HostTraceVariant::plan), and grows when a call needs more, so the bytes
stay allocated between calls and an allocation's address moves only when
its offset or the buffer does. A replay inside a stream capture, or under
one still running on the stream, takes a buffer of its own.
"""

from __future__ import annotations

import bisect
import dataclasses
import heapq
import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.cuda._host_trace import declined
from torch.cuda._host_trace_lower_tape import (
    LoweredEagerCall,
    LoweredView,
    PointerSlot,
    PredictedOutput,
    ScalarSlot,
)
from torch.cuda._host_trace_program import LEAVES
from torch.fx.experimental.symbolic_shapes import free_symbols


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from torch.cuda._host_trace_capture import CapturedTape
    from torch.cuda._host_trace_lower_tape import LoweredTape
    from torch.cuda._host_trace_program import IntegerProgram


# the caching allocator's block granularity: a temporary's offset in the
# buffer keeps its address 256-aligned, as the trace assumed
_BLOCK = 512

# split_runs' policies, the default first
SPLITS = ("walk", "coarse", "peak")


@dataclass(frozen=True)
class StepMemory:
    tensors: tuple[int, ...]  # allocations made as tensors, in tape order
    # the run's own temporaries in tape order: (allocation, its seq, the seq
    # of the last launch that uses it, or its own seq if none does)
    temporaries: tuple[tuple[int, int, int], ...]
    # the bases (allocation k, eager output len(allocations) + j) whose last
    # step this is
    drops: tuple[int, ...]
    # "eager" memory, a run with temporaries: its allocations k in tape order,
    # each temporary's free (-1 - k) right after its last use; else empty
    order: tuple[int, ...] = ()
    # the argument positions whose last step this is
    arguments: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryPlan:
    outputs: tuple[int, ...]  # the output allocations, in tape order
    # per step of the tape, then one for the allocations after its last step
    steps: tuple[StepMemory, ...]
    # "held": (allocation, its offset's row, an eager call reads or writes
    # it) in the buffer of `buffer`'s row's bytes; the steps make the rest
    planned: tuple[tuple[int, int, bool], ...] = ()
    buffer: int = -1
    # "planned": a planned temporary k -> (its run's arena allocation, its
    # offset's row, its bytes' row, its seq, its last use)
    relocated: Mapping[int, tuple[int, int, int, int, int]] = dataclasses.field(default_factory=dict)
    # "planned": each arena and the rows of its `valid` conditions
    arenas: tuple[tuple[Arena | PackedArena, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class Arena:
    """A run's planned temporaries: allocation `allocation` (its first) becomes
    the arena, the others views of it."""

    step: int
    allocation: int
    seq: int  # the first temporary's
    last: int  # the last use of any
    # per size class, largest first at the trace: (bytes row, ((allocation,
    # slot), ...), slot count)
    classes: tuple[tuple[int, tuple[tuple[int, int], ...], int], ...]
    lives: Mapping[int, tuple[int, int]]  # each temporary's (seq, last use)
    renames: tuple[tuple[sympy.Symbol, sympy.Symbol], ...]  # see _canonical

    def offsets(self, values: Sequence[int]) -> tuple[dict[int, int], int]:
        """Each temporary's offset and the arena's bytes at rows `values`."""
        offsets, base = {}, 0
        for nbytes, members, count in self.classes:
            size = -(-values[nbytes] // _BLOCK) * _BLOCK
            for k, slot in members:
                offsets[k] = base + slot * size
            base += count * size
        return offsets, base

    def rows(self, lowered: LoweredTape) -> tuple[dict[int, int], int, list[int]]:
        """Each temporary's offset row, the arena's bytes row, and the
        conditions the plan needs at a call: a member's bytes within its
        class's, and the renamed symbols equal."""
        emit, row = lowered.program.emit, lowered.lowering.row
        pad, mask = emit("constant", _BLOCK - 1), emit("constant", -_BLOCK)
        offsets, base = {}, emit("constant", 0)
        conditions = [emit("eq", row(x), row(r)) for x, r in self.renames]
        for nbytes, members, count in self.classes:
            size = emit("bitand", emit("add", nbytes, pad), mask)
            for k, slot in members:
                offsets[k] = base if slot == 0 else emit("add", base, _times(emit, size, slot))
                if lowered.allocations[k].nbytes != nbytes:
                    conditions.append(emit("le", lowered.allocations[k].nbytes, nbytes))
            base = emit("add", base, _times(emit, size, count))
        return offsets, base, conditions


# A polynomial over a run's symbols: {exponents: integer coefficient}.
Form = dict[tuple[int, ...], int]


@dataclass(frozen=True)
class PackedArena:
    """A run's planned temporaries packed first fit across sizes ("packed").
    An offset is a sum of temporaries' padded sizes (0 or the end of a block
    live at its allocation), its rows the sum of their padded bytes rows. A
    candidate is taken where it is ordered against every live block for all
    values of the symbols at or above their bounds: each padded size is
    within [b, b + 511] for its bytes b, a polynomial in the symbols, which a
    call's `valid` checks (with the bounds)."""

    step: int
    allocation: int
    seq: int
    last: int
    lives: Mapping[int, tuple[int, int]]
    renames: tuple[tuple[sympy.Symbol, sympy.Symbol], ...]  # see _canonical
    # the forms' variables: a leaf symbol or a floor (any function) of them
    symbols: tuple[sympy.Expr, ...]
    bounds: tuple[int, ...]  # per symbol, its least value
    exprs: Mapping[int, sympy.Expr]  # each temporary's bytes
    nbytes: Mapping[int, int]  # each temporary's bytes row
    forms: Mapping[int, Mapping[int, int]]  # each temporary's offset: {temporary: count}
    total: Mapping[int, int]
    # (f, g): f <= g holds at the trace but is not proven, which a call checks
    guards: tuple[tuple[Mapping[int, int], Mapping[int, int]], ...] = ()

    def offsets(self, values: Sequence[int]) -> tuple[dict[int, int], int]:
        at = lambda f: sum(c * (-(-values[self.nbytes[k]] // _BLOCK) * _BLOCK) for k, c in f.items())
        return {k: at(f) for k, f in self.forms.items()}, at(self.total)

    def rows(self, lowered: LoweredTape) -> tuple[dict[int, int], int, list[int]]:
        emit, row = lowered.program.emit, lowered.lowering.row
        pad, mask = emit("constant", _BLOCK - 1), emit("constant", -_BLOCK)
        padded = {k: emit("bitand", emit("add", n, pad), mask) for k, n in self.nbytes.items()}

        def sum_row(f: Mapping[int, int]) -> int:
            out = emit("constant", 0)
            for k, c in sorted(f.items()):
                out = emit("add", out, _times(emit, padded[k], c))
            return out

        conditions = [emit("eq", row(x), row(r)) for x, r in self.renames]
        conditions += [emit("ge", row(x), emit("constant", b)) for x, b in zip(self.symbols, self.bounds)]
        conditions += [emit("eq", self.nbytes[k], row(e)) for k, e in self.exprs.items()]
        conditions += [emit("le", sum_row(f), sum_row(g)) for f, g in self.guards]
        return {k: sum_row(f) for k, f in self.forms.items()}, sum_row(self.total), conditions


def _times(emit: Callable[..., int], row: int, n: int) -> int:
    return row if n == 1 else emit("multiply", row, emit("constant", n))


def _at(f: Form, xs: Sequence[int]) -> int:
    return sum(c * math.prod(x**e for x, e in zip(xs, m)) for m, c in f.items())


def _add(f: Mapping, g: Mapping, sign: int = 1) -> dict:
    h = dict(f)
    for m, c in g.items():
        h[m] = h.get(m, 0) + sign * c
    return {m: c for m, c in h.items() if c}


def _nonpositive(f: Form, bounds: Sequence[int]) -> bool:
    """f <= 0 wherever every symbol is at or above its bound: a sufficient
    test, each monomial being nonnegative and nondecreasing there."""
    return all(c <= 0 for m, c in f.items() if any(m)) and _at(f, bounds) <= 0


def plan_memory(lowered: LoweredTape, memory: str = "eager") -> MemoryPlan:
    allocations, steps = lowered.allocations, lowered.steps
    n_alloc = len(allocations)
    escapes = set()
    for out in lowered.outputs:
        if isinstance(out, ScalarSlot):
            continue
        kind, k = out.base if isinstance(out, LoweredView) else out
        if kind in ("allocation", "eager"):
            escapes.add(k if kind == "allocation" else n_alloc + k)
    # argument position -> the steps that use it; an output's is never dropped
    args_used: dict[int, set[int]] = {}
    for out in lowered.outputs:
        ref = out.base if isinstance(out, LoweredView) else out
        if isinstance(ref, tuple) and ref[0] == "argument":
            args_used[ref[1]] = {len(steps)}
    used: dict[int, set[int]] = {}  # base -> the steps that use it
    touched = set()  # the bases eager calls read, write or return
    last_seq = [a.seq for a in allocations]
    ends = []  # each step's last seq
    for i, step in enumerate(steps):
        if isinstance(step, range):
            ends.append(lowered.launches[step.stop - 1].seq)
            for lo in lowered.launches[step.start : step.stop]:
                # every root the launch holds, addressed or not: an entry for its op (a fold's or a
                # redispatch's) may address any of them (they match the launch's roots)
                refs = [s.root for s in lo.slots if isinstance(s, PointerSlot)] + [lowered.lowering.roots[id(r)] for r in lo.launch.roots]
                for kind, k in refs:
                    if kind == "argument":
                        args_used.setdefault(k, set()).add(i)
                        continue
                    base = k if kind == "allocation" else n_alloc + k
                    used.setdefault(base, set()).add(i)
                    if base < n_alloc:
                        last_seq[base] = lo.seq
            continue
        ends.append(step.seq)
        bases = [n_alloc + p.root for p in step.outputs if not isinstance(p, int)]
        for leaf in step.leaves:
            if not isinstance(leaf, LoweredView):
                continue
            kind, k = leaf.base
            if kind == "argument":
                args_used.setdefault(k, set()).add(i)
            else:
                bases.append(k if kind == "allocation" else n_alloc + k)
        for b in bases:
            used.setdefault(b, set()).add(i)
            touched.add(b)
    # a keyed site's operands reach its nodes through its table
    position = {id(rec.root): rec.position for rec in lowered.tape.inputs}
    step_of = {j: i for i, s in enumerate(steps) if isinstance(s, range) for j in s}
    for site in lowered.sites:
        for t in site.site.operands:
            if id(t._root) in position and site.nodes:
                args_used.setdefault(position[id(t._root)], set()).add(step_of[site.nodes[0]])
    last_arg: list[list[int]] = [[] for _ in range(len(steps) + 1)]
    for a, s in args_used.items():
        last_arg[max(s)].append(a)
    # host steps come first, out of tape order, and allocate nothing
    first = sum(isinstance(s, LoweredEagerCall) and s.call.host for s in steps)
    holding = [first + bisect.bisect(ends[first:], a.seq) for a in allocations]
    local = {
        k
        for k in range(n_alloc)
        if k not in escapes
        and k not in touched
        and holding[k] < len(steps)
        and isinstance(steps[holding[k]], range)
        and used.get(k, set()) <= {holding[k]}
    }
    last = {k: holding[k] for k in range(n_alloc)}
    for b, s in used.items():
        last[b] = max(s)
    pool: set[int] = set()
    planned: tuple[tuple[int, int, bool], ...] = ()
    buffer = -1
    if memory == "held":
        # a keyed site's scratch bytes are its table entry's, not a row
        scratch = {b for site in lowered.sites for _, b in site.scratch.values()}
        pool = {k for k in range(n_alloc) if k not in escapes and k not in scratch}
        spans = {}
        for k in sorted(pool):
            start = (holding[k], allocations[k].seq)
            end = (holding[k], last_seq[k]) if k in local else (last[k], _AFTER)
            spans[k] = (allocations[k].nbytes, start, end)
        offsets, buffer = _layout(lowered.program, spans)
        planned = tuple((k, offsets[k], k in touched) for k in sorted(pool))
    made: list[list[int]] = [[] for _ in range(len(steps) + 1)]
    for k in range(n_alloc):
        if k not in pool:
            made[holding[k]].append(k)
    dropped: list[list[int]] = [[] for _ in range(len(steps) + 1)]
    for b, s in last.items():
        if b not in escapes and b not in local and (b not in pool or b in touched):
            dropped[s].append(b)
    plan = []
    for i in range(len(steps) + 1):
        tensors = tuple(k for k in made[i] if k not in local)
        temporaries = tuple(
            (k, allocations[k].seq, last_seq[k]) for k in made[i] if k in local
        )
        drops = tuple(dropped[i])
        order: tuple[int, ...] = ()
        if memory == "eager" and temporaries:
            events = [(allocations[k].seq, 0, k) for k in tensors]
            for k, seq, last_use in temporaries:
                events += [(seq, 0, k), (last_use, 1, -1 - k)]
            order = tuple(e for *_, e in sorted(events))
        arguments = tuple(last_arg[i]) if i < len(steps) else ()
        plan.append(StepMemory(tensors, temporaries, drops, order, arguments))
    outputs = tuple(sorted(k for k in escapes if k < n_alloc))
    return MemoryPlan(outputs, tuple(plan), planned, buffer)


# a tensor's span ends after the step of its last use
_AFTER = float("inf")


def _layout(
    program: IntegerProgram, spans: Mapping[int, tuple[int, tuple, tuple]]
) -> tuple[dict[int, int], int]:
    """Each allocation's offset row in one buffer, and the buffer's bytes'
    row, for `spans` (allocation -> (bytes row, start, end)); two whose spans
    overlap never share bytes, at any call. At the traced call the offsets
    are first fit's in start order. Ranked by that offset, an allocation
    sits at the highest end of the earlier ranked ones it overlaps, which
    is its first-fit offset at the traced call."""
    v = program.values

    def size(row: int) -> int:
        return -(-v[row] // _BLOCK) * _BLOCK

    first_fit = {}
    live: list[tuple[int, int, tuple]] = []  # (offset, end, span end)
    for k, (row, start, end) in sorted(spans.items(), key=lambda x: x[1][1]):
        live = sorted(b for b in live if b[2] > start)
        at = 0
        for offset, top, _ in live:
            # an empty one too goes in a nonempty gap, above any it ties
            if offset - at >= max(size(row), 1):
                break
            at = max(at, top)
        first_fit[k] = at
        live.append((at, at + size(row), end))
    ranked = sorted(spans, key=lambda k: (first_fit[k], spans[k][1]))
    zero = program.emit("constant", 0)
    pad, mask = program.emit("constant", _BLOCK - 1), program.emit("constant", -_BLOCK)
    offsets: dict[int, int] = {}
    ends: list[int] = []
    under: list[set[int]] = []  # per ranked allocation, the earlier ones it overlaps
    covered: set[int] = set()
    for i, k in enumerate(ranked):
        row, start, end = spans[k]
        overlaps = {j for j in range(i) if spans[ranked[j]][1] < end and start < spans[ranked[j]][2]}
        # j is below another overlapping one that overlaps it
        kept: list[int] = []
        for j in sorted(overlaps, reverse=True):
            if not any(j in under[h] for h in kept):
                kept.append(j)
        at = program.emit("max", zero, *(ends[j] for j in kept)) if kept else zero
        if v[at] != first_fit[k]:
            raise AssertionError(f"allocation {k} at {v[at]}, first fit {first_fit[k]}")
        offsets[k] = at
        ends.append(program.emit("add", at, program.emit("bitand", program.emit("add", row, pad), mask)))
        under.append(overlaps)
        covered |= overlaps
    tops = [ends[i] for i in range(len(ranked)) if i not in covered]
    return offsets, program.emit("max", zero, *tops) if tops else zero


def check_plan(lowered: LoweredTape, plan: MemoryPlan) -> None:
    """That every allocation a step uses is live there in the plan: at each launch, those its pointer
    slots and its keyed site's table (operands, scratch) reach, a temporary of its run between its
    seq and its last use, a tensor from its step to the one that drops it; at an eager step, its
    leaves' tensors. A plan that frees an allocation before a use lets a later one take its bytes
    while it is in use. Build time only."""
    n = len(lowered.allocations)
    made, spans, dropped = {}, {}, {}
    for i, step in enumerate(plan.steps):
        made.update((k, i) for k in step.tensors)
        spans.update((k, (i, seq, last)) for k, seq, last in step.temporaries)
        dropped.update((b, i) for b in step.drops)
    held = {k for k, *_ in plan.planned} | set(plan.relocated)
    tables: dict[int, list[int]] = {}
    for site in lowered.sites:
        for j in site.nodes:
            tables.setdefault(j, []).extend(b for _, b in (*site.operands, *site.scratch.values()) if 0 <= b < n)

    def live(b: int, i: int, seq: float) -> bool:
        if b in held or b >= n:
            return True
        if b in spans:
            run, first, last = spans[b]
            return run == i and first <= seq <= last
        return made.get(b, len(plan.steps)) <= i <= dropped.get(b, len(plan.steps))

    for i, step in enumerate(lowered.steps):
        if isinstance(step, range):
            uses = [(lowered.launches[j].seq, [s.base for s in lowered.launches[j].slots if isinstance(s, PointerSlot) and s.base is not None] + tables.get(j, [])) for j in step]
        else:
            uses = [(step.seq, [k for kind, k in (leaf.base for leaf in step.leaves if isinstance(leaf, LoweredView)) if kind == "allocation"])]
        for seq, bases in uses:
            if bad := [b for b in bases if not live(b, i, seq)]:
                raise AssertionError(f"host_trace: allocations {bad} are used at step {i} (seq {seq}) outside their lifetimes in the memory plan")


def _nbytes(lowered: LoweredTape) -> list[int]:
    """Each base's bytes at the traced call as the splits and the "auto"
    choice weigh them: allocation k's, then eager output j's at
    len(allocations) + j, as its predictions extend. A keyed site's scratch
    buffer weighs 0: a library's workspace, live for its site's launches
    only, as eager allocates and frees it per call, takes one slot of a run
    buffer, which its run's sites share, and a split (a graph segment, and
    addresses that move between replays) or eager order to save that slot
    costs more than it saves."""
    v, n = lowered.program.values, len(lowered.allocations)
    scratch = {b for site in lowered.sites for _, b in site.scratch.values()}
    nbytes = [0 if k in scratch else v[a.nbytes] for k, a in enumerate(lowered.allocations)]
    nbytes += [0] * len(lowered.eager_roots)
    for step in lowered.steps:
        if not isinstance(step, LoweredEagerCall):
            continue
        for p in step.outputs:
            if isinstance(p, PredictedOutput) and all(v[r] for r in p.sizes):
                extent = 1 + v[p.offset]
                extent += sum((v[s] - 1) * v[d] for s, d in zip(p.sizes, p.strides))
                nbytes[n + p.root] = max(nbytes[n + p.root], extent * p.dtype.itemsize)
    return nbytes


def _argument_nbytes(lowered: LoweredTape, freed: Collection[int]) -> dict[int, int]:
    """Each freed tensor argument's bytes at the traced call, by position."""
    nbytes = {}
    for rec in lowered.tape.inputs:
        if rec.position not in freed:
            continue
        # the call's metadata, a twin's call's too (its records hold their trace's hints)
        a = lowered.tape.args[rec.position]
        sizes, strides, offset = (a.shape, a.stride(), a.storage_offset()) if isinstance(a, torch.Tensor) else a[:3]
        extent = 1 + offset + sum((s - 1) * d for s, d in zip(sizes, strides))
        nbytes[rec.position] = extent * rec.dtype.itemsize if all(sizes) else 0
    return nbytes


def _peak(
    lowered: LoweredTape,
    plan: MemoryPlan,
    nbytes: Sequence[int],
    arguments: Mapping[int, int],
    after: float = -1,
) -> tuple[int, int, int]:
    """The replay's peak live bytes under `plan`, counting the `arguments` it
    frees (position -> bytes) until their last step, the step reaching it and
    the seq of the allocation that does (-1 at a step's start, its run
    buffer or an eager call's outputs). Only points past seq `after` count;
    a step-level point sits at its step's last seq."""
    n, steps = len(lowered.allocations), lowered.steps
    live, peak = sum(arguments.values()), (0, -1, -1)
    for i, m in enumerate(plan.steps):
        step = steps[i] if i < len(steps) else None
        if isinstance(step, range):
            end: float = lowered.launches[step.stop - 1].seq
        else:
            end = step.seq if step is not None else math.inf
        if m.order:
            for e in m.order:
                k = e if e >= 0 else -1 - e
                live += nbytes[k] if e >= 0 else -nbytes[k]
                seq = lowered.allocations[k].seq
                if live > peak[0] and seq > after:
                    peak = (live, i, seq)
        else:
            live += sum(nbytes[k] for k in m.tensors)
            _, buffer = place(m.temporaries, [nbytes[k] for k, *_ in m.temporaries])
            if end > after:
                peak = max(peak, (live + buffer, i, -1))
        if isinstance(step, LoweredEagerCall):
            roots = (p.root for p in step.outputs if isinstance(p, PredictedOutput))
            live += sum(nbytes[n + j] for j in roots)
            if end > after:
                peak = max(peak, (live, i, -1))
        live -= sum(nbytes[b] for b in m.drops)
        live -= sum(arguments.get(a, 0) for a in m.arguments)
    return peak


def _fixed_sizes(lowered: LoweredTape, freed: Collection[int]) -> bool:
    """Whether every byte count split_runs weighs (each allocation's, each
    eager call's predicted output's and each freed argument's past its
    offset, which no caller vouches for) is the same at every call the tape
    serves."""
    fixed: list[bool] = []
    for op, *operands in lowered.program.instructions:
        fixed.append(op == "constant" or (op not in LEAVES and all(fixed[x] for x in operands)))
    rows = [a.nbytes for a in lowered.allocations]
    for step in lowered.steps:
        for p in step.outputs if isinstance(step, LoweredEagerCall) else ():
            if isinstance(p, PredictedOutput):
                rows += [*p.sizes, *p.strides, p.offset]
    args = [(r.sizes, r.strides) for r in lowered.tape.inputs if r.position in freed]
    return all(fixed[r] for r in rows) and not free_symbols(args)


def _site_starts(lowered: LoweredTape) -> dict[int, int]:
    """A launch inside a keyed site (past its first) -> the site's first."""
    starts: dict[int, int] = {}
    for site in lowered.sites:
        if site.nodes:
            first = min(site.nodes)
            for j in range(first + 1, max(site.nodes) + 1):
                starts[j] = min(starts.get(j, first), first)
    return starts


def _last_uses(lowered: LoweredTape, run: range, cut: int, m: StepMemory, arguments: Mapping[int, int]) -> dict:
    """Per tensor run step `m` drops or argument it frees, just after the last
    launch before `cut` that uses it: the peak's allocation may come after an
    earlier one that splitting there does not lower."""
    held = set(m.drops)
    held_args = {a for a in m.arguments if arguments.get(a)}
    last_use = {}
    for j in range(run.start, cut):
        for slot in lowered.launches[j].slots:
            if not isinstance(slot, PointerSlot):
                continue
            if slot.root[0] == "argument" and slot.root[1] in held_args:
                last_use[("argument", slot.root[1])] = j + 1
            elif slot.base in held:
                last_use[slot.base] = j + 1
    return last_use


def _split_run(lowered: LoweredTape, i: int, c: int) -> LoweredTape:
    """The tape with run step i split before launch c."""
    run = lowered.steps[i]
    steps = (*lowered.steps[:i], range(run.start, c), range(c, run.stop), *lowered.steps[i + 1 :])
    opaque = {k + (k > i): o for k, o in lowered.opaque.items()}
    return dataclasses.replace(lowered, steps=steps, opaque=opaque)


def _split_at_the_peak(
    lowered: LoweredTape, freed: Collection[int] = ()
) -> tuple[LoweredTape, MemoryPlan]:
    """split_runs' "peak": the tape with runs split where a tensor or `freed`
    argument held to its run's end is live at the replay's peak, and its
    "eager" plan: while
    the peak is an allocation inside a run that drops either, the run is
    split just before that allocation's first launch or just after the last
    earlier launch that uses one of them, whichever lowers the peak most,
    which drops those used only before it once the first part is queued.
    Decided at the traced call's sizes; each split costs a graph launch per
    replay."""
    nbytes, arguments = _nbytes(lowered), _argument_nbytes(lowered, freed)
    inside = _site_starts(lowered)
    plan = plan_memory(lowered)
    while True:
        peak, i, seq = _peak(lowered, plan, nbytes, arguments)
        run = lowered.steps[i] if 0 <= i < len(lowered.steps) else None
        m = plan.steps[i]
        drops = m.drops or any(arguments.get(a) for a in m.arguments)
        if not isinstance(run, range) or seq < 0 or not drops:
            return lowered, plan
        launches = range(run.start, run.stop)
        cut = next((j for j in launches if lowered.launches[j].seq > seq), run.stop)
        best = None
        for c in sorted({cut, *_last_uses(lowered, run, cut, m, arguments).values()}):
            if c in (run.start, run.stop) or c in inside:
                continue
            split = _split_run(lowered, i, c)
            split_plan = plan_memory(split)
            split_peak = _peak(split, split_plan, nbytes, arguments)[0]
            if split_peak < peak and (best is None or split_peak < best[0]):
                best = (split_peak, split, split_plan)
        if best is None:
            return lowered, plan
        _, lowered, plan = best


def _rejoin(
    lowered: LoweredTape, cuts: list[int], nbytes: Sequence[int], arguments: Mapping[int, int], bound: int
) -> tuple[LoweredTape, MemoryPlan]:
    """The tape with each run starting at one of `cuts`, first first, joined
    to the run before it if the replay's peak stays at most `bound`."""
    for c in sorted(cuts):
        i = next(k for k, step in enumerate(lowered.steps) if isinstance(step, range) and step.start == c)
        prev = lowered.steps[i - 1]
        if i in lowered.opaque or not isinstance(prev, range) or prev.stop != c:
            continue
        steps = (*lowered.steps[: i - 1], range(prev.start, lowered.steps[i].stop), *lowered.steps[i + 1 :])
        opaque = {k - (k > i): o for k, o in lowered.opaque.items()}
        joined = dataclasses.replace(lowered, steps=steps, opaque=opaque)
        if _peak(joined, plan_memory(joined), nbytes, arguments)[0] <= bound:
            lowered = joined
    return lowered, plan_memory(lowered)


def split_runs(
    lowered: LoweredTape, freed: Collection[int] = (), later: bool | None = None, splits: str = "peak"
) -> tuple[LoweredTape, MemoryPlan]:
    """`splits` (SPLITS) "walk" and "coarse": the tape with runs split where a tensor or `freed` argument held to
    its run's end is live at a peak, and its "eager" plan. Walking the peaks
    in order (the replay's peak, then the peak after it, ...): while the peak
    is an allocation inside a run that drops either, the run is split just
    before that allocation's first launch if that lowers the peak (a later
    one by 1% of the replay's peak), else just after the last earlier launch
    that uses one of them, whichever lowers it most (or before the keyed site
    holding the launch), which drops those used only before it once the
    first part is queued; else the walk moves past it. Lowering
    each later peak too keeps the plan fitting other sizes, whose peak can sit
    elsewhere, and lets a backward drop its saved tensors as it goes. `later`
    (by default, whether some size varies) walks past a peak it cannot lower;
    otherwise the walk stops there, since a split past it lowers no point up
    to it and so not the replay's peak. Each split costs a graph launch per
    replay, and a backward can take hundreds at the replay's peak that each
    drop a few saved tensors: "coarse", once the replay's peak is walked,
    undoes its splits, first first, while the replay's peak stays within
    1/200 of what they reached. "peak": _split_at_the_peak. Decided at the
    traced call's sizes."""
    if splits == "peak":
        return _split_at_the_peak(lowered, freed)
    nbytes, arguments = _nbytes(lowered), _argument_nbytes(lowered, freed)
    site_start = _site_starts(lowered)
    plan = plan_memory(lowered)
    top, after = _peak(lowered, plan, nbytes, arguments)[0], -1
    if later is None:
        later = not _fixed_sizes(lowered, freed)
    peak_cuts: list[int] = []  # where the replay's peak's splits start
    rejoined = False
    while True:
        peak, i, seq = _peak(lowered, plan, nbytes, arguments, after)
        if i < 0:
            return lowered, plan
        run = lowered.steps[i] if i < len(lowered.steps) else None
        m = plan.steps[i]
        drops = m.drops or any(arguments.get(a) for a in m.arguments)
        best = None
        if isinstance(run, range) and seq >= 0 and drops and (after >= 0 or not rejoined):
            cut = next((j for j in range(run.start, run.stop) if lowered.launches[j].seq > seq), run.stop)
            for cuts in ((cut,), _last_uses(lowered, run, cut, m, arguments).values()):
                for c in sorted({site_start.get(c, c) for c in cuts}):
                    if not run.start < c < run.stop:
                        continue
                    split = _split_run(lowered, i, c)
                    split_plan = plan_memory(split)
                    split_top = _peak(split, split_plan, nbytes, arguments)[0]
                    gain = peak - _peak(split, split_plan, nbytes, arguments, after)[0]
                    if split_top <= top and gain > (0 if after < 0 else top // 100) and (best is None or gain > best[0]):
                        best = (gain, split, split_plan, split_top, c)
                if best is not None:
                    break
        if best is not None:
            _, lowered, plan, top, c = best
            if after < 0:
                peak_cuts.append(c)
            continue
        if after < 0 and peak_cuts and splits == "coarse":
            lowered, plan = _rejoin(lowered, peak_cuts, nbytes, arguments, top + top // 200)
            top, peak_cuts, rejoined = _peak(lowered, plan, nbytes, arguments)[0], [], True
            continue
        if not later:
            return lowered, plan
        if seq >= 0:
            after = seq
        elif isinstance(run, range):
            after = lowered.launches[run.stop - 1].seq
        else:
            after = run.seq if run is not None else math.inf


def auto_memory(
    lowered: LoweredTape, freed: Collection[int] = (), splits: str = "peak"
) -> tuple[LoweredTape, MemoryPlan]:
    """The "auto" plan: "run_buffer"'s if its peak is at most a margin,
    max(64 MiB, 5%), above "eager"'s (after split_runs) and no run buffer is
    larger than the margin; else "eager"'s. "run_buffer" takes split_runs'
    splits too if they lower its own peak; at fixed sizes, where split_runs
    stops at "eager"'s peak, its later splits too if they lower it by 1% of
    "eager"'s peak (a run buffer holds more past that peak), but for
    `splits` "peak". Decided at the traced call's
    sizes. Eager order frees and retakes blocks mid-run, so a replay's
    addresses move with the pool's state and it repatches the nodes that
    read them; a run buffer's rarely move. But each buffer is a block of its
    own size, which the pool keeps per variant when the shapes vary."""
    nbytes, arguments = _nbytes(lowered), _argument_nbytes(lowered, freed)
    fixed = _fixed_sizes(lowered, freed)
    split, eager = split_runs(lowered, freed, later=not fixed, splits=splits)
    peak = _peak(split, eager, nbytes, arguments)[0]
    tape, buffered = lowered, plan_memory(lowered, "run_buffer")
    buffered_peak = _peak(tape, buffered, nbytes, arguments)[0]
    candidates = [(split, 0)]
    if fixed and splits != "peak":
        candidates.append((split_runs(lowered, freed, later=True, splits=splits)[0], peak // 100))
    for candidate, gain in candidates:
        if candidate is lowered:
            continue
        candidate_buffered = plan_memory(candidate, "run_buffer")
        candidate_peak = _peak(candidate, candidate_buffered, nbytes, arguments)[0]
        if candidate_peak < buffered_peak - gain:
            tape, buffered, buffered_peak = candidate, candidate_buffered, candidate_peak
    margin = max(64 << 20, peak // 20)
    largest = max(
        place(m.temporaries, [nbytes[k] for k, *_ in m.temporaries])[1]
        for m in buffered.steps
    )
    if largest <= margin and buffered_peak <= peak + margin:
        return tape, buffered
    return split, eager


def place(
    temporaries: Sequence[tuple[int, int, int]], nbytes: Sequence[int]
) -> tuple[list[int], int]:
    """Each temporary's offset in one buffer (`nbytes` and the result follow
    `temporaries`), and the buffer's size: a temporary takes the lowest gap
    among those live at its allocation (first fit). The native commit
    (HostTraceVariant::allocate) places as this does, per call."""
    offsets = [0] * len(nbytes)
    live: list[tuple[int, int, int]] = []  # (offset, end, last use)
    total = 0
    for i, ((_, seq, last), n) in enumerate(zip(temporaries, nbytes)):
        size = -(-n // _BLOCK) * _BLOCK
        if size == 0:
            continue
        live = sorted(b for b in live if b[2] > seq)
        at = 0
        for offset, end, _ in live:
            if offset - at >= size:
                break
            at = max(at, end)
        offsets[i] = at
        live.append((at, at + size, last))
        total = max(total, at + size)
    return offsets, total


def size_classes(
    lowered: LoweredTape, packed: bool = False, reuse: bool = True, split: bool = True, scratch: bool = False
) -> tuple[MemoryPlan, list[Arena | PackedArena]]:
    """The "run_buffer" plan and each run's arenas, decided at the traced
    call: a run's temporaries but its keyed sites' scratch buffers, in one
    arena or, when the run buffer (first fit, as the native commit places
    per call) is smaller at the trace that way, split by time around the
    scratch buffers (_around_scratch). An arena is planned by _holes
    ("planned") or _pack ("packed"), else by size classes: one class per
    symbolic byte count, its slots assigned in tape order (the lowest free
    one; a slot frees after its holder's last use). reuse=False skips _holes
    and _offline (size classes only), split=False the split around scratch.
    scratch=True puts the scratch buffers in the arena too, each at its traced
    bytes (a constant row), which a key's binding must not exceed."""
    plan = plan_memory(lowered, "run_buffer")
    scratch = set() if scratch else {b for site in lowered.sites for _, b in site.scratch.values()}
    values = lowered.program.values
    arenas: list[Arena | PackedArena] = []
    for i, m in enumerate(plan.steps):
        temporaries = [t for t in m.temporaries if t[0] not in scratch]
        if not temporaries:
            continue
        kept = [t for t in m.temporaries if t[0] in scratch]
        one = [_one_arena(lowered, i, temporaries, packed, reuse)]
        parts = [_one_arena(lowered, i, g, packed, reuse) for g in _around_scratch(temporaries, kept, values, lowered)] if split else []
        if len(parts) > 1 and _buffer(parts, kept, lowered) < _buffer(one, kept, lowered):
            one = parts
        arenas.extend(one)
    return plan, arenas


def _around_scratch(
    temporaries: Sequence[tuple[int, int, int]], kept: Sequence[tuple[int, int, int]], values: Sequence[int], lowered: LoweredTape
) -> list[list[tuple[int, int, int]]]:
    """The temporaries live while some scratch buffer with bytes at the trace
    is, then those between two such buffers, a group per gap. The native
    commit places a keyed site's scratch buffer by first fit around the
    run's arenas, so above any arena live with it: a gap's arena, live with
    none, can take the scratch buffers' bytes."""
    cuts = sorted((seq, last) for k, seq, last in kept if values[lowered.allocations[k].nbytes] > 0)

    def apart(t: tuple[int, int, int], c: tuple[int, int]) -> bool:
        # as place frees: the earlier one's last use at most the later's seq
        return (t[1] < c[0] and t[2] <= c[0]) or (c[0] < t[1] and c[1] <= t[1])

    over, gaps = [], {}
    for t in temporaries:
        if all(apart(t, c) for c in cuts):
            gaps.setdefault(bisect.bisect_left([c[0] for c in cuts], t[1]), []).append(t)
        else:
            over.append(t)
    return [g for g in [over, *gaps.values()] if g]


def _buffer(arenas: Sequence[Arena | PackedArena], kept: Sequence[tuple[int, int, int]], lowered: LoweredTape) -> int:
    """The run buffer's bytes at the trace with these arenas: first fit."""
    values = lowered.program.values
    ts = sorted([(a.allocation, a.seq, a.last, a.offsets(values)[1]) for a in arenas] + [(k, seq, last, values[lowered.allocations[k].nbytes]) for k, seq, last in kept], key=lambda t: t[1])
    return place([t[:3] for t in ts], [t[3] for t in ts])[1]


def _one_arena(lowered: LoweredTape, step: int, temporaries: Sequence[tuple[int, int, int]], packed: bool, reuse: bool = True) -> Arena | PackedArena:
    values = lowered.program.values
    first = temporaries[0]
    lives = {k: (seq, last) for k, seq, last in temporaries}
    head = (step, first[0], first[1], max(t[2] for t in temporaries))
    arena = None
    if packed:
        arena = _pack(lowered, temporaries, head, lives)
    elif reuse:
        planned = [a for a in (_holes(lowered, temporaries, head, lives), _offline(lowered, temporaries, head, lives)) if a is not None]
        arena = min(planned, key=lambda a: (a.offsets(values)[1], len(a.guards)), default=None)
    if arena is not None:
        return arena
    # its size is its largest member's bytes row at the trace, which the
    # others' must not exceed at a call
    exprs, renames = _canonical(lowered, [t[0] for t in temporaries])
    by_class: dict[sympy.Expr, list[tuple[int, int, int]]] = {}
    for t in temporaries:
        by_class.setdefault(exprs[t[0]], []).append(t)
    classes = []
    for members in by_class.values():
        nbytes = max((lowered.allocations[t[0]].nbytes for t in members), key=lambda r: values[r])
        free: list[int] = []
        busy: list[tuple[int, int]] = []  # (last use, slot)
        slots, count = [], 0
        for k, seq, last in members:
            while busy and busy[0][0] <= seq:
                heapq.heappush(free, heapq.heappop(busy)[1])
            if free:
                slot = heapq.heappop(free)
            else:
                slot, count = count, count + 1
            slots.append((k, slot))
            heapq.heappush(busy, (last, slot))
        classes.append((nbytes, tuple(slots), count))
    classes.sort(key=lambda c: (-values[c[0]], c[0]))
    return Arena(*head, tuple(classes), lives, renames)


def _bytes_expr(lowered: LoweredTape, k: int) -> sympy.Expr:
    """Allocation k's bytes as computeStorageNbytes counts a nonempty one."""
    from torch.cuda._host_trace_tape import _sym_expr

    rec = lowered.tape.allocs[k]
    extent = 1 + sum((_sym_expr(n) - 1) * _sym_expr(st) for n, st in zip(rec.sizes, rec.strides))
    return sympy.expand(rec.dtype.itemsize * extent)


def _leaf_values(lowered: LoweredTape, symbols: Collection[sympy.Expr]) -> dict[sympy.Expr, int]:
    """The leaf symbols among `symbols` at the build's call, their rows'
    values: a respecified tape's are its call's, its hints the trace's."""
    from torch.cuda._host_trace_tape import _sym_expr

    values = lowered.program.values
    return {x: values[lowered.lowering.row(v)] for v, _ in lowered.lowering._leaves if (x := _sym_expr(v)) in symbols}


def _canonical(
    lowered: LoweredTape, ks: Sequence[int]
) -> tuple[dict[int, sympy.Expr], tuple[tuple[sympy.Symbol, sympy.Symbol], ...]]:
    """Each allocation's bytes with the leaf symbols equal at the build's call
    renamed to one of them, and the renames, which a call checks: per-layer
    symbols (a weight's dimension, one per layer) then share a size."""
    exprs = {k: _bytes_expr(lowered, k) for k in ks}
    symbols = set().union(*(e.free_symbols for e in exprs.values()))
    hints = _leaf_values(lowered, symbols)
    first: dict[int, sympy.Symbol] = {}
    renames = []
    for x in sorted(symbols, key=str):
        if x in hints and first.setdefault(hints[x], x) != x:
            renames.append((x, first[hints[x]]))
    sub = dict(renames)
    return {k: sympy.expand(e.xreplace(sub)) for k, e in exprs.items()}, tuple(renames)


@dataclass(frozen=True)
class _Forms:
    """A run's temporaries' bytes as polynomials with integer coefficients in
    its canonical leaf symbols (see _canonical) and its floors (any sympy
    function), each a variable of its own at least 0."""

    renames: tuple[tuple[sympy.Symbol, sympy.Symbol], ...]
    symbols: tuple[sympy.Expr, ...]  # the variables, as expressions a row lowers
    bounds: tuple[int, ...]  # per variable, its least value
    exprs: Mapping[int, sympy.Expr]  # each temporary's bytes
    low: Mapping[int, Form]  # each temporary's bytes over the variables
    padded: Mapping[int, int]  # each temporary's padded bytes at the trace
    # each temporary -> the first with the same bytes, whose padded size its
    # equals at a call (`valid` checks each bytes against its expression)
    rep: Mapping[int, int]
    # each temporary whose bytes' coefficients are all multiples of _BLOCK:
    # its padded size is its bytes at every call
    exact: frozenset[int]

    def le(self, f: Mapping[int, int], g: Mapping[int, int]) -> bool:
        """f <= g for all values of the variables at or above their bounds,
        f and g being sums of padded sizes ({temporary: count}): a padded
        size is within [b, b + 511] for its bytes b (b itself when exact)."""
        zero = (0,) * len(self.symbols)
        bound: Form = {}
        for k, c in self.same(_add(f, g, -1)).items():
            bound = _add(bound, {m: c * v for m, v in _add(self.low[k], {zero: _BLOCK - 1} if c > 0 and k not in self.exact else {}).items()})
        return _nonpositive(bound, self.bounds)

    def same(self, f: Mapping[int, int]) -> dict[int, int]:
        """f over the representatives."""
        out: dict[int, int] = {}
        for k, c in f.items():
            out[self.rep[k]] = out.get(self.rep[k], 0) + c
        return {k: c for k, c in out.items() if c}

    def at_trace(self, f: Mapping[int, int]) -> int:
        return sum(c * self.padded[k] for k, c in f.items())


def _forms(lowered: LoweredTape, temporaries: Sequence[tuple[int, int, int]]) -> _Forms | None:
    """None when a size is not such a polynomial, or a temporary is empty at
    the trace."""
    from torch.cuda._host_trace_tape import _sym_expr

    bound_of: dict[sympy.Expr, int] = {}
    for v, leaf in lowered.lowering._leaves:
        x = _sym_expr(v)
        if isinstance(x, sympy.Symbol) and isinstance(leaf, tuple):
            bound_of.setdefault(x, 1 if leaf[0] == "size" else 0)
    exprs, renames = _canonical(lowered, [t[0] for t in temporaries])
    hint_of = _leaf_values(lowered, set().union(*(e.free_symbols for e in exprs.values())))
    calls = {a for e in exprs.values() for a in sympy.preorder_traversal(e) if isinstance(a, sympy.Function)}
    atom = {a: sympy.Dummy(f"atom{i}") for i, a in enumerate(sorted(calls, key=str))}
    polys = {k: sympy.expand(e.xreplace(atom)) for k, e in exprs.items()}
    of: dict[sympy.Expr, sympy.Expr] = {}
    for a, d in atom.items():
        if not a.free_symbols <= bound_of.keys():
            return None
        bound_of[d], hint_of[d], of[d] = 0, int(a.xreplace(hint_of)), a
    symbols = tuple(sorted(set().union(*(e.free_symbols for e in polys.values())), key=str))
    if any(x not in bound_of or hint_of[x] < bound_of[x] for x in symbols):
        return None
    zero = (0,) * len(symbols)
    low: dict[int, Form] = {}
    for k, e in polys.items():
        if not symbols:
            terms = [(zero, e)]
        elif e.is_polynomial(*symbols):
            terms = sympy.Poly(e, *symbols).terms()
        else:
            return None
        if any(not c.is_Integer for _, c in terms):
            return None
        low[k] = {m: int(c) for m, c in terms if c}
    values = lowered.program.values
    hints = tuple(hint_of[x] for x in symbols)
    if any(values[lowered.allocations[k].nbytes] != _at(f, hints) for k, f in low.items()):
        return None
    padded = {k: -(-values[lowered.allocations[k].nbytes] // _BLOCK) * _BLOCK for k in exprs}
    bounds = tuple(bound_of[x] for x in symbols)
    first: dict[sympy.Expr, int] = {}
    rep = {k: first.setdefault(e, k) for k, e in exprs.items()}
    exact = frozenset(k for k, f in low.items() if all(c % _BLOCK == 0 for c in f.values()))
    return _Forms(renames, tuple(of.get(x, x) for x in symbols), bounds, exprs, low, padded, rep, exact)


def _arena(lowered: LoweredTape, head: tuple[int, int, int, int], lives: Mapping[int, tuple[int, int]], forms: _Forms, offsets, total, guards=()) -> PackedArena:
    nbytes = {k: lowered.allocations[k].nbytes for k in forms.exprs}
    return PackedArena(*head, lives, forms.renames, forms.symbols, forms.bounds, forms.exprs, nbytes, offsets, total, tuple(guards))


def _pack(
    lowered: LoweredTape, temporaries: Sequence[tuple[int, int, int]], head: tuple[int, int, int, int], lives: Mapping[int, tuple[int, int]]
) -> PackedArena | None:
    """First fit in tape order with offsets as sums of padded sizes."""
    F = _forms(lowered, temporaries)
    if F is None:
        return None
    le = F.le
    forms: dict[int, dict[int, int]] = {}
    live: list[tuple[dict[int, int], dict[int, int], int]] = []  # (offset, end, last use)
    total: dict[int, int] = {}
    for k, seq, last in temporaries:
        live = [b for b in live if b[2] > seq]
        fits = []
        for at in ({}, *(b[1] for b in live)):
            end = _add(at, {F.rep[k]: 1})
            if all(le(end, b[0]) or le(b[1], at) for b in live):
                fits.append(at)
        if fits:
            at = min(fits, key=F.at_trace)
        else:
            # above every live block: the sum of their ends
            at = {}
            for b in live:
                at = _add(at, b[1])
        forms[k] = at
        end = _add(at, {F.rep[k]: 1})
        live.append((at, end, last))
        if le(total, end):
            total = end
        elif not le(end, total):
            total = _add(total, end)
    return _arena(lowered, head, lives, F, forms, total)


def _holes(
    lowered: LoweredTape, temporaries: Sequence[tuple[int, int, int]], head: tuple[int, int, int, int], lives: Mapping[int, tuple[int, int]]
) -> PackedArena | None:
    """Best fit over free holes in tape order ("planned"). A hole is an
    offset and a size, both sums of padded sizes. A temporary takes the
    smallest free hole whose size is at least its own for all values of the
    symbols (_Forms.le), else the smallest that is at the trace, which a call
    then checks (a guard); the hole's remainder stays free, a freed block
    merges with the free holes next to it, and with no hole the temporary
    goes on top of the arena (into the free hole there, when that is no
    larger)."""
    F = _forms(lowered, temporaries)
    if F is None:
        return None
    key = lambda f: tuple(sorted(f.items()))
    free: dict[tuple, tuple[dict[int, int], dict[int, int]]] = {}  # offset key -> (offset, size)
    busy: list[tuple[int, dict[int, int], dict[int, int]]] = []  # (last use, offset, size)
    offsets: dict[int, dict[int, int]] = {}
    guards: list[tuple[dict[int, int], dict[int, int]]] = []
    total: dict[int, int] = {}

    def release(at: dict[int, int], size: dict[int, int]) -> None:
        end = _add(at, size)
        after = free.pop(key(end), None)
        if after is not None:
            size = _add(size, after[1])
        for k, (o, s) in list(free.items()):
            if key(_add(o, s)) == key(at):
                del free[k]
                at, size = o, _add(s, size)
                break
        free[key(at)] = (at, size)

    for k, seq, last in temporaries:
        for b in [b for b in busy if b[0] <= seq]:
            busy.remove(b)
            release(b[1], b[2])
        need = {F.rep[k]: 1}
        holes = sorted(free.values(), key=lambda h: (F.at_trace(h[1]), key(h[0])))
        hole = next((h for h in holes if F.le(need, h[1])), None)
        if hole is None:
            hole = next((h for h in holes if F.at_trace(h[1]) >= F.padded[k]), None)
            if hole is not None:
                guards.append(({k: 1}, hole[1]))
        if hole is not None:
            at = hole[0]
            del free[key(at)]
            rest = _add(hole[1], need, -1)
            if rest:
                free[key(_add(at, need))] = (_add(at, need), rest)
        else:
            # the free hole on top grows when it is no larger than the temporary
            top = next((h for h in holes if key(_add(*h)) == key(total) and F.le(h[1], need)), None)
            at = total if top is None else top[0]
            if top is not None:
                del free[key(at)]
            total = _add(at, need)
        offsets[k] = at
        busy.append((last, at, need))
    return _arena(lowered, head, lives, F, offsets, total, guards)


def _offline(
    lowered: LoweredTape, temporaries: Sequence[tuple[int, int, int]], head: tuple[int, int, int, int], lives: Mapping[int, tuple[int, int]]
) -> PackedArena | None:
    """Greedy by size: largest at the trace first, each at the lowest offset
    (0 or the end of a placed block live with it) clear of every placed block
    live with it at the trace. An ordering not proven for all values of the
    symbols (_Forms.le) is a guard, and so is the total's: the end largest at
    the trace."""
    F = _forms(lowered, temporaries)
    if F is None:
        return None
    at_trace = F.at_trace
    placed: list[tuple[dict[int, int], dict[int, int], int, int]] = []  # (offset, end, seq, last)
    offsets: dict[int, dict[int, int]] = {}
    guards: list[tuple[dict[int, int], dict[int, int]]] = []

    def order(f: dict[int, int], g: dict[int, int]) -> None:
        if not F.le(f, g) and (f, g) not in guards:
            guards.append((f, g))

    for k, seq, last in sorted(temporaries, key=lambda t: (-F.padded[t[0]], t[1])):
        need = {F.rep[k]: 1}
        live = [b for b in placed if seq < b[3] and b[2] < last]
        for at in sorted([{}, *(b[1] for b in live)], key=at_trace):
            end = _add(at, need)
            if all(at_trace(end) <= at_trace(b[0]) or at_trace(b[1]) <= at_trace(at) for b in live):
                break
        for b in live:
            order(end, b[0]) if at_trace(end) <= at_trace(b[0]) else order(b[1], at)
        offsets[k] = at
        placed.append((at, end, seq, last))
    total = max((b[1] for b in placed), key=at_trace)
    for b in placed:
        order(b[1], total)
    return _arena(lowered, head, lives, F, offsets, total, guards)


def _moved(emit: Callable[..., int], relocated: Mapping[int, tuple[int, ...]], launch: Any) -> Any:
    """The launch with each pointer into a planned temporary at its arena
    plus the temporary's offset row."""

    def move(slot: object) -> object:
        if isinstance(slot, PointerSlot) and slot.base in relocated:
            k, at = relocated[slot.base][:2]
            return PointerSlot(("allocation", k), emit("add", slot.displacement, at), None, k)
        return slot

    return dataclasses.replace(launch, slots=tuple(map(move, launch.slots)))


def relocate_entries(lowered: LoweredTape, relocated: Mapping[int, tuple[int, ...]], entries: list, args: Sequence[Any]) -> tuple[Any, list]:
    """A fold's or a redispatch's entries (site, predicate, launches) with
    their launches relocated as the variant's are, and the program compiled
    again over the call `args`, at whose values the program's rows are."""
    from torch.cuda._host_trace_program import compile_program

    program = lowered.lowering.program
    program.inputs = tuple(args)
    try:
        entries = [(site, predicate, tuple(_moved(program.emit, relocated, lo) for lo in launches)) for site, predicate, launches in entries]
        return compile_program(program), entries
    finally:
        program.inputs = ()


def arena_addresses(lowered: LoweredTape, arenas: Sequence[Arena | PackedArena], addresses: Sequence[int]) -> list[int]:
    """The capture's placeholder addresses with each planned temporary at its
    arena's placeholder plus its offset at the trace: the nodes then hold what
    the relocated launches compute there."""
    addresses = list(addresses)
    for arena in arenas:
        offsets, _ = arena.offsets(lowered.program.values)
        for k, offset in offsets.items():
            addresses[k] = addresses[arena.allocation] + offset
    return addresses


def relocate(
    captured: CapturedTape, plan: MemoryPlan, arenas: Sequence[Arena | PackedArena]
) -> tuple[CapturedTape, MemoryPlan]:
    """The captured tape with each planned temporary's pointers at its arena
    plus its offset's row, and each arena allocated as its run's only
    temporary but its keyed sites' scratch buffers: the "planned" plan. Must
    run before the tape releases its traced call's arguments (the program is
    compiled again over them)."""
    from torch.cuda._host_trace_program import compile_program

    lowered = captured.lowered
    program = lowered.program
    emit = program.emit
    one = emit("constant", 1)
    valid = lowered.valid
    relocated: dict[int, tuple[int, int, int, int, int]] = {}
    allocations = list(lowered.allocations)
    steps = list(plan.steps)
    kept_arenas = []
    for arena in arenas:
        rows, base, conditions = arena.rows(lowered)
        kept_arenas.append((arena, tuple(conditions)))
        for c in conditions:
            valid = emit("and", valid, c)
        for k, at in rows.items():
            relocated[k] = (arena.allocation, at, lowered.allocations[k].nbytes, *arena.lives[k])
        offsets, total = arena.offsets(program.values)
        if program.values[base] != total or any(program.values[rows[k]] != o for k, o in offsets.items()):
            raise AssertionError("host_trace: an arena's rows disagree with its plan at the trace")
        a = allocations[arena.allocation]
        allocations[arena.allocation] = dataclasses.replace(a, dtype=torch.uint8, sizes=(base,), strides=(one,), nbytes=base, arena=True)

    for i in {arena.step for arena in arenas}:
        own = [(a.allocation, a.seq, a.last) for a in arenas if a.step == i]
        kept = [t for t in steps[i].temporaries if t[0] not in relocated]
        steps[i] = dataclasses.replace(steps[i], temporaries=tuple(sorted(own + kept, key=lambda t: t[1])), order=())

    launches = tuple(_moved(emit, relocated, lo) for lo in lowered.launches)
    sites = tuple(
        dataclasses.replace(
            site,
            operands=tuple(
                (emit("add", row, relocated[b][1]), relocated[b][0]) if b in relocated else (row, b)
                for row, b in site.operands
            ),
            scratch={j: (emit("add", row, relocated[b][1]), relocated[b][0]) if b in relocated else (row, b) for j, (row, b) in site.scratch.items()},
            arena_scratch=frozenset(j for j, (_, b) in site.scratch.items() if b in relocated),
        )
        for site in lowered.sites
    )
    if program.values[valid] != 1:
        # F11: the arenas' conditions are the trace's call's (its rows' values); one failing there plans nothing
        raise declined("an arena's conditions fail at the traced call")
    program.inputs = tuple(lowered.tape.args)
    try:
        compiled = compile_program(program)
    finally:
        program.inputs = ()
    new = dataclasses.replace(lowered, compiled=compiled, valid=valid, allocations=tuple(allocations), launches=launches, sites=sites)
    moved = {id(old): lo for old, lo in zip(lowered.launches, launches)}
    segments = tuple(
        dataclasses.replace(s, launches=tuple(dataclasses.replace(c, launch=moved[id(c.launch)]) for c in s.launches))
        for s in captured.segments
    )
    return (
        dataclasses.replace(captured, lowered=new, segments=segments),
        dataclasses.replace(plan, steps=tuple(steps), relocated=relocated, arenas=tuple(kept_arenas)),
    )
