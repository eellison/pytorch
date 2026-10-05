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

"eager" (HostTraceReplay's default) makes eager's requests: a run's
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
is HostTraceReplay's `splits` (SPLITS): "walk" (the default) splits wherever
that lowers the replay's peak or, when the sizes vary, a later one; "coarse"
then undoes the replay's peak's splits that each lower it by little, a
graph launch fewer each; "peak" splits only at the replay's peak.

"auto" (Inductor's default) picks one of the two per variant, by
auto_memory.

"run_buffer" makes fewer allocator calls and holds more memory: each
allocation is made at the start of its step, and a run's temporaries are
placed first fit in one buffer freed once the run's graph is queued; two
share bytes only if their lifetimes (allocation to last use) do not
overlap. A tensor cannot reuse a temporary's bytes, and each distinct
buffer size is its own block, which fragments the allocator's pool when
the shapes vary.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.cuda._host_trace_lower_tape import (
    LoweredEagerCall,
    LoweredView,
    PointerSlot,
    PredictedOutput,
    ScalarSlot,
)
from torch.cuda._host_trace_program import LEAVES
from torch.cuda._host_trace_tape import _hint
from torch.fx.experimental.symbolic_shapes import free_symbols


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from torch.cuda._host_trace_lower_tape import LoweredTape


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
                for slot in lo.slots:
                    if not isinstance(slot, PointerSlot):
                        continue
                    if slot.root[0] == "argument":
                        args_used.setdefault(slot.root[1], set()).add(i)
                    elif slot.base is not None:
                        used.setdefault(slot.base, set()).add(i)
                        if slot.base < n_alloc:
                            last_seq[slot.base] = lo.seq
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
    made: list[list[int]] = [[] for _ in range(len(steps) + 1)]
    for k in range(n_alloc):
        made[holding[k]].append(k)
    dropped: list[list[int]] = [[] for _ in range(len(steps) + 1)]
    for b, s in last.items():
        if b not in escapes and b not in local:
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
    return MemoryPlan(tuple(sorted(k for k in escapes if k < n_alloc)), tuple(plan))


def _nbytes(lowered: LoweredTape) -> list[int]:
    """Each base's bytes at the traced call: allocation k's, then eager output
    j's at len(allocations) + j, as its predictions extend."""
    v, n = lowered.program.values, len(lowered.allocations)
    nbytes = [v[a.nbytes] for a in lowered.allocations]
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
        sizes, strides = [_hint(s) for s in rec.sizes], [_hint(s) for s in rec.strides]
        extent = 1 + _hint(rec.offset) + sum((s - 1) * d for s, d in zip(sizes, strides))
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
    inside: set[int] = set()  # the launch boundaries inside a keyed site
    for site in lowered.sites:
        if site.nodes:
            inside.update(range(min(site.nodes) + 1, max(site.nodes) + 1))
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
        # also just after the last launch before `cut` that uses each dropped
        # tensor or argument: the peak's allocation may come after an earlier
        # one that splitting there does not lower
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
        best = None
        for c in sorted({cut, *last_use.values()}):
            if c in (run.start, run.stop) or c in inside:
                continue
            steps = (*lowered.steps[:i], range(run.start, c), range(c, run.stop))
            opaque = {k + (k > i): o for k, o in lowered.opaque.items()}
            split = dataclasses.replace(
                lowered, steps=steps + lowered.steps[i + 1 :], opaque=opaque
            )
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
    site_start: dict[int, int] = {}  # a launch inside a keyed site -> its first
    for site in lowered.sites:
        if site.nodes:
            first = min(site.nodes)
            for j in range(first + 1, max(site.nodes) + 1):
                site_start[j] = min(site_start.get(j, first), first)
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
            # else just after the last launch before `cut` that uses each dropped
            # tensor or argument: the peak's allocation may come after an earlier
            # one that splitting there does not lower
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
            for cuts in ((cut,), last_use.values()):
                for c in sorted({site_start.get(c, c) for c in cuts}):
                    if not run.start < c < run.stop:
                        continue
                    steps = (*lowered.steps[:i], range(run.start, c), range(c, run.stop))
                    opaque = {k + (k > i): o for k, o in lowered.opaque.items()}
                    split = dataclasses.replace(
                        lowered, steps=steps + lowered.steps[i + 1 :], opaque=opaque
                    )
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
