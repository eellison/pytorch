"""Host tracing (private): one op of a variant dispatched again, alone.

A call that a variant's graph guards hold has every op's input metadata at
its formula in the variant's symbols; where an op's own guards (its
selector) fail, its dispatch (OpRec.redo: its traced host, or its Triton or
CuTe DSL launcher) runs again on a fresh trace, over tensors at those
formulas' values at the call and the planned roots. Its launches become an
entry of the selector, lowered as a fold's are, with every guard the run
recorded as its predicate. No Python of the traced function runs, and no
other op. FoldRefused where the op's outputs, allocations or launch topology
differ: the graph's metadata changed, which a trace decides.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._ops import OpOverload
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import FoldRefused, lower_entries
from torch.cuda._host_trace_program import compile_program
from torch.cuda._host_trace_tape import (
    _active,
    _ALLOC_ALIGNMENT,
    _ALLOC_SHIFT,
    _ALLOC_TAG,
    _capture,
    _hint,
    _ir_census,
    _PLACEHOLDER_LOW,
    _PLACEHOLDER_TAG,
    _Root,
    _sym_expr,
    _Trace,
    _TracedTensor,
    _TraceMode,
    fx_config,
    Memcpy,
)
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.cuda._host_trace_lower_tape import LoweredLaunch, LoweredMemset, LoweredSelector, LoweredTape
    from torch.cuda._host_trace_tape import OpRec


def redispatch(
    lowered: LoweredTape, args: Sequence[Any], values: Sequence[int], unselected: Sequence[int]
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...]]]]:
    """Selectors `unselected` (native site indices) of the variant, whose
    graph guards hold the call `args` (at rows `values`), as entries from
    their ops dispatched again: fold's result, with no trace."""
    program = lowered.lowering.program
    rows = len(program.values)
    # the ops' input formulas are evaluated at the call, each new row too
    program.inputs, program.values = tuple(args), list(values)
    try:
        selected = []
        for i in unselected:
            selector = lowered.selectors[i - len(lowered.sites)]
            selected.append((i, *_redo(lowered, selector)))
        values = program.values
    except (Declined, FoldRefused) as e:
        compiled = compile_program(program) if len(program.values) > rows else None
        if isinstance(e, Declined):
            raise FoldRefused(f"a lowering declined: {e}", compiled) from e
        if e.program is None:
            e.program = compiled
        raise
    finally:
        program.inputs = ()
    return lower_entries(lowered, args, values, selected)


def _redo(lowered: LoweredTape, selector: LoweredSelector) -> tuple[list[Any], list[tuple[int, Any]]]:
    from torch.cuda._host_trace_cute import intercepting as cute_intercepting
    from torch.cuda._host_trace_triton_launch import intercepting

    old, lo = lowered.tape, lowered.lowering
    op: OpRec = old.ops[selector.op]
    if op.kind != "traced" or op.redo is None or not isinstance(old.shape_env, _ir.Env):
        raise FoldRefused(f"{op.func} has no traced dispatch to run again")
    tr = _Trace(old.device)
    env = tr.shape_env
    # no fresh symbol shares a name with the variant's: one the dispatch did
    # not get from its arguments (a closure's) is refused, not renamed
    env.unique_ids |= old.shape_env.unique_ids
    # the fresh trace's symbol -> its formula in the variant's symbols
    mapping: dict[sympy.Symbol, sympy.Expr] = {}
    fresh: dict[sympy.Expr, torch.SymInt] = {}
    roots: dict[int, _Root] = {}
    back: dict[int, _Root] = {}  # by the fresh root's id
    tensors: dict[int, _TracedTensor] = {}

    def value(v: Any) -> Any:
        if isinstance(v, (torch.SymBool, torch.SymFloat)):
            # a heuristic's constexpr, say: its formula is not an input row
            raise FoldRefused(f"{op.func} takes a symbolic {type(v).__name__}")
        if not isinstance(v, torch.SymInt):
            return v
        e = _sym_expr(v)
        if e not in fresh:
            fresh[e] = env.symbol(lo.program.values[lo.row(e)], f"r{len(fresh)}", positive=bool(e.is_positive))
            mapping[_sym_expr(fresh[e])] = e
        return fresh[e]

    def root(r: _Root) -> _Root:
        if id(r) not in roots:
            if r.kind == "argument":
                base = lo.program.values[lo.row(r.sym)]
                sym = env.symbol(_PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW), f"{r.name}.base")
                mapping[_sym_expr(sym)] = _sym_expr(r.sym)
            else:
                (q,) = _sym_expr(r.sym).free_symbols
                fq = env.symbol(_hint(r.sym) // _ALLOC_ALIGNMENT, f"{r.name}.base/{_ALLOC_ALIGNMENT}")
                mapping[_sym_expr(fq)] = q
                sym = _ALLOC_ALIGNMENT * fq
            roots[id(r)] = _Root(r.name, sym, r.kind)
            back[id(roots[id(r)])] = r
            if r.kind == "argument":
                # the input's record, as trace() has it: an as_strided bound, an overlap, copy-on-write
                rec = next(i for i in old.inputs if i.root is r)
                layout = [value(n) for n in rec.sizes], [value(s) for s in rec.strides], value(rec.offset)
                rec = replace(rec, sizes=layout[0], strides=layout[1], offset=layout[2], root=roots[id(r)])
                tr.inputs.append(rec)
                tr.arguments[id(rec.root)] = rec
        return roots[id(r)]

    def tensor(t: _TracedTensor) -> _TracedTensor:
        if id(t) not in tensors:
            sizes, strides = [value(n) for n in t.shape], [value(s) for s in t._sym_strides]
            tensors[id(t)] = _TracedTensor(root(t._root), sizes, strides, value(t._sym_offset), t.dtype, t.device)
        return tensors[id(t)]

    # the hints: the formulas at the call, rows that fail at no other call
    with lo.symbolic.total():
        call = pytree.tree_map(lambda a: tensor(a) if isinstance(a, _TracedTensor) else value(a), op.call)
    if len(op.allocs):
        q = old.allocs[op.allocs.start].q
        tr.first_alloc = ((_hint(q) * _ALLOC_ALIGNMENT ^ _ALLOC_TAG) >> _ALLOC_SHIFT) - 1
    prior = getattr(_active, "trace", None)
    _active.trace = tr
    # an operator's dispatch runs where _TraceMode dispatched it, a launcher where the host called it
    mode = contextlib.nullcontext() if isinstance(op.func, OpOverload) else _TraceMode(tr)
    try:
        # as trace() runs a host: Triton and CuTe DSL launches intercepted, any other launch refused
        with torch.cuda.device(old.device), _capture(old.device), intercepting(), cute_intercepting():
            tr.stream = torch.cuda.current_stream(old.device)
            with mode, fx_config.patch(backed_size_oblivious=False):  # type: ignore[attr-defined]
                op.redo(*call)
    except Exception as e:
        if not isinstance(e, Declined) and torch.cuda._host_trace.raise_unexpected:
            raise
        raise FoldRefused(f"{op.func} declines again: {type(e).__name__}: {e}") from e
    finally:
        _active.trace = prior
    if tr.declined is not None or _ir_census(tr):
        raise FoldRefused(f"{op.func} declines again: {tr.declined or _ir_census(tr)}")
    if len(tr.ops) != 1 or tr.sites or tr.eager_outputs:
        raise FoldRefused(f"{op.func} dispatches otherwise")
    new = tr.ops[0]
    if new.kind != "traced" or len(new.launches) != len(op.launches) or len(new.allocs) != len(op.allocs):
        raise FoldRefused(f"{op.func} dispatches otherwise")
    for a, b in zip(tr.allocs, old.allocs[op.allocs.start : op.allocs.stop]):
        mapping[_sym_expr(a.q)] = _sym_expr(b.q)
        back[id(a.root)] = b.root

    def rename(v: Any) -> Any:
        e = _sym_expr(v)
        if not e.free_symbols <= mapping.keys():
            raise FoldRefused(f"{op.func} dispatched on a symbol of no argument")
        e = e.xreplace(mapping)
        return int(e) if e.is_Integer else e

    def same(x: Any, y: Any) -> bool:
        if isinstance(x, _TracedTensor) != isinstance(y, _TracedTensor):
            return False
        if not isinstance(x, _TracedTensor):
            return rename(x) == _sym_expr(y) if isinstance(x, torch.SymInt) or isinstance(y, torch.SymInt) else x == y
        layout = [*map(rename, x.shape), *map(rename, x._sym_strides), rename(x._sym_offset)]
        want = [*map(_sym_expr, y.shape), *map(_sym_expr, y._sym_strides), _sym_expr(y._sym_offset)]
        return back.get(id(x._root)) is y._root and x.dtype == y.dtype and layout == want

    for a, b in zip(tr.allocs, old.allocs[op.allocs.start : op.allocs.stop]):
        layout = (a.dtype, tuple(map(rename, a.sizes)), tuple(map(rename, a.strides)))
        if layout != (b.dtype, tuple(map(_sym_expr, b.sizes)), tuple(map(_sym_expr, b.strides))):
            raise FoldRefused(f"{op.func} allocates otherwise")
    if len(new.outputs) != len(op.outputs) or not all(map(same, new.outputs, op.outputs)):
        raise FoldRefused(f"{op.func} returned other metadata")
    launches = []
    olds = [rec for _, rec in old.launches[op.launches.start : op.launches.stop]]
    for n, (_, x), y in zip(selector.nodes, tr.launches, olds, strict=True):
        if isinstance(x, KernelLaunch) and (x.rng or x.rng_increment or x.descriptors or x.cpu_scalars):
            raise FoldRefused(f"an RNG, TMA or CPU scalar launch in {op.func}")
        if isinstance(x, Memcpy):
            raise FoldRefused(f"a memcpy in {op.func}")
        if any(id(r) not in back for r in x.roots):
            raise FoldRefused(f"a launch in {op.func} of a root of no role")
        owned = tuple(back[id(r)] for r in x.roots)
        topology = type(x) is type(y) and {id(r) for r in owned} == {id(r) for r in y.roots}
        if topology and isinstance(x, KernelLaunch):
            topology = (x.attributes, x.programmatic) == (y.attributes, y.programmatic)
        elif topology:
            topology = x.element_size == y.element_size and rename(x.height) == _sym_expr(y.height)
        if not topology:
            raise FoldRefused(f"another launch topology in {op.func}")
        if isinstance(x, KernelLaunch):
            geometry = (tuple(map(rename, x.grid)), tuple(map(rename, x.block)), rename(x.smem))
            x = replace(x, slots=tuple(map(rename, x.slots)), roots=owned, grid=geometry[0], block=geometry[1], smem=geometry[2])
        else:
            x = replace(x, slots=(rename(x.slots[0]),), roots=owned, width=rename(x.width), height=rename(x.height), pitch=rename(x.pitch))
        launches.append((lowered.launches[n].seq, x))
    guards = [g for g in (rename(g.expr) for g in env.guards) if g is not sympy.true]
    if sympy.false in guards:
        raise AssertionError(f"host_trace: {op.func}'s dispatch again guards a condition the call fails")
    return guards, launches
