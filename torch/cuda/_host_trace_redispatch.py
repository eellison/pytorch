"""Host tracing (private): one op of a variant dispatched again, alone.

A call that a variant's graph guards hold has every op's input metadata at
its formula in the variant's symbols; where an op's own guards (its
selector) fail, its dispatch (OpRec.redo: its traced host, or its Triton or
CuTe DSL launcher) runs again on a fresh trace, over tensors at those
formulas' values at the call and the planned roots. Its launches become an
entry of the selector, lowered as a fold's are, with every guard the run
recorded as its predicate. No Python of the traced function runs, and no
other op. Where its launch or allocation count differs (a split reduction:
a memset, a kernel and two temporaries for one kernel), the entry is a
section: its launches replace the op's in the graph (a form's piece), each
allocation an output is the variant's, and the others are its temporaries,
sized at the call by formulas in the variant's symbols. FoldRefused where the
op's outputs or launch topology differ otherwise: the call traces again.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode
from torch.cuda import _host_trace_hint_audit as _hint_audit, _host_trace_ir as _ir
from torch.cuda._host_trace import _drop_tracebacks, Declined, declined
from torch.cuda._host_trace_launch import compiled_cluster, KernelLaunch
from torch.cuda._host_trace_lower_tape import FoldRefused, HintsFail, IRRename, lower_entry, LoweredSection, PointerSlot
from torch.cuda._host_trace_program import compile_program
from torch.cuda._host_trace_tape import (
    _active,
    _ALLOC_ALIGNMENT,
    _capture,
    _CAPTURE_ERRORS,
    _InputRec,
    _ir_census,
    _PLACEHOLDER_LOW,
    _PLACEHOLDER_TAG,
    _placeholder,
    _Root,
    _sym_expr,
    _sym_key,
    _Trace,
    _TracedTensor,
    _TraceMode,
    CallValues,
    fx_config,
    Memcpy,
    Memset,
    OpaqueCall,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch.cuda._host_trace_lower_tape import LoweredLaunch, LoweredMemset, LoweredSelector, LoweredTape
    from torch.cuda._host_trace_opaque import KeyedSite
    from torch.cuda._host_trace_tape import OpRec


def redispatch(
    lowered: LoweredTape, args: Sequence[Any], values: Sequence[int], unselected: Sequence[int], dispatches: dict | None = None, opaque: Sequence[Any] = ()
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...], Any]]]:
    """Selectors `unselected` (native site indices) of the variant, whose
    graph guards hold the call `args` (at rows `values`), as entries from
    their ops dispatched again: fold's result, with no trace. An op whose redo
    would read what an earlier redo of this variant read (Template), where
    that entry's predicate holds, takes that entry with its own roots. A
    redo routes calls to the replay's `opaque` providers, as its trace did."""
    lo = lowered.lowering
    program = lo.program
    rows = len(program.values)
    # the ops' input formulas are evaluated at the call, each new row too
    program.inputs, program.values = tuple(args), list(values)
    entries, recorded = [], []
    base = len(lowered.allocations) + len(lowered.eager_roots) + lo.sections
    try:
        # one that refused before goes first: refusing again costs no other op's dispatch
        with _hosting(lowered.tape.device):
            for i in sorted(unselected, key=lambda i: i not in lo.refused):
                selector = lowered.selectors[i - len(lowered.sites)]
                op = lowered.tape.ops[selector.op]
                if selector.op not in lo.reads:
                    lo.reads[selector.op] = _reads(lowered, op, selector)
                reads = lo.reads[selector.op]
                template = lo.templates.get(reads[0]) if reads is not None else None
                if template is not None and not all(reads[1][p] is r for p, r in template.pinned):
                    # it reads a root of its own op at a position where this op's is another
                    template = None
                if template is not None and program.values[template.predicate] == 1 and _bound(lowered, reads[2], program.values):
                    entry, after = _instantiate(lowered, i, template, reads[1], base, selector, op)
                    if torch.cuda._host_trace.verify_redo_templates:
                        full = lower_entry(lowered, i, *_full_redo(lowered, i, selector, dispatches, opaque), base)[0]
                        _assert_same_entry(entry, full, program.values)
                    lo.instances += 1
                    entries.append(entry)
                    recorded.append((i, template.guards))
                    base = after
                    continue
                guards, launches, section = _full_redo(lowered, i, selector, dispatches, opaque)
                entry, base = lower_entry(lowered, i, guards, launches, section, base)
                lo.redos += 1
                entries.append(entry)
                recorded.append((i, tuple(guards)))
                if reads is not None and (template := _template(lowered, entry, guards, launches, section, reads[1], op)) is not None:
                    lo.templates[reads[0]] = template
        entries.sort(key=lambda e: unselected.index(e[0]))
        compiled = compile_program(program)
        lo.entry_guards += recorded
    except (Declined, FoldRefused) as e:
        compiled = compile_program(program) if len(program.values) > rows else None
        if isinstance(e, Declined):
            raise FoldRefused(f"a lowering declined: {e}", compiled) from e
        if e.program is None:
            e.program = compiled
        raise
    finally:
        program.inputs = ()
    return compiled, entries


def _full_redo(lowered: LoweredTape, i: int, selector: LoweredSelector, dispatches: dict | None, opaque: Sequence[Any] = ()) -> tuple[list[Any], list[tuple[int, Any]], Section | None]:
    try:
        try:
            return _redo(lowered, selector, hosted=True, dispatches=dispatches, opaque=opaque)
        except HintsFail:
            return _redo(lowered, selector, ir=False, hosted=True, opaque=opaque)
    except FoldRefused as e:
        lowered.lowering.refused.add(i)
        e.op = selector.op
        raise


class _Unclassified(Exception):
    """A read a redo template's key cannot name: the op is redone in full."""


@dataclass(frozen=True)
class Template:
    """A redo's entry, by what the redo read (_reads), for another op of the
    variant that reads the same: its predicate row and guards, its lowered
    launches, per launch (slot, role) with role ("root", k) the k-th root of
    the read walk or ("temp", m) the section's m-th temporary, its section
    (temporaries' bytes rows), its drops as positions in its op's
    allocations, and the roots (position, root) a guard, scalar, dimension
    or displacement of it reads, which another op takes only where its root
    at that position is the same."""

    predicate: int
    guards: tuple[Any, ...]
    rows: tuple[Any, ...]
    roles: tuple[tuple[tuple[int, tuple[str, int]], ...], ...]
    section: LoweredSection | None
    drops: tuple[int, ...]
    pinned: tuple[tuple[int, _Root], ...]


def _reads(lowered: LoweredTape, op: OpRec, selector: LoweredSelector) -> tuple[Any, list[_Root], tuple[int, ...]] | None:
    """What a redo of `op` reads, as a key, its roots in reading order and its
    keyed sites (indices in lowered.sites), with no fresh symbol: the op, its
    call's structure and constants, each SymInt / SymBool as the variant's
    node, each root's kind and layout formulas (by position), each tensor's
    layout, then the op's own allocations and outputs, the launch chain it
    replaces and its keyed sites' calls (_one_role's reads). Only an
    operator's traced dispatch: one operator for every op of the key, whose
    redos run in the same Python and global state (the variant's contract,
    as its own trace assumed), so its records are a function of these reads.
    A launcher's redo (a Triton or CuTe DSL closure) and anything the walk
    cannot name (a SymFloat, a tensor the trace does not track) is None: the
    op is redone in full."""
    func = op.func
    if op.kind != "traced" or op.redo is None or not isinstance(func, OpOverload) or not lowered.lowering.ir:
        return None
    old = lowered.tape
    items: list[Any] = [func, repr(pytree.tree_structure(op.call)), len(selector.nodes)]
    roots: list[_Root] = []
    seen: dict[int, int] = {}

    def value(v: Any) -> None:
        if isinstance(v, torch.SymFloat) or (isinstance(v, torch.Tensor) and not isinstance(v, _TracedTensor)):
            raise _Unclassified
        if isinstance(v, torch.SymBool):
            items.append(("bool", v.node.node) if v.node.constant is None else (bool, bool(v.node.constant)))
        elif isinstance(v, torch.SymInt):
            items.append(("int", _sym_key(v)))
        else:
            items.append((type(v), repr(v) if isinstance(v, float) else v))

    def layout(sizes: Any, strides: Any, offset: Any) -> None:
        for n in (*sizes, *strides, offset):
            value(n)

    def root(r: _Root) -> None:
        if id(r) in seen:
            items.append(("root", seen[id(r)]))
            return
        seen[id(r)] = len(roots)
        roots.append(r)
        items.append(("new root", r.kind))
        if r.kind == "argument":
            rec = next(i for i in old.inputs if i.root is r)
            layout(rec.sizes, rec.strides, rec.offset)
            items.append((rec.dtype, rec.extent, rec.cow))
        elif r.kind == "allocation":
            a = next(a for a in old.allocs if a.root is r)
            layout(a.sizes, a.strides, 0)
            items.append(a.dtype)
        else:
            o = next(o for o in old.eager_outputs if o._root is r)
            layout(o.shape, o._sym_strides, o._sym_offset)
            items.append(o.dtype)

    def tensor(t: _TracedTensor) -> None:
        root(t._root)
        layout(t.shape, t._sym_strides, t._sym_offset)
        items.append((t.dtype, t.device))

    try:
        for leaf in pytree.tree_leaves(op.call):
            tensor(leaf) if isinstance(leaf, _TracedTensor) else value(leaf)
        # what the redo compares its result with: the op's allocations and outputs, and the chain a section replaces
        items.append("own")
        for a in old.allocs[op.allocs.start : op.allocs.stop]:
            root(a.root)
        for o in op.outputs:
            tensor(o) if isinstance(o, _TracedTensor) else value(o)
        for _, rec in old.launches[op.launches.start : op.launches.stop]:
            fixed = (rec.function, rec.attributes, rec.programmatic) if isinstance(rec, KernelLaunch) else (getattr(rec, "element_size", None),)
            items.append((type(rec), *fixed))
        span = {id(rec) for _, rec in old.launches[op.launches.start : op.launches.stop]}
        sites = tuple(j for j, ls in enumerate(lowered.sites) if ls.site.nodes and id(ls.site.nodes[0]) in span)
        for j in sites:
            site = lowered.sites[j].site
            if site.call is None:
                raise _Unclassified
            items.append(("site", site.op, site.provider, site.state, repr(site.call[0]), site.call[1], site.rng, tuple(site.scratch)))
            for t in site.operands:
                tensor(t)
            for v in site.scalars:
                value(v)
            for t, _ in site.scratch.values():
                root(t._root)
    except _Unclassified:
        return None
    key = tuple(items)
    try:
        hash(key)
    except TypeError:
        return None
    return key, roots, sites


def _bound(lowered: LoweredTape, sites: tuple[int, ...], values: Sequence[int]) -> bool:
    """Whether each keyed site's key binds at the call: where one does not, a
    full redo records an opaque call, no site, and refuses."""
    for j in sites:
        site = lowered.sites[j]
        key = site.key([values[r] for r in site.rows])
        if site.site.provider.refusal(key) is not None or site.site.provider.bind(key) is None:
            return False
    return True


def _template(lowered: LoweredTape, entry: tuple, guards: list[Any], launches: list[tuple[int, Any]], redone: Section | None, roots: list[_Root], op: OpRec) -> Template | None:
    """The full redo's entry as a Template, or None where another op reading
    the same could not take it: a pointer into a root of no role. A guard,
    scalar, dimension or displacement that reads one of its roots (an
    alignment specialization on a pointer) pins that root (Template.pinned)."""
    lo = lowered.lowering
    _, predicate, rows, section = entry
    if any(isinstance(g, sympy.Basic) for g in guards):
        return None
    names: dict[str, int] = {}  # a root's symbol -> its position
    for p, r in enumerate(roots):
        k = _sym_key(r.sym)
        names.update(dict.fromkeys(k.free_symbols if isinstance(k, _ir.Node) else (), p))
    pinned: set[int] = set()

    def pin(v: Any) -> None:
        k = v if isinstance(v, _ir.Node) else _sym_key(v)
        if isinstance(k, _ir.Node):
            pinned.update(names[n] for n in k.free_symbols if n in names)

    for g, _ in guards:
        pin(g)
    for n in (n for t in (redone.temporaries if redone else ()) for n in (*t.sizes, *t.strides)):
        pin(n)
    position = {lo.roots[id(r)]: k for k, r in enumerate(roots)}
    temps = {b: m for m, (b, _) in enumerate(section.temporaries)} if section else {}
    roles = []
    for row, (_, x) in zip(rows, launches, strict=True):
        role = []
        values = x.slots if isinstance(x, (KernelLaunch, Memcpy)) else (x.slots[0],)
        for k, (slot, v) in enumerate(zip(row.slots, values)):
            if isinstance(slot, PointerSlot):
                own = next((r for r in x.roots if _sym_key(r.sym) is not None and lo.lowering.split(_sym_key(v), _sym_key(r.sym)) is not None), None)
                if own is None:
                    return None
                pin(lo.lowering.split(_sym_key(v), _sym_key(own.sym)))
                if slot.root in position:
                    role.append((k, ("root", position[slot.root])))
                elif slot.root[0] == "section" and slot.base in temps:
                    role.append((k, ("temp", temps[slot.base])))
                else:
                    return None
            else:
                pin(v)
        dims = (*x.grid, *x.block, x.smem) if isinstance(x, KernelLaunch) else (x.nbytes,) if isinstance(x, Memcpy) else (x.width, x.height, x.pitch)
        for d in dims:
            pin(d)
        roles.append(tuple(role))
    drops = tuple(d - op.allocs.start for d in section.drops) if section else ()
    return Template(predicate, tuple(guards), rows, tuple(roles), section, drops, tuple((p, roots[p]) for p in sorted(pinned)))


def _instantiate(lowered: LoweredTape, i: int, t: Template, roots: list[_Root], base: int, selector: LoweredSelector, op: OpRec) -> tuple[tuple, int]:
    """Template t as selector site i's entry: its launches with each role's
    pointer slot into this op's root at that position (or a fresh base per
    section temporary), at this selector's seqs; and the next free base."""
    lo = lowered.lowering
    refs = [lo.roots[id(r)] for r in roots]
    seqs = [lowered.launches[n].seq for n in selector.nodes]
    temps = tuple((base + m, row) for m, (_, row) in enumerate(t.section.temporaries)) if t.section else ()
    rows = []
    for j, (row, role) in enumerate(zip(t.rows, t.roles)):
        slots = list(row.slots)
        for k, (kind, p) in role:
            s = slots[k]
            if kind == "temp":
                slots[k] = PointerSlot(("section", temps[p][0]), s.displacement, None, temps[p][0])
                continue
            ref = refs[p]
            address = lo.emit("add", lo.row(roots[p].sym), s.displacement) if ref[0] == "argument" else None
            slots[k] = PointerSlot(ref, s.displacement, address, lo.base(ref))
        rows.append(replace(row, seq=seqs[0] if t.section else seqs[j], slots=tuple(slots)))
    section = LoweredSection(temps, tuple(op.allocs.start + d for d in t.drops)) if t.section else None
    return (i, t.predicate, tuple(rows), section), base + len(temps)


def _assert_same_entry(a: tuple, b: tuple, values: Sequence[int]) -> None:
    """verify_redo_templates: an instantiated entry and the full redo's are
    one entry at the call: predicate, each launch's slots (roots, bases and
    values), dimensions and extents, the section's bytes and drops."""

    def view(entry: tuple) -> tuple:
        _, predicate, rows, section = entry
        out: list[Any] = [values[predicate]]
        for row in rows:
            slots = tuple(
                (s.root, s.base, values[s.displacement], None if s.address is None else values[s.address]) if isinstance(s, PointerSlot) else values[s.row]
                for s in row.slots
            )
            if isinstance(row.launch, KernelLaunch):
                dims = tuple(values[r] for r in (*row.grid, *row.block, row.smem))
                out.append((type(row), row.launch.function, row.launch.attributes, slots, dims))
            else:
                extents = (values[row.nbytes],) if hasattr(row, "nbytes") else tuple(values[r] for r in (row.width, row.height, row.pitch))
                out.append((type(row), slots, extents))
        out.append(None if section is None else (tuple(values[r] for _, r in section.temporaries), section.drops))
        return tuple(out)

    if view(a) != view(b):
        raise AssertionError(f"host_trace: a redo template's entry for site {a[0]} differs from the full redo's: {view(a)} != {view(b)}")


_redo_fakes = threading.local()


@contextlib.contextmanager
def _hosting(device: torch.device) -> Iterator[None]:
    # as trace() runs a host: Triton and CuTe DSL launches intercepted, any other launch refused
    from torch.cuda._host_trace_cute import intercepting as cute_intercepting
    from torch.cuda._host_trace_triton_launch import intercepting

    with torch.cuda.device(device), _capture(device), intercepting(), cute_intercepting():
        yield


def _redo(
    lowered: LoweredTape,
    selector: LoweredSelector,
    ir: bool = True,
    hosted: bool = False,
    dispatches: dict[Any, list[tuple[_Trace, list[torch.SymInt], list[_Root]]]] | None = None,
    opaque: Sequence[Any] = (),
) -> tuple[list[Any], list[tuple[int, Any]], Section | None]:
    # ir off renames in sympy
    # dispatches: the family's dispatches by what they read of their call, each with its fresh symbols and roots
    old, lo = lowered.tape, lowered.lowering
    op: OpRec = old.ops[selector.op]
    if op.kind not in ("traced", "eager") or op.redo is None or torch.cuda._host_trace.symbolic != "ir":
        raise FoldRefused(f"{op.func} has no traced dispatch to run again")
    # the fresh symbols' env, as _Trace makes one: only a redo that dispatches builds the _Trace and its tensors
    env = _ir.Env()
    # no fresh symbol shares a name with the variant's: one the dispatch did
    # not get from its arguments (a closure's) is refused, not renamed
    env.unique_ids |= old.shape_env.unique_ids
    env._id_skips.update(getattr(old.shape_env, "_id_skips", {}))
    # an IR variant's entry takes the fresh trace's nodes renamed in the variant's context
    ir = ir and lo.ir
    key = _sym_key if ir else _sym_expr
    renamer = IRRename(old.shape_env, env, f"{op.func} dispatched on a symbol of no argument")
    # the fresh trace's symbol -> its formula in the variant's symbols (by name, a node, in ir)
    mapping: dict[Any, Any] = renamer.mapping if ir else {}
    fresh: dict[Any, Any] = {}
    roots: dict[int, _Root] = {}
    back: dict[int, _Root] = {}  # by the fresh root's id
    # the call's tensors (by the variant's tensor's id), the inputs and the outer tensors, built where it dispatches
    tensors: dict[int, tuple] = {}
    inputs: list[_InputRec] = []
    outer: dict[int, tuple] = {}
    made: list[torch.SymInt] = []
    at_call: list[int] = []  # each of `made`'s value: its formula's at the call, or a root's placeholder
    # what the dispatch reads of its call, in order: constants, the fresh symbols, roots and tensors
    shape: list[Any] = []

    def bind(sym: torch.SymInt, to: Any) -> None:
        if ir:
            n = _sym_key(to)
            mapping[sym.node.node.args[0]] = old.shape_env.ctx.const(n) if isinstance(n, int) else n
        else:
            mapping[_sym_expr(sym)] = _sym_expr(to)

    def value(v: Any) -> Any:
        if isinstance(v, torch.SymFloat) or (isinstance(v, torch.SymBool) and not (ir and isinstance(v.node, _ir.IRSymNode))):
            # a SymFloat's representation is the symbolic-float lane's (land/core/symfloat)
            raise FoldRefused(f"{op.func} takes a symbolic {type(v).__name__}")
        if isinstance(v, torch.SymBool) and v.node.constant is None:
            # a condition (a constexpr's, say): a fresh 0/1 symbol whose formula is where(condition, 1, 0) in the
            # variant's context, read by the op as `b != 0`; its guards are the op's, as a SymInt's
            k = v.node.node
            if k not in fresh:
                at_call.append(lo.program.values[lo.row(k)])
                b = env.symbol(at_call[-1], f"r{len(fresh)}")
                made.append(b)
                ctx = old.shape_env.ctx
                mapping[b.node.node.args[0]] = ctx.where(k, ctx.const(1), ctx.const(0))
                fresh[k] = b, b != 0
            shape.append(fresh[k][0])
            return fresh[k][1]
        if isinstance(v, torch.SymBool):
            v = bool(v.node.constant)
        if not isinstance(v, torch.SymInt):
            # 1, 1.0 and True are equal, as are 0.0 and -0.0
            shape.append((type(v), repr(v) if isinstance(v, float) else v))
            return v
        k = key(v)
        if k not in fresh:
            at_call.append(lo.program.values[lo.row(k)])
            fresh[k] = env.symbol(at_call[-1], f"r{len(fresh)}", positive=bool(_sym_expr(v).is_positive))
            made.append(fresh[k])
            bind(fresh[k], v)
        shape.append(fresh[k])
        return fresh[k]

    def root(r: _Root) -> _Root:
        if id(r) not in roots:
            if r.kind == "argument":
                base = lo.program.values[lo.row(r.sym)]
                at_call.append(_PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW))
                sym = env.symbol(at_call[-1], f"{r.name}.base")
                made.append(sym)
                bind(sym, r.sym)
            else:
                at_call.append(_placeholder(r) // _ALLOC_ALIGNMENT)
                fq = env.symbol(at_call[-1], f"{r.name}.base/{_ALLOC_ALIGNMENT}")
                made.append(fq)
                if ir:
                    (q,) = _sym_key(r.sym).free_symbols
                    mapping[fq.node.node.args[0]] = old.shape_env.ctx.symbols[q]
                else:
                    (q,) = _sym_expr(r.sym).free_symbols
                    mapping[_sym_expr(fq)] = q
                sym = _ALLOC_ALIGNMENT * fq
            roots[id(r)] = _Root(r.name, sym, r.kind, r.index, r.host)
            back[id(roots[id(r)])] = r
            if r.kind == "argument":
                # the input's record, as trace() has it: an as_strided bound, an overlap, copy-on-write
                rec = next(i for i in old.inputs if i.root is r)
                layout = [value(n) for n in rec.sizes], [value(s) for s in rec.strides], value(rec.offset)
                rec = replace(rec, sizes=layout[0], strides=layout[1], offset=layout[2], root=roots[id(r)])
                inputs.append(rec)
                shape.append((rec.dtype, rec.extent, rec.cow))
            else:
                # the storage, as trace() has it: an as_strided bound
                if r.kind == "allocation":
                    a = next(a for a in old.allocs if a.root is r)
                    sizes, strides, offset, dtype = a.sizes, a.strides, 0, a.dtype
                else:
                    o = next(o for o in old.eager_outputs if o._root is r)
                    sizes, strides, offset, dtype = o.shape, o._sym_strides, o._sym_offset, o.dtype
                layout = [value(n) for n in sizes], [value(s) for s in strides], value(offset)
                outer[id(roots[id(r)])] = (roots[id(r)], *layout, dtype, old.device)
                shape.append(dtype)
        shape.append(roots[id(r)])
        return roots[id(r)]

    def tensor(t: _TracedTensor) -> _TracedTensor:
        if id(t) not in tensors:
            sizes, strides = [value(n) for n in t.shape], [value(s) for s in t._sym_strides]
            tensors[id(t)] = (root(t._root), sizes, strides, value(t._sym_offset), t.dtype, t.device)
            shape.append((t.dtype, t.device))
        shape.append(t)
        return t

    # the hints: the formulas at the call, rows that fail at no other call
    with lo.symbolic.total():
        call = pytree.tree_map(lambda a: tensor(a) if isinstance(a, _TracedTensor) else value(a), op.call)
    # a dispatch that read the same of its call, whose guards hold at this one, is this one's: the op's records are a
    # function of what it reads (an entry's are taken wherever its predicate holds)
    sig, cached, nodes = None, None, 0
    resolved: dict[int, _TracedTensor] = {}  # the plain tensors the body reached resolved to arguments (_resolver)
    if dispatches is not None and ir and not any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in pytree.tree_leaves(op.call)):
        index = {k: i for i, k in enumerate([*map(id, (*made, *roots.values())), *tensors])}
        read = tuple((index[id(x)], x.node.node.args[0] in env.ctx.positive) if isinstance(x, torch.SymInt) else (index[id(x)],) if isinstance(x, (_Root, _TracedTensor)) else x for x in shape)
        sig = (op.func if isinstance(op.func, OpOverload) else op.redo, op.kind, repr(pytree.tree_structure(op.call)), read)
        try:
            hash(sig)
        except TypeError:
            sig = None
    if sig in (dispatches or {}):
        consts = _ir.Ctx()
        hints = [consts.const(x) for x in at_call]
        # the latest first: a layer takes the one before's
        for c in reversed(dispatches[sig]):
            at = {s.node.node.args[0]: h for s, h in zip(c[1], hints)}
            with contextlib.suppress(KeyError, ArithmeticError, ValueError):
                if all(consts.transfer(g, c[0].shape_env.ctx, at, {}).op == "true" for g, _ in c[0].shape_env.records):
                    cached = c
                    break
    if cached is not None:
        tr, made_then, roots_then = cached
        env = tr.shape_env
        renamer = IRRename(old.shape_env, env, renamer.refusal)
        renamer.mapping.update({s.node.node.args[0]: mapping[m.node.node.args[0]] for s, m in zip(made_then, made)})
        mapping = renamer.mapping
        back = {id(s): back[id(m)] for s, m in zip(roots_then, roots.values())}
    else:
        # FakeTensorMode() extracts the whole stack: a thread's redos reuse one, each with a fresh converter (a nested
        # redo makes its own)
        fake = vars(_redo_fakes).pop("mode", None) or FakeTensorMode(allow_fallback_kernels=False)
        fake.fake_tensor_converter = FakeTensorConverter(export=False)
        tr = _Trace(old.device, opaque=opaque, fake_mode=fake, shape_env=env)
        tr.values = CallValues(env, {m.node.node.args[0]: x for m, x in zip(made, at_call)})
        tr.inputs, tr.arguments = inputs, {id(i.root): i for i in inputs}
        tr.outer = {k: _TracedTensor(*o) for k, o in outer.items()}
        if torch.cuda._host_trace.redo_resolves_arguments:
            tr.resolve = _resolver(old, lo, tr, root, made, at_call, inputs, resolved)
        built = {k: _TracedTensor(*t) for k, t in tensors.items()}
        call = pytree.tree_map_only(_TracedTensor, lambda a: built[id(a)], call)
        if len(op.allocs):
            tr.first_alloc = old.allocs[op.allocs.start].root.index
        prior = getattr(_active, "trace", None)
        _active.trace = tr
        # an operator's dispatch runs where _TraceMode dispatched it, a launcher where the host called it
        mode = contextlib.nullcontext() if isinstance(op.func, OpOverload) else _TraceMode(tr)
        try:
            # hosted: inside the caller's _hosting, one capture for all its redos
            with contextlib.nullcontext() if hosted else _hosting(old.device):
                tr.stream = torch.cuda.current_stream(old.device)
                with mode, fx_config.patch(backed_size_oblivious=False):  # type: ignore[attr-defined]
                    op.redo(*call)
                from cuda.bindings import runtime

                graph = _check_cuda_bindings(runtime.cudaStreamGetCaptureInfo(tr.stream.cuda_stream))[2]
                # less the capture's own node (tape._capture's _sleep(0))
                nodes = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, numNodes=0))[1] - 1
                if nodes and hosted:
                    raise declined(f"the host enqueued {nodes} operations the trace does not record")
        except Exception as e:
            refused = isinstance(e, Declined) or (hosted and getattr(e, "error_code", None) in _CAPTURE_ERRORS)
            if not refused and torch.cuda._host_trace.raise_unexpected:
                raise
            raise FoldRefused(f"{op.func} declines again: {type(e).__name__}: {e}") from e
        finally:
            _active.trace = prior
            _redo_fakes.mode = fake
    if tr.declined is not None or _ir_census(tr):
        _drop_tracebacks(tr.declined)
        raise FoldRefused(f"{op.func} declines again: {tr.declined or _ir_census(tr)}")
    # its records are the entry's predicate
    _hint_audit.freeze(env, len(env.records))
    eager = op.kind == "eager"
    # the op's keyed sites (FZ12), in order
    span = {id(rec): j for j, (_, rec) in enumerate(old.launches[op.launches.start : op.launches.stop])}
    theirs = sorted((s for s in old.sites if s.nodes and id(s.nodes[0]) in span), key=lambda s: span[id(s.nodes[0])])
    if theirs and any(type(rec) is OpaqueCall for _, rec in tr.launches):
        # the call bound the sites' keys first (_fill): a nested call whose key none binds is another call
        raise FoldRefused(f"another keyed site in {op.func}: a call at a key no site binds")
    if len(tr.ops) != 1 or (tr.eager_outputs and not eager) or tr.ops[0].kind != op.kind:
        raise FoldRefused(f"{op.func} dispatches otherwise")
    # one whose guards read a symbol the dispatch made itself is never taken
    names = {s.node.node.args[0] for s in made}
    if sig is not None and cached is None and not nodes and not tr.sites and not resolved and all(g.free_symbols <= names for g, _ in env.records):
        dispatches.setdefault(sig, []).append((tr, made, list(roots.values())))
    new = tr.ops[0]
    if eager:
        # the call's fresh outputs are the variant's, one to one
        (_, call), (_, own) = tr.launches[0], old.launches[op.launches.start]
        for a, b in zip(call.outputs, own.outputs):
            if a._root.kind == "eager" and b._root.kind == "eager":
                if back.setdefault(id(a._root), b._root) is not b._root or sum(r is b._root for r in back.values()) > 1:
                    raise FoldRefused(f"{op.func} returned other metadata")
                if ir:
                    ((fq,), (q,)) = _sym_key(a._root.sym).free_symbols, _sym_key(b._root.sym).free_symbols
                    mapping[fq] = old.shape_env.ctx.symbols[q]
                else:
                    ((fq,), (q,)) = _sym_expr(a._root.sym).free_symbols, _sym_expr(b._root.sym).free_symbols
                    mapping[fq] = q
    # the op's keyed sites serve the redo's nested calls, one to one in order and in the same role: a site evaluates
    # its key from the call's rows, so the entry holds the op's other launches and allocations
    if len(tr.sites) != len(theirs) or not all(map(_one_role, tr.sites, theirs)):
        raise FoldRefused(f"another keyed site in {op.func}")
    buffers = [(x._root, y._root) for s, v in zip(tr.sites, theirs) for x, y in zip(s.operands[len(s.call[1]) :], v.operands[len(v.call[1]) :])]
    buffers += [(s.scratch[j][0]._root, v.scratch[j][0]._root) for s, v in zip(tr.sites, theirs) for j in s.scratch]
    theirs_buffers = {id(t._root) for v in theirs for t in (*v.operands[len(v.call[1]) :], *(t for t, _ in v.scratch.values()))}
    keyed = {id(n) for s in (*tr.sites, *theirs) for n in s.nodes}
    mine = {id(a.root): a for a in tr.allocs}
    olds = old.allocs[op.allocs.start : op.allocs.stop]
    allocated = {id(b.root): b for b in olds}
    for x, y in buffers:
        if id(x) not in mine or id(y) not in allocated:
            raise FoldRefused(f"a keyed site's buffer {op.func} does not allocate")
        bind(mine[id(x)].q, allocated[id(y)].q)
        back[id(x)] = y
    # the op's own: its allocations and launches but its sites'
    ours = [(k, b) for k, b in zip(op.allocs, olds) if id(b.root) not in theirs_buffers]
    olds = [b for _, b in ours]
    tr_allocs = [a for a in tr.allocs if id(a.root) not in {id(x) for x, _ in buffers}]
    redone = [(seq, x) for seq, x in tr.launches if id(x) not in keyed]

    def shaped(x: Any, y: Any) -> bool:
        # what a graph exec fixes at instantiation: a kernel's launch attributes (a cluster), programmatic edge and
        # compiled cluster, a memset's element size
        if type(x) is not type(y):
            return False
        if isinstance(x, KernelLaunch):
            same = x.function == y.function or compiled_cluster(x.function) == compiled_cluster(y.function)
            return (x.attributes, x.programmatic) == (y.attributes, y.programmatic) and same
        if isinstance(x, Memcpy):
            return x.kind == y.kind
        return not isinstance(x, Memset) or x.element_size == y.element_size

    section: Section | None = None
    chain = [rec for _, rec in old.launches[op.launches.start : op.launches.stop] if id(rec) not in keyed]
    counts = len(redone) == len(chain) and len(tr_allocs) == len(olds)
    # another launch chain (more or fewer launches or allocations, or another topology: a split count's cluster), or a
    # memcpy, which only a section's own node holds
    if not counts or not all(map(shaped, (x for _, x in redone), chain)) or any(isinstance(x, Memcpy) for _, x in redone):
        if eager or not ir:
            raise FoldRefused(f"{op.func} dispatches otherwise")
        if not getattr(torch._C, "_host_trace_section_entries", False):
            raise FoldRefused(f"{op.func} dispatches otherwise; the native replay takes no section entries")
        if not selector.nodes:
            raise FoldRefused(f"{op.func} dispatches otherwise; it has no launch a section replaces")
        if selector.nodes != tuple(range(selector.nodes[0], selector.nodes[-1] + 1)):
            # a keyed site's nodes among the op's (FZ12): a piece replaces one chain
            raise FoldRefused(f"{op.func} dispatches otherwise; its launches are not one chain")
        seq = lowered.launches[selector.nodes[0]].seq
        # an allocation an output is maps to the variant's for that output; the rest are the section's own
        roles = {id(b.root): b for b in olds}
        mine = {id(a.root): a for a in tr_allocs}
        pairs = {id(x._root): roles[id(y._root)] for x, y in zip(new.outputs, op.outputs) if isinstance(x, _TracedTensor) and isinstance(y, _TracedTensor) and id(x._root) in mine and id(y._root) in roles}
        if len({id(b) for b in pairs.values()}) != len(pairs):
            raise FoldRefused(f"{op.func} returns one allocation as two")
        temps: list[Temporary] = []
        ctx = old.shape_env.ctx
        for a in tr_allocs:
            if id(a.root) in pairs:
                b = pairs[id(a.root)]
                bind(a.q, b.q)
                back[id(a.root)] = b.root
                continue
            # a symbol of the variant's context no guard reads: the temporary's address, as an allocation's
            mapping[a.q.node.node.args[0]] = ctx.sym(f"section{len(ctx.symbols)}", _placeholder(a.root) // _ALLOC_ALIGNMENT)
            temp = _Root(a.root.name, renamer(a.root.sym), "allocation", a.root.index)
            back[id(a.root)] = temp
            temps.append(Temporary(temp, a.dtype, tuple(map(renamer, a.sizes)), tuple(map(renamer, a.strides)), seq))
        # the op's allocations no output of the section is: it does not make them
        paired = {id(b) for b in pairs.values()}
        section = Section(tuple(temps), tuple(k for k, b in ours if id(b) not in paired))
        olds = [pairs[id(a.root)] for a in tr_allocs if id(a.root) in pairs]
        allocs = [a for a in tr_allocs if id(a.root) in pairs]
    else:
        allocs = tr_allocs
        for a, b in zip(allocs, olds):
            bind(a.q, b.q)
            back[id(a.root)] = b.root

    def rename(v: Any) -> Any:
        if ir:
            return renamer(v)
        e = _sym_expr(v)
        if not e.free_symbols <= mapping.keys():
            raise FoldRefused(f"{op.func} dispatched on a symbol of no argument")
        e = e.xreplace(mapping)
        return int(e) if e.is_Integer else e

    if ir:
        # Env records, as the variant's own guards, and its checks of the call's values (Tape.checks)
        records = [*env.records, *((c.node.node, c.node.written) for c in tr.checks)]
        guards = [r for r in map(renamer.record, records) if r[0].op != "true"]
        false = any(r[0].op == "false" for r in guards)
    else:
        exprs = [*(g.expr for g in env.guards), *(c.node.expr for c in tr.checks)]
        guards = [g for g in map(rename, exprs) if g is not sympy.true]
        false = sympy.false in guards
    if false:
        raise AssertionError(f"host_trace: {op.func}'s dispatch again guards a condition the call fails")
    # a symbol the redo's guards pin (s == 1 at a size-1 call): its layouts and the variant's agree under them
    ctx, pins = old.shape_env.ctx if ir else None, {}
    for g, _ in guards if ir else ():
        c, ts = ctx.as_terms(g.args[0]) if g.op == "eq" else (0, {})
        if len(ts) == 1 and (t := next(iter(ts))).op == "sym" and abs(ts[t]) == 1:
            pins[t.args[0]] = ctx.const(-c * ts[t])
    subs = {**ctx.symbols, **pins} if pins else {}

    def pinned(v: Any) -> Any:
        v = ctx.transfer(v, ctx, subs, {}) if isinstance(v, _ir.Node) else v
        return v.args[0] if isinstance(v, _ir.Node) and v.op == "const" else v

    def same(x: Any, y: Any) -> bool:
        if isinstance(x, _TracedTensor) != isinstance(y, _TracedTensor):
            return False
        if not isinstance(x, _TracedTensor):
            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                return x is y
            return rename(x) == key(y) if isinstance(x, torch.SymInt) or isinstance(y, torch.SymInt) else x == y
        layout = [*map(rename, x.shape), *map(rename, x._sym_strides), rename(x._sym_offset)]
        want = [*map(key, y.shape), *map(key, y._sym_strides), key(y._sym_offset)]
        agree = layout == want or (pins and list(map(pinned, layout)) == list(map(pinned, want)))
        return back.get(id(x._root)) is y._root and x.dtype == y.dtype and agree

    for a, b in zip(allocs, olds):
        # as same() compares an output: under the redo's pins ((1, n) against (s, n) at s == 1)
        layout, want = [*map(rename, a.sizes), *map(rename, a.strides)], [*map(key, b.sizes), *map(key, b.strides)]
        if a.dtype != b.dtype or (layout != want and not (pins and list(map(pinned, layout)) == list(map(pinned, want)))):
            raise FoldRefused(f"{op.func} allocates otherwise")
    if len(new.outputs) != len(op.outputs) or not all(map(same, new.outputs, op.outputs)):
        raise FoldRefused(f"{op.func} returned other metadata")
    if not all(same(x, y) for s, v in zip(tr.sites, theirs) for x, y in zip((*s.operands, *s.scalars), (*v.operands, *v.scalars))):
        raise FoldRefused(f"another keyed site in {op.func}")
    if eager:
        # a Triton or CuTe DSL launch: its function, then its grid and options
        # or streams, compared as its arguments are (as fold compares them)
        split = [(c.target[:2], c.target[2:]) if isinstance(c.target, tuple) else (c.target, ()) for c in (call, own)]
        new_leaves, new_spec = pytree.tree_flatten((split[0][1], call.args, call.kwargs))
        old_leaves, old_spec = pytree.tree_flatten((split[1][1], own.args, own.kwargs))
        if split[0][0] != split[1][0] or call.generator is not own.generator or new_spec != old_spec or not all(map(same, new_leaves, old_leaves)):
            raise FoldRefused(f"another eager call of {op.func}")
        # the variant's own call stands; only the guards that select it are new
        return guards, [], None
    launches = []
    olds = [rec for _, rec in old.launches[op.launches.start : op.launches.stop]]
    descriptors = getattr(torch._C, "_host_trace_entry_descriptors_enabled", bool)()
    if section is not None:
        # a section's launches take the op's place: each reads roots the op's own launches hold (the plan keeps
        # them live there) or its temporaries
        held = {id(r) for y in olds for r in y.roots} | {id(t.root) for t in section.temporaries}
        for _, x in redone:
            if isinstance(x, KernelLaunch) and (x.rng or x.rng_increment or x.cpu_scalars or (x.descriptors and not descriptors)):
                raise FoldRefused(f"an RNG, TMA or CPU scalar launch in {op.func}'s section")
            if isinstance(x, Memcpy) and x.kind != "D2D":
                raise FoldRefused(f"a host memcpy in {op.func}'s section")
            owned = tuple(back.get(id(r)) for r in x.roots)
            if any(r is None or id(r) not in held for r in owned):
                raise FoldRefused(f"a launch in {op.func}'s section of a root its op does not hold")
            launches.append((seq, _renamed(x, owned, rename)))
        return guards, launches, section
    for n, (_, x), y in zip(selector.nodes, redone, chain, strict=True):
        if isinstance(x, KernelLaunch) and (x.rng or x.rng_increment or x.cpu_scalars):
            raise FoldRefused(f"an RNG or CPU scalar launch in {op.func}")
        if isinstance(x, KernelLaunch) and x.descriptors and not descriptors:
            # an entry holds no TMA descriptor without entry descriptors
            raise FoldRefused(f"a TMA launch in {op.func}")
        if isinstance(x, Memcpy):
            raise FoldRefused(f"a memcpy in {op.func}")
        if any(id(r) not in back for r in x.roots):
            raise FoldRefused(f"a launch in {op.func} of a root of no role")
        owned = tuple(back[id(r)] for r in x.roots)
        topology = type(x) is type(y) and {id(r) for r in owned} == {id(r) for r in y.roots}
        if topology and isinstance(x, KernelLaunch):
            topology = (x.attributes, x.programmatic) == (y.attributes, y.programmatic)
            topology = topology and (x.function == y.function or compiled_cluster(x.function) == compiled_cluster(y.function))
        elif topology:
            topology = x.element_size == y.element_size and rename(x.height) == key(y.height)
        if not topology:
            raise FoldRefused(f"another launch topology in {op.func}")
        launches.append((lowered.launches[n].seq, _renamed(x, owned, rename)))
    return guards, launches, None


def _resolver(
    old: Any, lo: Any, tr: _Trace, root: Callable[[_Root], _Root], made: list, at_call: list, inputs: list, resolved: dict[int, _TracedTensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """redo_resolves_arguments: a plain CUDA tensor the redo's body reaches
    becomes the stand-in of the call's tensor argument of the same storage
    object and view (dtype, sizes, strides, storage offset), where exactly one
    is: that argument's root at its input record's layout in fresh symbols, as
    the redo builds any argument it reads. Any other tensor is left as it is
    (it declines as before)."""
    by_storage: dict[int, list[tuple[_InputRec, torch.Tensor]]] = {}
    for rec in old.inputs:
        a = lo.program.inputs[rec.position]
        if isinstance(a, torch.Tensor):
            by_storage.setdefault(a.untyped_storage()._cdata, []).append((rec, a))

    def resolve(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, _TracedTensor) or t.device.type != "cuda":
            return t
        if id(t) in resolved:
            return resolved[id(t)]
        view = (t.dtype, t.shape, t.stride(), t.storage_offset())
        found = [rec for rec, a in by_storage.get(t.untyped_storage()._cdata, ()) if (a.dtype, a.shape, a.stride(), a.storage_offset()) == view]
        if len(found) != 1:
            return t
        n = len(made)
        with lo.symbolic.total():
            r = root(found[0].root)
        for m, x in zip(made[n:], at_call[n:]):
            tr.values.bind(m, x)
        rec = next(i for i in inputs if i.root is r)
        tr.arguments[id(r)] = rec
        resolved[id(t)] = _TracedTensor(r, list(rec.sizes), list(rec.strides), rec.offset, t.dtype, t.device)
        return resolved[id(t)]

    return resolve


@dataclass(frozen=True)
class Temporary:
    """A section's temporary: its root (a symbol of the variant's context no
    guard reads), dtype, sizes and strides in the variant's symbols, and the
    seq of the op's first launch, which the section replaces."""

    root: _Root
    dtype: torch.dtype
    sizes: tuple[Any, ...]
    strides: tuple[Any, ...]
    seq: int


@dataclass(frozen=True)
class Section:
    """A redo's launches in its op's place: its temporaries, and the op's
    allocations (indices) it does not make."""

    temporaries: tuple[Temporary, ...]
    drops: tuple[int, ...]


def _one_role(s: KeyedSite, v: KeyedSite) -> bool:
    """Whether a redo's keyed site s is the variant's site v's call but for its
    operands' and scalars' formulas (_redo compares those): the op, provider,
    library state, call structure and scratch buffers. An RNG site, or a
    symbolic bool or float scalar, is no site's."""
    symbolic = (torch.SymBool, torch.SymFloat)
    if s.rng or v.rng or s.call is None or any(isinstance(a, symbolic) for a in (*s.scalars, *v.scalars)):
        return False
    same = (s.op, s.state, s.call, len(s.operands), len(s.scalars)) == (v.op, v.state, v.call, len(v.operands), len(v.scalars))
    return same and s.provider is v.provider and s.scratch.keys() <= v.scratch.keys()


def _renamed(x: Any, roots: tuple[_Root, ...], rename: Callable[[Any], Any]) -> Any:
    # a kernel launch, memset or memcpy with its values renamed and its roots `roots`
    if isinstance(x, KernelLaunch):
        geometry = (tuple(map(rename, x.grid)), tuple(map(rename, x.block)), rename(x.smem))
        return replace(x, slots=tuple(map(rename, x.slots)), roots=roots, grid=geometry[0], block=geometry[1], smem=geometry[2])
    if isinstance(x, Memcpy):
        return replace(x, slots=tuple(map(rename, x.slots)), roots=roots, nbytes=rename(x.nbytes))
    return replace(x, slots=(rename(x.slots[0]),), roots=roots, width=rename(x.width), height=rename(x.height), pitch=rename(x.pitch))
