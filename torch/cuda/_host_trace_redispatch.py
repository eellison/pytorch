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

import collections
import contextlib
import dataclasses
import threading
import time
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
from torch.cuda._host_trace_program import _read_leaf, _step, compile_program, LEAVES, Status
from torch.cuda._host_trace_tape import (
    _active,
    _ALLOC_ALIGNMENT,
    _capture,
    _CAPTURE_ERRORS,
    _InputRec,
    _ir_census,
    _library_body,
    _metadata,
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
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from contextlib import AbstractContextManager

    from torch.cuda._host_trace_lower_tape import LoweredLaunch, LoweredMemset, LoweredSelector, LoweredTape
    from torch.cuda._host_trace_opaque import KeyedSite
    from torch.cuda._host_trace_tape import OpRec


class RedoCall:
    """What HostTraceReplay's redo_context gets around a library body's redo that reads lifted state: the call's flat
    arguments (the redo's stand-ins at the lifted positions, the call's own objects elsewhere), the op's operator, and its
    rank among the tape's ops of that operator. A context that reads the rank makes the redo its op's alone: its entry is
    no other op's (redo template, family dispatch, shared confirmation)."""

    def __init__(self, args: tuple[Any, ...], func: Any, rank: int) -> None:
        self.args, self.func, self._rank = args, func, rank
        self.rank_read = False

    @property
    def rank(self) -> int:
        self.rank_read = True
        return self._rank


@dataclass(frozen=True)
class RedoOptions:
    """HostTraceReplay's redo_policy, lifted, redo_context and static_prefix."""

    policy: str = "inputs_only"
    lifted: frozenset[int] = frozenset()
    context: Callable[[RedoCall], AbstractContextManager[Any]] | None = None
    static_prefix: int = 0


REDO_POLICIES = ("inputs_only", "strict", "refuse")


# test and diagnosis: redos a dispatch of the family served (no body run), and redo bodies run
dispatch_hits = 0
redo_bodies = 0
# test and diagnosis: _for_instance's calls, those of another read walk length (no renaming), cache hits, computed instances
# (their pins, renamed guards, guards trimmed of their pinned renames, declined ones and seconds)
for_instance_stats: collections.Counter = collections.Counter()


def redispatch(
    lowered: LoweredTape,
    args: Sequence[Any],
    values: Sequence[int],
    unselected: Sequence[int],
    dispatches: dict | None = None,
    opaque: Sequence[Any] = (),
    static_prefix: int = 0,
    redo: RedoOptions = RedoOptions(),
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...], Any]]]:
    """Selectors `unselected` (native site indices) of the variant, whose
    graph guards hold the call `args` (at rows `values`), as entries from
    their ops dispatched again: fold's result, with no trace. An op whose redo
    would read what an earlier redo of this variant read (Template), where
    that entry's predicate holds, takes that entry with its own roots. A
    redo of an op holding keyed sites routes calls to the replay's `opaque`
    providers, as its trace did. The first `static_prefix` arguments are
    the replay's static prefix (a model's parameters, the same tensors every
    call): a template keys their layouts by value, and another layer's op
    takes it with its own (_for_instance). A library body redoes under
    `redo`'s policy (_library_redo_refusal)."""
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
                if redo.policy == "strict" and (why := _python_decided(lowered, selector)) is not None:
                    lo.refused.add(i)
                    e = FoldRefused(why)
                    e.op = selector.op
                    raise e
                # before a template too: another op's template is this op's redo
                if _library_body(op.func) and (why := _library_redo_refusal(lowered, selector, opaque, redo)) is not None:
                    lo.refused.add(i)
                    e = FoldRefused(why if isinstance(why, _OwnRefusal) else f"{op.func} dispatches otherwise; {why}")
                    e.op = selector.op
                    raise e
                declared = _declared(lowered, static_prefix)
                if selector.op not in lo.reads:
                    lo.reads[selector.op] = _reads(lowered, op, selector, declared)
                reads = lo.reads[selector.op]
                template = lo.templates.get(reads[0]) if reads is not None else None
                if template is not None:
                    # its roots at this op's positions: the same object at a pin, another static-prefix argument (its
                    # guards renamed onto it, its keyed layout values checked), else the full redo
                    template = _for_instance(lowered, template, reads[1], static_prefix, reads[3])
                if template is not None and template.untrimmed is not None and program.values[template.untrimmed] != program.values[template.predicate]:
                    raise AssertionError(f"host_trace: {op.func}'s instance trimmed of its pinned renames disagrees with the untrimmed one at the call")
                if template is not None and program.values[template.predicate] == 1 and _bound(lowered, reads[2], program.values):
                    entry, after = _instantiate(lowered, i, template, reads[1], base, selector, op)
                    if torch.cuda._host_trace.verify_redo_templates:
                        full = lower_entry(lowered, i, *_full_redo(lowered, i, selector, dispatches, opaque, redo), base)[0]
                        _assert_same_entry(entry, full, program.values)
                    lo.instances += 1
                    entries.append(entry)
                    recorded.append((i, template.guards))
                    base = after
                    continue
                guards, launches, section = _full_redo(lowered, i, selector, dispatches, opaque, redo)
                entry, base = lower_entry(lowered, i, guards, launches, section, base)
                lo.redos += 1
                entries.append(entry)
                recorded.append((i, tuple(guards)))
                if reads is not None and selector.op not in lo.rank_read and (template := _template(lowered, entry, guards, launches, section, reads[1], op)) is not None:
                    lo.templates[reads[0]] = replace(template, substituted=reads[3])
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


def _symbols(v: Any, out: set) -> set:
    """The free symbols (IR names) of a record's values: SymInts, IR nodes, roots, and dataclass records' fields."""
    if isinstance(v, torch.SymInt):
        k = _sym_key(v)
        if isinstance(k, _ir.Node):
            out |= k.free_symbols
        elif isinstance(k, sympy.Basic):
            out |= {str(x) for x in k.free_symbols}
    elif isinstance(v, _ir.Node):
        out |= v.free_symbols
    elif isinstance(v, sympy.Basic):
        out |= {str(x) for x in v.free_symbols}
    elif isinstance(v, _Root):
        _symbols(v.sym, out)
    elif isinstance(v, (tuple, list, frozenset, set)):
        for x in v:
            _symbols(x, out)
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        for f in dataclasses.fields(v):
            if f.name not in ("owner", "images", "function", "abi", "provider"):
                _symbols(getattr(v, f.name), out)
    return out


def _python_decided(lowered: LoweredTape, selector: LoweredSelector) -> str | None:
    """redo_policy "strict": a failing own guard of the op that a library body's Python decided on (recorded or deduped
    there, not in an ATen op's C++ host; Env.python_decided), whose outcome may have reached Python state other code reads,
    which a redo of this op alone leaves as the trace set it. Rows per own guard (selector.rows); where they do not line
    up, any such guard counts."""
    op, env = lowered.tape.ops[selector.op], lowered.tape.shape_env
    decided = getattr(env, "python_decided", ())
    if not decided:
        return None
    values = lowered.lowering.program.values
    rows = selector.rows if len(selector.rows) == len(op.guards) else (None,) * len(op.guards)
    for g, row in zip(op.guards, rows):
        if g in decided and (row is None or row >= len(values) or values[row] != 1):
            return f"redo_policy strict: {op.func}'s failing guard {repr(env.records[g][0])[:80]} was decided by a library body's Python"
    return None


def _lifted_symbols(tape: Any, lifted: frozenset[int]) -> set:
    """The symbols of the call's arguments at the lifted positions: tensors' layouts and bases, ints."""
    out: set = set()
    for rec in tape.inputs:
        if rec.position in lifted:
            _symbols((rec.sizes, rec.strides, rec.offset, rec.root.sym), out)
    for rec in tape.int_inputs:
        if rec.position in lifted:
            _symbols(rec.sym, out)
    return out


def _lifted(lowered: LoweredTape, redo: RedoOptions) -> set:
    """_lifted_symbols, once per lowering."""
    lo = lowered.lowering
    if lo.lifted_symbols is None:
        lo.lifted_symbols = _lifted_symbols(lowered.tape, redo.lifted)
    return lo.lifted_symbols


def _reads_lifted(lowered: LoweredTape, k: int, redo: RedoOptions) -> bool:
    """Whether op k's records (launches, allocations, owned guards) read a symbol of the lifted arguments (cached per lowering)."""
    lo, tape = lowered.lowering, lowered.tape
    if not redo.lifted:
        return False
    if k not in lo.reads_lifted:
        op, env = tape.ops[k], tape.shape_env
        records: set = set()
        for j in op.launches:
            _symbols(tape.launches[j][1], records)
        for a in op.allocs:
            _symbols((tape.allocs[a].q, tape.allocs[a].sizes, tape.allocs[a].strides), records)
        for g in op.guards:
            _symbols(env.records[g][0] if isinstance(env, _ir.Env) else env.guards[g].expr, records)
        lo.reads_lifted[k] = not records.isdisjoint(_lifted(lowered, redo))
    return lo.reads_lifted[k]


def _tensor_symbols(lowered: LoweredTape) -> set:
    """The symbols of the tape's tensor layouts: its inputs' and every allocation's (once per lowering)."""
    lo, tape = lowered.lowering, lowered.tape
    if lo.tensor_symbols is None:
        out: set = set()
        for rec in tape.allocs:
            _symbols((rec.q, rec.sizes, rec.strides), out)
        for rec in tape.inputs:
            _symbols((rec.sizes, rec.strides, rec.offset, rec.root.sym), out)
        lo.tensor_symbols = out
    return lo.tensor_symbols


def _scalar_read(lowered: LoweredTape, k: int, allowed: set) -> str | None:
    """A free symbol of op k's records (launches, allocations, owned guards) that reached it as a Python scalar: a call
    int input that is not one of the op's own call values, or a symbol of no input. A redo of a library body reads such
    a value from wherever its body found it (a holder its traced function filled from an argument), at the call, and
    bakes it into the entry with no guard. Symbols of the op's call, its allocations, and the layouts of tensors (call
    inputs, other ops' allocations) are not: a tensor a body reaches is tracked (resolved to its argument, or declined);
    nor are `allowed` (the lifted arguments' symbols, which a redo_context installs)."""
    tape = lowered.tape
    op = tape.ops[k]
    own: set = set()
    for leaf in pytree.tree_leaves(op.call):
        if isinstance(leaf, _TracedTensor):
            _symbols((tuple(leaf.shape), tuple(leaf.stride()), leaf.storage_offset(), leaf._root), own)
        elif isinstance(leaf, torch.SymInt):
            _symbols(leaf, own)
    for a in op.allocs:
        _symbols((tape.allocs[a].q, tape.allocs[a].sizes, tape.allocs[a].strides), own)
    tensors = _tensor_symbols(lowered)
    records: set = set()
    for j in op.launches:
        _symbols(tape.launches[j][1], records)
    env = tape.shape_env
    for g in op.guards:
        _symbols(env.records[g][0] if isinstance(env, _ir.Env) else env.guards[g].expr, records)
    return next((s for s in sorted(records - own - allowed) if s not in tensors), None)


def _library_redo_refusal(lowered: LoweredTape, selector: LoweredSelector, opaque: Sequence[Any], redo: RedoOptions) -> str | None:
    """Why a library body's redo is refused, or None (once per traced op, cached): a symbol of its records that came as a
    Python scalar (_scalar_read); else its confirmation, the op redone once at the trace call itself (the variant's rows
    evaluated at the trace's own arguments) giving back its recorded launches launch for launch with no section, and no
    guard beyond its trace's but over its own symbols. A body that read a holder records a constant where its trace has a
    symbol there, whatever the values. A lifted argument's symbol counts as the op's own where a redo_context installs it
    (the redo and the confirmation run in it); redo_policy "refuse" refuses every library body."""
    lo, tape = lowered.lowering, lowered.tape
    k = selector.op
    if k in lo.library_redo:
        return lo.library_redo[k]
    why = None
    if redo.policy == "refuse":
        why = _OwnRefusal(f"redo_policy refuse: {tape.ops[k].func} is a library body")
    elif (sym := _scalar_read(lowered, k, set() if redo.context is None else _lifted(lowered, redo))) is not None:
        lifted = redo.context is None and sym in _lifted(lowered, redo)
        why = f"its records read {sym}, " + ("lifted state, with no redo_context" if lifted else "which its call does not carry (a Python value a redo would bake)")
    else:
        start = time.perf_counter()
        reads = _reads(lowered, tape.ops[k], selector, _declared(lowered, redo.static_prefix)) if lo.ir else None
        lo.reads[k] = reads
        candidates = lo.confirmed.get(reads[0], ()) if reads is not None else ()
        if any(_records_match(lowered, j, roots, k, reads[1], redo) for j, roots in candidates):
            _LIBRARY_COUNTS["shared"] += 1
        else:
            why = _confirmation(lowered, selector, opaque, redo)
            _LIBRARY_COUNTS["confirmations"] += 1
            _LIBRARY_COUNTS["confirmations with no confirmed key" if not candidates else "confirmations past a confirmed key"] += 1
            if why is None and reads is not None and k not in lo.rank_read:
                lo.confirmed.setdefault(reads[0], []).append((k, reads[1]))
        _LIBRARY_COUNTS["confirmation_s"] += time.perf_counter() - start
    lo.library_redo[k] = why
    _LIBRARY_COUNTS["refused" if why else "confirmed"] += 1
    return why


def _records_match(lowered: LoweredTape, a: int, roots_a: list[_Root], b: int, roots_b: list[_Root], redo: RedoOptions) -> bool:
    """Whether op b's records are op a's with a's read-walk roots renamed onto b's by position (a bijection; their base
    symbols one for one), the static prefix's layout symbols at their values (redo templates key them so): every launch
    field, allocation and owned guard, constants exactly. Ops of one redo template key redo alike (the property templates
    rest on), so b's confirmation is a's: b redone at its trace call is a's redo renamed, which gives back a's records
    renamed, b's."""
    tape = lowered.tape
    ctx = tape.shape_env.ctx
    if len(roots_a) != len(roots_b):
        _LIBRARY_COUNTS["share_miss:roots"] += 1
        return False
    rename: dict[str, str] = {}
    for x, y in zip(roots_a, roots_b):
        kx, ky = _sym_key(x.sym), _sym_key(y.sym)
        if not isinstance(kx, _ir.Node) or not isinstance(ky, _ir.Node) or len(kx.free_symbols) != 1 or len(ky.free_symbols) != 1:
            _LIBRARY_COUNTS["share_miss:root base"] += 1
            return False
        (nx,), (ny,) = kx.free_symbols, ky.free_symbols
        if rename.setdefault(nx, ny) != ny:
            _LIBRARY_COUNTS["share_miss:bijection"] += 1
            return False
    if len(set(rename.values())) != len(rename):
        _LIBRARY_COUNTS["share_miss:bijection"] += 1
        return False
    declared = _declared(lowered, redo.static_prefix)
    hints = {n: ctx.const(int(ctx.symbols[n].hint)) for n in declared}
    at_b = collections.ChainMap(hints, ctx.symbols)
    from_a = collections.ChainMap({nx: ctx.symbols[ny] for nx, ny in rename.items()}, at_b)
    position_a = {id(r): p for p, r in enumerate(roots_a)}
    position_b = {id(r): p for p, r in enumerate(roots_b)}

    def canon(v: Any, symbols: Any) -> Any:
        k = _sym_key(v) if isinstance(v, (torch.SymInt, torch.SymBool, torch.SymFloat)) else v
        if isinstance(k, _ir.Node):
            return ctx.transfer(k, ctx, symbols, {})
        if isinstance(k, _Root):
            return ("root", (position_a if symbols is from_a else position_b)[id(k)])  # KeyError: a root of no read
        if isinstance(k, (tuple, list)):
            return tuple(canon(x, symbols) for x in k)
        if dataclasses.is_dataclass(k) and not isinstance(k, type):
            # a field the class leaves out of its own equality (compare=False: a TMA descriptor's last-encode cache) is no record
            fields = (f for f in dataclasses.fields(k) if f.compare and f.name not in ("name", "witness", "owner", "images", "abi"))
            return (type(k), *(canon(getattr(k, f.name), symbols) for f in fields))
        return k

    def same(x: Any, y: Any) -> bool:
        try:
            return bool(canon(x, from_a) == canon(y, at_b))
        except Exception:  # a root of no read, a formula that does not transfer, a field with no equality: not shared
            return False

    op_a, op_b = tape.ops[a], tape.ops[b]
    if len(op_a.launches) != len(op_b.launches) or len(op_a.allocs) != len(op_b.allocs) or len(op_a.guards) != len(op_b.guards):
        _LIBRARY_COUNTS["share_miss:counts"] += 1
        return False
    env = tape.shape_env
    if any(getattr(rec, "cpu_scalars", ()) for _, rec in tape.launches[op_b.launches.start : op_b.launches.stop]):
        _LIBRARY_COUNTS["share_miss:cpu scalars"] += 1
        return False  # re-read from a CPU tensor at each call (KernelLaunch.cpu_scalars, compare=False): its own
    for (_, x), (_, y) in zip(tape.launches[op_a.launches.start : op_a.launches.stop], tape.launches[op_b.launches.start : op_b.launches.stop]):
        if not same(x, y):
            field = next((f.name for f in dataclasses.fields(x) if f.compare and not same(getattr(x, f.name), getattr(y, f.name))), "type") if dataclasses.is_dataclass(x) and type(x) is type(y) else "type"
            _LIBRARY_COUNTS[f"share_miss:launch {field}"] += 1
            return False
    for p, q in zip(tape.allocs[op_a.allocs.start : op_a.allocs.stop], tape.allocs[op_b.allocs.start : op_b.allocs.stop]):
        if not same((p.dtype, p.sizes, p.strides, p.root), (q.dtype, q.sizes, q.strides, q.root)):
            _LIBRARY_COUNTS["share_miss:allocation"] += 1
            return False
    # the owned guards as sets (their order is the trace's recording order, which dedup with earlier ops moves)
    try:
        guards_a = {canon(env.records[g][0], from_a) for g in op_a.guards}
        guards_b = {canon(env.records[h][0], at_b) for h in op_b.guards}
    except Exception:
        guards_a, guards_b = None, ()
    if guards_a != guards_b:
        _LIBRARY_COUNTS["share_miss:guard"] += 1
        return False
    return True


def _confirmation(lowered: LoweredTape, selector: LoweredSelector, opaque: Sequence[Any], redo: RedoOptions) -> str | None:
    lo, tape = lowered.lowering, lowered.tape
    op = tape.ops[selector.op]
    program = lo.program
    inputs = _trace_call(tape, program.inputs)
    values = _trace_values(lo, inputs)
    if values is None:
        return "confirmation: the variant's rows do not evaluate at its trace call"
    global redo_bodies
    bodies = redo_bodies  # a confirmation's body is counted as one (_LIBRARY_COUNTS), not as a redispatch's redo body
    with _rolled_back(lo):
        program.inputs, program.values = inputs, values
        try:
            guards, launches, section = _redo(lowered, selector, hosted=True, dispatches=None, opaque=opaque, redo=redo)
        except FoldRefused as e:
            return _OwnRefusal(str(e))  # the op's redo refuses itself, for its own reason
        except (HintsFail, Declined) as e:
            return f"confirmation: its redo at its trace call failed ({type(e).__name__}: {str(e)[:160]})"
        finally:
            _LIBRARY_COUNTS["confirmation_bodies"] += redo_bodies - bodies
            redo_bodies = bodies
    keyed = {id(n) for site in tape.sites for n in site.nodes}
    chain = [rec for _, rec in tape.launches[op.launches.start : op.launches.stop] if id(rec) not in keyed]
    if len(launches) != len(chain):
        return f"confirmation: {len(launches)} launches at its trace call for its {len(chain)}"
    # the static prefix's layouts are the same every call: a value the redo read off a declared view (close_over) at its
    # constant layout equals the trace's at their values
    declared = _declared(lowered, redo.static_prefix) if lo.ir else frozenset()
    fold = _at_values(tape.shape_env.ctx, declared) if declared else None
    fold_x, roots = fold, None
    if section is not None:
        # a section (a memcpy only a section's node holds, say) confirms where it gives back the trace's records but for its
        # scratch: its temporaries are the op's dropped allocations, in allocation order (notes/SECTION_CONFIRMATION.md)
        if not lo.ir or isinstance(paired := _section_pairing(tape, op, section), str):
            return f"confirmation: its redo at its trace call takes another chain ({paired if lo.ir else 'sympy'})"
        roots, scratch = paired
        ctx = tape.shape_env.ctx
        fold, fold_x, rename = _at_values(ctx, declared), _at_values(ctx, declared, scratch), _at_values(ctx, frozenset(), scratch)
        for t, k in zip(section.temporaries, section.drops):
            a = tape.allocs[k]
            if t.dtype != a.dtype or not _same_value((t.sizes, t.strides), (tuple(a.sizes), tuple(a.strides)), lambda u, v: fold_x(u) == fold(v)):
                return f"confirmation: its redo at its trace call takes another chain (a temporary is not its allocation's layout)"
    for j, ((_, x), y) in enumerate(zip(launches, chain)):
        if (diff := _launch_difference(x, y, fold, fold_x=fold_x, roots=roots)) is not None:
            return f"confirmation: launch {j} differs at its trace call ({diff})"
    env = tape.shape_env
    recorded = {node for node, _ in env.records} if isinstance(env, _ir.Env) else {g.expr for g in env.guards}
    own = _symbols(op.call, set())
    for a in op.allocs:
        _symbols((tape.allocs[a].q, tape.allocs[a].sizes, tape.allocs[a].strides), own)
    # an IR redo's guards are (node, written) pairs, a sympy one's its expressions
    exprs = [g[0] if isinstance(g, tuple) else g for g in guards]
    if section is not None:
        # renamed, not folded: the membership test below judges a guard by its symbols, and a declared symbol at its value makes it a constant
        exprs = [rename(g) for g in exprs]
    # and a guard over what a declared view's read reads besides: the static prefix's layouts and bases (the same every call)
    for rec in tape.inputs if declared else ():
        if rec.position < redo.static_prefix:
            _symbols(rec.root.sym, own)
    extra = [g for g in exprs if g not in recorded and not ((free := _symbols(g, set())) and free <= own | declared)]
    if extra:
        return f"confirmation: {len(extra)} guards its trace did not record ({repr(extra[0])[:80]})"
    return None


@contextlib.contextmanager
def _rolled_back(lo: Any) -> Any:
    """The variant's program and lowering caches as they were before the block (a confirmation's redo appends rows and
    memoizes their expressions; none of it is the variant's): the rows, their values and dedup keys, the domains, the
    inputs, the expression memos, guard rows, sources and leaves."""
    program, lowering = lo.program, lo.lowering
    n = len(program.instructions)
    domains = None if program.domains is None else len(program.domains)
    inputs, values = program.inputs, program.values
    guard_rows, sources, leaves = len(lowering.guard_rows), set(lowering.sources), len(lo._leaves)
    try:
        yield
    finally:
        for row in program.instructions[n:]:
            program._rows.pop(row, None)
        del program.instructions[n:]
        if domains is not None:
            del program.domains[domains:]
        program.inputs, program.values = inputs, values
        for cache in (lowering._memo, lowering._float_memo):
            for key in [k for k, row in cache.items() if row >= n]:
                del cache[key]
        lowering.guard_rows[:] = [r for r in lowering.guard_rows if r < n]
        for key in set(lowering.sources) - sources:
            del lowering.sources[key]
        del lo._leaves[leaves:]
        if torch.cuda._host_trace.verify_redo_templates and (
            len(program.instructions) != n or len(lowering.guard_rows) > guard_rows
            or any(row >= n for cache in (lowering._memo, lowering._float_memo) for row in cache.values())
        ):
            raise AssertionError("host_trace: a confirmation left a row of its own in the variant's lowering")


def _trace_values(lo: Any, inputs: tuple) -> list[int] | None:
    """Every row's value at the trace call (a copy: a confirmation's redo appends its own): the rows' values at the trace
    call are a function of the rows and the trace call's metadata, so the lowering keeps them and evaluates only the rows
    added since (rows are only appended; a confirmation's own are rolled back past them)."""
    program = lo.program
    cached, last = lo.trace_values or (None, None)
    rows = program.instructions
    # a rollback cuts a suffix of the rows: a kept value past the cut was of a row it removed (another row object now)
    if cached is None or len(cached) > len(rows) or (cached and rows[len(cached) - 1] is not last):
        cached = _values_at(program, inputs)
    elif len(cached) < len(rows):
        cached = _values_at(program, inputs, cached)
    lo.trace_values = None if cached is None else (cached, rows[len(cached) - 1] if cached else None)
    if cached is not None and torch.cuda._host_trace.verify_redo_templates and cached != _values_at(program, inputs):
        raise AssertionError("host_trace: the kept trace-call row values disagree with a fresh evaluation")
    return None if cached is None else list(cached)


def _values_at(program: Any, inputs: tuple, prefix: list[int] | None = None) -> list[int] | None:
    """Every row's value at `inputs`, as IntegerProgram.emit evaluates a row (None where one fails); with `prefix`, the
    first rows' values, evaluating only the rest."""
    values: list[int] = list(prefix or ())
    for row in program.instructions[len(values) :]:
        op, *operands = row
        if op == "constant":
            value = operands[0]
        elif op in LEAVES:
            if not 0 <= operands[0] < len(inputs) or (value := _read_leaf(row, inputs[operands[0]])) is None:
                return None
        else:
            value = _step(op, [values[x] for x in operands])
            if isinstance(value, Status):
                return None
        values.append(value)
    return values


def _trace_call(tape: Any, current: Sequence[Any]) -> tuple:
    """The trace call's arguments: a released tensor (release_args' sizes, strides, offset, dtype, base) as the call's own
    tensor where that has exactly this layout at this address (a static argument: a body's read of module state resolves
    to it by object, as at the call), else as a tensor at that address and layout over a storage it does not own (only its
    metadata and address are read)."""
    args = []
    for i, a in enumerate(tape.args):
        if isinstance(a, tuple) and len(a) == 5 and isinstance(a[3], torch.dtype):
            now = current[i] if i < len(current) else None
            if isinstance(now, torch.Tensor) and now.dtype == a[3] and _metadata(now)[:3] == (tuple(a[0]), tuple(a[1]), a[2]) and _metadata(now)[4] == a[4]:
                a = now
            else:
                shape, stride, offset, dtype, base = a
                itemsize = torch.empty((), dtype=dtype).element_size()
                numel = offset + sum((n - 1) * st for n, st in zip(shape, stride)) + 1 if all(shape) else offset
                storage = torch._C._construct_storage_from_data_pointer(base, tape.device, numel * itemsize)
                a = torch.empty(0, dtype=dtype, device=tape.device).set_(storage, offset, shape, stride)
        args.append(a)
    return tuple(args)


def _at_values(ctx: Any, declared: frozenset[str], renamed: Mapping[str, Any] | None = None) -> Callable[[Any], Any]:
    """A value as a node with the `declared` symbols at their values (and the `renamed` symbols as those nodes)."""
    hints = {name: ctx.const(int(ctx.symbols[name].hint)) for name in declared}
    symbols = collections.ChainMap(dict(renamed or {}), hints, ctx.symbols)
    names = declared | set(renamed or ())

    def fold(v: Any) -> Any:
        k = _sym_key(v)
        if isinstance(k, int):
            return ctx.const(k)
        if not isinstance(k, _ir.Node) or not k.free_symbols & names:
            return k
        try:
            return ctx.transfer(k, ctx, symbols, {})
        except (KeyError, ArithmeticError, ValueError):
            return k

    return fold


def _launch_difference(
    x: Any, y: Any, fold: Callable[[Any], Any] | None = None, *, fold_x: Callable[[Any], Any] | None = None, roots: Mapping[int, Any] | None = None
) -> str | None:
    """The first field where a redo's record x and the trace's y differ (values by node, roots by identity), or None;
    with `fold`, values equal as fold gives them count as equal (x's through `fold_x`, if given); `roots` maps a root of x
    (by id) to the trace's root it stands for (a section's temporary to its allocation)."""
    fx = fold_x or fold
    same = lambda u, v: _sym_key(u) == _sym_key(v) or (fold is not None and fx(u) == fold(v))  # noqa: E731
    if type(x) is not type(y):
        return f"{type(x).__name__} for {type(y).__name__}"
    if [id((roots or {}).get(id(r), r)) for r in getattr(x, "roots", ())] != [id(r) for r in getattr(y, "roots", ())]:
        return "roots"
    if isinstance(x, KernelLaunch):
        if x.function != y.function:
            return "function"
        for name, a, b in (("grid", x.grid, y.grid), ("block", x.block, y.block), ("smem", (x.smem,), (y.smem,)), ("slots", x.slots, y.slots)):
            if len(a) != len(b) or not all(same(u, v) for u, v in zip(a, b)):
                return name
        # and the rest of the record (TMA descriptors, parameter layout, attributes, RNG): not the name, the CUfunction's own
        # abi / owner / images, or what the class leaves out of its equality (caches, the capture's witness)
        skip = ("name", "function", "abi", "grid", "block", "smem", "slots", "roots", "owner", "images")
        for f in dataclasses.fields(x):
            if f.compare and f.name not in skip and not _same_value(getattr(x, f.name), getattr(y, f.name), same):
                return f.name
        return None
    for f in dataclasses.fields(x):
        if f.name in ("roots", "name"):
            continue
        u, v = getattr(x, f.name), getattr(y, f.name)
        us, vs = (u, v) if isinstance(u, tuple) else ((u,), (v,))
        if len(us) != len(vs) or not all(same(p, q) for p, q in zip(us, vs)):
            return f.name
    return None


def _section_pairing(tape: Any, op: Any, section: Section) -> tuple[dict[int, Any], dict[str, Any]] | str:
    """A section's temporaries paired with the op's dropped allocations in allocation order (no search): per pair the
    temporary's root standing for the allocation's, and its base symbol renamed to the allocation's; why not, where the
    counts differ, an output is among the drops, or a base is not one symbol."""
    if len(section.temporaries) != len(section.drops):
        return "its temporaries do not pair with its allocations"
    outputs = {id(o._root) for o in op.outputs if isinstance(o, _TracedTensor)}
    ctx = tape.shape_env.ctx
    roots: dict[int, Any] = {}
    scratch: dict[str, Any] = {}
    for t, k in zip(section.temporaries, section.drops):
        a = tape.allocs[k]
        if id(a.root) in outputs:
            return "an output among its temporaries"
        kt, ka = _sym_key(t.root.sym), _sym_key(a.root.sym)
        if not isinstance(kt, _ir.Node) or not isinstance(ka, _ir.Node) or len(kt.free_symbols) != 1 or len(ka.free_symbols) != 1:
            return "a temporary's base is not one symbol"
        ((nt,), (na,)) = kt.free_symbols, ka.free_symbols
        roots[id(t.root)] = a.root
        scratch[nt] = ctx.symbols[na]
    return roots, scratch


def _same_value(u: Any, v: Any, same: Callable[[Any, Any], bool]) -> bool:
    """A record field's values equal: SymInts and ints by `same`, tuples elementwise, dataclasses by their compared fields."""
    if isinstance(u, (tuple, list)) and isinstance(v, (tuple, list)):
        return len(u) == len(v) and all(_same_value(p, q, same) for p, q in zip(u, v))
    if dataclasses.is_dataclass(u) and not isinstance(u, type):
        return type(u) is type(v) and all(_same_value(getattr(u, f.name), getattr(v, f.name), same) for f in dataclasses.fields(u) if f.compare)
    if isinstance(u, (int, torch.SymInt, _ir.Node)) and isinstance(v, (int, torch.SymInt, _ir.Node)) and not isinstance(u, bool):
        return same(u, v)
    return u is v or u == v


class _Resolving(TorchFunctionMode):
    """A redo's resolver (_resolver) at the torch-function layer."""

    def __init__(self, resolve: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.resolve = resolve

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        args, kwargs = pytree.tree_map_only(torch.Tensor, self.resolve, (args, kwargs or {}))
        return func(*args, **kwargs)


class _OwnRefusal(str):
    """A confirmation's redo refused itself: its reason, as the op's redo would give it."""


# library-body redos decided, by outcome (tests and the startup harness read them)
_LIBRARY_COUNTS: collections.Counter = collections.Counter()


def _full_redo(
    lowered: LoweredTape, i: int, selector: LoweredSelector, dispatches: dict | None, opaque: Sequence[Any] = (), redo: RedoOptions = RedoOptions()
) -> tuple[list[Any], list[tuple[int, Any]], Section | None]:
    try:
        try:
            return _redo(lowered, selector, hosted=True, dispatches=dispatches, opaque=opaque, redo=redo)
        except HintsFail:
            return _redo(lowered, selector, ir=False, hosted=True, opaque=opaque, redo=redo)
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
    # the read walk's roots of the op whose redo it is (_for_instance maps them onto another op's)
    roots: tuple[_Root, ...] = ()
    # the declared symbols its key took at their values (_reads): pinned there where another op takes it
    substituted: frozenset[str] = frozenset()
    # verify_redo_templates: an instance's predicate with every renamed guard (_for_instance trims those renamed only through
    # pinned layout symbols), checked equal to its predicate at each call that takes it
    untrimmed: int | None = None


def _reads(lowered: LoweredTape, op: OpRec, selector: LoweredSelector, declared: frozenset[str] = frozenset()) -> tuple[Any, list[_Root], tuple[int, ...]] | None:
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
    op is redone in full. A SymInt over `declared` symbols alone (the static
    prefix's layouts) is keyed by its value; an argument's extent by its
    length; then the roots its recorded launch chain holds (a body's reads
    of module state: weights, the step's metadata), so an op of another layer
    takes the template with its own roots at those positions."""
    func = op.func
    if op.kind != "traced" or op.redo is None or not isinstance(func, OpOverload) or not lowered.lowering.ir:
        return None
    old = lowered.tape
    ctx = old.shape_env.ctx
    items: list[Any] = [func, repr(pytree.tree_structure(op.call)), len(selector.nodes)]
    roots: list[_Root] = []
    seen: dict[int, int] = {}
    # the declared symbols a formula over others too is keyed at (their hints): pinned at them where taken (_for_instance)
    substituted: set[str] = set()

    def value(v: Any) -> None:
        if isinstance(v, torch.SymFloat) or (isinstance(v, torch.Tensor) and not isinstance(v, _TracedTensor)):
            raise _Unclassified
        if isinstance(v, torch.SymBool):
            items.append(("bool", v.node.node) if v.node.constant is None else (bool, bool(v.node.constant)))
        elif isinstance(v, torch.SymInt):
            k = _sym_key(v)
            fs = k.free_symbols if isinstance(k, _ir.Node) else frozenset()
            if fs and fs <= declared:
                items.append(("value", int(k.hint)))
            elif fs & declared:
                # a formula over a static-prefix layout symbol and a call's (a weight's dimension times the rows): each
                # layer's weight has its own symbol of the same value, so the declared ones are keyed at their values
                hints = {name: ctx.const(int(ctx.symbols[name].hint)) for name in fs & declared}
                substituted.update(hints)
                try:
                    items.append(("int", ctx.transfer(k, ctx, collections.ChainMap(hints, ctx.symbols), {})))
                except (KeyError, ArithmeticError, ValueError) as e:
                    raise _Unclassified from e
            else:
                items.append(("int", k))
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
            # the extent's length: its place is the root's (a run's address, not a read)
            items.append((rec.dtype, rec.extent[1] - rec.extent[0], rec.cow))
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
        items.append("chain roots")
        for _, rec in old.launches[op.launches.start : op.launches.stop]:
            for r in getattr(rec, "roots", ()):
                root(r)
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
    return key, roots, sites, frozenset(substituted)


def _declared(lowered: LoweredTape, static_prefix: int) -> frozenset[str]:
    """The symbols of the static prefix's layouts (sizes, strides, offsets)."""
    lo = lowered.lowering
    got = lo.declared.get(static_prefix)
    if got is None:
        names: set[str] = set()
        for rec in lowered.tape.inputs:
            if rec.position < static_prefix:
                for v in (*rec.sizes, *rec.strides, rec.offset):
                    k = _sym_key(v)
                    if isinstance(k, _ir.Node):
                        names |= k.free_symbols
        got = lo.declared[static_prefix] = frozenset(names)
    return got


def _for_instance(lowered: LoweredTape, t: Template, roots: list[_Root], static_prefix: int, substituted: frozenset[str] = frozenset()) -> Template | None:
    """Template t for the op whose read walk gave `roots`, or None (a full
    redo). Each position where t's op read a static-prefix argument: this op
    reads one too, and its layout symbols and t's op's hold the values the key
    took (value pins, rows at each call: a changed prefix misses); where it is
    another argument, t's guards reading t's op's (its address: an alignment
    check, its layout) are renamed onto it and join the predicate, and a pin
    there moves to it. A pin on any other root needs the same object."""
    lo, old = lowered.lowering, lowered.tape
    for_instance_stats["calls"] += 1
    if len(t.roots) != len(roots):
        for_instance_stats["unrenamed"] += 1
        # (a substituted symbol needs its pins: a full redo)
        return t if not (t.substituted or substituted) and all(roots[p] is r for p, r in t.pinned) else None
    cache_key = (id(t), tuple(map(id, roots)), substituted)
    if cache_key in lo.instance_templates:
        for_instance_stats["cached"] += 1
        return lo.instance_templates[cache_key]
    start = time.perf_counter()
    recs = lo.first_inputs
    ctx = old.shape_env.ctx
    subs: dict[str, _ir.Node] = {}
    # the declared symbols both keys took at their values hold them at the call (a static prefix's layouts)
    pins: list[Any] = [(ctx.eq(ctx.symbols[name], ctx.const(int(ctx.symbols[name].hint))), None) for name in sorted(t.substituted | substituted)]
    moved: set[int] = set()
    # the layout symbols pinned at their values (both ops': their keys hold them at the same values), and the static-prefix
    # roots whose layouts are pinned (each root's pins one row, made once per lowering: _root_pin_row)
    pinned: set[str] = set()
    root_pins: list[Any] = []
    out: Template | None = t
    for p, (mine, theirs) in enumerate(zip(t.roots, roots)):
        a = recs.get(id(mine)) if mine.kind == "argument" else None
        if a is None or a.position >= static_prefix:
            continue
        b = recs.get(id(theirs)) if theirs.kind == "argument" else None
        if b is None or b.position >= static_prefix:
            out = None
            break
        for rec in (a, b):
            if all(rec is not r for r in root_pins):
                root_pins.append(rec)
            for v in (*rec.sizes, *rec.strides, rec.offset):
                k = _sym_key(v)
                if isinstance(k, _ir.Node) and k.op == "sym":
                    pinned.add(k.args[0])
        if theirs is not mine:
            moved.add(p)
            for x, y in zip((a.root.sym, *a.sizes, *a.strides, a.offset), (b.root.sym, *b.sizes, *b.strides, b.offset)):
                kx, ky = _sym_key(x), _sym_key(y)
                if isinstance(kx, _ir.Node) and kx.op == "sym":
                    subs[kx.args[0]] = ky if isinstance(ky, _ir.Node) else ctx.const(int(ky))
    if out is not None and not all(roots[p] is r or p in moved for p, r in t.pinned):
        out = None
    if out is not None and (pins or subs or root_pins):
        full, memo = collections.ChainMap(subs, ctx.symbols), {}
        # a template guard renamed only through pinned layout symbols is the guard itself under the pins (each symbol
        # and its rename are pinned at one value), which t.predicate holds already: only a guard reading a renamed symbol
        # no pin holds (a root's address: its alignment checks) is renamed and conjoined
        renaming = subs.keys() - pinned
        renamed = [(ctx.transfer(g, ctx, full, memo), None) for g, _ in t.guards if isinstance(g, _ir.Node) and g.free_symbols & renaming]

        def instance_predicate(renamed: list, grouped: bool = True) -> int | None:
            guards = [g for g in (*renamed, *pins) if g[0].op != "true"]
            if not grouped:
                guards += [g for r in root_pins for g in _root_pins(lowered, r) if g[0].op != "true"]
            predicate = t.predicate
            if guards:
                try:
                    with lo.lowering.total() as domains:
                        predicate = lo.emit("and", predicate, lo.lowering.conjunction(guards, lo.addresses))
                    for ok in domains:
                        predicate = lo.emit("and", predicate, ok)
                except Declined:
                    return None  # a renamed guard false at the hints: this op takes another branch
            if grouped:
                for r in root_pins:
                    if (row := _root_pin_row(lowered, r)) is not None:
                        predicate = lo.emit("and", predicate, row)
            return predicate

        predicate = instance_predicate(renamed)
        if torch.cuda._host_trace.verify_redo_templates:
            # the untrimmed predicate (every guard reading a renamed symbol renamed) holds at this call exactly when
            # the trimmed one does
            every = [(ctx.transfer(g, ctx, full, memo), None) for g, _ in t.guards if isinstance(g, _ir.Node) and g.free_symbols & subs.keys()]
            untrimmed = instance_predicate(every, grouped=False)
            values = lo.program.values
            if (predicate is None) != (untrimmed is None) or (predicate is not None and values[predicate] != values[untrimmed]):
                raise AssertionError(f"host_trace: {t} trimmed of its pinned renames disagrees with the untrimmed instance")
        for_instance_stats["trimmed"] += sum(1 for g, _ in t.guards if isinstance(g, _ir.Node) and g.free_symbols & subs.keys() and not g.free_symbols & renaming)
        out = None if predicate is None else replace(t, predicate=predicate, pinned=tuple((p, roots[p]) for p, _ in t.pinned), roots=tuple(roots), untrimmed=untrimmed if torch.cuda._host_trace.verify_redo_templates else None)
        for_instance_stats["pins"] += len(pins) + sum(len(_root_pins(lowered, r)) for r in root_pins)
        for_instance_stats["renamed"] += len(renamed)
    for_instance_stats["computed"] += 1
    for_instance_stats["declined"] += out is None
    for_instance_stats["computed_s"] += time.perf_counter() - start
    lo.instance_templates[cache_key] = out
    return out


def _root_pins(lowered: LoweredTape, rec: Any) -> list[tuple[Any, None]]:
    """The pins of a static-prefix argument record (LoweredTape.lowering.first_inputs'): each of its layout formulas
    equal to its value. Made once per lowering."""
    lo = lowered.lowering
    got = lo.root_pins.get(id(rec))
    if got is None:
        ctx = lowered.tape.shape_env.ctx
        got = lo.root_pins[id(rec)] = [(ctx.eq(k, ctx.const(int(k.hint))), None) for v in (*rec.sizes, *rec.strides, rec.offset) if isinstance(k := _sym_key(v), _ir.Node)]
    return got


def _root_pin_row(lowered: LoweredTape, rec: Any) -> int | None:
    """The row that is 1 when the record's pins (_root_pins) hold, or None for none; made once per lowering: every
    instance taking a template onto or from the root conjoins it."""
    lo = lowered.lowering
    if id(rec) not in lo.root_pin_rows:
        pins = [g for g in _root_pins(lowered, rec) if g[0].op != "true"]
        row = None
        if pins:
            with lo.lowering.total() as domains:
                row = lo.lowering.conjunction(pins, lo.addresses)
            for ok in domains:
                row = lo.emit("and", row, ok)
        lo.root_pin_rows[id(rec)] = row
    return lo.root_pin_rows[id(rec)]


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
    return Template(predicate, tuple(guards), rows, tuple(roles), section, drops, tuple((p, roots[p]) for p in sorted(pinned)), tuple(roots))


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
    redo: RedoOptions = RedoOptions(),
) -> tuple[list[Any], list[tuple[int, Any]], Section | None]:
    # ir off renames in sympy
    # dispatches: the family's dispatches by what they read of their call, each with its fresh symbols and roots
    old, lo = lowered.tape, lowered.lowering
    op: OpRec = old.ops[selector.op]
    # the op's keyed sites (FZ12), in order: its redo routes calls to the providers, as its trace did
    span = {id(rec): j for j, (_, rec) in enumerate(old.launches[op.launches.start : op.launches.stop])}
    theirs = sorted((s for s in old.sites if s.nodes and id(s.nodes[0]) in span), key=lambda s: span[id(s.nodes[0])])
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
                # what a redo reads of an input's extent (its place is the variant's trace's): its length, whether it is
                # empty, and (the dispatch key's) its overlap with the other inputs (_Trace._eager_call's written/read
                # test; test_a_redo_reads_extents_only_for_overlap keeps it the only reader)
                first, last = rec.extent
                shape.append((rec.dtype, last - first, first <= last, rec.cow))
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
        # a library body that reads lifted state redoes inside redo_context, its holders holding stand-ins for the lifted
        # arguments (built as the call's are, so its read of them is a read of the call)
        stand_ins: dict[int, Any] | None = None
        if redo.context is not None and _library_body(op.func) and _reads_lifted(lowered, selector.op, redo):
            stand_ins = {}
            for rec in old.inputs:
                if rec.position in redo.lifted:
                    stand_ins[rec.position] = (root(rec.root), [value(n) for n in rec.sizes], [value(s) for s in rec.strides], value(rec.offset), rec.dtype, old.device)
            for irec in old.int_inputs:
                if irec.position in redo.lifted:
                    stand_ins[irec.position] = value(irec.sym)
            shape.append(("redo_context", tuple(sorted(stand_ins))))
    # a dispatch that read the same of its call, whose guards hold at this one, is this one's: the op's records are a
    # function of what it reads (an entry's are taken wherever its predicate holds)
    sig, cached, nodes = None, None, 0
    resolved: dict[int, _TracedTensor] = {}  # the plain tensors the body reached resolved to arguments (_resolver)
    if dispatches is not None and ir and not any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in pytree.tree_leaves(op.call)):
        index = {k: i for i, k in enumerate([*map(id, (*made, *roots.values())), *tensors])}
        read = tuple((index[id(x)], x.node.node.args[0] in env.ctx.positive) if isinstance(x, torch.SymInt) else (index[id(x)],) if isinstance(x, (_Root, _TracedTensor)) else x for x in shape)
        live = [(i, r.extent) for i, r in enumerate(inputs) if r.extent[0] <= r.extent[1]]
        overlaps = tuple((i, j) for i, a in live for j, b in live if i < j and a[0] <= b[1] and b[0] <= a[1])
        sig = (op.func if isinstance(op.func, OpOverload) else op.redo, op.kind, repr(pytree.tree_structure(op.call)), read, overlaps)
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
                # its guards and its checks of the call's values (both become the entry's guards)
                held = [*(g for g, _ in c[0].shape_env.records), *(ch.node.node for ch in c[0].checks)]
                if all(consts.transfer(g, c[0].shape_env.ctx, at, {}).op == "true" for g in held):
                    cached = c
                    break
    global dispatch_hits, redo_bodies
    if cached is not None:
        dispatch_hits += 1
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
        tr = _Trace(old.device, opaque=opaque if theirs else (), fake_mode=fake, shape_env=env)
        tr.values = CallValues(env, {m.node.node.args[0]: x for m, x in zip(made, at_call)})
        tr.inputs, tr.arguments = inputs, {id(i.root): i for i in inputs}
        tr.outer = {k: _TracedTensor(*o) for k, o in outer.items()}
        if torch.cuda._host_trace.redo_resolves_arguments or torch.cuda._host_trace._CLOSURES:
            tr.resolve = _resolver(old, lo, op, tr, root, value, made, at_call, resolved, redo.static_prefix)
        built = {k: _TracedTensor(*t) for k, t in tensors.items()}
        call = pytree.tree_map_only(_TracedTensor, lambda a: built[id(a)], call)
        if len(op.allocs):
            tr.first_alloc = old.allocs[op.allocs.start].root.index
        prior = getattr(_active, "trace", None)
        _active.trace = tr
        # an operator's dispatch runs where _TraceMode dispatched it, a launcher where the host called it
        mode = contextlib.nullcontext() if isinstance(op.func, OpOverload) else _TraceMode(tr)
        context = contextlib.ExitStack()
        if tr.resolve is not None:
            # at each torch function the body calls, before autograd sees a plain tensor
            context.enter_context(_Resolving(tr.resolve))
        if stand_ins is not None:
            args = list(lo.program.inputs)
            for p, x in stand_ins.items():
                args[p] = _TracedTensor(*x) if isinstance(x, tuple) else x
            redo_call = RedoCall(tuple(args), op.func, sum(o.func == op.func for o in old.ops[: selector.op]))
            context.enter_context(redo.context(redo_call))  # type: ignore[misc]
        redo_bodies += 1
        try:
            # hosted: inside the caller's _hosting, one capture for all its redos
            with contextlib.nullcontext() if hosted else _hosting(old.device):
                tr.stream = torch.cuda.current_stream(old.device)
                with mode, fx_config.patch(backed_size_oblivious=False), context:  # type: ignore[attr-defined]
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
            if stand_ins is not None and redo_call.rank_read:
                lo.rank_read.add(selector.op)
    if tr.declined is not None or _ir_census(tr):
        _drop_tracebacks(tr.declined)
        raise FoldRefused(f"{op.func} declines again: {tr.declined or _ir_census(tr)}")
    # its records are the entry's predicate
    _hint_audit.freeze(env, len(env.records))
    eager = op.kind == "eager"
    if theirs and any(type(rec) is OpaqueCall for _, rec in tr.launches):
        # the call bound the sites' keys first (_fill): a nested call whose key none binds is another call
        raise FoldRefused(f"another keyed site in {op.func}: a call at a key no site binds")
    if len(tr.ops) != 1 or (tr.eager_outputs and not eager) or tr.ops[0].kind != op.kind:
        raise FoldRefused(f"{op.func} dispatches otherwise")
    # one whose guards read a symbol the dispatch made itself is never taken
    names = {s.node.node.args[0] for s in made}
    if sig is not None and cached is None and not nodes and not tr.sites and not resolved and selector.op not in lo.rank_read and all(g.free_symbols <= names for g, _ in env.records):
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
        # the op's allocations no output of the section is: it does not make them, so no other launch may read one (a library
        # body's scratch a later op reaches through Python state)
        paired = {id(b) for b in pairs.values()}
        dropped = {id(b.root) for _, b in ours if id(b) not in paired}
        if any(id(r) in dropped for j, (_, rec) in enumerate(old.launches) if j not in op.launches for r in getattr(rec, "roots", ())):
            raise FoldRefused(f"{op.func}'s section drops an allocation another op reads")
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
    old: Any, lo: Any, op: OpRec, tr: _Trace, root: Callable[[_Root], _Root], value: Callable[[Any], Any], made: list, at_call: list, resolved: dict[int, _TracedTensor], static_prefix: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A plain tensor the redo's body reaches, as a stand-in on one of the
    call's arguments' roots (in fresh symbols, as the redo builds any argument
    it reads): with redo_resolves_arguments, one that is a tensor argument (the
    object), at that argument's input record's layout; one close_over declared
    on the storage of a static-prefix argument (the same object every call), at
    its own layout, constant while declared, on that argument's root (where
    several static-prefix arguments share the storage, the one the op's
    recorded launches read). Any other tensor is left as it is (it declines as
    before)."""
    ht = torch.cuda._host_trace
    by_object: dict[int, _InputRec] = {}
    by_storage: dict[int, list[_InputRec]] = {}
    for rec in old.inputs:
        a = lo.program.inputs[rec.position]
        if isinstance(a, torch.Tensor):
            if ht.redo_resolves_arguments or (rec.position < static_prefix and ht.closed_over(a)):
                by_object.setdefault(id(a), rec)
            if rec.position < static_prefix and ht._CLOSURES:
                by_storage.setdefault(a.untyped_storage()._cdata, []).append(rec)
    read = {id(r) for _, launch in old.launches[op.launches.start : op.launches.stop] for r in getattr(launch, "roots", ())}

    def on_storage(t: torch.Tensor) -> _InputRec | None:
        recs = by_storage.get(t.untyped_storage()._cdata, ())
        roots = {id(rec.root): rec for rec in recs}
        if len(roots) > 1:
            roots = {k: rec for k, rec in roots.items() if k in read}
        return next(iter(roots.values())) if len(roots) == 1 else None

    def stand_in(t: torch.Tensor, rec: _InputRec, layout: Callable[[], tuple]) -> _TracedTensor:
        n = len(made)
        with lo.symbolic.total():
            r = root(rec.root)
            sizes, strides, offset = layout()
        for m, x in zip(made[n:], at_call[n:]):
            tr.values.bind(m, x)
        tr.arguments.setdefault(id(r), next(i for i in tr.inputs if i.root is r))
        return _TracedTensor(r, sizes, strides, offset, t.dtype, t.device)

    def resolve(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, _TracedTensor) or t.device.type != "cuda":
            return t
        if id(t) not in resolved:
            if (rec := by_object.get(id(t))) is not None:
                resolved[id(t)] = stand_in(t, rec, lambda: ([value(s) for s in rec.sizes], [value(s) for s in rec.strides], value(rec.offset)))
            elif by_storage and ht.closed_over(t) and (rec := on_storage(t)) is not None:
                resolved[id(t)] = stand_in(t, rec, lambda: (list(t.shape), list(t.stride()), t.storage_offset()))
            else:
                return t
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
