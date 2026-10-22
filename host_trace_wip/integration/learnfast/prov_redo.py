"""RETIRED on step_ms34+ (core agent 10-20; memset/notes/INTEGRATION_RETIREMENT.md): the core covers it. Kept for older cores.

PROTOTYPE (learnfast, 10-14; the user's idea via the coordinator): opt-in "redo only ops whose records depend on
their own inputs", in place of no_lib_redo's blanket refusal of library-body redos. Diagnostic / measurement use.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import prov_redo

A library-body op (no_lib_redo.library_body: a non-ATen operator or a dispatch unit) is redone at a redispatch only if
  1. statically, every symbol of its records (launches, allocations, the guards it owns) is one of its own call's or its
     own allocations' (opdeps "args_only"), or, with PROV_CLOSURE=1 (closure over the declared inputs on), also of the
     replay's static prefix ("args_or_declared"); and
  2. its first redispatch in the process confirms it: the op redone once at the variant's own trace values (each symbol's
     hint), outside any context, must give back its recorded launches (function, grid, block, smem, every slot, roots) and
     decide no guard the trace did not record, with no section. A read of anything else shows as a constant where the
     trace has a symbol, a tensor the redo cannot track, or another chain: the op (by operator name) is refused for the
     process. The confirmation never makes an entry.
Everything else keeps no_lib_redo's refusal. No redo_context.

PROV_SECTION_MEMO=1: diag27 also imports section_memo.py (check_section refusals memoized per selector), after its own
check_section patches.
STATS counts decisions."""
from __future__ import annotations

import collections
import dataclasses
import os
import sys
import weakref

import torch
import torch.cuda._host_trace_ir as IR
import torch.cuda._host_trace_redispatch as _rd
import torch.cuda._host_trace_replay as _hr
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import FoldRefused
from torch.cuda._host_trace_tape import _sym_key, _TracedTensor
from torch.utils import _pytree as pytree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_lib_redo as _nlr  # noqa: E402
import opdeps  # noqa: E402

CLOSURE = os.environ.get("PROV_CLOSURE") == "1"
SECTION_MEMO = os.environ.get("PROV_SECTION_MEMO") == "1"
RELAX_GUARDS = os.environ.get("PROV_RELAX_GUARDS") == "1"
# LIFT (10-15, the user's design): symbols of the adapter's lifted per-step state (metadata tensors, lifted ints: step
# positions [n_p + n_a + 2, len - n_vp)) count as op inputs; the redo binds them at their holders (diag27 DIAG27_LIFT)
LIFT = os.environ.get("PROV_LIFT") == "1"
STATS: collections.Counter = collections.Counter()
REFUSED_BY: collections.Counter = collections.Counter()
CONFIRMED: dict[str, bool] = {}
CONFIRM_WHY: dict[str, str] = {}
_CLASS: dict[int, tuple] = {}  # id(tape) -> (weakref(tape), rows)
_ORIGIN: dict[int, dict] = {}  # id(tape) -> symbol -> (kind, name): diagnostics


def _replay():
    f = sys._getframe(1)
    while f is not None and not isinstance(f.f_locals.get("self"), _hr.HostTraceReplay):
        f = f.f_back
    return None if f is None else f.f_locals["self"]


def _lifted_range(rp, tape):
    """the adapter's lifted positions, from its step function's closure (n_p, n_a, n_vp)"""
    step = getattr(rp, "fn", None)
    if step is None or step.__closure__ is None:
        return None, None
    cl = {n: c.cell_contents for n, c in zip(step.__code__.co_freevars, step.__closure__)}
    if "n_p" not in cl or "proxy" not in cl:
        return None, None
    n_a = cl.get("n_a", len(getattr(cl["proxy"], "attrs", ())))
    n_args = 1 + max([r.position for r in tape.inputs] + [r.position for r in tape.int_inputs])
    return cl["n_p"] + n_a + 2, n_args - cl.get("n_vp", 0)


def _rows(tape):
    ent = _CLASS.get(id(tape))
    if ent is not None and ent[0]() is tape:
        return ent[1]
    rp = _replay()
    opdeps.STATIC["n"] = getattr(rp, "static_prefix", 0) if rp is not None else 0
    opdeps.LIFTED["lo"], opdeps.LIFTED["hi"] = _lifted_range(rp, tape) if LIFT else (None, None)
    rows = opdeps.classify(tape)
    lifted = {s for s, (kind, _) in opdeps.LAST_ORIGIN.items() if kind == "lifted"}
    _CLASS[id(tape)] = (weakref.ref(tape), (rows, lifted))
    _ORIGIN[id(tape)] = dict(opdeps.LAST_ORIGIN)
    return rows, lifted


def _pinned_lifted(lowered, selector, lifted):
    """a failing guard of the op that pins a lifted symbol (s == c: a concretized per-step value that may have escaped
    into Python state another op reads), with its provenance. Rows per own guard (LoweredSelector.rows, as
    retrace_causes reads them); if they do not line up with the op's guards, any pin the op owns counts (conservative)"""
    op = lowered.tape.ops[selector.op]
    env, values = lowered.tape.shape_env, lowered.lowering.program.values
    rows = selector.rows if len(selector.rows) == len(op.guards) else (None,) * len(op.guards)
    for g, row in zip(op.guards, rows):
        if row is not None and values[row] == 1:
            continue
        node = env.records[g][0]
        if node.op != "eq" or not node.free_symbols & lifted:
            continue
        # a pin as _redo reads one: c + k*s == 0 over one symbol (s == c / k), the symbol a lifted one; an equality of a
        # derived value (min / floordiv / bit_length of it: a launch-shape decision) is a branch, not a concretized value
        try:
            c, ts = env.ctx.as_terms(node.args[0])
        except Exception:
            ts = None
        if ts is not None and len(ts) == 1 and (t := next(iter(ts))).op == "sym" and t.args[0] in lifted:
            where = getattr(sys.modules.get("diag27_live"), "GUARD_WHERE", None)
            if where is not None and where.get((id(env), g)) == {"aten"}:
                # recorded only inside an ATen op's C++ symbolic evaluation within this op (the op owns the guard): the redo
                # re-evaluates it at the call, and check_section checks the op's outputs; no Python copy of the value
                STATS["pin_exempt_aten"] += 1
                continue
            sloc = env.slocs[g] if g < len(getattr(env, "slocs", ())) else None
            stacks = getattr(sys.modules.get("diag27_live"), "GUARD_STACKS", None)
            stack = stacks.get((id(env), g)) if stacks is not None else None
            return f"{repr(node)[:60]} at {str(sloc)[:120]}" + (f"\n  stack: {' <- '.join(reversed(stack[-25:]))}" if stack else "")
    return None


class _Hints:
    """program.values for a redo at the trace's own values: a row key's hint"""

    def __getitem__(self, k):
        k = _sym_key(k) if isinstance(k, torch.SymInt) else k
        return k if isinstance(k, int) else int(k.hint)


def _same_launch(x, y, owned, same=None):
    same = same or (lambda u, v: _sym_key(u) == _sym_key(v))
    if type(x) is not type(y):
        return False
    if isinstance(x, KernelLaunch):
        if x.function != y.function or [id(r) for r in owned] != [id(r) for r in y.roots]:
            return False
        a = [*x.grid, *x.block, x.smem, *x.slots]
        b = [*y.grid, *y.block, y.smem, *y.slots]
        return len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
    # a memset / memcpy: its roots, then every other field as values (SymInts by node, not by ==)
    if [id(r) for r in owned] != [id(r) for r in y.roots]:
        return False
    for f in dataclasses.fields(x):
        if f.name in ("roots", "name"):
            continue
        u, v = getattr(x, f.name), getattr(y, f.name)
        if isinstance(u, tuple) != isinstance(v, tuple):
            return False
        us, vs = (u, v) if isinstance(u, tuple) else ((u,), (v,))
        if len(us) != len(vs) or not all(same(p, q) for p, q in zip(us, vs)):
            return False
    return True


def _declared_same(env, declared):
    """with closure: two values agree when they do with every declared input's symbol at its value (the static prefix is
    the same tensors every call, so its layout symbols are constants of the entry: a redo that reads a declared storage
    at its constant offset matches the trace's symbolic offset)"""
    if not declared:
        return None
    ctx = env.ctx
    subs = {**ctx.symbols, **{n: ctx.const(int(ctx.symbols[n].hint)) for n in declared if n in ctx.symbols}}
    memo = {}

    def fold(v):
        k = _sym_key(v)
        if isinstance(k, IR.Node):
            k = ctx.transfer(k, ctx, subs, memo)
            return k.args[0] if k.op == "const" else k
        return k

    def same(u, v):
        return _sym_key(u) == _sym_key(v) or fold(u) == fold(v)

    same.fold = fold
    return same


def _launch_diff(x, y):
    """the first field where two launches differ (diagnostic text)"""
    if type(x) is not type(y):
        return f"type {type(x).__name__} vs {type(y).__name__}"
    if not isinstance(x, KernelLaunch):
        for f in dataclasses.fields(x):
            u, v = getattr(x, f.name), getattr(y, f.name)
            if f.name == "roots":
                if [id(r) for r in u] != [id(r) for r in v]:
                    return f"roots {[r.name for r in u]} vs {[r.name for r in v]}"
                continue
            if f.name != "name" and str([_sym_key(t) for t in (u if isinstance(u, tuple) else (u,))]) != str([_sym_key(t) for t in (v if isinstance(v, tuple) else (v,))]):
                return f"{f.name}: redo {str(u)[:70]} vs trace {str(v)[:70]}"
        return "non-kernel record"
    if x.function != y.function:
        return "function"
    if [id(r) for r in x.roots] != [id(r) for r in y.roots]:
        return f"roots {[r.name for r in x.roots]} vs {[r.name for r in y.roots]}"
    for name, a, b in (("grid", x.grid, y.grid), ("block", x.block, y.block), ("smem", (x.smem,), (y.smem,)), ("slots", x.slots, y.slots)):
        if len(a) != len(b):
            return f"{name} length {len(a)} vs {len(b)}"
        for j, (u, v) in enumerate(zip(a, b)):
            if _sym_key(u) != _sym_key(v):
                return f"{name}[{j}]: redo {str(_sym_key(u))[:70]} vs trace {str(_sym_key(v))[:70]}"
    return "?"


def confirm(lowered, i, selector, opaque):
    """The op redone at the variant's trace values reproduces its records (see the module note)."""
    lo, old = lowered.lowering, lowered.tape
    op = old.ops[selector.op]
    values = lo.program.values
    try:
        lo.program.values, lo.__dict__["row"] = _Hints(), (lambda k: k)
        guards, launches, section = _rd._redo(lowered, selector, hosted=True, dispatches=None, opaque=opaque)
    except Exception as e:  # FoldRefused, Declined, HintsFail: not reproduced
        return False, f"redo at the trace's values failed: {type(e).__name__}: {str(e)[:200]}"
    finally:
        lo.program.values = values
        lo.__dict__.pop("row", None)
    if section is not None:
        return False, "another chain at the trace's values"
    env = old.shape_env
    recorded = {node for node, _ in env.records}
    extra = [g for g, _ in guards if g not in recorded]
    # a guard only on declared inputs' symbols (an alignment or layout check of a module parameter) is the static
    # prefix's contract, which the trace checks per call as the argument contract, not as a recorded guard
    declared = set()
    if CLOSURE or RELAX_GUARDS:
        rp = _replay()
        n = getattr(rp, "static_prefix", 0) if rp is not None and CLOSURE else 0
        for rec in old.inputs:
            if rec.position < n:
                opdeps.syms((rec.sizes, rec.strides, rec.offset, rec.root.sym), declared)
    if RELAX_GUARDS:
        # relaxed rule (coordinator 10-14): an extra guard may use only the op's own call's / own allocations' symbols (or
        # the declared ones with closure): another form of a decision on its inputs, not a read of anything else
        own = set()
        for leaf in pytree.tree_leaves(op.call):
            if isinstance(leaf, _TracedTensor):
                opdeps._tensor_syms(leaf, own)
            elif isinstance(leaf, torch.SymInt):
                opdeps.syms(leaf, own)
        for a in op.allocs:
            opdeps.syms((old.allocs[a].q, old.allocs[a].sizes, old.allocs[a].strides), own)
        declared |= own
        if LIFT:
            declared |= _rows(old)[1]
    rest = [g for g in extra if not (g.free_symbols and g.free_symbols <= declared)]
    if rest:
        return False, f"{len(rest)} guards the trace did not record (of {len(extra)}): {[repr(g)[:80] for g in rest[:3]]}"
    if extra:
        STATS["declared_only_guards"] += len(extra)
    # a keyed site's nodes (a library body's cuBLAS GEMM, patched per key from its table) are the site's, not the op's
    # chain: _redo returns the op's other launches, as here
    keyed = {id(n) for site in old.sites for n in site.nodes}
    chain = [rec for _, rec in old.launches[op.launches.start : op.launches.stop] if id(rec) not in keyed]
    if len(launches) != len(chain):
        return False, f"{len(launches)} launches for {len(chain)}"
    decl_only = set()
    if CLOSURE:
        rp = _replay()
        n = getattr(rp, "static_prefix", 0) if rp is not None else 0
        for rec in old.inputs:
            if rec.position < n:
                opdeps.syms((rec.sizes, rec.strides, rec.offset, rec.root.sym), decl_only)
    same = _declared_same(env, decl_only)
    for j, ((_, x), y) in enumerate(zip(launches, chain)):
        if not _same_launch(x, y, getattr(x, "roots", ()), same):
            if os.environ.get("PROV_DUMP_DIFF") == "1":
                nm = lambda r: getattr(r, "name", type(r).__name__)[:40]
                print(f"prov_redo: DIFF at {j}/{len(chain)} redo={[nm(r) for _, r in launches]} trace={[nm(r) for r in chain]}", flush=True)
                print(f"prov_redo: DIFF roots redo={[[q.name for q in getattr(r, 'roots', ())] for _, r in launches]} trace={[[q.name for q in getattr(r, 'roots', ())] for r in chain]}", flush=True)
            d = _launch_diff(x, y)
            if same is not None and isinstance(x, KernelLaunch):
                for u, v in zip((*x.grid, *x.block, x.smem, *x.slots), (*y.grid, *y.block, y.smem, *y.slots)):
                    if _sym_key(u) != _sym_key(v):
                        try:
                            d += f" | folded redo {str(same.fold(u))[:60]} vs trace {str(same.fold(v))[:60]}"
                        except Exception as e:
                            d += f" | fold failed {type(e).__name__}: {str(e)[:80]}"
                        break
            org = _ORIGIN.get(id(old), {})
            tags = sorted({f"{n}={org[n][0]}:{org[n][1]}" for n in org if n in d})[:4]
            if os.environ.get("PROV_DUMP_DIFF") == "1":
                names = {org[n][1] for n in org if n in d}
                for rec in old.inputs:
                    if rec.name in names:
                        hint = lambda v: (int(_sym_key(v).hint) if isinstance(_sym_key(v), IR.Node) else _sym_key(v))
                        print(f"prov_redo: DIFF input {rec.name} pos {rec.position} {rec.dtype} sizes {[hint(v) for v in rec.sizes]} "
                              f"strides {[hint(v) for v in rec.strides]} offset {hint(rec.offset)} root {rec.root.name}", flush=True)
                leaves = [f"{getattr(t, '_root', None) and t._root.name}:{[str(_sym_key(v))[:20] for v in t.shape]}" for t in pytree.tree_leaves(op.call) if isinstance(t, _TracedTensor)]
                print(f"prov_redo: DIFF op.call tensors {leaves}", flush=True)
            return False, f"a launch differs ({getattr(y, 'name', type(y).__name__)[:60]}): {d} {tags}"
    return True, "reproduced"


if not getattr(_rd._full_redo, "_prov_redo", False):
    _inner = _rd._full_redo  # no_lib_redo's wrapper (it refuses library bodies)

    def _full_redo(lowered, i, selector, dispatches, opaque=(), *rest, **kwargs):  # rest: a later core's arguments (ms34's redo)
        op = lowered.tape.ops[selector.op]
        if not _nlr.library_body(op.func):
            return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)
        name = str(op.func)
        try:
            rows, lifted = _rows(lowered.tape)
            row = rows[selector.op]
            verdict = row["verdict"]
        except Exception as e:  # the prototype's own failure: refuse as before
            STATS["classify_error"] += 1
            if STATS["classify_error"] == 1:
                print(f"prov_redo: classify failed: {type(e).__name__}: {e}", flush=True)
            return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)
        lift_ok = LIFT and verdict == "args_or_lifted" and (CLOSURE or "declared" not in row["why"])
        if lift_ok and (pin := _pinned_lifted(lowered, selector, lifted)) is not None:
            STATS["refused_pin"] += 1
            REFUSED_BY[(name, "pin")] += 1
            if STATS["refused_pin"] <= 3:
                print(f"prov_redo: {name}: failing pin on a lifted symbol {pin}: refused", flush=True)
            return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)
        if not (verdict == "args_only" or (CLOSURE and verdict == "args_or_declared") or lift_ok):
            STATS["refused_static"] += 1
            REFUSED_BY[(name, verdict)] += 1
            return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)  # no_lib_redo refuses it
        if name not in CONFIRMED:
            try:
                ok, why = confirm(lowered, i, selector, opaque)
            except Exception as e:
                ok, why = False, f"confirmation raised {type(e).__name__}: {str(e)[:200]}"
            CONFIRMED[name], CONFIRM_WHY[name] = ok, why
            STATS["confirmed" if ok else "confirm_failed"] += 1
            print(f"prov_redo: {name} ({verdict}): {why}", flush=True)
        if not CONFIRMED[name]:
            STATS["refused_confirm"] += 1
            REFUSED_BY[(name, "confirm")] += 1
            return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)
        STATS["allowed"] += 1
        return _nlr._orig(lowered, i, selector, dispatches, opaque, *rest, **kwargs)  # the core's own redo

    _full_redo._prov_redo = True
    _rd._full_redo = _full_redo


# A redo runs the op's host as its trace did: with Python's max / min held as their symbolic forms (the trace holds them,
# HostTraceReplay.trace_builtin_minmax, default on). Without it a redo's min(s, 608) branches (a guard and `s`) where the
# trace recorded min(s, 608): another form of the same launch, which the confirmation (rightly) cannot match.
if LIFT and os.environ.get("PROV_REDO_MINMAX", "1") == "1" and not getattr(_rd._redo, "_minmax", False):
    _orig_redo_fn = _rd._redo

    def _redo_minmax(*a, **k):
        with _hr._builtin_minmax:
            return _orig_redo_fn(*a, **k)

    _redo_minmax._minmax = True
    _rd._redo = _redo_minmax
