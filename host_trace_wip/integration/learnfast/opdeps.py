"""opdeps (learnfast diagnostic, 10-14): per top-level op of a traced Tape, do its records depend only on its own inputs?

An op's records: its launches (grid, block, smem, every slot, every root's address), its allocations' sizes and strides,
and the guards it owns (Tape.owners). Its inputs: every traced tensor and int leaf of OpRec.call (sizes, strides, storage
offset, root address) plus its own allocations' address symbols. A record symbol outside those is classified by where
it comes from: a step int input (forward-context ints the adapter makes symbolic), a step tensor input's metadata or
address that no argument of the op carries (module state, kv cache, metadata tensors read from context), another op's
allocation, or other (made during the op). An op whose records include an eager / opaque call ("other" / "eager"
kind) hides its body's reads: never args-only.

`shared`: argument symbols that a non-argument step input also carries (one symbol for input_ids' and a metadata
tensor's token count under trusted layouts): a body read of that input would look like an argument read in the
records, so the static criterion cannot tell them apart for that op.

Use: classify(tape) -> list of per-op dicts; summarize(rows) -> per-func counts."""
from __future__ import annotations

import collections
import dataclasses

import sympy
import torch
import torch.cuda._host_trace_ir as IR
import torch.cuda._host_trace_tape as TT
from torch.utils import _pytree as pytree


def syms(v, out=None):
    out = set() if out is None else out
    if isinstance(v, torch.SymInt):
        k = TT._sym_key(v)
        if isinstance(k, IR.Node):
            out |= k.free_symbols
        elif isinstance(k, sympy.Basic):
            out |= {str(s) for s in k.free_symbols}
    elif isinstance(v, IR.Node):
        out |= v.free_symbols
    elif isinstance(v, sympy.Basic):
        out |= {str(s) for s in v.free_symbols}
    elif isinstance(v, TT._Root):
        syms(v.sym, out)
    elif isinstance(v, (tuple, list, frozenset, set)):
        for x in v:
            syms(x, out)
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        for f in dataclasses.fields(v):
            if f.name in ("owner", "images", "function", "abi", "provider"):
                continue
            syms(getattr(v, f.name), out)
    return out


def _tensor_syms(t, out):
    syms(tuple(t.shape), out)
    syms(tuple(t.stride()), out)
    syms(t.storage_offset(), out)
    syms(t._root.sym, out)


STATIC = {"n": 0}  # the replay's static_prefix (declared inputs: module params, attrs), set by the caller
# LIFT (10-15): step argument positions [lo, hi) are lifted per-step state (the adapter's metadata tensors and lifted ints,
# installed at their holders in the forward context); None: no lifted classification
LIFTED = {"lo": None, "hi": None}
LAST_ORIGIN: dict = {}  # symbol -> (kind, name) of the last classify()


def _lifted(pos):
    lo, hi = LIFTED["lo"], LIFTED["hi"]
    return lo is not None and lo <= pos < hi


def classify(tape):
    env = tape.shape_env
    records = getattr(env, "records", None)
    # where each step-level symbol comes from
    origin = {}
    for rec in tape.int_inputs:
        for s in syms(rec.sym):
            origin.setdefault(s, ("lifted" if _lifted(rec.position) else "int_input", rec.name))
    input_syms = {}
    for rec in tape.inputs:
        ss = syms((rec.sizes, rec.strides, rec.offset, rec.root.sym))
        input_syms[id(rec.root)] = ss
        kind = "declared" if rec.position < STATIC["n"] else "lifted" if _lifted(rec.position) else "tensor_input"
        for s in ss:
            origin.setdefault(s, (kind, rec.name))
    LAST_ORIGIN.clear()
    LAST_ORIGIN.update(origin)
    alloc_op = {}
    for k, op in enumerate(tape.ops):
        for a in op.allocs:
            alloc_op[a] = k
    for a, rec in enumerate(tape.allocs):
        for s in syms(rec.q):
            origin.setdefault(s, ("alloc_of_op", alloc_op.get(a)))
    owned = collections.defaultdict(list)
    for i, o in enumerate(tape.owners):
        if o is not None:
            owned[o].append(i)
    # per symbol, the input roots whose layout holds it; the int inputs' symbols (once, not per op)
    sym_roots = collections.defaultdict(set)
    for rid, ss in input_syms.items():
        for s in ss:
            sym_roots[s].add(rid)
    int_syms = {s for s, (kind, _) in origin.items() if kind == "int_input"}
    rows = []
    for k, op in enumerate(tape.ops):
        arg_syms, arg_roots = set(), set()
        for leaf in pytree.tree_leaves(op.call):
            if isinstance(leaf, TT._TracedTensor):
                _tensor_syms(leaf, arg_syms)
                arg_roots.add(id(leaf._root))
            elif isinstance(leaf, torch.SymInt):
                syms(leaf, arg_syms)
        own = set()
        for a in op.allocs:
            syms((tape.allocs[a].q, tape.allocs[a].sizes, tape.allocs[a].strides), own)
        rec_syms, launch_roots = set(), set()
        for j in op.launches:
            rec = tape.launches[j][1]
            syms(rec, rec_syms)
            for r in getattr(rec, "roots", ()) or ():
                launch_roots.add(id(r))
        guard_syms = set()
        if records is not None:
            for i in owned.get(k, ()):
                syms(records[i][0], guard_syms)
        extra = (rec_syms | guard_syms) - arg_syms - own
        why = collections.Counter()
        names = collections.defaultdict(set)
        for s in extra:
            kind, name = origin.get(s, ("other", None))
            why[kind] += 1
            names[kind].add(str(name))
        # an argument symbol also an int input's or in the layout of an input root the op does not take as an argument
        shared = {s for s in arg_syms if s in int_syms or not sym_roots.get(s, set()) <= arg_roots}
        if op.kind != "traced":
            verdict = "opaque_records"
        elif not extra:
            verdict = "args_only"
        elif set(why) <= {"declared"}:
            verdict = "args_or_declared"  # args-only under closure over the declared inputs
        elif set(why) <= {"declared", "lifted"}:
            verdict = "args_or_lifted"  # args-only once the lifted per-step state is bound per redo (and closure if declared)
        else:
            verdict = "reads_outside"
        rows.append(dict(k=k, func=str(getattr(op.func, "__name__", None) or op.func)[:90], kind=op.kind, launches=len(op.launches),
                         guards=len(owned.get(k, ())), verdict=verdict, why=dict(why), names={a: sorted(b)[:4] for a, b in names.items()},
                         shared=len(shared), selector=bool(owned.get(k))))
    return rows


def summarize(rows):
    out = collections.defaultdict(lambda: collections.Counter())
    names = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        key = r["func"]
        out[key][r["verdict"]] += 1
        out[key]["with_guards"] += r["selector"]
        out[key]["shared"] += bool(r["shared"])
        for kind, ns in r["names"].items():
            names[key][kind] |= set(ns)
    return {k: dict(v, names={a: sorted(b)[:5] for a, b in names[k].items()}) for k, v in out.items()}


def classify_nested(tape, nested):
    """The same criterion for nested op calls (a route op inside a vLLM custom op's body): `nested` holds, per nested
    _Trace.op call, (func, call, launch range, alloc range, guard-record range); its guards are the ones recorded while
    it ran (Tape.owners attributes them to the enclosing top-level op)."""
    env = tape.shape_env
    records = getattr(env, "records", None)
    origin = {}
    for rec in tape.int_inputs:
        for s in syms(rec.sym):
            origin.setdefault(s, ("int_input", rec.name))
    input_syms = {}
    for rec in tape.inputs:
        ss = syms((rec.sizes, rec.strides, rec.offset, rec.root.sym))
        input_syms[id(rec.root)] = ss
        for s in ss:
            origin.setdefault(s, ("declared" if rec.position < STATIC["n"] else "tensor_input", rec.name))
    for a, rec in enumerate(tape.allocs):
        for s in syms(rec.q):
            origin.setdefault(s, ("alloc", a))
    int_syms = {s for s, (kind, _) in origin.items() if kind == "int_input"}
    rows = []
    for func, call, lr, ar, gr in nested:
        arg_syms, arg_roots = set(), set()
        for leaf in pytree.tree_leaves(call):
            if isinstance(leaf, TT._TracedTensor):
                _tensor_syms(leaf, arg_syms)
                arg_roots.add(id(leaf._root))
            elif isinstance(leaf, torch.SymInt):
                syms(leaf, arg_syms)
        own = set()
        for a in ar:
            syms((tape.allocs[a].q, tape.allocs[a].sizes, tape.allocs[a].strides), own)
        rec_syms, opaque = set(), False
        for j in lr:
            rec = tape.launches[j][1]
            opaque |= isinstance(rec, TT.EagerCall)
            syms(rec, rec_syms)
        guard_syms = set()
        if records is not None:
            for i in gr:
                syms(records[i][0], guard_syms)
        extra = (rec_syms | guard_syms) - arg_syms - own
        why = collections.Counter()
        names = collections.defaultdict(set)
        for s in extra:
            kind, name = origin.get(s, ("other", None))
            why[kind] += 1
            names[kind].add(str(name))
        nonarg = set()
        for rid, ss in input_syms.items():
            if rid not in arg_roots:
                nonarg |= ss
        verdict = "opaque_records" if opaque else "args_only" if not extra else "args_or_declared" if set(why) <= {"declared"} else "reads_outside"
        rows.append(dict(func=str(getattr(func, "__name__", None) or func)[:90], kind="nested", launches=len(lr), guards=len(gr),
                         verdict=verdict, why=dict(why), names={a: sorted(b)[:4] for a, b in names.items()},
                         shared=len(arg_syms & (nonarg | int_syms)), selector=len(gr) > 0))
    return rows
