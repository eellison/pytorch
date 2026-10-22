"""PROTOTYPE (learnfast 10-17; labelled in notes/NONDEFAULTS_learnfast.md): cross-variant reuse of redo results.

User-approved direction (10-16): a per-op result is keyed by what the op's redo reads and its size class, and is reused by
every variant of the entry where its guards hold, instead of running the redo body again. Core already has the mechanism
for one case: _redo's dispatch cache (HostTraceReplay._dispatches, one dict per contract) takes a cached fresh trace of the
op's body wherever its guards hold at the call's values and renames it into the variant. On 27B it almost never hits:
  1. its key holds each argument's trace-time extent (absolute first/last byte), so the same op over the same buffers in
     another variant (traced at another call) or another layer misses; the trace reads extents only through emptiness and
     the written/read overlap test (_host_trace_tape.py ~1124), so this key holds (dtype, length, empty, cow) per argument
     plus the pairwise overlap relation of the redo's input records instead;
  2. a redo whose body reached state outside its call (the learnfast prototypes' stand-ins: lifted per-step state,
     closure-resolved weights, RD5) is never stored. Here every root()/value() call _redo's own closures make after the
     call's reads (the resolver's) is recorded as a variant-independent event: ("root", argument position) or ("int",
     position) / ("lay", position, field, k). A later redo of the same call key replays the events with its own closures
     (same position, the same static-prefix object at a static position), and takes the cached trace only when the full
     read signature (call + events, normalized as in 1) is equal and every guard record and value check of the cached
     trace holds at this call's values. Otherwise the replay is undone and the body runs.
Scope: per contract (the dict core passes as `dispatches`), as core's own cache. Soundness rests on core's dispatch-cache
contract (an op's records are a function of what it reads) extended to the recorded events; validated by --check runs.
Patched textually into a copy of step_ms16's _redo (exact-match asserts); off unless imported (diag27 DIAG27_XVR=1).
Validation: bitwise --check runs (n27wf prefill / mixed) with the prototype on."""
import collections
import contextlib
import inspect
import os
import sys
import time

import torch
import torch.cuda._host_trace_redispatch as RD
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace_tape import _Root, _TracedTensor

STATS: collections.Counter = collections.Counter()
SECS: collections.Counter = collections.Counter()
_CACHE: dict = {}
VALIDATE = os.environ.get("XVR_VALIDATE") == "1"
_CUR: list = []  # the _State of each active (possibly nested) redo
_HITS: list = []
_FORCE_MISS = [False]  # id(contract dispatches dict) -> (that dict, {call key: [entry, ...]})


class _State:
    def __init__(self, old, lo):
        self.phase, self.depth, self.events, self.ok = "call", 0, [], True
        self.old, self.lo, self.prefix, self.src, self.call_key = old, lo, None, None, None
        _CUR.append(self)


# where a redo may read an input record's extent: the written/read overlap test in _Trace._eager_call (emptiness and
# overlap only), the copied _redo's read signature, and this module's normalized key (_norm_read). Any other reader makes
# the dispatch unstorable (soundness of the normalized key).
import torch.cuda._host_trace_tape as _TT  # noqa: E402

_TAPE_SRC = inspect.getsource(_TT).splitlines()
_OVERLAP_LINES = {i + 1 for i, ln in enumerate(_TAPE_SRC) if ".extent[" in ln and ("others = [" in ln or "overlap = any(" in ln)}
if len(_OVERLAP_LINES) != 2:
    raise AssertionError(f"xvr: the overlap test's extent reads moved: {_OVERLAP_LINES}")
_ALLOWED = {(_TT.__file__, n) for n in _OVERLAP_LINES}
_ALLOWED_FILES = {RD.__file__ + ":xvr", __file__}


class _Watched(_TT._InputRec):
    def __getattribute__(self, name):
        if name == "extent":
            f = sys._getframe(1)
            if f.f_code.co_filename not in _ALLOWED_FILES and (f.f_code.co_filename, f.f_lineno) not in _ALLOWED:
                STATS[f"extent_other_read|{f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno}"] += 1
                if _CUR:
                    _CUR[-1].ok = False
        return object.__getattribute__(self, name)


def _watched(rec):
    import dataclasses

    return _Watched(**{f.name: getattr(rec, f.name) for f in dataclasses.fields(rec)})


def _static_prefix():
    import torch.cuda._host_trace_replay as _hr

    f = sys._getframe(2)
    while f is not None and not isinstance(f.f_locals.get("self"), _hr.HostTraceReplay):
        f = f.f_back
    return 0 if f is None else (f.f_locals["self"].static_prefix or 0)


def _sources(xs):
    # a variant SymInt object (the resolvers pass the int inputs' and input records' own objects) -> its source: the int
    # input at position p, or field f's k-th entry of the input record at position p
    if xs.src is None:
        src = {}
        for irec in xs.old.int_inputs:
            src.setdefault(id(irec.sym), ("int", irec.position))
        for rec in xs.old.inputs:
            for f, vals in (("s", rec.sizes), ("t", rec.strides), ("o", (rec.offset,))):
                for k_, v in enumerate(vals):
                    src.setdefault(id(v), ("lay", rec.position, f, k_))
        xs.src = src
    return xs.src


def _wrap(xs, value_in, root_in, tensor_in):
    """_redo's value/root/tensor, recording the resolver's top-level calls (phase "extra") as events."""

    def value(v):
        if xs.phase == "extra" and xs.depth == 0:
            if isinstance(v, torch.SymInt):
                d = _sources(xs).get(id(v))
                if d is None:
                    xs.ok = False
                xs.events.append(d)
            elif isinstance(v, (int, bool, float)):
                xs.events.append(("const", type(v), v))
            else:
                xs.ok = False
                xs.events.append(None)
        xs.depth += 1
        try:
            return value_in(v)
        finally:
            xs.depth -= 1

    def root(r):
        _ = value  # a free variable named value: the learnfast resolvers read it from root's closure
        if xs.phase == "extra" and xs.depth == 0:
            rec = next((i for i in xs.old.inputs if i.root is r), None) if r.kind == "argument" else None
            if rec is None:
                xs.ok = False
                xs.events.append(None)
            else:
                if xs.prefix is None:
                    xs.prefix = _static_prefix()
                p = rec.position
                xs.events.append(("root", p, xs.lo.program.inputs[p] if p < xs.prefix else None))
        xs.depth += 1
        try:
            return root_in(r)
        finally:
            xs.depth -= 1

    def tensor(t):
        xs.depth += 1
        try:
            return tensor_in(t)
        finally:
            xs.depth -= 1

    return value, root, tensor


def _norm_read(shape, made, roots, tensors, inputs, env, start=0):
    """core's `read` over shape[start:], with each argument's absolute extent replaced by what the trace reads of it, plus
    the overlap relation of the input records (both non-empty and overlapping)"""
    index = {k: i for i, k in enumerate([*map(id, (*made, *roots.values())), *tensors])}
    out = []
    for x in shape[start:]:
        if isinstance(x, torch.SymInt):
            out.append((index[id(x)], x.node.node.args[0] in env.ctx.positive))
        elif isinstance(x, (_Root, _TracedTensor)):
            out.append((index[id(x)],))
        elif isinstance(x, tuple) and len(x) == 3 and isinstance(x[1], tuple) and len(x[1]) == 2 and isinstance(x[0], torch.dtype):
            (a, b) = x[1]
            out.append((x[0], b - a, a <= b, x[2]))
        else:
            out.append(x)
    ex = [r.extent for r in inputs]
    ov = tuple((i, j) for i in range(len(ex)) for j in range(i + 1, len(ex))
               if ex[i][0] <= ex[i][1] and ex[j][0] <= ex[j][1] and ex[i][0] <= ex[j][1] and ex[j][0] <= ex[i][1])
    return tuple(out), ov


def _replay(xs, events, value, root, old, lo):
    """the events with this redo's own closures; False where a position or a static object differs"""
    by_pos = {}
    for rec in old.inputs:
        by_pos.setdefault(rec.position, rec)
    ints = {irec.position: irec for irec in old.int_inputs}
    with lo.symbolic.total():
        for e in events:
            if e[0] == "root":
                rec = by_pos.get(e[1])
                if rec is None or next(i for i in old.inputs if i.root is rec.root).position != e[1]:
                    return False
                if e[2] is not None and lo.program.inputs[e[1]] is not e[2]:
                    return False
                root(rec.root)
            elif e[0] == "int":
                irec = ints.get(e[1])
                if irec is None:
                    return False
                value(irec.sym)
            elif e[0] == "lay":
                rec = by_pos.get(e[1])
                if rec is None:
                    return False
                vals = {"s": rec.sizes, "t": rec.strides, "o": (rec.offset,)}[e[2]]
                if e[3] >= len(vals):
                    return False
                value(vals[e[3]])
            else:
                value(e[2])
    return True


_FAILS: collections.Counter = collections.Counter()


def _holds(c_env, checks, made_then, at_call, at_then=None, fn=None):
    consts = _ir.Ctx()
    hints = [consts.const(x) for x in at_call]
    at = {s.node.node.args[0]: h for s, h in zip(made_then, hints)}
    try:
        for g, _ in c_env.records:
            if consts.transfer(g, c_env.ctx, at, {}).op != "true":
                if at_then is not None:
                    idx = {s.node.node.args[0]: i for i, s in enumerate(made_then)}
                    syms = sorted(g.free_symbols)
                    k = (fn, _ir.render(g)[:120] if hasattr(_ir, "render") else repr(g)[:120])
                    _FAILS[k] += 1
                    if _FAILS[k] <= 2:
                        print(f"xvr: guard miss {fn}: {k[1]} syms {[(n, idx.get(n), at_then[idx[n]] if n in idx else None, at_call[idx[n]] if n in idx else None) for n in syms][:6]}", flush=True)
                return False
        for c in checks:
            if consts.transfer(c.node.node, c_env.ctx, at, {}).op != "true":
                STATS[f"why_check|{fn}"] += 1
                if STATS[f"why_check|{fn}"] <= 2:
                    print(f"xvr: value check miss {fn}: {_ir.render(c.node.node)[:200]}", flush=True)
                return False
    except (KeyError, ArithmeticError, ValueError):
        return False
    return True


def lookup(*a):
    xs = a[0]
    try:
        return _lookup(*a)
    except Exception as e:  # a prototype bug falls back to the body (state restored in _lookup)
        STATS["lookup_error"] += 1
        xs.call_key, xs.ok = None, False
        if STATS["lookup_error"] <= 3:
            import traceback
            print("xvr: lookup error", repr(e)[:300], traceback.format_exc()[-1500:], flush=True)
        return None


def store(*a):
    try:
        return _store(*a)
    except Exception as e:
        STATS["store_error"] += 1
        if STATS["store_error"] <= 3:
            import traceback
            print("xvr: store error", repr(e)[:300], traceback.format_exc()[-1500:], flush=True)


def _lookup(xs, dispatches, op, value, root, made, at_call, shape, tensors, roots, inputs, fresh, outer, back, mapping, env, ir):
    if dispatches is None or not ir or not xs.ok:
        return None
    if _FORCE_MISS[0]:
        return None
    t0 = time.perf_counter()
    try:
        call_read = _norm_read(shape, made, roots, tensors, inputs, env)
        key = (op.func if isinstance(op.func, torch._ops.OpOverload) else op.redo, op.kind, repr(RD.pytree.tree_structure(op.call)), call_read)
        try:
            hash(key)
        except TypeError:
            STATS["unhashable"] += 1
            return None
        xs.call_key = key
    finally:
        SECS["lookup"] += time.perf_counter() - t0
    scope = _CACHE.get(id(dispatches))
    cands = scope[1].get(key) if scope is not None and scope[0] is dispatches else None
    fn = str(op.func)[:60]
    if not cands:
        STATS[f"miss_new|{fn}"] += 1
        return None
    t0 = time.perf_counter()
    snap = (len(made), len(at_call), len(shape), len(inputs), set(roots), set(fresh), set(tensors), set(outer), set(back), set(mapping))
    restored = False
    try:
        for c in reversed(cands):
            events, full_read, tr, made_then, roots_then = c[:5]
            xs.phase = "replay"
            ok = _replay(xs, events, value, root, xs.old, xs.lo)
            if not ok:
                STATS[f"why_replay|{fn}"] += 1
            elif len(made) != len(made_then) or len(roots) != len(roots_then):
                STATS[f"why_len|{fn}"] += 1
            else:
                mine = _norm_read(shape, made, roots, tensors, inputs, env)
                if mine != full_read:
                    STATS[f"why_read|{fn}"] += 1
                    if STATS[f"why_read|{fn}"] <= 3:
                        a_, b_ = mine[0], full_read[0]
                        j = next((j for j, (x, y) in enumerate(zip(a_, b_)) if x != y), min(len(a_), len(b_)))
                        print(f"xvr: read differs {fn}: len {len(a_)} vs {len(b_)}, first at {j}: {a_[j:j + 3]} vs {b_[j:j + 3]}; overlap equal {mine[1] == full_read[1]}", flush=True)
                elif _holds(tr.shape_env, tr.checks, made_then, at_call, c[5] if len(c) > 5 else None, fn):
                    STATS[f"hit|{fn}"] += 1
                    if events:
                        STATS[f"hit_with_events|{fn}"] += 1
                    _HITS.append(fn)
                    return c
                else:
                    STATS[f"why_guards|{fn}"] += 1
            _restore(snap, made, at_call, shape, inputs, roots, fresh, tensors, outer, back, mapping)
        restored = True
        STATS[f"miss_guards|{fn}"] += 1
        return None
    except Exception:
        if not restored:
            _restore(snap, made, at_call, shape, inputs, roots, fresh, tensors, outer, back, mapping)
        raise
    finally:
        xs.phase = "extra"
        SECS["replay"] += time.perf_counter() - t0


def _restore(snap, made, at_call, shape, inputs, roots, fresh, tensors, outer, back, mapping):
    n_made, n_at, n_shape, n_inputs, k_roots, k_fresh, k_tensors, k_outer, k_back, k_mapping = snap
    del made[n_made:], at_call[n_at:], shape[n_shape:], inputs[n_inputs:]
    for d, keep in ((roots, k_roots), (fresh, k_fresh), (tensors, k_tensors), (outer, k_outer), (back, k_back), (mapping, k_mapping)):
        for k in [k for k in d if k not in keep]:
            del d[k]


def _store(xs, dispatches, op, cached, nodes, tr, made, roots, shape, tensors, inputs, env, at_call=()):
    if dispatches is None or cached is not None or not xs.ok or xs.call_key is None or _FORCE_MISS[0]:
        if not xs.ok and xs.call_key is not None:
            STATS[f"not_ok|{str(op.func)[:60]}"] += 1
        return
    names = {s.node.node.args[0] for s in made}
    if nodes or tr.sites or not all(g.free_symbols <= names for g, _ in env.records):
        STATS[f"unstorable|{str(op.func)[:60]}"] += 1
        return
    scope = _CACHE.get(id(dispatches))
    if scope is None or scope[0] is not dispatches:
        scope = _CACHE[id(dispatches)] = (dispatches, {})
    full_read = _norm_read(shape, made, roots, tensors, inputs, env)
    try:
        hash(full_read)
    except TypeError:
        STATS["unhashable_full"] += 1
        return
    scope[1].setdefault(xs.call_key, []).append((tuple(xs.events), full_read, tr, list(made), list(roots.values()), list(at_call)))
    STATS[f"stored|{str(op.func)[:60]}"] += 1
    if xs.events:
        kinds = {e[0] for e in xs.events if e}
        STATS[f"stored_with_events|{str(op.func)[:60]}|{','.join(sorted(kinds))}"] += 1


# ---- the patched copy of core's _redo ----
_MOD = inspect.getsource(RD)
_SRC = _MOD[_MOD.index("\ndef _redo(") + 1 : _MOD.index("\ndef _resolver(")]


def _sub(src, old, new):
    if src.count(old) != 1:
        raise AssertionError(f"xvr: pattern not found exactly once in step_ms16 _redo: {old[:80]!r}")
    return src.replace(old, new)


_SRC = _sub(_SRC, "def _redo(", "def _xvr_redo(")
_SRC = _sub(_SRC, """                rec = replace(rec, sizes=layout[0], strides=layout[1], offset=layout[2], root=roots[id(r)])
""", """                rec = _XVR._watched(replace(rec, sizes=layout[0], strides=layout[1], offset=layout[2], root=roots[id(r)]))
""")
_SRC = _sub(_SRC, """    # the hints: the formulas at the call, rows that fail at no other call
    with lo.symbolic.total():
        call = pytree.tree_map(lambda a: tensor(a) if isinstance(a, _TracedTensor) else value(a), op.call)
""", """    _xs = _XVR._State(old, lo)
    value, root, tensor = _XVR._wrap(_xs, value, root, tensor)
    # the hints: the formulas at the call, rows that fail at no other call
    with lo.symbolic.total():
        call = pytree.tree_map(lambda a: tensor(a) if isinstance(a, _TracedTensor) else value(a), op.call)
    _xs.phase = "extra"
""")
_SRC = _sub(_SRC, """    if cached is not None:
        tr, made_then, roots_then = cached
""", """    if cached is None:
        _hit = _XVR.lookup(_xs, dispatches, op, value, root, made, at_call, shape, tensors, roots, inputs, fresh, outer, back, mapping, env, ir)
        if _hit is not None:
            cached = _hit[2:5]
    if cached is not None:
        tr, made_then, roots_then = cached
""")
_SRC = _sub(_SRC, """        dispatches.setdefault(sig, []).append((tr, made, list(roots.values())))
""", """        dispatches.setdefault(sig, []).append((tr, made, list(roots.values())))
    _XVR.store(_xs, dispatches, op, cached, nodes, tr, made, roots, shape, tensors, inputs, env, at_call)
""")
_ns = dict(vars(RD))
_ns["_XVR"] = sys.modules[__name__]
exec(compile("from __future__ import annotations\n" + _SRC, RD.__file__ + ":xvr", "exec"), _ns)
_xvr_redo = _ns["_xvr_redo"]


# prov_redo's PROV_REDO_MINMAX wrapper (Python's min / max held symbolic in a redo) stays around this copy
_MINMAX = getattr(RD._redo, "_minmax", False)


def _canon(out):
    import re

    guards, launches, section = out
    names: dict = {}

    def norm(x):
        return re.sub(r"section\d+", lambda m: names.setdefault(m.group(0), f"S{len(names)}"), repr(x))

    return (sorted(norm(g) for g in guards), [norm(x) for x in launches], norm(section))


def _redo(*a, **k):
    import torch.cuda._host_trace_replay as _hr

    t0 = time.perf_counter()
    depth = len(_CUR)
    n_hits = len(_HITS)
    try:
        with _hr._builtin_minmax if _MINMAX else contextlib.nullcontext():
            out = _xvr_redo(*a, **k)
            if VALIDATE and len(_HITS) > n_hits and not _FORCE_MISS[0]:
                fn = _HITS[-1]
                del _CUR[depth:]
                _FORCE_MISS[0] = True
                try:
                    fresh = _xvr_redo(*a, **k)
                except Exception as e:
                    STATS[f"validate_fresh_raised|{fn}"] += 1
                    print(f"xvr: VALIDATE fresh redo raised where the cache hit ({fn}): {type(e).__name__}: {str(e)[:300]}", flush=True)
                    raise
                finally:
                    _FORCE_MISS[0] = False
                co, cf = _canon(out), _canon(fresh)
                if co[1:] != cf[1:]:
                    # launches or section differ: take the fresh redo's
                    STATS[f"validate_mismatch|{fn}"] += 1
                    if STATS[f"validate_mismatch|{fn}"] <= 3:
                        print(f"xvr: VALIDATE launch mismatch {fn}: launches {co[1] == cf[1]} section {co[2] == cf[2]}\n hit   {str(co[1:])[:1500]}\n fresh {str(cf[1:])[:1500]}", flush=True)
                    return fresh
                if co[0] != cf[0]:
                    # the same launches under other guards: each trace holds under its own guards (a library's memo
                    # cache can take another host path); the hit's guards all held at this call (_holds)
                    hx, fx = sorted(set(co[0]) - set(cf[0])), sorted(set(cf[0]) - set(co[0]))
                    STATS[f"validate_guards_differ|{fn}"] += 1
                    STATS[f"validate_guards_hit_only|{fn}"] += len(hx)
                    STATS[f"validate_guards_fresh_only|{fn}"] += len(fx)
                    if STATS[f"validate_guards_differ|{fn}"] <= 4:
                        print(f"xvr: VALIDATE guards differ {fn}: hit-only {[g[:160] for g in hx][:6]} fresh-only {[g[:160] for g in fx][:6]}", flush=True)
                STATS[f"validate_ok|{fn}"] += 1
            return out
    finally:
        del _CUR[depth:]
        SECS["redo_total"] += time.perf_counter() - t0


if not getattr(RD._redo, "_xvr", False):
    _redo._xvr = True
    RD._redo = _redo
    import atexit

    atexit.register(lambda: print("XVR", dict(STATS), {k: round(v, 2) for k, v in SECS.items()}, flush=True))


# ---- diagnostic (cheap counters): why a redispatch's op does not take its variant's template (within-variant sharing) ----
_TPL_FN = {"fn": None}
_inner_fi = RD._for_instance


def _fi_diag(lowered, t, roots, static_prefix, *rest):
    out = _inner_fi(lowered, t, roots, static_prefix, *rest)
    fn = _TPL_FN["fn"] = None
    try:
        reads = lowered.lowering.reads
        f = sys._getframe(1)
        while f is not None and f.f_code.co_name != "redispatch":
            f = f.f_back
        if f is not None and "op" in f.f_locals:
            fn = str(f.f_locals["op"].func)[:60]
    except Exception:
        pass
    _TPL_FN["fn"] = fn
    if out is None:
        STATS[f"tpl_for_instance_none|{fn}"] += 1
    else:
        STATS[f"tpl_for_instance_ok|{fn}"] += 1
        vals = lowered.lowering.program.values
        if vals[out.predicate] != 1:
            STATS[f"tpl_predicate_false|{fn}"] += 1
            own = vals[t.predicate] == 1
            STATS[f"tpl_predicate_false_own_{'true' if own else 'false'}|{fn}"] += 1
            if own and os.environ.get("XVR_TPLDIAG") == "1" and STATS[f"tpl_diag_printed|{fn}"] < 4:
                STATS[f"tpl_diag_printed|{fn}"] += 1
                _tpl_diag(lowered, t, roots, static_prefix, fn)
    return out


RD._for_instance = _fi_diag


def _tpl_diag(lowered, t, roots, static_prefix, fn):
    """XVR_TPLDIAG: which renamed guard / value pin of a template fails for another layer's op at the same call"""
    lo, old = lowered.lowering, lowered.tape
    recs, ctx = lo.first_inputs, old.shape_env.ctx
    try:
        subs, pins, moved = {}, [], []
        for p, (mine, theirs) in enumerate(zip(t.roots, roots)):
            a = recs.get(id(mine)) if mine.kind == "argument" else None
            if a is None or a.position >= static_prefix:
                continue
            b = recs.get(id(theirs)) if theirs.kind == "argument" else None
            if b is None:
                continue
            for rec in (a, b):
                for v in (*rec.sizes, *rec.strides, rec.offset):
                    k = RD._sym_key(v)
                    if isinstance(k, _ir.Node):
                        pins.append(ctx.eq(k, ctx.const(int(k.hint))))
            if theirs is not mine:
                moved.append((p, a.position, b.position))
                for x, y in zip((a.root.sym, *a.sizes, *a.strides, a.offset), (b.root.sym, *b.sizes, *b.strides, b.offset)):
                    kx, ky = RD._sym_key(x), RD._sym_key(y)
                    if isinstance(kx, _ir.Node) and kx.op == "sym":
                        subs[kx.args[0]] = ky if isinstance(ky, _ir.Node) else ctx.const(int(ky))
        import collections as _c
        full = _c.ChainMap(subs, ctx.symbols)
        bad = []
        for g, _ in t.guards:
            if not (isinstance(g, _ir.Node) and g.free_symbols & subs.keys()):
                continue
            r = ctx.transfer(g, ctx, full, {})
            row = lo.row(r) if r.op not in ("true", "false") else None
            val = (r.op == "true") if row is None else lo.program.values[row]
            if val != 1 and val is not True:
                bad.append((_ir.render(g)[:200], _ir.render(r)[:200]))
        badp = []
        for g in pins:
            if g.op in ("true", "false"):
                if g.op == "false":
                    badp.append(_ir.render(g)[:120])
                continue
            if lo.program.values[lo.row(g)] != 1:
                badp.append(_ir.render(g)[:120])
        print(f"xvr: TPLDIAG {fn}: moved {moved[:8]} renamed-guard failures {len(bad)} {bad[:4]} pin failures {len(badp)} {badp[:4]}", flush=True)
    except Exception as e:
        import traceback
        print("xvr: TPLDIAG error", repr(e)[:200], traceback.format_exc()[-800:], flush=True)


# ---- diagnostic XVR_KEYDIFF=1 (installed by diag27 after tpl_layer): what distinguishes two layers' GDN template keys
_KEYS: dict = {}  # id(lowering) -> [(op index, key)]


def install_keydiff():
    inner = RD._reads

    def reads(lowered, op, selector, *rest):
        r = inner(lowered, op, selector, *rest)
        if r is not None and "gdn" in str(op.func):
            lst = _KEYS.setdefault(id(lowered.lowering), [])
            lst.append(r[0])
            if len(lst) in (2, 4, 7) and STATS["keydiff_printed"] < 6:
                STATS["keydiff_printed"] += 1
                a, b = lst[0], lst[-1]
                diffs = [(j, x, y) for j, (x, y) in enumerate(zip(a, b)) if x != y]
                print(f"xvr: KEYDIFF gdn layer key 0 vs {len(lst) - 1}: len {len(a)} vs {len(b)}, {len(diffs)} diffs: "
                      + "; ".join(f"[{j}] {str(x)[:90]} | {str(y)[:90]}" for j, x, y in diffs[:8])
                      + f"; context before first: {[str(z)[:60] for z in a[max(0, diffs[0][0] - 3):diffs[0][0]]] if diffs else ''}", flush=True)
        return r

    RD._reads = reads
