"""PROTOTYPE (learnfast, 10-15; coordinator item (a)): templates shared across vLLM layers, on a head with the template core
patch (step_ms10+: _reads keys static-prefix layouts by value and walks the chain roots; _for_instance renames a template onto
another layer's static-prefix arguments). tpl_closure.py is the same for heads without the patch.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import tpl_layer

Two key relaxations, each sound by a one-time confirmation:
  1. layer name: a constant of the op's call naming a vLLM layer (a str or LayerName that is a key of the forward context's
     no_compile_layers: the GDN / attention ops' layer_name) is keyed out; the op is keyed by the roots the name resolves to
     (core's chain roots: the layer's weights and caches, closure-resolved).
  2. declared offset: a static-prefix argument root whose offset is one plain symbol (a layer's view into a cache buffer the
     layers share: GDN conv / SSM state, attention KV) is keyed without that offset. An instance whose argument there has another
     offset takes the template with each pointer slot into that root moved by the offsets' difference in bytes; core's
     _for_instance renames the guards onto the instance's argument and value-pins both offsets.
Confirmation: the first instantiation of each (key, layer) (layer = the instance's roots) also runs the full redo and compares
with core's _assert_same_entry; on a mismatch the full redo's entry is used and that (key, layer) stays on full redos (no
template given to it, none stored from it). After that the layer instantiates without re-verifying. A per-layer Python scalar
that reaches a launch is a mismatch at that first confirmation. Templates with neither relaxation are core's, unconfirmed."""
import collections
import dataclasses
import sys
import time

import torch.cuda._host_trace_ir as IR
import torch.utils._pytree as pytree
import torch.cuda._host_trace_redispatch as _rd
import torch.cuda._host_trace_replay as _hr
from torch.cuda._host_trace_tape import _sym_key

STATS: collections.Counter = collections.Counter()
KEYS: dict = {}  # func -> distinct key hashes (diagnostic)
CURRENT = {"reads": None}
CONFIRMED: set = set()
FAILED: set = set()
_ORIGIN: dict = {}  # id(instance template) -> (instance template, the roots of the op whose redo made its template)
_NAMES: set = set()
LAYER = ("layer name",)
OFFSET = ("declared offset",)
DECLVAL = ("declared value",)
import os as _os  # noqa: E402

DECLVAL_ON = _os.environ.get("TPL_DECLVAL") == "1"


def _layer_names():
    if not _NAMES:
        try:
            from vllm.forward_context import get_forward_context

            _NAMES.update(get_forward_context().no_compile_layers)
        except Exception:
            pass
    return _NAMES


def _prefix():
    f = sys._getframe(1)
    while f is not None and not isinstance(f.f_locals.get("self"), _hr.HostTraceReplay):
        f = f.f_back
    return 0 if f is None else f.f_locals["self"].static_prefix


def _rec_of(old, root):
    return next((i for i in old.inputs if i.root is root), None)


def _shiftable(rec):
    k = _sym_key(rec.offset)
    return isinstance(k, IR.Node) and k.op == "sym" and k.hint is not None


def _ids(roots):
    return tuple(map(id, roots))


class _ReadsDict(dict):
    def __getitem__(self, k):
        v = super().__getitem__(k)
        CURRENT["reads"] = v
        return v


_orig_reads = _rd._reads


def _reads(lowered, op, selector, declared=frozenset()):
    lo = lowered.lowering
    if not isinstance(lo.reads, _ReadsDict):
        lo.reads = _ReadsDict(lo.reads)
    r = _orig_reads(lowered, op, selector, declared)
    if r is None:
        return None
    key, roots, sites, *rest = r  # step_ms22+: a 4th element (core's substituted static-prefix symbols)
    names, items, old, prefix = _layer_names(), list(key), lowered.tape, _prefix()
    c = 0
    for j, x in enumerate(key):
        if not isinstance(x, tuple) or len(x) != 2:
            continue
        if x[0] is str and x[1] in names or getattr(x[1], "value", None) in names:
            items[j] = LAYER
        elif x[0] == "new root":
            rt = roots[c]
            c += 1
            rec = _rec_of(old, rt) if rt.kind == "argument" else None
            if rec is not None and rec.position < prefix and _shiftable(rec):
                items[j + len(rec.sizes) + len(rec.strides) + 1] = OFFSET
    if DECLVAL_ON and declared:
        # TPL_DECLVAL (10-18): a mixed expression over static-prefix layout symbols and others (an allocation's 128*s26*T,
        # s26 a layer's own weight dimension) keyed with each static-prefix symbol at its value, as a pure static-prefix
        # expression already is (core's ("value", hint)); marked so the per-(key, layer) confirmation runs
        ctx = old.shape_env.ctx
        for j, x in enumerate(items):
            if isinstance(x, tuple) and len(x) == 2 and x[0] == "int" and isinstance(x[1], IR.Node):
                fs = x[1].free_symbols & declared
                if fs and not fs >= x[1].free_symbols:
                    subs = {n: ctx.const(int(ctx.symbols[n].hint)) for n in fs}
                    items[j] = ("int", ctx.transfer(x[1], ctx, collections.ChainMap(subs, ctx.symbols), {}))
                    if DECLVAL not in items:
                        items.append(DECLVAL)
    key2 = tuple(items)
    STATS["declval" if DECLVAL in key2 else "no_declval"] += 1
    STATS["layer_keyout" if LAYER in key2 else "no_layer_name"] += 1
    STATS["offset_keyout" if OFFSET in key2 else "no_offset"] += 1
    KEYS.setdefault(str(op.func)[:60], set()).add(hash(key2))
    return (key2, roots, sites, *rest)


_orig_fi = _rd._for_instance


def _for_instance(lowered, t, roots, static_prefix, *rest):
    reads = CURRENT["reads"]
    if reads is not None and (reads[0], _ids(roots)) in FAILED:
        return None
    out = _orig_fi(lowered, t, roots, static_prefix, *rest)
    if out is not None and out is not t:
        _ORIGIN[id(out)] = (out, t.roots)
    return out


def _shifted(lowered, t, entry, roots, origin):
    old, lo = lowered.tape, lowered.lowering
    delta = {}
    for p, (r_i, r_j) in enumerate(zip(origin, roots)):
        if r_i is r_j or r_i.kind != "argument":
            continue
        a, b = _rec_of(old, r_i), _rec_of(old, r_j)
        if a is not None and b is not None and _shiftable(a) and _shiftable(b):
            d = (int(_sym_key(b.offset).hint) - int(_sym_key(a.offset).hint)) * a.dtype.itemsize
            if d:
                delta[p] = d
    if not delta:
        return entry
    STATS["shifted"] += 1
    i, predicate, rows, section = entry
    out = []
    for row, role in zip(rows, t.roles):
        slots = list(row.slots)
        for k, (kind, p) in role:
            if kind != "temp" and p in delta:
                sl = slots[k]
                disp = lo.emit("add", sl.displacement, lo.emit("constant", delta[p]))
                addr = lo.emit("add", lo.row(roots[p].sym), disp) if sl.address is not None else None
                slots[k] = dataclasses.replace(sl, displacement=disp, address=addr)
        out.append(dataclasses.replace(row, slots=tuple(slots)))
    return (i, predicate, tuple(out), section)


# TPL_CONFIRM=oplayer (coordinator 10-15, guarded form of lever (i); labelled, the user may veto): the full confirmation runs once
# per (op, layer) across variants; a later instantiation of that (op, layer) under another variant passes a cheap check instead:
# its fingerprint (the key with every symbolic node as a placeholder, so per-layer Python constants and constant layouts stay; the
# layer name; each root's kind, argument position, offset value and dtype) equals the confirmed one, else the full confirmation.
# TPL_CONFIRM=key (default): per (key, layer) as above. TPL_CONFIRM_VALIDATE=1 (with oplayer): the full confirmation runs every time
# too, and a pass of the cheap check whose full confirmation fails is counted (cheap_ok_full_failed; expected 0).
import os  # noqa: E402

CONFIRM = os.environ.get("TPL_CONFIRM", "key")
VALIDATE = os.environ.get("TPL_CONFIRM_VALIDATE") == "1"
_OPLAYER: dict = {}  # (func, layer id) -> the confirmed fingerprint


def _placeholder(x):
    if isinstance(x, IR.Node):
        return "sym"
    if isinstance(x, tuple):
        return tuple(_placeholder(y) for y in x)
    return x


def _layer_of(lowered, op, roots):
    old, names, layer = lowered.tape, _layer_names(), None
    for leaf in pytree.tree_leaves(op.call):
        v = leaf if isinstance(leaf, str) else getattr(leaf, "value", None)
        if isinstance(v, str) and v in names:
            layer = v
    meta = []
    for r in roots:
        rec = _rec_of(old, r) if r.kind == "argument" else None
        off = _sym_key(rec.offset) if rec is not None else None
        meta.append((r.kind, rec.position if rec is not None else None, int(off.hint) if isinstance(off, IR.Node) and off.hint is not None else off,
                     str(rec.dtype) if rec is not None else None))
    prefix = _prefix()
    ident = (layer, tuple(m[1] for m in meta if m[1] is not None and m[1] < prefix))
    return ident, tuple(meta)


_orig_inst = _rd._instantiate


def _instantiate(lowered, i, t, roots, base, selector, op):
    STATS["instantiated"] += 1
    entry, after = _orig_inst(lowered, i, t, roots, base, selector, op)
    o = _ORIGIN.get(id(t))
    entry = _shifted(lowered, t, entry, roots, o[1] if o is not None and o[0] is t else t.roots)
    key = CURRENT["reads"][0] if CURRENT["reads"] is not None else None
    if key is None or not (LAYER in key or OFFSET in key or DECLVAL in key):
        return entry, after
    ck = (key, _ids(roots))
    if ck in CONFIRMED:
        STATS["trusted"] += 1
        return entry, after
    cheap_ok, olk, fp = False, None, None
    if CONFIRM == "oplayer":
        ident, meta = _layer_of(lowered, op, roots)
        olk, fp = (str(op.func), ident), (_placeholder(key), meta)
        cheap_ok = _OPLAYER.get(olk) == fp
        STATS["cheap_ok" if cheap_ok else ("cheap_differs" if olk in _OPLAYER else "cheap_first")] += 1
        if cheap_ok and not VALIDATE:
            CONFIRMED.add(ck)
            return entry, after
    f = sys._getframe(1)
    while f.f_code.co_name != "redispatch":
        f = f.f_back
    t0 = time.perf_counter()
    full = _rd.lower_entry(lowered, i, *_rd._full_redo(lowered, i, selector, f.f_locals["dispatches"], f.f_locals["opaque"]), base)
    STATS["confirm_s"] += time.perf_counter() - t0
    try:
        _rd._assert_same_entry(entry, full[0], lowered.lowering.program.values)
    except AssertionError as e:
        STATS["confirm_mismatch"] += 1
        if cheap_ok:
            STATS["cheap_ok_full_failed"] += 1
        FAILED.add(ck)
        if STATS["confirm_mismatch"] <= 3:
            print(f"tpl_layer: confirmation mismatch {str(op.func)[:50]}: {str(e)[:400]}", flush=True)
        return full
    STATS["confirmed"] += 1
    CONFIRMED.add(ck)
    if olk is not None and olk not in _OPLAYER:
        _OPLAYER[olk] = fp
    return entry, after


_orig_tpl = _rd._template


def _template(lowered, entry, guards, launches, redone, roots, op):
    r = CURRENT["reads"]
    if r is not None and (r[0], _ids(roots)) in FAILED:
        return None
    return _orig_tpl(lowered, entry, guards, launches, redone, roots, op)


if not getattr(_rd._reads, "_tpl_layer", False):
    _reads._tpl_layer = True
    _rd._reads, _rd._for_instance, _rd._instantiate, _rd._template = _reads, _for_instance, _instantiate, _template
