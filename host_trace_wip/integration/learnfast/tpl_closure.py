"""PROTOTYPE (learnfast, 10-15; coordinator: item (b), templates keyed by closure roots): one redo per op class instead of one
per layer instance.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import tpl_closure

Core's redispatch templates (_reads key -> Template -> _instantiate) never match across layers of a vLLM step:
  - the key holds every root's layout as the variant's nodes, and each declared input (the static prefix: weights, buffers,
    caches) has its own offset / size symbols, so two layers' weights never give one key;
  - an op whose redo reaches roots outside its call (closure-resolved declared storages, lifted metadata tensors) gets no
    template at all (_template: a pointer into a root of no role).
This wraps _reads:
  1. a SymInt of the key whose symbols are all declared inputs' is keyed by its value (the static prefix is the same tensors
     every call: its layout symbols are constants of the entry; core's own contract for static_prefix);
  2. the roots the op's own recorded launch chain holds are appended to the read walk (in chain order, kind and layout in the
     key as above), so a template's roles cover the closure / lifted roots, and an instance of the same key takes its own
     roots at those positions (the layers' chains are one structure: the key holds the chain's kernels and root layouts).
Pins (a guard, dimension or displacement reading a root's symbol) are core's: such a root must be the same object for an
instance to take the template. Equality: torch.cuda._host_trace.verify_redo_templates=True makes every instantiation also
run the full redo and assert one entry (core's _assert_same_entry). STATS counts keys and instantiations."""
import collections
import os
import sys
import time

import torch
import torch.cuda._host_trace_ir as IR
import torch.cuda._host_trace_redispatch as _rd
import torch.cuda._host_trace_replay as _hr
from torch.cuda._host_trace_tape import _sym_key

STATS: collections.Counter = collections.Counter()
KEYS: dict = {}  # func -> distinct key hashes (diagnostic)
_FIRST: dict = {}


def _replay():
    f = sys._getframe(1)
    while f is not None and not isinstance(f.f_locals.get("self"), _hr.HostTraceReplay):
        f = f.f_back
    return None if f is None else f.f_locals["self"]


def _declared_names(old):
    rp = _replay()
    n = getattr(rp, "static_prefix", 0) if rp is not None else 0
    out = set()
    for rec in old.inputs:
        if rec.position < n:
            for v in (*rec.sizes, *rec.strides, rec.offset, rec.root.sym):
                k = _sym_key(v)
                if isinstance(k, IR.Node):
                    out |= k.free_symbols
    return out


# TPL_LAYER_KEYOUT=1 (coordinator 10-15, item (a)): a str constant of the op's call that names a vLLM layer (a key of the forward
# context's no_compile_layers: the GDN / attention ops' layer_name) is keyed out; the op is keyed by the roots the name resolves to
# (the chain roots below: its weights and caches, closure-resolved). Sound by construction with TPL_CONFIRM_ONCE: each (key, layer)
# is confirmed once, at its first instantiation, by a full redo compared with core's _assert_same_entry; a per-layer Python scalar
# that reaches a launch is a mismatch there and that layer stays on full redos (its templates are never stored).
LAYER_KEYOUT = os.environ.get("TPL_LAYER_KEYOUT") == "1"
# TPL_OFFSET_SHIFT=1 (with the key-out): a static-prefix argument read through the op's chain whose offset is one plain symbol (a
# layer's view into a cache buffer the layers share: GDN conv / SSM state) is keyed without that offset; an instance whose argument
# there has another offset takes the template with each pointer slot into that root moved by the offsets' difference in bytes
# (displacement + (o_j - o_i) * itemsize); its guards are renamed onto o_j and both offsets value-pinned (_specialize). Checked by
# the same one-time confirmation per (key, layer).
OFFSET_SHIFT = os.environ.get("TPL_OFFSET_SHIFT", "1" if LAYER_KEYOUT else "0") == "1"


def _prefix():
    rp = _replay()
    return getattr(rp, "static_prefix", 0) if rp is not None else 0


def _shiftable(rec):
    k = _sym_key(rec.offset)
    return isinstance(k, IR.Node) and k.op == "sym" and k.hint is not None
CONFIRM_ONCE = os.environ.get("TPL_CONFIRM_ONCE", "1" if LAYER_KEYOUT else "0") == "1"
_NAMES: set = set()


def _layer_names():
    if not _NAMES:
        try:
            from vllm.forward_context import get_forward_context

            _NAMES.update(get_forward_context().no_compile_layers)
        except Exception:
            pass
    return _NAMES


def _const_item(item, declared):
    # an argument root's (dtype, extent, cow): its extent is the trace's absolute byte range; keyed by its length
    if (isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], torch.dtype) and isinstance(item[1], tuple)
            and len(item[1]) == 2 and all(isinstance(x, int) for x in item[1])):
        return (item[0], ("bytes", item[1][1] - item[1][0]), item[2])
    if os.environ.get("TPL_VALUE_KEY", "1") == "1" and isinstance(item, tuple) and len(item) == 2 and item[0] == "int" and isinstance(item[1], IR.Node):
        fs = item[1].free_symbols
        if fs and fs <= declared and item[1].hint is not None:
            return ("const", int(item[1].hint))
    return item


if not getattr(_rd._reads, "_tpl_closure", False):
    _orig_reads = _rd._reads

    def _reads(lowered, op, selector):
        r = _orig_reads(lowered, op, selector)
        if r is None:
            STATS["no_key"] += 1
            return None
        key, roots, sites = r
        old = lowered.tape
        declared = _declared_names(old)
        items = [_const_item(x, declared) for x in key]
        if LAYER_KEYOUT:
            names = _layer_names()
            items = [("layer name",) if isinstance(x, tuple) and len(x) == 2 and (x[0] is str and x[1] in names or getattr(x[1], "value", None) in names) else x
                     for x in items]
            STATS["layer_keyout" if ("layer name",) in items else "no_layer_name"] += 1
        seen = {id(x): k for k, x in enumerate(roots)}
        roots = list(roots)
        items.append("chain roots")
        for _, rec in old.launches[op.launches.start : op.launches.stop]:
            for rt in getattr(rec, "roots", ()) or ():
                if id(rt) in seen:
                    items.append(("root", seen[id(rt)]))
                    continue
                seen[id(rt)] = len(roots)
                roots.append(rt)
                items.append(("new root", rt.kind))
                if rt.kind == "argument":
                    ir = next(i for i in old.inputs if i.root is rt)
                    for v in (*ir.sizes, *ir.strides, ir.offset):
                        k = _sym_key(v)
                        items.append(_const_item(("int", k), declared) if isinstance(k, IR.Node) else (type(k), k))
                    if OFFSET_SHIFT and ir.position < _prefix() and _shiftable(ir):
                        items[-1] = ("declared offset",)
                    items.append(_const_item((ir.dtype, ir.extent, ir.cow), declared))
                elif rt.kind == "allocation":
                    a = next(a for a in old.allocs if a.root is rt)
                    for v in (*a.sizes, *a.strides):
                        k = _sym_key(v)
                        items.append(("int", k) if isinstance(k, IR.Node) else (type(k), k))
                    items.append(a.dtype)
        key2 = tuple(items)
        try:
            hash(key2)
        except TypeError:
            return r
        STATS["keys"] += 1
        KEYS.setdefault(str(op.func)[:60], set()).add(hash(key2))
        fn = str(op.func)[:60]
        if os.environ.get("TPL_DUMP_KEY") and os.environ["TPL_DUMP_KEY"] in fn and fn not in _FIRST:
            print(f"tpl_closure: key dump {fn} " + " | ".join(f"{j}:{str(x)[:90]}" for j, x in enumerate(key2)), flush=True)
        prev = _FIRST.setdefault(fn, key2)
        dk = "diffs_shown:" + fn
        if prev is not key2 and os.environ.get("TPL_DIFF_FUNC", "") in fn and STATS[dk] < int(os.environ.get("TPL_DIFFS", "3")):
            STATS[dk] += 1
            if len(prev) == len(key2):
                d = [(j, str(a)[:100], str(b)[:100]) for j, (a, b) in enumerate(zip(prev, key2)) if a != b][:12]
                print(f"tpl_closure: key diff {fn[:50]} n={len(key2)} {d}", flush=True)
            else:
                print(f"tpl_closure: key length {fn[:50]} {len(prev)} vs {len(key2)}", flush=True)
        return key2, roots, sites

    _reads._tpl_closure = True
    _rd._reads = _reads

    _orig_inst = _rd._instantiate

    def _instantiate(lowered, i, t, roots, base, selector, op):
        STATS["instantiated"] += 1
        got = _orig_inst(lowered, i, t, roots, base, selector, op)
        if OFFSET_SHIFT:
            got = (_shifted(lowered, t, got[0], roots), got[1])
        if not CONFIRM_ONCE:
            return got
        key = CURRENT["reads"][0] if CURRENT["reads"] is not None else None
        if key is None or not (("layer name",) in key or ("declared offset",) in key):
            return got  # no key-out: the template's own soundness (renamed guards, value pins), as before
        ck = (key, tuple(map(id, roots)))
        if ck in CONFIRMED:
            STATS["trusted"] += 1
            return got
        f = sys._getframe(1)
        while f is not None and f.f_code.co_name != "redispatch":
            f = f.f_back
        loc = f.f_locals
        t0 = time.perf_counter()
        full = _rd.lower_entry(lowered, i, *_rd._full_redo(lowered, i, selector, loc["dispatches"], loc["opaque"]), base)
        STATS["confirm_s"] += time.perf_counter() - t0
        try:
            _rd._assert_same_entry(got[0], full[0], lowered.lowering.program.values)
        except AssertionError as e:
            STATS["confirm_mismatch"] += 1
            FAILED.add(ck)
            if STATS["confirm_mismatch"] <= 3:
                print(f"tpl_closure: confirmation mismatch {str(op.func)[:50]}: {str(e)[:300]}", flush=True)
            return full
        STATS["confirmed"] += 1
        CONFIRMED.add(ck)
        return got

    _rd._instantiate = _instantiate


# Pins on declared inputs. A template's guards reading a declared argument's address / layout symbols (CuTe's alignment
# checks of a weight's data pointer: (base + offset) % 16 == 0) pin that root, so no other layer could take it. A declared
# input's symbols are constants of the entry, and the key already holds its layout values; so for an instance whose root at a
# pinned declared position is another declared argument, the template is specialized: those guards renamed onto that
# instance's argument (its root's address symbol and its record's layout symbols) and added to the predicate (lowered at
# this call, as the entry's own guards are), the pins moved to its roots. Anything else at a pin keeps core's refusal.
import dataclasses  # noqa: E402

CURRENT = {"reads": None}
CONFIRMED: set = set()  # (key, roots ids) confirmed once
FAILED: set = set()  # (key, roots ids) whose confirmation mismatched: full redos, no stored template
_SPECIAL: dict = {}


def _rec_of(old, root):
    return next((i for i in old.inputs if i.root is root), None)


def _sym_pairs(rec_i, rec_j):
    out = {}
    for a, b in zip((rec_i.root.sym, rec_i.offset, *rec_i.sizes, *rec_i.strides), (rec_j.root.sym, rec_j.offset, *rec_j.sizes, *rec_j.strides)):
        ka, kb = _sym_key(a), _sym_key(b)
        if isinstance(ka, IR.Node) and ka.op == "sym":
            out[ka.args[0]] = kb
    return out


def _specialize(lowered, t, reads):
    """t for the instance reading `reads`: every position where the template's own instance read a declared argument and
    this one reads another, its guards renamed onto this one's argument (address and layout symbols), and value pins (each
    declared layout symbol keyed by value equal to its value, both instances': the key took them as constants, these rows
    check it at each call, so a changed static prefix misses instead of reusing); pins moved to this one's roots."""
    lo, old = lowered.lowering, lowered.tape
    rp = _replay()
    n = getattr(rp, "static_prefix", 0) if rp is not None else 0
    ctx = old.shape_env.ctx
    roots_i = _TPL_ROOTS.get(id(t))
    if roots_i is None or len(roots_i) != len(reads[1]):
        return None
    subs, vpins = {}, []
    for r_i, r_j in zip(roots_i, reads[1]):
        if r_i.kind != "argument":
            continue
        ri, rj = _rec_of(old, r_i), _rec_of(old, r_j)
        if ri is None or ri.position >= n:
            continue
        if rj is None or rj.position >= n:
            return None
        for rec in (ri, rj):
            for v in (rec.offset, *rec.sizes, *rec.strides):
                k = _sym_key(v)
                if isinstance(k, IR.Node) and k.hint is not None:
                    vpins.append(ctx.eq(k, ctx.const(int(k.hint))))
        if r_j is not r_i:
            for name, node in _sym_pairs(ri, rj).items():
                subs[name] = node if isinstance(node, IR.Node) else ctx.const(int(node))
    pins = []
    for p, r in t.pinned:
        r_j = reads[1][p]
        if r_j is not r and _rec_of(old, r) is not None and _rec_of(old, r).position < n:
            pins.append((p, r_j))
        elif r_j is r:
            pins.append((p, r))
        else:
            return None
    full = {**ctx.symbols, **subs}
    memo, renamed = {}, []
    if subs:
        for g, _ in t.guards:
            if isinstance(g, IR.Node) and g.free_symbols & subs.keys():
                renamed.append((ctx.transfer(g, ctx, full, memo), None))
    guards = renamed + [(v, None) for v in vpins if not (isinstance(v, IR.Node) and v.op == "true")]
    if not guards:
        return _origin(dataclasses.replace(t, pinned=tuple(pins)), roots_i)
    addresses = frozenset(lo.key(rec.root.sym) for rec in old.inputs)
    with lo.lowering.total() as domains:
        pred = lo.lowering.conjunction(guards, addresses)
    predicate = lo.emit("and", t.predicate, pred)
    for ok in domains:
        predicate = lo.emit("and", predicate, ok)
    return _origin(dataclasses.replace(t, predicate=predicate, pinned=tuple(pins)), roots_i)


def _origin(t, roots_i):
    _TPL_ROOTS[id(t)] = roots_i
    _KEEP.append(t)
    return t


def _shifted(lowered, t, entry, roots):
    """entry (an instantiation of t) with each pointer slot into a static-prefix argument whose offset differs from t's own
    instance's moved by the difference (TPL_OFFSET_SHIFT)."""
    roots_i = _TPL_ROOTS.get(id(t))
    if roots_i is None:
        return entry
    old, lo = lowered.tape, lowered.lowering
    delta = {}
    for p, (r_i, r_j) in enumerate(zip(roots_i, roots)):
        if r_i is r_j or r_i.kind != "argument":
            continue
        a, b = _rec_of(old, r_i), _rec_of(old, r_j)
        if a is None or b is None or not (_shiftable(a) and _shiftable(b)):
            continue
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


_TPL_ROOTS: dict = {}  # id(template) -> the reads roots of the instance whose redo made it


class _ReadsDict(dict):
    def __getitem__(self, k):
        v = super().__getitem__(k)
        CURRENT["reads"] = v
        return v


class _TplDict(dict):
    lowered = None

    def get(self, key, default=None):
        t = super().get(key, default)
        reads = CURRENT["reads"]
        if t is not None and reads is not None and (key, tuple(map(id, reads[1]))) in FAILED:
            return None
        if t is None or reads is None or reads[0] != key or id(t) not in _TPL_ROOTS:
            return t
        ck = (id(t), tuple(id(r) for r in reads[1]))
        if ck not in _SPECIAL:
            try:
                _SPECIAL[ck] = _specialize(self.lowered, t, reads)
                STATS["specialized" if _SPECIAL[ck] is not None else "not_specializable"] += 1
            except Exception as e:  # a renamed guard false at the hints, a symbol the context lacks: core's refusal
                _SPECIAL[ck] = None
                STATS["specialize_failed"] += 1
                if STATS["specialize_failed"] <= 3:
                    print(f"tpl_closure: specialize failed {type(e).__name__}: {str(e)[:200]}", flush=True)
        if _SPECIAL[ck] is None:
            return None  # this instance cannot take it: the full redo (core's refusal)
        return _SPECIAL[ck]


if not getattr(_rd._reads, "_tpl_dicts", False):
    _reads_inner = _rd._reads

    def _reads_with_dicts(lowered, op, selector):
        lo = lowered.lowering
        if not isinstance(lo.reads, _ReadsDict):
            lo.reads = _ReadsDict(lo.reads)
        if not isinstance(lo.templates, _TplDict):
            td = _TplDict(lo.templates)
            td.lowered = lowered
            lo.templates = td
        return _reads_inner(lowered, op, selector)

    _reads_with_dicts._tpl_closure = True
    _reads_with_dicts._tpl_dicts = True
    _rd._reads = _reads_with_dicts


# remember the reading instance's roots of each template (the specialization maps its roots onto another instance's)
if not getattr(_rd._template, "_tpl_roots", False):
    _tpl_inner = _rd._template

    def _template(lowered, entry, guards, launches, redone, roots, op):
        r = CURRENT["reads"]
        if r is not None and (r[0], tuple(map(id, roots))) in FAILED:
            return None
        t = _tpl_inner(lowered, entry, guards, launches, redone, roots, op)
        if t is not None:
            _TPL_ROOTS[id(t)] = list(roots)
            _KEEP.append(t)
        return t

    _template._tpl_roots = True
    _rd._template = _template
_KEEP: list = []  # the templates _TPL_ROOTS names (ids stay valid)
