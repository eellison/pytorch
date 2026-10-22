# adapter_hook.py: stock SGLang + the hook, generic over attention backends and KV / recurrent-state pools (hybrid
# models such as Qwen3.8-Flash-Next: QSA sparse attention + GDN + PLE). The region starts at model.forward: SGLang's
# prep and every sub-backend's init_forward_metadata run eagerly as shipped. Per call, the entry's arguments are
#   - the model's parameters and buffers (one per tensor, swapped in by _reparametrize_module) and its modules' plain
#     CUDA tensor attributes;
#   - every CUDA tensor held by the attention backends and the two pools (token_to_kv_pool, req_to_token_pool) and
#     their sub-objects, found by walking attributes, lists and tuples (KV caches, QSA buffers, mamba conv / ssm state,
#     PLE / n-gram pools, workspaces), swapped into their owners inside the step;
#   - the ForwardBatch's CUDA tensors and ints, and each sub-backend's forward_metadata's CUDA tensors and ints (walked
#     through dataclasses, msgspec Structs and plain objects), ints as symbolic ints.
# Bools, enums, None, strings and CPU tensors' dtype / shape go into the entry's key; a CPU tensor or list the region
# reads is baked (logged in summary()["baked_host_reads"]). No engine source is edited; attention goes through
# FlashInfer's API with the trtllm-gen fork (INTEG_ATTN=fork) or FlashInfer's traced C++ launcher (INTEG_ATTN=cpp).
import collections
import copy
import dataclasses
import os
import traceback

import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.nn.utils.stateless import _reparametrize_module

FAMILIES = tuple(os.environ.get("ARMF_FAMILIES", "blas,attention,rng,reduce").split(","))
EXTERN = [n for n in os.environ.get("ARMF_EXTERN", "fused_inplace_qknorm,apply_rope_inplace,_run_activation_inplace,store_cache").split(",") if n]
MEMORY = os.environ.get("ARMF_MEMORY", "run_buffer")
ATTN = os.environ.get("INTEG_ATTN", "fork")
ht.raise_unexpected = bool(os.environ.get("ARMF_HT_RAISE"))
if ATTN == "cpp":
    import flashinfer.trtllm_cpp  # noqa: F401
elif ATTN == "fork":
    import flashinfer.trtllm_trace  # noqa: F401

DECLINE_SITES = collections.Counter()
_orig_declined_init = ht.Declined.__init__


def _declined_init(exc, *args, **kwargs):
    _orig_declined_init(exc, *args, **kwargs)
    frames = [f for f in traceback.extract_stack()[:-1] if "/torch/" not in f.filename and not f.filename.endswith("adapter_hook.py")]
    where = f"{os.path.basename(frames[-1].filename)}:{frames[-1].lineno} {frames[-1].line}" if frames else "?"
    DECLINE_SITES[(str(exc)[:200], where[:200])] += 1


ht.Declined.__init__ = _declined_init
BAKED = collections.Counter()


def _cuda(x):
    return isinstance(x, torch.Tensor) and x.is_cuda


def _sglang_obj(v):
    return hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, torch.nn.Module, type)) and type(v).__module__.startswith("sglang")


def _backends(b):
    out = [b]
    for s in list(getattr(b, "attn_backend_list", None) or []) + [getattr(b, n, None) for n in ("full_attn_backend", "linear_attn_backend")]:
        if s is not None and all(s is not o for o in out):
            out.append(s)
    return out


def _object_slots(roots, depth=3):
    """(owner, attribute, index or None) of every CUDA tensor held by the roots and their SGLang sub-objects, through
    attributes, lists and tuples, to `depth` levels."""
    slots, seen = [], set()

    def walk(o, d):
        if id(o) in seen:
            return
        seen.add(id(o))
        for k, v in list(vars(o).items()):
            if _cuda(v):
                slots.append((o, k, None))
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if _cuda(x):
                        slots.append((o, k, i))
                    elif d > 0 and _sglang_obj(x):
                        walk(x, d - 1)
            elif d > 0 and _sglang_obj(v):
                walk(v, d - 1)

    for r in roots:
        if r is not None:
            walk(r, depth)
    return slots


def _fields(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclasses.fields(obj)]
    if hasattr(type(obj), "__struct_fields__"):
        return list(type(obj).__struct_fields__)
    return list(vars(obj))


def _structural(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, torch.Tensor):
        return ("cpu", str(v.dtype), tuple(v.shape), v.is_pinned())
    if hasattr(v, "name") and hasattr(v, "value") and type(v).__module__.startswith("sglang"):  # an enum
        return (type(v).__name__, v.name)
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, len(v))
    return type(v).__name__


class _Flat:
    """One object's walk: CUDA tensors and ints become argument leaves (paths), the rest structure (key) and baked
    values; nested dataclasses / msgspec Structs / SGLang objects are walked to `depth`."""

    @staticmethod
    def walk(obj, prefix, tensors, ints, key, depth):
        for n in _fields(obj):
            v = getattr(obj, n, None)
            p = prefix + (n,)
            if _cuda(v):
                tensors.append((p, v))
            elif type(v) is int:
                ints.append((p, v))
            elif isinstance(v, (list, tuple)) and any(_cuda(x) for x in v):
                key.append((p, (type(v).__name__, len(v), tuple(_cuda(x) for x in v))))
                for k, x in enumerate(v):
                    if _cuda(x):
                        tensors.append((p + (k,), x))
            elif depth > 0 and (dataclasses.is_dataclass(v) or hasattr(type(v), "__struct_fields__")) and not isinstance(v, type):
                key.append((p, type(v).__name__))
                _Flat.walk(v, p, tensors, ints, key, depth - 1)
            else:
                key.append((p, _structural(v)))


def _rebuild(obj, repl, prefix, depth, log):
    """obj with the leaves in repl (path -> value) replaced, rebuilt bottom up (dataclasses copied, msgspec Structs
    replaced, plain objects copied); reads of its baked non-tensor fields are logged."""
    changes = {}
    for n in _fields(obj):
        p = prefix + (n,)
        if p in repl:
            changes[n] = repl[p]
            continue
        v = getattr(obj, n, None)
        if isinstance(v, (list, tuple)) and any(p + (k,) in repl for k in range(len(v))):
            new = [repl.get(p + (k,), x) for k, x in enumerate(v)]
            changes[n] = type(v)(new) if isinstance(v, tuple) else new
            continue
        if depth > 0 and (dataclasses.is_dataclass(v) or hasattr(type(v), "__struct_fields__")) and not isinstance(v, type):
            if any(k[: len(p)] == p for k in repl):
                changes[n] = _rebuild(v, repl, p, depth - 1, log)
    if hasattr(type(obj), "__struct_fields__"):
        import msgspec

        return msgspec.structs.replace(obj, **changes)
    s = copy.copy(obj)
    for k, v in changes.items():
        object.__setattr__(s, k, v)
    if log:
        _log_reads(s, set(changes))
    return s


_LOGGED = {}


def _log_reads(s, replaced):
    base = type(s)
    cls = _LOGGED.get(base)
    if cls is None:
        names = frozenset(_fields(s))

        def __getattribute__(self, name, _names=names, _base=base):
            v = object.__getattribute__(self, name)
            if name in _names and v is not None and not _cuda(v) and not isinstance(v, (int, bool, torch.SymInt)):
                BAKED[f"{_base.__name__}.{name}"] += 1
            return v

        try:
            cls = _LOGGED[base] = type(f"Hooked{base.__name__}", (base,), {"__getattribute__": __getattribute__})
        except TypeError:
            cls = _LOGGED[base] = base
    if cls is not base:
        try:
            object.__setattr__(s, "__class__", cls)
        except TypeError:
            pass


class ArmS:
    def __init__(self, model_runner):
        self.mr = model_runner
        self.model = model_runner.model
        self.orig_forward = self.model.forward
        self.backend = model_runner.attn_backend
        index, self.params, self.names, self.slots = {}, [], [], []
        for mn, m in self.model.named_modules():
            for k, t in [*m._parameters.items(), *m._buffers.items()]:
                if t is None:
                    continue
                if id(t) not in index:
                    index[id(t)] = len(self.params)
                    self.params.append(t.detach())
                self.names.append(f"{mn}.{k}" if mn else k)
                self.slots.append(index[id(t)])
        pb = {id(t) for t in self.params}
        self.attrs = [(m, k) for m in self.model.modules() for k, v in vars(m).items() if _cuda(v) and id(v) not in pb]
        self.backends = _backends(self.backend)
        roots = [*self.backends, getattr(model_runner, "token_to_kv_pool", None), getattr(model_runner, "req_to_token_pool", None)]
        self.obj_slots = [s for s in _object_slots(roots) if id(getattr(s[0], s[1]) if s[2] is None else getattr(s[0], s[1])[s[2]]) not in pb]
        extern = tuple(op.default for n in EXTERN if (op := getattr(torch.ops.sglang, n, None)) is not None)
        indexed = {torch.ops.sglang.store_cache.default: ("indices", "k_cache", "v_cache")} if hasattr(torch.ops.sglang, "store_cache") else {}
        indexed = {k: v for k, v in indexed.items() if k in extern}
        self.provider = HarvestProvider(FAMILIES + ("extern",), extern_ops=extern, indexed=indexed) if extern else HarvestProvider(FAMILIES)
        self.entries, self.outs, self.calls = {}, {}, collections.Counter()
        self.installed = False

    def install(self):
        self.installed = True
        self.model.forward = self.forward

    def uninstall(self):
        self.model.forward = self.orig_forward
        self.installed = False

    def _slot_values(self):
        return [getattr(o, k) if i is None else getattr(o, k)[i] for o, k, i in self.obj_slots]

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        if args or any(v is not None for v in kwargs.values()):
            self.calls["eager_kwargs"] += 1
            return self.orig_forward(input_ids, positions, forward_batch, *args, **kwargs)
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or (mode.is_extend() and not mode.is_target_verify())):
            self.calls["eager_mode"] += 1
            return self.orig_forward(input_ids, positions, forward_batch)
        kind = "decode" if mode.is_decode() else "extend"
        fb_t, fb_i, fb_k = [], [], []
        _Flat.walk(forward_batch, ("fb",), fb_t, fb_i, fb_k, 0)
        fb_t = [(p, v) for p, v in fb_t if p[1] not in ("input_ids", "positions")]
        mds = [getattr(b, "forward_metadata", None) for b in self.backends]
        md_t, md_i, md_k = [], [], []
        for j, md in enumerate(mds):
            if md is None:
                md_k.append(((f"md{j}",), None))
                continue
            md_k.append(((f"md{j}",), type(md).__name__))
            _Flat.walk(md, (f"md{j}",), md_t, md_i, md_k, 2)
        statics = self._slot_values()
        attrs = [getattr(m, k) for m, k in self.attrs]
        skey = tuple(_cuda(v) for v in statics) + tuple(_cuda(v) for v in attrs)
        key = (kind, tuple(p for p, _ in fb_t), tuple(p for p, _ in fb_i), tuple(fb_k), tuple(p for p, _ in md_t), tuple(p for p, _ in md_i), tuple(md_k), skey)
        if not all(skey):
            self.calls["eager_static_not_cuda"] += 1
            return self.orig_forward(input_ids, positions, forward_batch)
        entry = self.entries.get(key)
        if entry is None:
            entry = self.entries[key] = R.HostTraceReplay(self._step(key), opaque=(self.provider,), memory=MEMORY)
        self.calls[kind] += 1
        flat_args = (*self.params, *attrs, *statics, input_ids, positions, *[v for _, v in fb_t], *[v for _, v in md_t], *[v for _, v in fb_i], *[v for _, v in md_i])
        self._ctx = (forward_batch, mds)
        try:
            flat = entry(*flat_args)
        finally:
            self._ctx = None
        template, names = self.outs[key]
        out = copy.copy(template)
        for n, t in zip(names, flat):
            setattr(out, n, t)
        return out

    def _step(self, key):
        kind, fb_tp, fb_ip, _, md_tp, md_ip, _, _ = key
        n_p, n_a, n_s = len(self.params), len(self.attrs), len(self.obj_slots)
        proxy = self

        def step(*flat):
            fb, mds = proxy._ctx
            i = 0
            params = flat[i:i + n_p]; i += n_p
            attrs = flat[i:i + n_a]; i += n_a
            statics = flat[i:i + n_s]; i += n_s
            input_ids, positions = flat[i], flat[i + 1]; i += 2
            repl = {}
            for p in fb_tp + md_tp + fb_ip + md_ip:
                repl[p] = flat[i]; i += 1
            fb_repl = {p: v for p, v in repl.items() if p[0] == "fb"}
            fb_repl[("fb", "input_ids")] = input_ids
            fb_repl[("fb", "positions")] = positions
            shell_fb = _rebuild(fb, fb_repl, ("fb",), 0, True)
            shell_mds = [None if md is None else _rebuild(md, {p: v for p, v in repl.items() if p[0] == f"md{j}"}, (f"md{j}",), 2, True)
                         for j, md in enumerate(mds)]
            saved_md = [getattr(b, "forward_metadata", None) for b in proxy.backends]
            saved_attrs = [(m, k, getattr(m, k)) for m, k in proxy.attrs]
            groups = collections.defaultdict(list)
            for (o, k, idx), t in zip(proxy.obj_slots, statics):
                groups[(id(o), k)].append((o, k, idx, t))
            saved_objs = []
            try:
                for b, md in zip(proxy.backends, shell_mds):
                    if md is not None:
                        b.forward_metadata = md
                for (m, k), t in zip(proxy.attrs, attrs):
                    setattr(m, k, t)
                for items in groups.values():
                    o, k = items[0][0], items[0][1]
                    old = getattr(o, k)
                    saved_objs.append((o, k, old))
                    if items[0][2] is None:
                        setattr(o, k, items[0][3])
                    else:
                        new = list(old)
                        for _, _, idx, t in items:
                            new[idx] = t
                        setattr(o, k, type(old)(new) if isinstance(old, tuple) else new)
                with _reparametrize_module(proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    out = proxy.orig_forward(input_ids, positions, shell_fb)
            finally:
                for b, md in zip(proxy.backends, saved_md):
                    if md is not None or hasattr(b, "forward_metadata"):
                        b.forward_metadata = md
                for m, k, v in saved_attrs:
                    setattr(m, k, v)
                for o, k, v in reversed(saved_objs):
                    setattr(o, k, v)
            names = [f.name for f in dataclasses.fields(out) if isinstance(getattr(out, f.name), torch.Tensor)]
            template = copy.copy(out)
            for n in names:
                setattr(template, n, None)
            proxy.outs[key] = (template, names)
            return tuple(getattr(out, n) for n in names)

        return step

    def counters(self):
        es = list(self.entries.values())
        out = {k: sum(getattr(e, k, 0) for e in es) for k in ("traces", "replays", "eager", "relowers", "redispatches", "learned")}
        out["variants"] = sum(len(e.variants) for e in es)
        out["calls"] = dict(self.calls)
        return out

    def summary(self):
        res = self.counters()
        res["harvests"] = self.provider.harvests
        res["bindings"] = len(self.provider.bindings)
        res["refused"] = sorted({f"{k[1]} {k[3]}: {w[:200]}" for k, w in self.provider.refused.items()})[:30]
        res["decline_sites"] = [f"{n}x {e} @ {w}" for (e, w), n in DECLINE_SITES.most_common(20)]
        res["baked_host_reads"] = dict(BAKED.most_common(30))
        res["static_slots"] = len(self.obj_slots)
        ents = {}
        for k, e in self.entries.items():
            pv = []
            for v in e.variants:
                lo = v.captured.lowered
                steps = list(getattr(lo, "steps", None) or [])
                calls = [getattr(st, "call", None) for st in steps if not isinstance(st, (int, range))]
                eager = collections.Counter(getattr(c, "name", None) or str(getattr(c, "target", "?")) for c in calls if type(c).__name__ == "EagerCall")
                pv.append({"learns": v.learns, "segments": len(v.captured.segments), "launches": len(v.captured.launches), "eager_ops": dict(eager.most_common(30))})
            ents[f"{k[0]}/{len(self.entries)}:{abs(hash(k)) % 10000}"] = {
                "traces": e.traces, "replays": e.replays, "eager": e.eager, "variants": len(e.variants), "per_variant": pv,
                "declines": sorted({str(d)[:240] for d in getattr(e, "declines", [])})[:12],
                "retrace_causes": {str(c): n for c, n in getattr(e, "retrace_causes", {}).items()},
                "redispatches": getattr(e, "redispatches", None)}
        res["entries"] = ents
        return res
