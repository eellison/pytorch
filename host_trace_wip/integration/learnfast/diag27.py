"""diag27.py SCRIPT [args]: runs SCRIPT (r27's drive_sweep27.py, read-only) with HostTraceReplay instrumentation for the learn:
per miss (trace) the call's token count, every failing guard of every variant of its family (graph guards with their owner op, or
the dispatch op and its redispatch refusal with the raise line and the redo's op kinds), and the time in trace / build /
redispatch / harvest; plus per slow call the time outside them. One JSON line per event in DIAG27_FILE."""
import json
import os
import runpy
import sys
import time
import traceback
import collections

import torch
import torch.cuda._host_trace_harvest as H
import torch.cuda._host_trace_lower_tape as LT
import torch.cuda._host_trace_replay as R

for _kv in filter(None, os.environ.get("DIAG27_SET", "").split(",")):
    _k, _v = _kv.split("=")
    setattr(torch.cuda._host_trace, _k, bool(int(_v)))
    print("diag27: torch.cuda._host_trace.%s = %s" % (_k, getattr(torch.cuda._host_trace, _k)), flush=True)
if os.environ.get("DIAG27_PORTED") == "1":
    # core agent 10-19: the adapters on the ms34 core redo API (snap/wip43), no learnfast redo prototypes (memset/coldstart/ported.py)
    sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/coldstart")
    import ported as _ported
    import atexit as _ax7
    _ax7.register(lambda: print("PORTED stats:", dict(_ported.STATS), "library:", dict(__import__("torch.cuda._host_trace_redispatch", fromlist=["_LIBRARY_COUNTS"])._LIBRARY_COUNTS), flush=True))
if os.environ.get("DIAG27_TRACE_PDL") == "0":
    torch.cuda._host_trace.trace_pdl = False
    print("diag27: torch.cuda._host_trace.trace_pdl = False", flush=True)
if os.environ.get("DIAG27_NO_PDL_EDGES") == "1":
    # core agent 10-18 (nsys hang probe): captured graphs drop programmatic (PDL) edges
    import torch.cuda._host_trace_capture as _HTC
    _HTC.keep_programmatic_edges = False
    print("diag27: _host_trace_capture.keep_programmatic_edges = False", flush=True)
if os.environ.get("DIAG27_SPLIT") in ("0", "1"):
    # the executor-level split pipeline (torch._C._host_trace_split, step_ms16+): on / off for the whole run
    print("diag27: split", os.environ["DIAG27_SPLIT"], "was", torch._C._host_trace_split(os.environ["DIAG27_SPLIT"] == "1"), flush=True)
if os.environ.get("DIAG27_GDN_MASKFILL") == "1":
    # proposed labelled overlay (learnfast/gdn_maskfill.py): Qwen GDN's prefill state reset as masked_fill_ (no mask-length pin)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gdn_maskfill as _gmf
    import atexit as _ax6
    _ax6.register(lambda: print("GDN_MASKFILL patched:", _gmf.PATCHED, flush=True))
if os.environ.get("DIAG27_FLA_SYMLEN") == "1":
    # the labelled fla overlay (learnfast/fla_symlen.py): .shape[0] for len() in fla's chunked-prefill ops
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fla_symlen as _fla
    import atexit as _ax4
    _ax4.register(lambda: print("FLA_SYMLEN patched modules:", len(_fla.PATCHED), flush=True))
if os.environ.get("DIAG27_GDN_EXPLICIT") == "1":
    # ms40 (10-22): the GDN fused core as a sibling op with explicit per-layer tensors (learnfast/gdn_explicit.py); imported after
    # gdn_maskfill so its hook is first and applies both rewrites
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gdn_explicit as _gex
    import atexit as _ax7
    _ax7.register(lambda: print("GDN_EXPLICIT patched:", _gex.PATCHED, flush=True))


if os.environ.get("DIAG27_REDOCTX") == "1" and os.environ.get("DIAG27_LIFT") != "1":
    # the prototype's redo runs inside the adapter's context: the adapters' interim refusals (no_lib_redo: no
    # redo_context; no_sections) are lifted for this run -- no_lib_redo sees a declared context, check_section is the
    # core's own again
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_lib_redo as _nlr
    import no_sections as _ns
    import torch.cuda._host_trace_replay as _hr0
    _nlr.REDO_CONTEXT = "learnfast REDOCTX prototype"
    _hr0.check_section = _ns._orig
if os.environ.get("DIAG27_SECTIONS_ON") == "1":
    # splicing fix (user 10-15): sections on, checked by the core's check_section (Delta 3 on a snap that has it); the adapters'
    # interim refusal (no_sections) lifted for this run
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_sections as _ns2
    import torch.cuda._host_trace_replay as _hr2
    _hr2.check_section = _ns2._orig
    print("diag27: sections on (core check_section)", flush=True)
if os.environ.get("DIAG27_PROV") == "1":
    # opt-in "redo only ops whose records depend on their own inputs" (prov_redo.py); PROV_CLOSURE follows DIAG27_CLOSURE
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("PROV_CLOSURE", os.environ.get("DIAG27_CLOSURE", "0"))
    import prov_redo as _prov
OUT = open(os.environ["DIAG27_FILE"], "a", buffering=1)
_t0 = time.perf_counter()
ACC = {"trace": 0.0, "build": 0.0, "redispatch": 0.0, "learn": 0.0, "slow": 0.0, "eager_sites": 0.0, "relower": 0.0, "redo_in_ctx": 0.0, "redo_body": 0.0, "resolver": 0.0}
CNT = {k: 0 for k in ACC}
STATE = {"T": None}


_QUIET = {"closed_over", "closed_over_storage", "redo_context", "check_row"} if os.environ.get("DIAG27_QUIET") == "1" else set()


def emit(**kw):
    # DIAG27_QUIET=1 (startup / timed runs): the per-redo diagnostic events are not written
    if kw.get("ev") in _QUIET:
        return
    kw["t"] = round(time.perf_counter() - _t0, 3)
    OUT.write(json.dumps(kw, default=str) + "\n")


def toks(args):
    return [int(x.shape[0]) for x in args if isinstance(x, torch.Tensor) and x.dim() == 1 and x.dtype in (torch.int32, torch.int64)][:6]


def wrap(cls, name, key):
    orig = getattr(cls, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            ACC[key] += time.perf_counter() - t
            CNT[key] += 1

    setattr(cls, name, w)


_orig_trace = R.trace


def _trace(fn, args, *a, **k):
    t = time.perf_counter()
    if os.environ.get("DIAG27_STATIC_SHAPES") == "1":
        # the replay's static prefix traced with static layouts (core's static_shapes: constants, guarded to their values)
        f = sys._getframe(1)
        while f is not None and not isinstance(f.f_locals.get("self"), R.HostTraceReplay):
            f = f.f_back
        n = getattr(f.f_locals["self"], "static_prefix", 0) if f is not None else 0
        if n:
            k["static_shapes"] = frozenset(k.get("static_shapes") or ()) | frozenset(range(n))
    try:
        tape = _orig_trace(fn, args, *a, **k)
        if os.environ.get("DIAG27_OPDEPS") == "1":
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import opdeps
                # the calling HostTraceReplay's static prefix (adapters test __init__'s signature: not wrapped)
                f = sys._getframe(1)
                while f is not None and not isinstance(f.f_locals.get("self"), R.HostTraceReplay):
                    f = f.f_back
                opdeps.STATIC["n"] = getattr(f.f_locals["self"], "static_prefix", 0) if f is not None else 0
                rows = opdeps.classify(tape)
                emit(ev="opdeps", toks=toks(args), n_ops=len(rows), summary=opdeps.summarize(rows),
                     rows=[r for r in rows if r["selector"] or r["verdict"] != "args_only"][:400])
                nrows = opdeps.classify_nested(tape, _NESTED.pop(id(tape.shape_env), []))
                emit(ev="opdeps_nested", toks=toks(args), n=len(nrows), summary=opdeps.summarize(nrows))
            except Exception as e:
                emit(ev="opdeps_error", err=repr(e)[:300], tb=traceback.format_exc()[-1500:])
        return tape
    finally:
        dt = time.perf_counter() - t
        ACC["trace"] += dt
        CNT["trace"] += 1
        emit(ev="trace", s=round(dt, 2), toks=toks(args))


R.trace = _trace

_NESTED: dict = {}
_OPD_STATIC = {"n": 0}
if os.environ.get("DIAG27_OPDEPS") == "1":
    # nested _Trace.op calls (inline in their top-level op): spans for opdeps.classify_nested
    import torch.cuda._host_trace_tape as _TT0

    _orig_tr_op = _TT0._Trace.op

    def _tr_op(self, func, args, kwargs, run, host=False, redo=None):
        if not self.in_op:
            return _orig_tr_op(self, func, args, kwargs, run, host, redo)
        env = self.shape_env
        start = (len(self.launches), len(self.allocs), len(env.owners))
        out = _orig_tr_op(self, func, args, kwargs, run, host, redo)
        _NESTED.setdefault(id(env), []).append((func, (args, kwargs), range(start[0], len(self.launches)), range(start[1], len(self.allocs)), range(start[2], len(env.owners))))
        return out

    _TT0._Trace.op = _tr_op
wrap(R.HostTraceReplay, "_build", "build")
wrap(R.HostTraceReplay, "_redispatch", "redispatch")
wrap(R.HostTraceReplay, "_eager_sites", "eager_sites")
wrap(R.HostTraceReplay, "_relower", "relower")
wrap(H.HarvestProvider, "learn", "learn")
wrap(R.HostTraceReplay, "_call_slow", "slow")

# where a FoldRefused is raised, and for redispatch's "dispatches otherwise" the redo's op count and kinds
_orig_fr = LT.FoldRefused.__init__


def _fr(self, msg, *a, **k):
    _orig_fr(self, msg, *a, **k)
    st = traceback.extract_stack(limit=3)[0]
    self._where = f"{os.path.basename(st.filename)}:{st.lineno}"
    try:
        f = sys._getframe(1)
        tr, op = f.f_locals.get("tr"), f.f_locals.get("op")
        if tr is not None and op is not None and "dispatches otherwise" in str(msg):
            self._detail = {
                "redo_ops": len(tr.ops),
                "redo_kinds": [o.kind for o in tr.ops][:8],
                "redo_funcs": [str(o.func) for o in tr.ops][:8],
                "op_kind": op.kind,
                "eager_outputs": bool(getattr(tr, "eager_outputs", None)),
                "ir": f.f_locals.get("ir"),
                "eager": f.f_locals.get("eager"),
                "redone": len(f.f_locals["redone"]) if "redone" in f.f_locals else None,
                "chain": len(f.f_locals["chain"]) if "chain" in f.f_locals else None,
                "tr_allocs": len(f.f_locals["tr_allocs"]) if "tr_allocs" in f.f_locals else None,
                "olds": len(f.f_locals["olds"]) if "olds" in f.f_locals else None,
                "redo_eager_reasons": [str(getattr(r, "reason", None))[:600] for _, r in tr.launches if type(r).__name__ == "EagerCall"][:4],
                "redo_declined": str(getattr(tr, "declined", None))[:400],
            }
    except Exception as e:  # diagnostics only
        self._detail = {"err": repr(e)}


LT.FoldRefused.__init__ = _fr

_orig_rd = R.HostTraceReplay._redispatch


def _unselected_ops(variant, unselected):
    # every failing selector at the call, by its op (retrace_causes shows the first only)
    try:
        low = variant.captured.lowered
        c = collections.Counter(str(variant.tape.ops[low.selectors[i - len(low.sites)].op].func) for i in unselected)
        return dict(c)
    except Exception as e:
        return {"error": repr(e)[:120]}


def _rd(self, variant, args, values, unselected, dispatches=None):
    r = _orig_rd(self, variant, args, values, unselected, dispatches)
    if r is not None:
        op = variant.tape.ops[r.op].func if r.op is not None else None
        emit(ev="redispatch_refused", toks=toks(args), op=str(op), why=str(r)[:400], where=getattr(r, "_where", None), detail=getattr(r, "_detail", None), unselected=len(unselected),
             unselected_ops=_unselected_ops(variant, unselected))
    else:
        emit(ev="redispatch_ok", toks=toks(args), unselected=len(unselected), unselected_ops=_unselected_ops(variant, unselected))
    return r


R.HostTraceReplay._redispatch = _rd


def causes(self, contract, args):
    """every variant of the family: failing graph guards (owner op) or the dispatch refusal"""
    from torch.cuda._host_trace_program import Status

    out = []
    for vi, variant in enumerate(self._families.get(contract, ())):
        low, tape = variant.captured.lowered, variant.tape
        env = tape.shape_env
        result = low.compiled.evaluate_inputs(tuple(args))
        if result is None or result[0] != Status.SUCCESS:
            out.append({"v": vi, "class": "other"})
            continue
        values = result[1]
        if values[low.valid] == 1:
            op, why = self._last_refusal.get(id(variant.native), (None, None))
            out.append({"v": vi, "class": "dispatch", "op": None if op is None else str(tape.ops[op].func), "why": (why or "")[:200]})
            continue
        fails = []
        for g, row in zip(tape.graph, low.lowering.lowering.guard_rows[: len(tape.graph)]):
            if values[row] != 1:
                text = R._ir.render(env.records[g][0]) if low.lowering.ir else str(env.guards[g].expr)
                owners = env.owner_sets[g] if env.owner_sets is not None else {env.owners[g]}
                ops = sorted({str(tape.ops[o].func) for o in owners if o is not None})
                fails.append((text[:160], ops[:3]))
        out.append({"v": vi, "class": "graph", "n_fail": len(fails), "fails": fails[:6], "learns": variant.learns})
    return out


_orig_miss = R.HostTraceReplay._miss


def _miss(self, contract, args, *a, **k):
    t, tr = time.perf_counter(), self.traces
    try:
        c = causes(self, contract, args) if self.traces else []
    except Exception as e:
        c = [{"err": repr(e)}]
    try:
        return _orig_miss(self, contract, args, *a, **k)
    finally:
        emit(ev="miss", toks=toks(args), traces_added=self.traces - tr, s=round(time.perf_counter() - t, 2), variants=len(self.variants), causes=c,
             acc={k: round(v, 2) for k, v in ACC.items()}, cnt=dict(CNT))


R.HostTraceReplay._miss = _miss

import torch.cuda._host_trace_tape as TT  # noqa: E402

_orig_th = TT._Trace._traced_host


def _th(self, func, args, kwargs, kernel, *a, **k):
    if getattr(self, "resolve", None) is None:
        return _orig_th(self, func, args, kwargs, kernel, *a, **k)

    def logged():
        try:
            return kernel()
        except Exception as e:
            if type(e).__name__ != "Declined":
                emit(ev="redo_body_error", func=str(func), err=repr(e)[:300], tb=traceback.format_exc()[-6000:])
            raise

    return _orig_th(self, func, args, kwargs, logged, *a, **k)


TT._Trace._traced_host = _th

# FNRESOLVE prototype: a redo resolves the plain tensors its body reaches (RD5's rule: same StorageImpl and view as exactly
# one call argument) at the torch-function layer, before autograd's ADInplaceOrView sees the real (non-inference) tensor
if os.environ.get("DIAG27_FNRESOLVE") == "1":
    from torch.overrides import TorchFunctionMode
    import torch.utils._pytree as _pt

    class _ResolveFn(TorchFunctionMode):
        def __init__(self, resolve):
            super().__init__()
            self.resolve = resolve

        def __torch_function__(self, func, types, args=(), kwargs=None):
            args, kwargs = _resolve_shallow(self.resolve, args, kwargs or {})
            return func(*args, **kwargs)

    _LEAF_TYPES = (int, float, bool, str, type(None), torch.dtype, torch.device, torch.SymInt, torch.SymFloat, torch.SymBool, torch.memory_format, torch.layout)

    def _resolve_seq(resolve, seq):
        # tensors at the top level or one list / tuple deep resolved directly; deeper containers through pytree
        out = None
        for i, a in enumerate(seq):
            if isinstance(a, torch.Tensor):
                r = resolve(a)
            elif type(a) in (list, tuple):
                if not any(isinstance(y, (torch.Tensor, list, tuple, dict)) for y in a):
                    continue
                if any(isinstance(y, (list, tuple, dict)) for y in a):
                    r = _pt.tree_map_only(torch.Tensor, resolve, a)
                else:
                    r = type(a)(resolve(y) if isinstance(y, torch.Tensor) else y for y in a)
            elif isinstance(a, dict) or (type(a) not in _LEAF_TYPES and not _pt._is_leaf(a)):
                r = _pt.tree_map_only(torch.Tensor, resolve, a)
            else:
                continue
            if r is not a:
                out = list(seq) if out is None else out
                out[i] = r
        return seq if out is None else type(seq)(out) if type(seq) in (list, tuple) else tuple(out)

    def _resolve_shallow(resolve, args, kwargs):
        args = _resolve_seq(resolve, args)
        if kwargs:
            vals = _resolve_seq(resolve, tuple(kwargs.values()))
            kwargs = dict(zip(kwargs.keys(), vals))
        return args, kwargs

    _orig_ec = TT._Trace.eager_call

    def _ec(self, func, args, kwargs):
        if getattr(self, "resolve", None) is None or getattr(self, "_fnresolve_on", False):
            return _orig_ec(self, func, args, kwargs)
        self._fnresolve_on = True
        try:
            with _ResolveFn(self.resolve):
                return _orig_ec(self, func, args, kwargs)
        finally:
            self._fnresolve_on = False

    TT._Trace.eager_call = _ec

    # the redo's resolver at the Triton launch intercept too (a kernel argument read straight from module state)
    import torch.cuda._host_trace_triton_launch as TL

    _orig_tl = TL._intercept

    import torch.cuda._host_trace_redispatch as RD

    _orig_resolver = RD._resolver

    _STATIC = {"n": 0}
    _orig_rd2 = R.HostTraceReplay._redispatch

    def _rd_static(self, *a, **k):
        _STATIC["n"] = getattr(self, "static_prefix", 0) or 0
        return _orig_rd2(self, *a, **k)

    R.HostTraceReplay._redispatch = _rd_static

    import contextlib as _ctxlib

    _ENTRY_FN = {"fn": None}
    _orig_rd3 = R.HostTraceReplay._redispatch

    def _rd_fn(self, *a, **k):
        _ENTRY_FN["fn"] = self.fn
        return _orig_rd3(self, *a, **k)

    R.HostTraceReplay._redispatch = _rd_fn
    REDOCTX_STATS = {"contexts": 0, "errors": 0}

    def _adapter_redo_context(step, flat):
        """REDOCTX prototype: the armV adapter's step-function state swap (its _step_hybrid / _step: forward context
        rebuilt from the step's metadata tensors and ints, attribute swap, parameter reparametrization), without
        running the model, on the redo's stand-ins for every argument of the step"""
        cl = {n: c.cell_contents for n, c in zip(step.__code__.co_freevars, step.__closure__ or ())}
        g = step.__globals__
        proxy, n_p, n_t = cl["proxy"], cl["n_p"], cl["n_t"]
        n_a, n_vp, smap = cl.get("n_a", len(proxy.attrs)), cl.get("n_vp", 0), cl.get("smap", ())
        fc, groups, sig, extra, esig = proxy._ctx
        params = flat[:n_p]
        attrs = flat[n_p:n_p + n_a]
        tensors = flat[n_p + n_a + 2:n_p + n_a + 2 + n_t]
        end = len(flat) - n_vp
        ints = iter(flat[n_p + n_a + 2 + n_t:end])
        amd = {}
        for (md, names), (_, msig) in zip(groups, sig):
            amd.update(dict.fromkeys(names, g["_md_rebuild"](md, msig, tensors, ints)))
        import copy as _copy
        from vllm.forward_context import override_forward_context
        sfc = _copy.copy(fc)
        sfc.attn_metadata = amd
        sfc.slot_mapping = {name: tensors[i] for name, i in smap}

        @_ctxlib.contextmanager
        def ctx():
            saved = proxy._swapped(attrs)
            try:
                with override_forward_context(sfc), g["_reparametrize_module"](proxy.model, {n: params[s] for n, s in zip(proxy.names, proxy.slots)}, tie_weights=False):
                    yield
            finally:
                for m, k, v in saved:
                    g["_set"](m, k, v)

        return ctx()

    _MOE_FUNCS = ("vllm.moe_forward.default", "vllm.moe_forward_shared.default")
    _IN_REDO = {"op": None, "logged": 0}
    if os.environ.get("DIAG27_LIFT") == "1":
        # diagnostic: which MoE layer a redo's body finds (vLLM's get_layer_from_name), first few redos only
        try:
            import vllm.model_executor.layers.fused_moe.runner.moe_runner as _mr

            _orig_glfn = _mr.get_layer_from_name

            def _glfn(layer_name):
                fc = None
                try:
                    from vllm.forward_context import get_forward_context
                    fc = get_forward_context()
                except Exception:
                    pass
                idx = getattr(fc, "moe_layer_index", None)
                layer = _orig_glfn(layer_name)
                if _IN_REDO["op"] is not None and _IN_REDO["logged"] < 6:
                    _IN_REDO["logged"] += 1
                    emit(ev="moe_layer_in_redo", layer_name=str(layer_name)[:80], counter=idx, resolved=getattr(layer, "layer_name", None))
                return layer

            _mr.get_layer_from_name = _glfn
        except Exception as e:
            print("diag27: moe layer hook not installed:", repr(e)[:200], flush=True)

    def _moe_index(old, op):
        """vLLM's MoE ops find their layer through forward_context.moe_layer_index, a counter each MoE call increments
        (layer_name "from_forward_context"): per call site a constant, the op's rank among the step's MoE calls. A redo
        of one MoE op holds that rank at the counter (its holder), as the trace's call saw it"""
        if str(op.func) not in _MOE_FUNCS:
            return None
        k = 0
        for o in old.ops:
            if o is op:
                return k
            if str(o.func) in _MOE_FUNCS:
                k += 1
        return None

    def _lifted_context(step, flat, moe_index=None):
        """LIFT: the adapter's forward context rebuilt from `flat`'s lifted positions (metadata tensors, lifted ints) and
        slot mapping, as its step function installs it; nothing else of the step"""
        cl = {n: c.cell_contents for n, c in zip(step.__code__.co_freevars, step.__closure__ or ())}
        g = step.__globals__
        proxy, n_p, n_t = cl["proxy"], cl["n_p"], cl["n_t"]
        n_a, n_vp, smap = cl.get("n_a", len(proxy.attrs)), cl.get("n_vp", 0), cl.get("smap", ())
        fc, groups, sig, extra, esig = proxy._ctx
        tensors = flat[n_p + n_a + 2:n_p + n_a + 2 + n_t]
        ints = iter(flat[n_p + n_a + 2 + n_t:len(flat) - n_vp])
        amd = {}
        for (md, names), (_, msig) in zip(groups, sig):
            amd.update(dict.fromkeys(names, g["_md_rebuild"](md, msig, tensors, ints)))
        import copy as _copy
        from vllm.forward_context import override_forward_context
        sfc = _copy.copy(fc)
        sfc.attn_metadata = amd
        sfc.slot_mapping = {name: tensors[i] for name, i in smap}
        if moe_index is not None and hasattr(sfc, "moe_layer_index"):
            sfc.moe_layer_index = moe_index
        return override_forward_context(sfc)

    # diagnostic: which ops a redispatch can template (core _reads keys) and how often an instance is instantiated
    TPL = collections.Counter()
    _orig_reads = RD._reads

    def _reads_diag(lowered, op, selector, *rest):  # rest: step_ms10's declared symbols
        r = _orig_reads(lowered, op, selector, *rest)
        TPL[(str(op.func)[:60], "key" if r is not None else "none")] += 1
        return r

    RD._reads = _reads_diag
    _orig_inst = RD._instantiate

    def _inst_diag(lowered, i, t, roots, base, selector, op):
        TPL[(str(op.func)[:60], "instantiated")] += 1
        return _orig_inst(lowered, i, t, roots, base, selector, op)

    RD._instantiate = _inst_diag
    _orig_tpl = RD._template

    def _tpl_diag(lowered, entry, guards, launches, redone, roots, op):
        t = _orig_tpl(lowered, entry, guards, launches, redone, roots, op)
        TPL[(str(op.func)[:60], "template" if t is not None else "no_template")] += 1
        if t is not None:
            TPL[(str(op.func)[:60], f"pinned={len(t.pinned)}:" + ",".join(sorted({r.kind for _, r in t.pinned})))] += 1
            if t.pinned and TPL[("pin_dump", "n")] < 2:
                TPL[("pin_dump", "n")] += 1
                names = set()
                for _, r in t.pinned:
                    k = TT._sym_key(r.sym)
                    names |= k.free_symbols if hasattr(k, "free_symbols") else set()
                gs = [repr(g)[:120] for g, _ in guards if hasattr(g, "free_symbols") and g.free_symbols & names]
                emit(ev="pin_dump", func=str(op.func)[:60], pinned=[r.name for _, r in t.pinned], guards=gs[:8], n_guards=len(gs))
        return t

    RD._template = _tpl_diag
    import atexit as _ax5
    _ax5.register(lambda: emit(ev="templates", counts={f"{a}|{b}": c for (a, b), c in TPL.items()}))

    _OP_INDEX: dict = {}

    def _needs_binding(old):
        """LIFT: only an op whose records read lifted per-step state (prov_redo's args_or_lifted) gets the holder binding
        and its stand-ins; every other redo (args_only, declared, ATen) runs as the core's"""
        if os.environ.get("DIAG27_LIFT") != "1" or os.environ.get("DIAG27_PROV") != "1":
            return True
        f = sys._getframe(2)
        while f is not None and "op" not in f.f_locals:
            f = f.f_back
        op = None if f is None else f.f_locals.get("op")
        if op is None:
            return True
        ent = _OP_INDEX.get(id(old))
        if ent is None or ent[0]() is not old:
            import weakref
            ent = _OP_INDEX[id(old)] = (weakref.ref(old), {id(o): k for k, o in enumerate(old.ops)})
        k = ent[1].get(id(op))
        try:
            rows, _ = _prov._rows(old)
        except Exception:
            return True
        return k is None or rows[k]["verdict"] == "args_or_lifted"

    def _resolver(old, lo, tr, root, made, at_call, inputs, resolved):
        tr._lf_ctx = (old, lo)
        if os.environ.get("DIAG27_REDOCTX") == "1" and _ENTRY_FN["fn"] is not None and _needs_binding(old):
            # stand-ins for every argument of the step, as the redo builds its op's own call: a tensor argument its
            # root at its input record's layout (fresh symbols bound to the variant's), an int argument a fresh SymInt
            # bound to the variant's int input, anything else the traced value; then the op's redo runs inside the
            # adapter's state swap on them
            cells = {n: c.cell_contents for n, c in zip(root.__code__.co_freevars, root.__closure__ or ())}
            value = cells["value"]
            frame = sys._getframe(1)
            while frame is not None and "op" not in frame.f_locals:  # _redo's frame (wrappers in between)
                frame = frame.f_back
            op = frame.f_locals.get("op")
            try:
                n0 = len(made)
                flat = list(old.args)
                lazy = os.environ.get("DIAG27_REDOCTX_LAZY") == "1"
                n_static = _STATIC["n"] if lazy else 0
                for pos in range(n_static):
                    # LAZY: the static prefix (module parameters, declared attributes: the same objects every call) is
                    # passed as the variant's own objects; a read of one by the redo resolves to its stand-in through
                    # RD5's identity-to-argument rule (returned below), so only the tensors the op reads get one
                    flat[pos] = lo.program.inputs[pos]
                with lo.symbolic.total():
                    for rec in old.inputs:
                        if rec.position < n_static:
                            continue
                        r = root(rec.root)
                        rec2 = next(i for i in inputs if i.root is r)
                        tr.arguments[id(r)] = rec2
                        x = old.args[rec.position]
                        dtype = x.dtype if isinstance(x, torch.Tensor) else x[3]
                        flat[rec.position] = TT._TracedTensor(r, list(rec2.sizes), list(rec2.strides), rec2.offset, dtype, old.device)
                    for irec in old.int_inputs:
                        if irec.position >= n_static:
                            flat[irec.position] = value(irec.sym)
                for m, x in zip(made[n0:], at_call[n0:]):
                    tr.values.bind(m, x)
                step = _ENTRY_FN["fn"]
                proxy = {n: c.cell_contents for n, c in zip(step.__code__.co_freevars, step.__closure__ or ())}["proxy"]
                stand_ins = tuple(flat)
                orig_redo = op.redo

                class _RedoDone(Exception):
                    pass

                def redo_in_ctx(*a, **k):
                    # the adapter's own step function on the stand-ins, its model forward replaced by the op's redo:
                    # every state swap the step makes before the forward (forward context, attributes, parameters)
                    # is live while the redo runs, and undone by the step's own finally / with blocks
                    op.redo = orig_redo  # this redo's stand-ins only: the OpRec keeps its own redo afterwards
                    held, orig_fwd = {}, proxy.orig_forward

                    def forward_is_the_redo(*_x, **_kw):
                        t1 = time.perf_counter()
                        try:
                            held["r"] = orig_redo(*a, **k)
                        finally:
                            ACC["redo_body"] += time.perf_counter() - t1
                            CNT["redo_body"] += 1
                        raise _RedoDone

                    proxy.orig_forward = forward_is_the_redo
                    t0 = time.perf_counter()
                    try:
                        step(*stand_ins)
                    except _RedoDone:
                        pass
                    finally:
                        proxy.orig_forward = orig_fwd
                        ACC["redo_in_ctx"] += time.perf_counter() - t0
                        CNT["redo_in_ctx"] += 1
                    if "r" not in held:
                        raise RuntimeError("the step function returned without calling its forward")
                    return held["r"]

                if os.environ.get("DIAG27_LIFT") == "1":
                    # LIFT (the user's design, 10-15): the holders of the lifted per-step state (the forward context's
                    # attention / GDN metadata, rebuilt by the adapter's _md_rebuild from the step's lifted tensors and
                    # ints, and its slot mapping) hold this redo's stand-ins while the op's redo runs: no step call, no
                    # parameter / attribute swap (closure resolves declared storages)
                    def redo_lifted(*a, **k):
                        op.redo = orig_redo
                        _IN_REDO["op"] = str(op.func)
                        t0 = time.perf_counter()
                        try:
                            with _lifted_context(step, stand_ins, _moe_index(old, op)):
                                t1 = time.perf_counter()
                                try:
                                    return orig_redo(*a, **k)
                                finally:
                                    ACC["redo_body"] += time.perf_counter() - t1
                                    CNT["redo_body"] += 1
                        finally:
                            _IN_REDO["op"] = None
                            ACC["redo_in_ctx"] += time.perf_counter() - t0
                            CNT["redo_in_ctx"] += 1

                    op.redo = redo_lifted
                else:
                    op.redo = redo_in_ctx
                REDOCTX_STATS["contexts"] += 1
                emit(ev="redo_context", op=str(op.func), n=len(flat)) if REDOCTX_STATS["contexts"] <= 5 else None
            except Exception as e:
                REDOCTX_STATS["errors"] += 1
                emit(ev="redo_context_error", err=repr(e)[:400], tb=traceback.format_exc()[-2500:])
            if os.environ.get("DIAG27_REDOCTX_LAZY") != "1":
                return lambda t: t
            # LAZY: the static prefix reached by the redo resolves through the branches below (CLOSURE: declared
            # storages; else RD5 / SAMEROOT identity-to-argument)
        if os.environ.get("DIAG27_CLOSURE") == "1":
            # CLOSURE prototype: the declared tensors (the entry's static prefix: module parameters, buffers, KV / mamba
            # state, which the adapter declares fixed) are closed over by object: a read of one is that tensor's own
            # (constant) address, whatever view or storage sharing; anything else is left plain and declines
            n = _STATIC["n"]
            decl = {}
            # several declared arguments of one object / storage (an attribute aliasing a parameter): the one root the
            # op's own recorded launches read, if exactly one; else the root they all share; else none (it declines)
            fr = sys._getframe(1)
            while fr is not None and "op" not in fr.f_locals:
                fr = fr.f_back
            op_roots = set()
            if fr is not None and fr.f_locals.get("op") is not None:
                opx = fr.f_locals["op"]
                for _, recx in old.launches[opx.launches.start : opx.launches.stop]:
                    op_roots.update(id(r) for r in (getattr(recx, "roots", ()) or ()))

            def pick(recs):
                roots_ = {id(r.root): r for r in recs}
                if len(roots_) == 1:
                    return recs[0]
                mine = [r for k, r in roots_.items() if k in op_roots]
                return mine[0] if len(mine) == 1 else None

            decl_l, stor_l = {}, {}
            for rec in old.inputs:
                if rec.position < n:
                    x = lo.program.inputs[rec.position]
                    if isinstance(x, torch.Tensor):
                        decl_l.setdefault(id(x), []).append(rec)
                        stor_l.setdefault(x.untyped_storage()._cdata, []).append(rec)
            decl = {k: r for k, v in decl_l.items() if (r := pick(v)) is not None}
            by_storage = {k: r for k, v in stor_l.items() if (r := pick(v)) is not None}
            storage_mode = os.environ.get("DIAG27_CLOSURE_STORAGE") == "1"

            def resolve_closed(t):
                if isinstance(t, TT._TracedTensor) or t.device.type != "cuda":
                    return t
                if id(t) in resolved:
                    return resolved[id(t)]
                rec0 = decl.get(id(t))
                if rec0 is None and os.environ.get("DIAG27_CLOSURE_OBJECTS") == "1" and id(t) in DECLARED:
                    view = (t.dtype, t.shape, t.stride(), t.storage_offset())
                    for rec in old.inputs:
                        x = lo.program.inputs[rec.position] if rec.position < n else None
                        if isinstance(x, torch.Tensor) and x.untyped_storage()._cdata == t.untyped_storage()._cdata and (x.dtype, x.shape, x.stride(), x.storage_offset()) == view:
                            rec0 = rec
                            break
                    if rec0 is None:
                        emit(ev="declared_no_prefix_twin", shape=list(t.shape))
                        return t
                if rec0 is None and storage_mode:
                    # a view of a declared (fixed) storage, any object: its address is that storage's base plus its own
                    # constant offset; its layout is its own constants
                    srec = by_storage.get(t.untyped_storage()._cdata)
                    if srec is None:
                        return t
                    k0 = len(made)
                    with lo.symbolic.total():
                        r = root(srec.root)
                    for m, x in zip(made[k0:], at_call[k0:]):
                        tr.values.bind(m, x)
                    rec = next(i for i in inputs if i.root is r)
                    tr.arguments[id(r)] = rec
                    resolved[id(t)] = TT._TracedTensor(r, list(t.shape), list(t.stride()), t.storage_offset(), t.dtype, t.device)
                    emit(ev="closed_over_storage", pos=srec.position, shape=list(t.shape)) if len(resolved) <= 6 else None
                    return resolved[id(t)]
                if rec0 is None:
                    return t
                k0 = len(made)
                with lo.symbolic.total():
                    r = root(rec0.root)
                    # this argument's own layout: several declared arguments can share one storage (root) at different
                    # offsets, and the redo's input record for the root is the first such argument's (_redo's root())
                    vfn = {n: c.cell_contents for n, c in zip(root.__code__.co_freevars, root.__closure__ or ())}["value"]
                    lay = [vfn(n) for n in rec0.sizes], [vfn(n) for n in rec0.strides], vfn(rec0.offset)
                for m, x in zip(made[k0:], at_call[k0:]):
                    tr.values.bind(m, x)
                rec = next(i for i in inputs if i.root is r)
                tr.arguments.setdefault(id(r), rec)
                resolved[id(t)] = TT._TracedTensor(r, list(lay[0]), list(lay[1]), lay[2], t.dtype, t.device)
                emit(ev="closed_over", pos=rec0.position) if len(resolved) <= 3 else None
                return resolved[id(t)]

            return resolve_closed
        if os.environ.get("DIAG27_RESOLVE_SAMEROOT") != "1":
            return _orig_resolver(old, lo, tr, root, made, at_call, inputs, resolved)
        # RD5's rule, with several candidate arguments of one StorageImpl and view taken when they are one root at one
        # layout in the variant (aliased arguments the trace recorded as one storage): any of them is the same stand-in
        by_storage = {}
        for rec in old.inputs:
            x = lo.program.inputs[rec.position]
            if isinstance(x, torch.Tensor):
                by_storage.setdefault(x.untyped_storage()._cdata, []).append((rec, x))

        def resolve(t):
            if isinstance(t, TT._TracedTensor) or t.device.type != "cuda":
                return t
            if id(t) in resolved:
                return resolved[id(t)]
            view = (t.dtype, t.shape, t.stride(), t.storage_offset())
            found = [rec for rec, x in by_storage.get(t.untyped_storage()._cdata, ()) if (x.dtype, x.shape, x.stride(), x.storage_offset()) == view]
            if not found:
                return t
            if len(found) > 1:
                ident = [rec for rec, x in by_storage.get(t.untyped_storage()._cdata, ()) if x is t]
                if len(ident) == 1:
                    emit(ev="ambiguous_by_identity", n=len(found), root=ident[0].root.name)
                    found = ident
            if len(found) > 1:
                same = len({id(r.root) for r in found}) == 1 and len({(tuple(map(str, r.sizes)), tuple(map(str, r.strides)), str(r.offset)) for r in found}) == 1
                emit(ev="ambiguous", n=len(found), same_root_layout=same, roots=sorted({r.root.name for r in found})[:4])
                if not same:
                    return t
            n = len(made)
            with lo.symbolic.total():
                r = root(found[0].root)
            for m, x in zip(made[n:], at_call[n:]):
                tr.values.bind(m, x)
            rec = next(i for i in inputs if i.root is r)
            tr.arguments[id(r)] = rec
            resolved[id(t)] = TT._TracedTensor(r, list(rec.sizes), list(rec.strides), rec.offset, t.dtype, t.device)
            return resolved[id(t)]

        return resolve

    def _resolver_timed(*a, **k):
        t = time.perf_counter()
        try:
            return _resolver(*a, **k)
        finally:
            ACC["resolver"] += time.perf_counter() - t
            CNT["resolver"] += 1

    RD._resolver = _resolver_timed
    _logged = set()

    def _tl(tr, jit, args, grid, warmup, kwargs):
        if getattr(tr, "resolve", None) is not None:
            args, kwargs = _pt.tree_map_only(torch.Tensor, tr.resolve, (args, kwargs))
            for name, t in list(zip(jit.arg_names, args)) + list(kwargs.items()):
                if isinstance(t, torch.Tensor) and type(t) is not TT._TracedTensor and (jit.fn.__name__, name) not in _logged:
                    _logged.add((jit.fn.__name__, name))
                    old, lo = tr._lf_ctx
                    same = []
                    for rec in old.inputs:
                        x = lo.program.inputs[rec.position]
                        if isinstance(x, torch.Tensor) and x.untyped_storage()._cdata == t.untyped_storage()._cdata:
                            same.append((rec.position, str(x.dtype), tuple(x.shape), x.stride(), x.storage_offset()))
                    emit(ev="unresolved", kernel=jit.fn.__name__, arg=name, view=(str(t.dtype), tuple(t.shape), t.stride(), t.storage_offset(), type(t).__name__, t.untyped_storage().nbytes()), same_storage_args=same[:8], n_same=len(same))
        return _orig_tl(tr, jit, args, grid, warmup, kwargs)

    TL._intercept = _tl

    # and at record_launch: a pointer the launch's grid callback put in META (causal_conv1d's batch_ptr from metadata)
    _orig_rl = TL.record_launch

    def _rl(tr, name, function, owner, abi, values, dims, launch_with):
        if getattr(tr, "resolve", None) is not None:
            values = {k: (tr.resolve(v) if isinstance(v, torch.Tensor) else v) for k, v in values.items()}
        return _orig_rl(tr, name, function, owner, abi, values, dims, launch_with)

    TL.record_launch = _rl

    # CuTe's pointee lookup also over a redo's outer roots (the variant's allocations and eager outputs the op reads)
    import torch.cuda._host_trace_cute as CU

    if hasattr(CU, "_RootIndex"):
        # a head with the root index (step333+): keep it, add a redo's outer roots (rule 3) after its finds
        _orig_ro = CU._roots_of

        def _roots_of(tr, v):
            found = _orig_ro(tr, v)
            outer = getattr(tr, "outer", None)
            if outer:
                key = TT._sym_key(v)
                held = set() if isinstance(key, int) else key.free_symbols
                have = {id(r) for r in found}
                found = list(found) + [t._root for t in outer.values() if id(t._root) not in have and TT._sym_key(t._root.sym).free_symbols & held]
            return found
    else:
        def _roots_of(tr, v):
            key = TT._sym_key(v)
            held = set() if isinstance(key, int) else key.free_symbols
            roots = (*(i.root for i in tr.inputs), *(a.root for a in tr.allocs), *(t._root for t in tr.eager_outputs), *(t._root for t in getattr(tr, "outer", {}).values()))
            seen, out = set(), []
            for r in roots:
                if id(r) not in seen and TT._sym_key(r.sym).free_symbols & held:
                    seen.add(id(r))
                    out.append(r)
            return out

    CU._roots_of = _roots_of

    if os.environ.get("DIAG27_CUTE_RESOLVE") == "1":
        # the redo's resolver at the CuTe launch intercept too (FA4's kv cache read from the layer: a declared storage)
        _orig_cu_int = CU._intercept

        def _cu_int(tr, compiled, args, kwargs, call, cute_args=()):
            if getattr(tr, "resolve", None) is not None:
                args, kwargs = _pt.tree_map_only(torch.Tensor, tr.resolve, (tuple(args), dict(kwargs)))
            return _orig_cu_int(tr, compiled, args, kwargs, call, cute_args)

        CU._intercept = _cu_int
    import torch.cuda._host_trace_memory as MEM

    _orig_cs = MEM.check_section

    def _cs(lowered, plan, nodes, launches, temporaries, drops):
        why = _orig_cs(lowered, plan, nodes, launches, temporaries, drops)
        if why is not None and "is not live" in why:
            try:
                b = int(why.split()[1])
                step = next((i for i, st in enumerate(lowered.steps) if isinstance(st, range) and nodes and nodes[0] in st), None)
                first, last = lowered.launches[nodes[0]].seq, lowered.launches[nodes[-1]].seq
                live = MEM._liveness(lowered, plan)
                a = lowered.allocations[b] if b < len(lowered.allocations) else None
                held = {k for k, *_ in plan.planned} | set(plan.relocated)
                made = [i for i, st in enumerate(plan.steps) if b in st.tensors]
                temps = [(i, sq, la) for i, st in enumerate(plan.steps) for k, sq, la in st.temporaries if k == b]
                drop = [i for i, st in enumerate(plan.steps) if b in st.drops]
                emit(ev="section_refused", why=why, base=b, n_alloc=len(lowered.allocations), alloc_seq=getattr(a, "seq", None), alloc_dtype=str(getattr(a, "dtype", None)),
                     step=step, first=first, last=last, live_first=live(b, step, first), live_last=live(b, step, last), in_drops=b in drops, held=b in held,
                     made_steps=made, temporaries=temps, drop_steps=drop, n_temps=len(temporaries), drops=sorted(drops)[:10], nodes=(nodes[0], nodes[-1], len(nodes)), n_launches=len(launches))
            except Exception as e:
                emit(ev="section_refused", why=why, err=repr(e))
        return why

    MEM.check_section = _cs

    # OPSPAN prototype: plan_memory keeps every allocation an op-guarded op's launches reach live through that op's last
    # launch, so a section (a redispatch whose launch chain differs) finds the bases it reaches live over the chain
    if os.environ.get("DIAG27_OPSPAN") == "1":
        import inspect, textwrap
        src = textwrap.dedent(inspect.getsource(MEM._orig_plan_memory if hasattr(MEM, "_orig_plan_memory") else MEM.plan_memory))
        anchor = "    # a keyed site's operands reach its nodes through its table\n"
        assert anchor in src
        ins = """    for sel in lowered.selectors:
        if sel.nodes:
            end = lowered.launches[sel.nodes[-1]].seq
            for j in sel.nodes:
                lo = lowered.launches[j]
                refs = [s.root for s in lo.slots if isinstance(s, PointerSlot)] + [lowered.lowering.roots[id(r)] for r in lo.launch.roots]
                for kind, k in refs:
                    if kind == "allocation" and k < n_alloc:
                        last_seq[k] = max(last_seq[k], end)
"""
        src = "from __future__ import annotations\n" + src.replace(anchor, ins + anchor)
        ns = {}
        exec(compile(src, MEM.__file__ + ":opspan", "exec"), MEM.__dict__, ns)
        MEM._orig_plan_memory = MEM.plan_memory
        MEM.plan_memory = ns["plan_memory"]

        _orig_build2 = R.HostTraceReplay._build

        def _build_mem(self, tape):
            v = _orig_build2(self, tape)
            try:
                low = v.captured.lowered
                nb = MEM._nbytes(low)
                out = {}
                for name, fn in (("base", MEM._orig_plan_memory), ("opspan", MEM.plan_memory)):
                    plan = fn(low, "run_buffer")
                    peak = MEM._peak(low, plan, nb, {})[0]
                    buf = sum(MEM.place(m.temporaries, [nb[k] for k, *_ in m.temporaries])[1] for m in plan.steps if m.temporaries)
                    out[name] = (peak, buf)
                emit(ev="opspan_mem", peak_base=out["base"][0], peak_opspan=out["opspan"][0], runbuf_base=out["base"][1], runbuf_opspan=out["opspan"][1],
                     allocs=len(low.allocations), toks=toks(tape.args) if getattr(tape, "args", None) is not None else None)
            except Exception as e:
                emit(ev="opspan_mem", err=repr(e)[:300])
            return v

        R.HostTraceReplay._build = _build_mem
        R.plan_memory = ns["plan_memory"]
        print("diag27: OPSPAN on", flush=True)
    R.check_section = _cs
    print("diag27: FNRESOLVE on (torch-function, Triton intercept, CuTe outer roots)", flush=True)

# each --check row emitted as it is made (a crash later in the run keeps the rows before it)
import importlib.abc, importlib.machinery  # noqa: E402


DECLARED: set = set()
_DECLARED_REFS: list = []
import gc  # noqa: E402


class _AdapterHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "adapter":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            cls = getattr(module, "ArmV", None)
            if cls is not None and hasattr(cls, "_check_hybrid"):
                _ch = cls._check_hybrid

                def _check_hybrid(self, *a, **k):
                    n = len(self.checks)
                    r = _ch(self, *a, **k)
                    for row in self.checks[n:]:
                        emit(ev="check_row", row={k2: v for k2, v in row.items() if not isinstance(v, (list, dict)) or k2 in ("stock_parts",)})
                    return r

                cls._check_hybrid = _check_hybrid
            if cls is not None and os.environ.get("DIAG27_CLOSURE_OBJECTS") == "1":
                _inst = cls.install

                def install(self, *a, **k):
                    r = _inst(self, *a, **k)
                    runner = getattr(self, "runner", None)
                    seen, ids, _held = set(), set(), []

                    def walk(o, depth):
                        if id(o) in seen or depth > 4:
                            return
                        seen.add(id(o))
                        if isinstance(o, torch.Tensor):
                            if o.device.type == "cuda":
                                ids.add(id(o))
                                _held.append(o)
                            return
                        if isinstance(o, torch.nn.Module):
                            for t in list(o.parameters(recurse=False)) + list(o.buffers(recurse=False)):
                                ids.add(id(t))
                                _held.append(t)
                            for v in vars(o).values():
                                if isinstance(v, (torch.Tensor, list, tuple, dict)):
                                    walk(v, depth + 1)
                            for m in o.children():
                                walk(m, depth)
                            return
                        if isinstance(o, (list, tuple)):
                            for v in o:
                                walk(v, depth + 1)
                        elif isinstance(o, dict):
                            for v in o.values():
                                walk(v, depth + 1)
                        elif depth < 3 and hasattr(o, "__dict__") and type(o).__module__.startswith("vllm"):
                            for v in vars(o).values():
                                walk(v, depth + 1)

                    if runner is not None:
                        walk(runner.model, 0)
                        walk(runner, 1)
                    DECLARED.update(ids)
                    # hold the declared objects: an id of a freed tensor could be reused by a per-step one
                    _DECLARED_REFS.extend(_held)
                    emit(ev="declared", n=len(ids))
                    return r

                cls.install = install

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _AdapterHook())

# SECTIONDIAG: each section entry's temporaries (formulas, size rows, values at creation), then their size rows
# evaluated at every later native call of that variant
if os.environ.get("DIAG27_SECTIONDIAG") == "1":
    import torch.cuda._host_trace_lower_tape as LT2
    import torch.cuda._host_trace_redispatch as RD

    _SECT: dict = {}
    _orig_le = LT2.lower_entry

    def _lower_entry(lowered, i, guards, launches, section, base):
        out = _orig_le(lowered, i, guards, launches, section, base)
        if section is not None:
            (site, predicate, rows, lsec), _ = out
            vals = lowered.lowering.program.values
            temps = []
            for t, (b, row) in zip(section.temporaries, lsec.temporaries):
                temps.append({"base": b, "row": row, "nbytes_at_A": vals[row] if row < len(vals) else None, "dtype": str(t.dtype),
                              "sizes": [str(x) for x in t.sizes][:6], "strides": [str(x) for x in t.strides][:6], "root": t.root.name})
            _SECT.setdefault(id(lowered), []).append((site, [(tt["base"], tt["row"]) for tt in temps]))
            emit(ev="section_made", site=site, drops=list(section.drops)[:10], n_temps=len(temps), temps=temps[:12], n_rows=len(rows))
        return out

    LT2.lower_entry = _lower_entry
    RD.lower_entry = _lower_entry

    _orig_cn = R.HostTraceReplay._call_native

    def _call_native(self, variant, args):
        low = variant.captured.lowered
        if id(low) in _SECT:
            try:
                res = variant.native.evaluate(tuple(args))
                if res is not None:
                    vals = res[0]
                    emit(ev="section_sizes_at_call", toks=toks(args), sizes=[(site, [(b, vals[row] if row < len(vals) else None) for b, row in tl]) for site, tl in _SECT[id(low)]][:8])
            except Exception as e:
                emit(ev="section_sizes_at_call", err=repr(e)[:200])
        return _orig_cn(self, variant, args)

    R.HostTraceReplay._call_native = _call_native

if os.environ.get("DIAG27_NO_SECTIONS") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_sections  # noqa: F401
    import atexit as _ax
    _ax.register(lambda: print("NO_SECTIONS refused", no_sections.REFUSED, flush=True))
if os.environ.get("DIAG27_NO_LIB_REDO") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_lib_redo  # noqa: F401
    import atexit as _ax2
    _ax2.register(lambda: print("NO_LIB_REDO refused", dict(no_lib_redo.REFUSED), flush=True))
import atexit  # noqa: E402

atexit.register(lambda: emit(ev="end", acc={k: round(v, 2) for k, v in ACC.items()}, cnt=dict(CNT)))
if os.environ.get("DIAG27_XVR") == "1":
    # PROTOTYPE (learnfast 10-17): cross-variant reuse of redo results (xvr.py; NONDEFAULTS_learnfast.md)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import xvr as _xvr
    atexit.register(lambda: emit(ev="xvr", stats=dict(_xvr.STATS), secs={k: round(v, 2) for k, v in _xvr.SECS.items()}))
if os.environ.get("DIAG27_TPL") == "1":
    # templates keyed by closure roots (learnfast/tpl_closure.py); equality: DIAG27_SET=...,verify_redo_templates=1
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch.cuda._host_trace_redispatch as _rd_t
    # a head with the template core patch (step_ms10+) takes tpl_layer (key-outs + confirmation on core's templates)
    if hasattr(_rd_t, "_for_instance"):
        import tpl_layer as _tpl
    else:
        import tpl_closure as _tpl
    atexit.register(lambda: emit(ev="tpl_closure", stats=dict(_tpl.STATS), distinct_keys={k: len(v) for k, v in _tpl.KEYS.items()}))
    if os.environ.get("XVR_KEYDIFF") == "1" and os.environ.get("DIAG27_XVR") == "1":
        _xvr.install_keydiff()
if os.environ.get("PROV_PIN_ATEN", "0") == "1":
    # pin rule (b) refinement (coordinator 10-15): each guard record (new or a dedup hit) is marked by where it was recorded:
    # inside a dispatched ATen op's C++ symbolic evaluation (torch._C._cuda_hostTraceRedispatch on the stack: a
    # CompositeExplicitAutograd body's host, e.g. a broadcast's size-1 check) or outside it (Python, incl. library code).
    # prov_redo exempts a failing pin only if every record of it was inside an ATen op of the redone op.
    import torch.cuda._host_trace_ir as _IRp
    GUARD_WHERE: dict = {}  # (id(env), record index) -> {"aten", "python"}
    _ATEN_DEPTH = [0]

    def _in_aten(orig):
        def run(*a, **k):
            _ATEN_DEPTH[0] += 1
            try:
                return orig(*a, **k)
            finally:
                _ATEN_DEPTH[0] -= 1
        return run

    # both entry points of an ATen op's C++ symbolic host (a composite's body; an op's CUDA host)
    for _n in ("_cuda_hostTraceRedispatch", "_cuda_hostTraceRedispatchOp"):
        if hasattr(torch._C, _n):
            setattr(torch._C, _n, _in_aten(getattr(torch._C, _n)))
    _env_cls = _IRp.Env
    _orig_record = _env_cls._record

    def _record_where(self, g, written):
        _orig_record(self, g, written)
        if g.op == "true":
            return
        w = written if written is not None else self.ctx.written.get(g.id)
        i = self._index.get(_IRp._record_key(g, w))
        if i is not None:
            GUARD_WHERE.setdefault((id(self), i), set()).add("aten" if _ATEN_DEPTH[0] else "python")

    _env_cls._record = _record_where
    sys.modules["diag27_live"] = sys.modules[__name__]
if os.environ.get("DIAG27_GUARD_STACKS") == "1":
    # diagnostic (coordinator 10-15: the GDN pin refusals): every guard's Python stack at its record, for prov_redo's pin message
    import traceback as _tb
    import torch.cuda._host_trace_ir as _IRg
    GUARD_STACKS: dict = {}
    _orig_gs = _IRg.guard_sloc

    def _guard_sloc_stack():
        env = sys._getframe(1).f_locals.get("self")
        if env is not None and hasattr(env, "slocs"):
            GUARD_STACKS[(id(env), len(env.slocs))] = [f"{os.path.basename(os.path.dirname(fs.filename))}/{os.path.basename(fs.filename)}:{fs.lineno} {fs.name}"
                                                       for fs in _tb.extract_stack(sys._getframe(1), limit=40)]
        return _orig_gs()

    _IRg.guard_sloc = _guard_sloc_stack
    sys.modules["diag27_live"] = sys.modules[__name__]
if os.environ.get("DIAG27_FAILING_GUARDS") == "1":
    # diagnostic (coordinator 10-15, warm-up coverage): at each redispatch attempt, the failing own guards of the first unselected
    # selector per op (the dispatch classes a token count leaves), as text with their values at the call
    import torch.cuda._host_trace_replay as _hrf
    _inner_fg, _FG = _hrf.HostTraceReplay._redispatch, collections.Counter()

    def _rd_fg(self, variant, args, values, unselected, *a, **k):
        try:
            low = variant.captured.lowered
            env, seen = low.tape.shape_env, set()
            for i in unselected:
                sel = low.selectors[i - len(low.sites)]
                op = low.tape.ops[sel.op]
                fn = str(op.func)[:60]
                if fn in seen or _FG[fn] >= int(os.environ.get("DIAG27_FAILING_GUARDS_N", "40")) or len(sel.rows) != len(op.guards):
                    continue
                seen.add(fn)
                _FG[fn] += 1
                fails = [repr(env.records[g][0])[:100] for g, row in zip(op.guards, sel.rows) if row is not None and row < len(values) and values[row] != 1]
                emit(ev="failing_guards", func=fn, toks=toks(args), n_guards=len(op.guards), fails=fails[:12])
        except Exception as e:
            emit(ev="failing_guards_error", err=repr(e)[:200])
        return _inner_fg(self, variant, args, values, unselected, *a, **k)

    _hrf.HostTraceReplay._redispatch = _rd_fg
if os.environ.get("DIAG27_GC_TIME") == "1":
    # GC seconds per run (core agent 10-15): gc.callbacks start -> stop, total and while a redispatch runs, collections by gen
    import gc as _gc
    import torch.cuda._host_trace_replay as _hrg
    GC_TIME = {"total_s": 0.0, "in_redispatch_s": 0.0, "n": [0, 0, 0]}
    _gc_t0, _in_rd = [0.0], [0]

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_t0[0] = time.perf_counter()
        else:
            dt = time.perf_counter() - _gc_t0[0]
            GC_TIME["total_s"] += dt
            GC_TIME["n"][info.get("generation", 0)] += 1
            if _in_rd[0]:
                GC_TIME["in_redispatch_s"] += dt

    _gc.callbacks.append(_gc_cb)
    _inner_gcrd = _hrg.HostTraceReplay._redispatch

    def _rd_gc(self, *a, **k):
        _in_rd[0] += 1
        try:
            return _inner_gcrd(self, *a, **k)
        finally:
            _in_rd[0] -= 1

    _hrg.HostTraceReplay._redispatch = _rd_gc
    atexit.register(lambda: emit(ev="gc_time", **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in GC_TIME.items()}))
    atexit.register(lambda: print("GC_TIME", GC_TIME, flush=True))
if os.environ.get("DIAG27_PROFILE_SLOW"):
    # coordinator 10-16 (eager-memory slow path): a deterministic profile of HostTraceReplay._call_slow (every slow call: traces,
    # redispatches, fills, native calls with eager-order allocation), dumped as pstats at exit; the analysis subtracts the
    # _redispatch subtree
    import cProfile
    import torch.cuda._host_trace_replay as _hrs
    _PROFS, _SDEPTH = cProfile.Profile(), [0]
    _inner_cs = _hrs.HostTraceReplay._call_slow

    def _cs_profiled(self, *a, **k):
        _SDEPTH[0] += 1
        if _SDEPTH[0] == 1:
            _PROFS.enable()
        try:
            return _inner_cs(self, *a, **k)
        finally:
            _SDEPTH[0] -= 1
            if _SDEPTH[0] == 0:
                _PROFS.disable()

    _hrs.HostTraceReplay._call_slow = _cs_profiled
    atexit.register(lambda: _PROFS.dump_stats(os.environ["DIAG27_PROFILE_SLOW"]))
if os.environ.get("DIAG27_PROFILE_PARTS"):
    # learnfast 10-18 (task 3 hand-off): separate deterministic profiles of the three slow-path parts, each enabled only when
    # no other is active: trace (R.trace), redispatch (HostTraceReplay._redispatch), settle (HostTraceReplay._settle);
    # dumped to <prefix>_{trace,redispatch,settle}.pstats
    import cProfile
    _PP = {n: cProfile.Profile() for n in ("trace", "redispatch", "settle")}
    _PP_ACTIVE = [None]

    def _pp_wrap(name, fn):
        def run(*a, **k):
            if _PP_ACTIVE[0] is not None:
                return fn(*a, **k)
            _PP_ACTIVE[0] = name
            _PP[name].enable()
            try:
                return fn(*a, **k)
            finally:
                _PP[name].disable()
                _PP_ACTIVE[0] = None
        return run

    R.trace = _pp_wrap("trace", R.trace)
    R.HostTraceReplay._redispatch = _pp_wrap("redispatch", R.HostTraceReplay._redispatch)
    R.HostTraceReplay._settle = _pp_wrap("settle", R.HostTraceReplay._settle)
    atexit.register(lambda: [_PP[n].dump_stats(f"{os.environ['DIAG27_PROFILE_PARTS']}_{n}.pstats") for n in _PP])
if os.environ.get("DIAG27_PLACES_CHECK") == "1":
    # equality check (10-16): every section_places call also scanned the uncached way; mismatches counted
    import torch.cuda._host_trace_replay as _hrp2
    from torch.cuda._host_trace_memory import _first_seqs as _fs
    PLACES = {"calls": 0, "mismatch": 0}
    _orig_sp = _hrp2.section_places

    def _scan(lowered, plan, nodes):
        step = next(i for i, s in enumerate(lowered.steps) if isinstance(s, range) and nodes[0] in s)
        m, first, last = plan.steps[step], lowered.span(nodes[0])[0], lowered.span(nodes[-1])[1]
        seqs = _fs(lowered)
        if not m.order:
            at = sum(seqs[k] < first for k in m.tensors)
            return at, at
        n = len(lowered.allocations)
        ends = {-1 - k: end for k, _, end in m.temporaries} | {-1 - n - a: seq for a, seq in m.freed}
        keys = [(seqs[e], 0) if e >= 0 else (ends[e], 1) for e in m.order]
        return sum(k < (first, 0) for k in keys), sum(k < (last, 2) for k in keys)

    def _sp_checked(lowered, plan, nodes):
        got = _orig_sp(lowered, plan, nodes)
        PLACES["calls"] += 1
        if got != _scan(lowered, plan, nodes):
            PLACES["mismatch"] += 1
        return got

    _hrp2.section_places = _sp_checked
    atexit.register(lambda: print("PLACES_CHECK", PLACES, flush=True))
if os.environ.get("DIAG27_PROFILE_REDISPATCH"):
    # coordinator 10-15 (lever ii): a deterministic profile of HostTraceReplay._redispatch only (every attempt, ok or refused),
    # dumped as pstats at exit; learnfast/notes/REDISPATCH_OVERHEAD.md reads it
    import cProfile
    import torch.cuda._host_trace_replay as _hrp
    _PROF, _PDEPTH = cProfile.Profile(), [0]
    _inner_rd = _hrp.HostTraceReplay._redispatch

    def _rd_profiled(self, *a, **k):
        _PDEPTH[0] += 1
        if _PDEPTH[0] == 1:
            _PROF.enable()
        try:
            return _inner_rd(self, *a, **k)
        finally:
            _PDEPTH[0] -= 1
            if _PDEPTH[0] == 0:
                _PROF.disable()

    _hrp.HostTraceReplay._redispatch = _rd_profiled
    atexit.register(lambda: _PROF.dump_stats(os.environ["DIAG27_PROFILE_REDISPATCH"]))
if os.environ.get("DIAG27_REFUSAL_MEMO") == "1":
    # fast refusal of a redispatch that would rebuild a refused section (learnfast/refusal_memo.py; equality: memo on / off)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import refusal_memo as _rmemo
    atexit.register(lambda: emit(ev="refusal_memo", stats=dict(_rmemo.STATS)))
if os.environ.get("DIAG27_SECTION_MEMO") == "1" or os.environ.get("PROV_SECTION_MEMO") == "1":
    # after every check_section patch above (FNRESOLVE's _cs): the memo sees the final verdict
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import section_memo as _smemo
    atexit.register(lambda: emit(ev="section_memo", stats=dict(_smemo.STATS)))
if os.environ.get("DIAG27_PROV") == "1":
    atexit.register(lambda: emit(ev="prov", stats=dict(_prov.STATS), confirmed=dict(_prov.CONFIRMED), why=dict(_prov.CONFIRM_WHY),
                                 refused_by={str(k): v for k, v in _prov.REFUSED_BY.items()}, closure=_prov.CLOSURE, memo=_prov.SECTION_MEMO))


if os.environ.get("DIAG27_REFUSE_REDO_OPS"):
    # core agent 10-18 (nsys hang probe C): ops whose name contains one of these substrings refuse their redispatch (the call
    # retraces instead), so no redispatch entry re-sets their nodes
    _REFUSE = tuple(filter(None, os.environ["DIAG27_REFUSE_REDO_OPS"].split(",")))
    _inner_rdx = R.redispatch

    def _rdx_refuse(lowered, args, values, unselected, *a, **k):
        for i in unselected:
            selector = lowered.selectors[i - len(lowered.sites)]
            op = lowered.tape.ops[selector.op]
            if any(x in str(op.func) for x in _REFUSE):
                e = LT.FoldRefused(f"{op.func}: redispatch refused by DIAG27_REFUSE_REDO_OPS")
                e.op = selector.op
                raise e
        return _inner_rdx(lowered, args, values, unselected, *a, **k)

    R.redispatch = _rdx_refuse
if os.environ.get("DIAG27_ESCAPE_P") in ("count", "enforce"):
    # core agent 10-19 (escape rule P evaluation): a redispatch whose unselected library-body op has a failing own guard that
    # its Python decided on (GUARD_WHERE has "python": recorded or deduped outside an ATen op's C++ host; needs
    # PROV_PIN_ATEN=1). "count" only counts; "enforce" refuses it (the call retraces), as core's rule P would
    import no_lib_redo as _nlr_p
    _ESC = collections.Counter()
    _ESC_OPS = collections.Counter()
    _inner_rdp = R.redispatch

    def _p_refusal(lowered, values, unselected):
        where = getattr(sys.modules.get("diag27_live"), "GUARD_WHERE", None)
        env = lowered.tape.shape_env
        for i in unselected:
            selector = lowered.selectors[i - len(lowered.sites)]
            op = lowered.tape.ops[selector.op]
            if not _nlr_p.library_body(op.func):
                continue
            rows = selector.rows if len(selector.rows) == len(op.guards) else (None,) * len(op.guards)
            for g, row in zip(op.guards, rows):
                failing = row is None or row >= len(values) or values[row] != 1
                if failing and where is not None and "python" in where.get((id(env), g), ()):
                    pin = env.records[g][0].op == "eq"
                    _ESC["pin" if pin else "branch"] += 1
                    return selector.op, f"escape rule P: {op.func}'s failing guard was decided by its Python ({'pin' if pin else 'branch'})"
        return None

    def _rdx_p(lowered, args, values, unselected, *a, **k):
        _ESC["redispatches"] += 1
        hit = _p_refusal(lowered, values, unselected)
        if hit is not None:
            _ESC["p_would_refuse"] += 1
            _ESC_OPS[str(lowered.tape.ops[hit[0]].func)] += 1
            if os.environ["DIAG27_ESCAPE_P"] == "enforce":
                e = LT.FoldRefused(hit[1])
                e.op = hit[0]
                raise e
        return _inner_rdp(lowered, args, values, unselected, *a, **k)

    R.redispatch = _rdx_p
    atexit.register(lambda: emit(ev="escape_p", mode=os.environ["DIAG27_ESCAPE_P"], stats=dict(_ESC), ops=dict(_ESC_OPS.most_common(30))))
if os.environ.get("DIAG27_EXEC_CENSUS") == "1":
    # core agent 10-18 (window-2 streams): exec_census of every HostTraceReplay seen, after the first slow call past t=70 s and
    # t=130 s (main thread) and at exit
    sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/streams")
    from exec_census import exec_census as _exec_census
    _EC_REPLAYS, _EC_MARKS = [], [70.0, 130.0]
    _inner_ec = R.HostTraceReplay._call_slow

    def _ec_dump(when):
        for k, r in enumerate(_EC_REPLAYS):
            try:
                c = _exec_census(r)
            except Exception as e:
                c = {"error": repr(e)[:300]}
            emit(ev="exec_census", when=when, replay=k, census=c)
            print("EXEC_CENSUS", when, k, json.dumps({x: c.get(x) for x in ("variants", "total")} if "total" in c else c), flush=True)

    def _cs_ec(self, *a, **k):
        if self not in _EC_REPLAYS:
            _EC_REPLAYS.append(self)
        try:
            return _inner_ec(self, *a, **k)
        finally:
            t = time.perf_counter() - _t0
            while _EC_MARKS and t >= _EC_MARKS[0]:
                _ec_dump(f"t>={_EC_MARKS.pop(0):.0f}s (at {t:.1f}s)")

    R.HostTraceReplay._call_slow = _cs_ec
    atexit.register(lambda: _ec_dump("exit"))
import torch.cuda._host_trace_ir as _IRc  # noqa: E402

if hasattr(_IRc, "sloc_memo_hits"):
    atexit.register(lambda: print("SLOC_MEMO_HITS", _IRc.sloc_memo_hits[0], flush=True))
# 10-18: core's dispatch-cache counters (step_ms21+: cross-variant redo reuse in core)
import torch.cuda._host_trace_redispatch as _RDc  # noqa: E402

if hasattr(_RDc, "dispatch_hits"):
    atexit.register(lambda: emit(ev="core_dispatch", dispatch_hits=_RDc.dispatch_hits, redo_bodies=_RDc.redo_bodies))
    atexit.register(lambda: print("CORE_DISPATCH hits", _RDc.dispatch_hits, "bodies", _RDc.redo_bodies, flush=True))
# 10-18 (core agent, redispatch-cost counts): DIAG27_RDCOST=1 times lower_entry / Lowering.conjunction / check_section and dumps
# core's for_instance_stats (wip25+) and the replays' redos / instances at exit
if os.environ.get("DIAG27_RDCOST") == "1":
    import torch.cuda._host_trace_lower as _LWc
    RDC = collections.Counter()

    def _rdc_wrap(name, fn):
        def run(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                RDC[name + "_s"] += time.perf_counter() - t
                RDC[name + "_n"] += 1
        return run

    _RDc.lower_entry = _rdc_wrap("lower_entry", _RDc.lower_entry)
    _LWc.Lowering.conjunction = _rdc_wrap("conjunction", _LWc.Lowering.conjunction)
    _REPLAYS = []
    _inner_rdc = R.HostTraceReplay._redispatch

    def _rd_rdc(self, *a, **k):
        if self not in _REPLAYS:
            _REPLAYS.append(self)
        return _inner_rdc(self, *a, **k)

    R.HostTraceReplay._redispatch = _rd_rdc

    def _rdc_dump():
        out = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in RDC.items()}
        fis = getattr(_RDc, "for_instance_stats", None)
        out["for_instance_stats"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (fis or {}).items()}
        out["replays"] = [{"redos": getattr(r, "redos", None), "redo_instances": getattr(r, "redo_instances", None), "redispatches": getattr(r, "redispatches", None)} for r in _REPLAYS]
        emit(ev="rdcost", **out)
        print("RDCOST", out, flush=True)

    atexit.register(_rdc_dump)
    if hasattr(R, "check_section"):
        R.check_section = _rdc_wrap("check_section", R.check_section)
# 10-17: seconds in core's _gc_untrack (the end-of-slow-call walk, _settle) per run; DIAG27_GC_UNTRACK=0 turns the walk off
ACC["gc_untrack"], CNT["gc_untrack"] = 0.0, 0
if os.environ.get("DIAG27_GC_UNTRACK") == "0":
    R.gc_untrack = False
    print("diag27: _host_trace_replay.gc_untrack = False", flush=True)
if hasattr(R, "_gc_untrack"):
    _orig_gcu = R._gc_untrack

    _GCU_LOG = os.environ.get("DIAG27_GCU_LOG") == "1"

    def _gcu_timed(*a, **k):
        if _GCU_LOG:  # core agent 10-18: per settle, gen-0 objects before the walk, walk counters, full walks, seconds
            import gc as _gcl
            g0 = len(_gcl.get_objects(generation=0))
            c0 = torch._C._host_trace_gc_untrack_counts() if hasattr(torch._C, "_host_trace_gc_untrack_counts") else None
        t = time.perf_counter()
        try:
            return _orig_gcu(*a, **k)
        finally:
            dt = time.perf_counter() - t
            ACC["gc_untrack"] += dt
            CNT["gc_untrack"] += 1
            if _GCU_LOG:
                c1 = torch._C._host_trace_gc_untrack_counts() if c0 is not None else None
                walks = getattr(R, "_gc_walks", None)
                emit(ev="gcu", n=CNT["gc_untrack"], gen0=g0, s=round(dt, 3), walked=None if c1 is None else c1[0] - c0[0],
                     old_skipped=None if c1 is None else c1[1] - c0[1], full_walks=None if walks is None else walks[2],
                     sealed_skipped=None if c1 is None or len(c1) < 3 else c1[2] - c0[2])

    R._gc_untrack = _gcu_timed


def _on_term(signum, frame):  # a timeout's SIGTERM: keep the timers (every atexit record), then die as SIGTERM would
    signal.signal(signum, signal.SIG_IGN)  # a second TERM (timeout forwards one too) re-entering here deadlocks on tqdm's lock
    try:
        atexit._run_exitfuncs()
    except Exception:
        emit(ev="end", acc={k: round(v, 2) for k, v in ACC.items()}, cnt=dict(CNT), signal=signum)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


import signal  # noqa: E402

signal.signal(signal.SIGTERM, _on_term)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.modules["diag27_live"] = sys.modules[__name__]  # the driver's range warm-up reads ACC / CNT per phase
runpy.run_path(script, run_name="__main__")
