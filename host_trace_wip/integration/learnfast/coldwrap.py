"""coldwrap.py SCRIPT [args]: runs SCRIPT (the vLLM lane's drive_sweep.py, read-only) with HarvestProvider.load waiting for
the side learn's merged bindings (PATH + ".done" written by cold8.sh once the merge is in place), and stamps the wall times of
process start, the load call, and the load's return into COLD_STAMPS (JSON lines)."""
import json, os, runpy, sys, time
T0 = float(os.environ.get("COLD_T0", time.time()))
OUT = open(os.environ["COLD_STAMPS"], "a", buffering=1)
def stamp(ev, **kw):
    OUT.write(json.dumps({"ev": ev, "t": round(time.time() - T0, 3), **kw}) + "\n")
stamp("python_start")
import torch.cuda._host_trace_harvest as H
_load = H.HarvestProvider.load
def load(self, path, versions=None):
    stamp("load_called")
    while not os.path.exists(path + ".done"):
        time.sleep(0.05)
    stamp("bindings_ready")
    n = _load(self, path, versions)
    stamp("load_done", n=n)
    return n
H.HarvestProvider.load = load
import importlib.abc, importlib.machinery
import torch.cuda._host_trace_replay as RP
_tr = RP.trace
_ntr = [0]
_PROF = os.environ.get("COLD_PROFILE") == "1"
def _trace(*a, **k):
    stamp("trace_begin")
    _ntr[0] += 1
    prof = None
    if _PROF:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    try:
        return _tr(*a, **k)
    finally:
        if prof is not None:
            prof.disable()
            prof.dump_stats(os.path.join(os.path.dirname(os.environ["COLD_STAMPS"]), "trace_%d.pstats" % _ntr[0]))
        stamp("trace_end", n=_ntr[0], fn=getattr(a[0], "__qualname__", str(a[0]))[:80] if a else None,
              ninputs=len(a[1]) if len(a) > 1 else None)
RP.trace = _trace
_bld = RP.HostTraceReplay._build
def _build(self, tape):
    stamp("build_begin")
    try:
        return _bld(self, tape)
    finally:
        stamp("build_end")
RP.HostTraceReplay._build = _build
def _timed(cls, name):
    f = getattr(cls, name)
    def g(*a, **k):
        stamp(name + "_begin")
        try:
            return f(*a, **k)
        finally:
            stamp(name + "_end")
    setattr(cls, name, g)
class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in ("adapter", "vllm.entrypoints.llm"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path) if name == "adapter" else None
        if name == "vllm.entrypoints.llm":
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    break
        if spec is None or spec.loader is None:
            return spec
        ex = spec.loader.exec_module
        def exec_module(m):
            ex(m)
            if name == "adapter" and hasattr(m, "ArmV"):
                _timed(m.ArmV, "__init__"); _timed(m.ArmV, "install")
            if name == "vllm.entrypoints.llm" and hasattr(m, "LLM"):
                _timed(m.LLM, "__init__")
        spec.loader.exec_module = exec_module
        return spec
sys.meta_path.insert(0, _Hook())
# ROOTS_CACHE prototype: _host_trace_cute._roots_of with a per-trace index from each root's base free symbols to the
# roots (built incrementally as the trace's inputs / allocations / eager outputs grow), same result and order
if os.environ.get("COLD_ROOTS_CACHE") == "1":
    import torch.cuda._host_trace_cute as CU
    import torch.cuda._host_trace_tape as TT
    _orig_roots_of = CU._roots_of
    _IDX = {}
    _CHECK = os.environ.get("COLD_ROOTS_CHECK") == "1"
    def _roots_of(tr, v):
        key = TT._sym_key(v)
        held = set() if isinstance(key, int) else key.free_symbols
        ent = _IDX.get(id(tr))
        if ent is None or ent[0] is not tr:
            ent = _IDX[id(tr)] = [tr, 0, 0, 0, {}, [], []]
        _, ni, na, ne, index, order, rank = ent
        def add(r, kind, j):
            pos = len(order); order.append(r); rank.append((kind, j))
            k = TT._sym_key(r.sym)
            for f in (k.free_symbols if not isinstance(k, int) else ()):
                index.setdefault(f, []).append(pos)
        for j in range(ni, len(tr.inputs)): add(tr.inputs[j].root, 0, j)
        for j in range(na, len(tr.allocs)): add(tr.allocs[j].root, 1, j)
        for j in range(ne, len(tr.eager_outputs)): add(tr.eager_outputs[j]._root, 2, j)
        ent[1], ent[2], ent[3] = len(tr.inputs), len(tr.allocs), len(tr.eager_outputs)
        hit = sorted({p for f in held for p in index.get(f, ())}, key=lambda p: rank[p])
        rs = [order[p] for p in hit]
        if _CHECK:
            want = _orig_roots_of(tr, v)
            if [id(x) for x in want] != [id(x) for x in rs]:
                raise AssertionError("roots cache mismatch")
        return rs
    CU._roots_of = _roots_of
if os.environ.get("COLD_NO_SECTIONS") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_sections  # noqa: F401
    import atexit as _ax
    _ax.register(lambda: print("NO_SECTIONS refused", no_sections.REFUSED, flush=True))
if os.environ.get("COLD_NO_LIB_REDO") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import no_lib_redo  # noqa: F401
    import atexit as _ax2
    _ax2.register(lambda: print("NO_LIB_REDO refused", dict(no_lib_redo.REFUSED), flush=True))
if os.environ.get("COLD_CAPMEMO") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import capmemo
    import atexit as _ax3
    _ax3.register(lambda: print("CAPMEMO", capmemo.STATS, flush=True))
import atexit
atexit.register(lambda: stamp("exit"))
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
