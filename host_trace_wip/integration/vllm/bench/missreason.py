# Phase 6 step 4c diagnostic: why a call whose variant's guards hold still traces. Runs bench/rangesweep.py (same args)
# with the slow path traced: each return of _fill / _learn / _relower / _call_native / _redispatch / _fold (its line,
# the variant's learns flag, the value) is kept for the call, and every _miss prints the call's events.
#   CUDA_VISIBLE_DEVICES=2 bash python_vllm_cpp.sh bench/missreason.py --out bench/out/range/X.json [rangesweep args]
import os, runpy, sys

import torch
import torch.cuda._host_trace_replay as R

WATCH = {"_fill", "_learn", "_relower", "_call_native", "_redispatch", "_fold"}
EV = []
_slow, _miss = R.HostTraceReplay._call_slow, R.HostTraceReplay._miss


def summ(v):
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, tuple) and v and isinstance(v[0], int):
        return ("outcome", v[0])
    return type(v).__name__


def local(frame, event, arg):
    if event == "return":
        var = frame.f_locals.get("variant")
        extra = {}
        if frame.f_code.co_name == "_learn":
            k = frame.f_locals.get("key")
            extra = {"op": str(frame.f_locals.get("op")), "refusal": None if k is None else frame.f_locals["provider"].refusal(k)}
        EV.append((frame.f_code.co_name, frame.f_lineno, getattr(var, "learns", None), id(var) % 10007 if var is not None else None, summ(arg), extra))
    elif event == "exception":
        EV.append((frame.f_code.co_name, frame.f_lineno, "exc", arg[0].__name__, str(arg[1])[:200]))
    return local


def tracer(frame, event, arg):
    return local if frame.f_code.co_name in WATCH and frame.f_code.co_filename.endswith("_host_trace_replay.py") else None


def slow(self, args, kwargs, searched):
    EV.clear()
    sys.settrace(tracer)
    try:
        return _slow(self, args, kwargs, searched)
    finally:
        sys.settrace(None)


def miss(self, contract, args, candidates=(), before=None):
    sys.settrace(None)
    ints = [a for a in args if isinstance(a, int)][-4:]
    T = args[483].shape[0] if len(args) > 483 and isinstance(args[483], torch.Tensor) else None
    fam = self._families.get(contract, ())
    print(f"MISS T{T} ints {ints} variants {len(fam)} learns {[v.learns for v in fam]} ids {[id(v) % 10007 for v in fam]} cands {len(candidates)} before {before is not None}", flush=True)
    for e in EV:
        print("   ", e, flush=True)
    return _miss(self, contract, args, candidates, before)


R.HostTraceReplay._call_slow, R.HostTraceReplay._miss = slow, miss
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "rangesweep.py"), *sys.argv[1:]]
G = runpy.run_path(sys.argv[0], run_name="__main__")
import collections
for k, e in G["V"].entries.items():
    for fam in e._families.values():
        for v in fam:
            lo = v.captured.lowered
            ops = collections.Counter(str(o.op).split(".")[-2] if "." in str(o.op) else str(o.op) for o in lo.opaque.values())
            print("CENSUS", str(k)[:40], "id", id(v) % 10007, "learns", v.learns, "sites", len(lo.sites), "opaque", dict(ops), flush=True)
