# Offline cuBLAS learn from a key set (no model): every key of a saved bindings file learned again by
# HarvestProvider.learn on synthetic operands (HostTraceReplay._learn, the warm-up path), timed per phase.
#   bash py.sh harness.py --keys KEYS.pkl --out out/x.pkl [--limit N] [--timers] [--sync] [--shard I/N]
import argparse
import collections
import json
import os
import pickle
import sys
import time
import types

p = argparse.ArgumentParser()
p.add_argument("--keys", required=True)
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--skip", type=int, default=0)
p.add_argument("--shard", default="", help="I/N: keys i with i %% N == I (round robin)")
p.add_argument("--shard-by", default="rr", choices=["rr", "block", "nk"], help="rr: round robin; block: contiguous; nk: whole (N, K) groups")
p.add_argument("--timers", action="store_true")
p.add_argument("--sync", action="store_true", help="synchronize around timed phases")
p.add_argument("--set", action="append", default=[], help="H.NAME=VALUE")
p.add_argument("--json", default="")
p.add_argument("--threads", type=int, default=1)
a = p.parse_args()

import torch  # noqa: E402
import torch.utils._pytree as pytree  # noqa: E402
import torch.cuda._host_trace_harvest as H  # noqa: E402
import torch.cuda._host_trace_replay as R  # noqa: E402
from torch.cuda._host_trace_opaque import OpaqueKey  # noqa: E402

for s in a.set:
    lhs, v = s.split("=", 1)
    setattr(H, lhs.split(".", 1)[1], eval(v))

t_start = time.perf_counter()
data = pickle.load(open(a.keys, "rb"))
entries = [k for k, _ in data["entries"]]
if a.shard:
    i, n = map(int, a.shard.split("/"))
    if a.shard_by == "rr":
        entries = entries[i::n]
    elif a.shard_by == "block":
        m = -(-len(entries) // n)
        entries = entries[i * m : (i + 1) * m]
    else:
        groups = sorted({tuple(k[3][1]) for k in entries})
        mine = set(groups[i::n])
        entries = [k for k in entries if tuple(k[3][1]) in mine]
entries = entries[a.skip :]
if a.limit:
    entries = entries[: a.limit]

T = collections.defaultdict(float)
C = collections.Counter()


def timed(name, fn, sync=False):
    def w(*args, **kw):
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            if sync:
                torch.cuda.synchronize()
            T[name] += time.perf_counter() - t
            C[name] += 1

    return w


if a.timers:
    C_ = torch._C
    for name in ("_cuda_hostTraceHarvestCapture", "_cuda_hostTraceCapturedNodes", "_cuda_hostTraceDigests", "_cuda_hostTraceLaunch",
                 "_cuda_hostTraceSlotQwords", "_cuda_hostTraceFunctionNames", "_cuda_hostTracePool", "_cuda_hostTraceArenaViews",
                 "_cuda_hostTraceStackBounds"):
        setattr(C_, name, timed(name, getattr(C_, name)))
    H.HarvestProvider._harvest = timed("_harvest", H.HarvestProvider._harvest, a.sync)
    H.HarvestProvider._compact = timed("_compact", H.HarvestProvider._compact)
    H._Mirrors.__init__ = timed("mirrors_init", H._Mirrors.__init__)
    H._Device.plug = timed("plug", H._Device.plug)
    H._signature = timed("_signature", H._signature)
    _cg = torch.cuda.CUDAGraph.reset
    torch.cuda.CUDAGraph.reset = timed("graph_reset", _cg)

provider = H.HarvestProvider(("blas",))
fake = types.SimpleNamespace(learned=0, learn_s=0.0)
learn = R.HostTraceReplay._learn


def one(k):
    k = H._local(k)
    alias, op_name, dtypes, sizes, strides, classes, scalars, device, state = k
    op = {"aten::mm": torch.ops.aten.mm.default, "aten::addmm": torch.ops.aten.addmm.default, "aten::bmm": torch.ops.aten.bmm.default}[op_name]
    align = tuple(0 if c == 256 else c for c in classes)
    key = OpaqueKey(op, dtypes, sizes, strides, align, scalars, device, state, alias)
    n_in = len(op._schema.arguments) if op_name != "aten::addmm" else 3
    n_in = {"aten::mm": 2, "aten::bmm": 2, "aten::addmm": 3}[op_name]
    leaves = [object()] * n_in
    spec = pytree.tree_flatten((tuple(leaves), {}))[1]
    return learn(fake, op, provider, (spec, frozenset(range(n_in))), key)


torch.cuda.init()
t0 = time.perf_counter()
cpu0 = time.process_time()
per = []
failed = 0
if a.threads <= 1:
    for k in entries:
        t = time.perf_counter()
        b = one(k)
        per.append(time.perf_counter() - t)
        failed += b is None
else:
    import concurrent.futures as cf

    def work(chunk):
        s = torch.cuda.Stream()
        out = []
        with torch.cuda.stream(s):
            for k in chunk:
                t = time.perf_counter()
                b = one(k)
                out.append((time.perf_counter() - t, b is None))
        return out

    chunks = [entries[i :: a.threads] for i in range(a.threads)]
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for res in ex.map(work, chunks):
            for dt, f in res:
                per.append(dt)
                failed += f
torch.cuda.synchronize()
t_learn = time.perf_counter() - t0
t1 = time.perf_counter()
saved = provider.save(a.out)
t_save = time.perf_counter() - t1
per.sort()
R_ = {
    "keys": len(entries),
    "learned": fake.learned,
    "failed": failed,
    "harvests": provider.harvests,
    "full_templates": sum(len(v) for v in provider.templates.values()),
    "learn_wall_s": t_learn,
    "learn_cpu_s": time.process_time() - cpu0,
    "per_key_ms_median": 1e3 * per[len(per) // 2] if per else None,
    "per_key_ms_p90": 1e3 * per[int(len(per) * 0.9)] if per else None,
    "per_key_ms_max": 1e3 * per[-1] if per else None,
    "save_s": t_save,
    "saved": saved,
    "total_s": time.perf_counter() - t_start,
    "timers_s": {k: round(v, 3) for k, v in sorted(T.items(), key=lambda x: -x[1])},
    "counts": dict(C),
    "loadavg": open("/proc/loadavg").read().split()[:3],
}
print(json.dumps(R_, indent=1))
if a.json:
    json.dump(R_, open(a.json, "w"), indent=1)
