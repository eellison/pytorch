# Per-op torch CPU cost before and after importing arm V's host-trace modules (and the adapter), same process.
import sys, time, importlib
import numpy as np, torch
_arr, _t = np.arange(1025, dtype=np.int32), torch.zeros(1024, dtype=torch.int32)
def probe(fn, n=5000):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[n // 2] * 1e6, 2)
def all_(tag):
    print(tag, {"torch_cpu": probe(lambda: torch.from_numpy(_arr)[:64].clone()), "slice": probe(lambda: _t[:64]), "add": probe(lambda: _t.add(1))}, flush=True)
all_("base")
for m in sys.argv[1:]:
    importlib.import_module(m)
    all_(m)
