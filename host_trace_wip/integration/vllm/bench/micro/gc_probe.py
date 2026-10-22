# Does a large unfrozen gc heap (arm V's 3.3M tracked objects) slow numpy / torch CPU ops? Probe before and after.
import gc, sys, time
import numpy as np, torch
_arr = np.arange(1025, dtype=np.int32)
def probe(fn, n=3000):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[n // 2] * 1e6, 2)
def all_():
    return {"python": probe(lambda: sum(i * i for i in range(200))), "numpy": probe(lambda: np.cumsum(_arr[:65])),
            "torch_cpu": probe(lambda: torch.from_numpy(_arr)[:64].clone())}
print("base", all_(), len(gc.get_objects()), flush=True)
keep = [[i] for i in range(int(sys.argv[1]))]
print("heap", all_(), len(gc.get_objects()), flush=True)
gc.freeze()
print("frozen", all_(), len(gc.get_objects()), flush=True)
gc.unfreeze(); gc.disable()
print("gc off", all_(), flush=True)
