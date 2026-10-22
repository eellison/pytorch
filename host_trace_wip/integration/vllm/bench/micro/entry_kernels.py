# Host cost of a HostTraceReplay hit vs the tape's kernel count (no profiler): K chained adds over n=489 args.
import time, torch
import torch.cuda._host_trace_replay as R
n = 489
xs = [torch.randn(64, device="cuda") for _ in range(n)]
def make(k):
    def fn(*a):
        y = a[0]
        for i in range(k):
            y = y + a[i % n]
        return (y,)
    return fn
for k in (1, 100, 470):
    e = R.HostTraceReplay(make(k), memory="run_buffer")
    for _ in range(4):
        e(*xs)
    ts = []
    for _ in range(200):
        torch.cuda.synchronize()
        t = time.perf_counter(); e(*xs); ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"k={k} replays={e.replays} variants={len(e.variants)} median_us {ts[100]*1e6:.1f} p10 {ts[20]*1e6:.1f}", flush=True)
