# Host cost of a HostTraceReplay hit, 489 args / 470 kernels: plain vs trusted (exact layouts) entries.
import time, torch
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_tape import TrustedInputs
n = 489
xs = [torch.randn(64, device="cuda") for _ in range(n)]
def fn(*a):
    y = a[0]
    for i in range(470):
        y = y + a[i % n]
    return (y,)
for mode in ("plain", "trusted"):
    tr = TrustedInputs(tuple((tuple(x.shape), tuple(x.stride())) for x in xs)) if mode == "trusted" else None
    e = R.HostTraceReplay(fn, memory="run_buffer", trusted=tr)
    for _ in range(4):
        e(*xs)
    ts = []
    for _ in range(200):
        torch.cuda.synchronize()
        t = time.perf_counter(); e(*xs); ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{mode} replays={e.replays} median_us {ts[100]*1e6:.1f} p10 {ts[20]*1e6:.1f}", flush=True)
