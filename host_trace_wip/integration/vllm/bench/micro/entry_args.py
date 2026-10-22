# Host cost of a HostTraceReplay hit vs the number of tensor arguments (no profiler): median of 300 replays.
import sys, time, torch
import torch.cuda._host_trace_replay as R
for n in (1, 50, 200, 489):
    xs = [torch.randn(64, device="cuda") for _ in range(n)]
    for mode in ("plain", "static"):
        e = R.HostTraceReplay(lambda *a: (a[0] * 2,), memory="run_buffer", static_shapes=range(n) if mode == "static" else ())
        for _ in range(4):
            e(*xs)
        ts = []
        for _ in range(300):
            torch.cuda.synchronize()
            t = time.perf_counter(); e(*xs); ts.append(time.perf_counter() - t)
        ts.sort()
        print(f"n={n} {mode} replays={e.replays} median_us {ts[150]*1e6:.1f} p10 {ts[30]*1e6:.1f}", flush=True)
