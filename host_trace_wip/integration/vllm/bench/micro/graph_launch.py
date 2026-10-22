# Host cost of cudaGraphLaunch for a graph of N small kernels (no profiler): median over 200 launches, GPU idle at each.
import time, torch
x = torch.zeros(1024, device="cuda")
for n in (1, 100, 470, 550):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x.add_(1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            x.add_(1)
    ts = []
    for _ in range(200):
        torch.cuda.synchronize()
        t = time.perf_counter(); g.replay(); ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"n={n} launch_us median {ts[100]*1e6:.1f} p10 {ts[20]*1e6:.1f}")
