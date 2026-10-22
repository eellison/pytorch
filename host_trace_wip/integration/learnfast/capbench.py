# capbench.py: where a capture_kernel_nodes call spends its time (trivial fill, and a 3-kernel fn), per phase, N calls
import time, torch
import torch.cuda._host_trace_capture as cap
from torch.cuda._host_trace_tape import _gc_hold
N = 200
x = torch.zeros(1 << 20, device="cuda")
def fn(s):
    x.fill_(1); x.mul_(2); x.add_(1)
for _ in range(10): cap.capture_kernel_nodes(fn)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(N): cap.capture_kernel_nodes(fn)
print("capture_kernel_nodes ms/call %.3f" % (1e3 * (time.perf_counter() - t) / N))
T = dict(begin=0, fn=0, end_pre=0, end_post=0, nodes=0, dtor=0)
for _ in range(N):
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with _gc_hold, cap.private_stream() as s, torch.cuda.stream(s):
        a = time.perf_counter(); g.capture_begin(capture_error_mode="thread_local"); b = time.perf_counter()
        fn(s); c = time.perf_counter()
        g.capture_end_pre(); d = time.perf_counter()
        g.capture_end_post(); e = time.perf_counter()
    nodes = cap.graph_nodes(g.raw_cuda_graph(), False); f = time.perf_counter()
    del g; h = time.perf_counter()
    for k, v in zip(T, (b - a, c - b, d - c, e - d, f - e, h - f)): T[k] += v
print({k: round(1e3 * v / N, 3) for k, v in T.items()})
