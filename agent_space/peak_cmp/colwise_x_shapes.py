import sys, torch, statistics
exec(open("agent_space/peak_cmp/colwise_probe.py").read().split("M, K = int(sys.argv[1])")[0])
def graph_bench(f, warmup=10, samples=30, calls=50):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
for M, K in ((16384, 7168), (32768, 4096), (65536, 2048), (65536, 8192)):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    fn = lambda x, w: mxfp8_colwise(rms(x, w))
    line = f"{M}x{K}:"
    for allow in (False, True):
        for cd in (False, True):
            torch._dynamo.reset(); metrics.reset()
            with config.patch({"triton.nested_reduction": True, "triton.nested_reduction_allow_x": allow, "coordinate_descent_tuning": cd, "triton.cudagraphs": False, "fx_graph_cache": False}):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
                us = graph_bench(lambda: compiled(x, w))
            line += f"  {'X' if allow else '3k'}{'/cd' if cd else '   '}={us:6.1f}us(k={metrics.generated_kernel_count})"
    print(line, flush=True)
