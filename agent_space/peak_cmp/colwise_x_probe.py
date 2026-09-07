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
for M, K in ((8192, 4096), (16384, 7168)):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    fn = lambda x, w: mxfp8_colwise(rms(x, w))
    torch._dynamo.reset()
    with config.patch({"triton.nested_reduction": False, "triton.cudagraphs": False, "fx_graph_cache": False}):
        ref = torch.compile(fn, fullgraph=True, dynamic=False)(x, w)
    for allow in (False, True):
        for cd in (False, True):
            torch._dynamo.reset(); metrics.reset()
            try:
                with config.patch({"triton.nested_reduction": True, "triton.nested_reduction_allow_x": allow, "coordinate_descent_tuning": cd, "triton.cudagraphs": False, "fx_graph_cache": False}):
                    compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
                    us = graph_bench(lambda: compiled(x, w))
                eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, ref))
                names = [l[4:l.index("(")][:20] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
                print(f"{M}x{K} allow_x={allow!s:5s} cd={cd!s:5s}: {us:7.1f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} exact={eq} {names}", flush=True)
            except Exception as ex:
                print(f"{M}x{K} allow_x={allow!s:5s} cd={cd!s:5s}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
