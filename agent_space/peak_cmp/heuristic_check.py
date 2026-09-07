import sys, torch, statistics, re
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
def graph_bench(f, warmup=20, samples=30, calls=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
for fmt in ("mxfp8", "nvfp4"):
    for K in (2048, 4096, 8192):
        M = 32 * 1024 * 1024 // K
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
        fn = lambda x, w, f=fmt: QUANT[f](rms(x, w))
        torch._dynamo.reset(); metrics.reset()
        with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
            us = graph_bench(lambda: compiled(x, w))
        k = [s for s in srcs if "@triton_heuristics" in s][0]
        kind = "persistent" if "persistent_reduction" in k else "looped"
        flag = "'nested_reduction': True" in k
        print(f"{fmt} {M}x{K} default: {kind} {us:6.2f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} meta_flag={flag}", flush=True)
