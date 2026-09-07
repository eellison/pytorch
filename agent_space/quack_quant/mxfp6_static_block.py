import statistics, torch
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz
def graph_bench(fn, warmup=20, samples=50, calls=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
def literal_block(x):
    return qz.to_mxfp6_e2m3(x, 32)
variants = (("quack shipped (dynamic=True, block symbolic)", qz.to_mxfp6_e2m3, True),
            ("dynamic=True, literal block_size", literal_block, True),
            ("dynamic=False", qz.to_mxfp6_e2m3, False))
for M, K in ((2048, 3072), (8192, 4096)):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = qz.to_mxfp6_e2m3(x)
    print(f"to_mxfp6_e2m3 {M}x{K}:")
    for label, fn, dyn in variants:
        torch._dynamo.reset(); metrics.reset()
        with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
            compiled = torch.compile(fn, fullgraph=True, dynamic=dyn)
            out, srcs = run_and_get_code(compiled, x)
            eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, ref))
            us = graph_bench(lambda: compiled(x))
        names = [l[4:l.index("(")][:16] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
        print(f"   {label:46s} {us:8.1f}us k={metrics.generated_kernel_count} eq={eq} {names}", flush=True)
