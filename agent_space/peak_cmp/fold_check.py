import sys, torch, contextlib, statistics
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
class _P(InductorChoices):
    @staticmethod
    def should_use_persistent_reduction(*a, **k): return True
def graph_bench(f, warmup=20, samples=50, calls=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
for fmt in ("nvfp4", "mxfp4"):
    fn = lambda x, w, fmt=fmt: QUANT[fmt](rms(x, w))
    torch._dynamo.reset()
    with config.patch({"triton.nested_reduction": False, "triton.cudagraphs": False, "fx_graph_cache": False}):
        ref = torch.compile(fn, fullgraph=True, dynamic=False)(x, w)
    for label, patches, persistent in (("nested persistent", {}, True), ("nested persistent cd", {"coordinate_descent_tuning": True}, True), ("nested default (looped)", {}, False), ("nested looped cd", {"coordinate_descent_tuning": True}, False)):
        torch._dynamo.reset(); metrics.reset()
        with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **patches}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
            us = graph_bench(lambda: compiled(x, w))
        eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, ref))
        k = [s for s in srcs if "@triton_heuristics" in s][0]; body = k[k.index("@triton.jit"):]
        print(f"{fmt} {label:26s}: {us:6.2f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} exact_vs_unnested={eq} splits={body.count('tl.split(')}")
        if label == "nested persistent":
            for line in body.splitlines():
                l = line.strip()
                if "tl.split(" in l or "broadcast_to(" in l or "inline_asm" in l: print("      ", l[:160])
