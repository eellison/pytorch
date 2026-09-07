"""Default-tuning form/warp sweep for nested RMSNorm->quant kernels across row widths."""
import sys, torch, contextlib, statistics, json
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
fmt = sys.argv_fmt = __import__("os").environ.get("FMT", "mxfp8")
class _P(InductorChoices):
    @staticmethod
    def should_use_persistent_reduction(*a, **k): return True
def graph_bench(f, warmup=20, samples=30, calls=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
rows = []
for K in (1024, 2048, 4096, 8192, 16384):
    M = 32 * 1024 * 1024 // K  # ~64MB bf16 per case
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    fn = lambda x, w: QUANT[fmt](rms(x, w))
    for label, persistent in (("looped", False), ("persistent", True)):
        torch._dynamo.reset(); metrics.reset()
        try:
            with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
                us = graph_bench(lambda: compiled(x, w))
            k = [s for s in srcs if "@triton_heuristics" in s][0]
            kind = "per" if "persistent_reduction" in k else "red"
            print(f"{fmt} {M}x{K}: {label:10s} {kind} {us:7.2f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}", flush=True)
            rows.append(dict(fmt=fmt, M=M, K=K, form=label, kind=kind, us=us))
        except Exception as ex:
            print(f"{fmt} {M}x{K}: {label:10s} ERROR {str(ex)[:120]}", flush=True)
json.dump(rows, open(f"agent_space/peak_cmp/form_sweep_{fmt}.json", "w"))
