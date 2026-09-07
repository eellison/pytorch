import sys, torch, contextlib, statistics
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
def mxfp8_quant_recip(normed):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // 32, 32)
    amax = g.abs().float().amax(dim=-1)
    raw = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw)
    inv = recip_ue8m0(scale)
    q = (g.float() * inv.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
    return q, swizzle_scale(scale)
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
ref = None
for qname, quant in (("div", mxfp8_quant), ("recip", mxfp8_quant_recip)):
    for kind, fn, args in (("fused", lambda x, w, q=quant: q(rms(x, w)), (x, w)), ("standalone", lambda x, q=quant: q(x), (x,))):
        for label, patches, persistent in (("persistent", {}, True), ("persistent_cd", {"coordinate_descent_tuning": True}, True), ("looped_cd", {"coordinate_descent_tuning": True}, False)):
            if kind == "standalone" and label != "persistent": continue
            torch._dynamo.reset(); metrics.reset()
            with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **patches}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, *args); torch.cuda.synchronize()
                us = graph_bench(lambda: compiled(*args))
            key = (kind, label)
            if qname == "div": globals().setdefault("refs", {})[key] = out
            eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, refs[key])) if qname == "recip" else "-"
            print(f"{qname:5s} {kind:10s} {label:14s}: {us:6.2f}us kernels={metrics.generated_kernel_count} exact_vs_div={eq}", flush=True)
