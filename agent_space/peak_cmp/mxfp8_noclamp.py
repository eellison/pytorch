import sys, torch, contextlib, statistics
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
xs = x.clone(); xs[1, :32] = float("inf"); xs[2, :8] = float("nan"); xs[5, :16] = -float("inf")
def mxfp8_noclamp(normed):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // 32, 32)
    amax = g.abs().float().amax(dim=-1)
    raw = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw)
    q = (g.float() * recip_ue8m0(scale).unsqueeze(-1)).to(torch.float8_e4m3fn).view(rows, hidden)
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
def same(a, b):
    ua, ub = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(ua, ub)
refs = {}
for qname, quant in (("clamp", mxfp8_quant), ("noclamp", mxfp8_noclamp)):
    for kind, fn, args in (("fused", lambda x, w, q=quant: q(rms(x, w)), (x, w)), ("standalone", lambda x, q=quant: q(x), (x,))):
        for label, patches, persistent in (("persistent_cd", {"coordinate_descent_tuning": True}, True), ("looped_cd", {"coordinate_descent_tuning": True}, False), ("persistent", {}, True)):
            if kind == "standalone" and label != "persistent": continue
            torch._dynamo.reset(); metrics.reset()
            with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **patches}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, _ = run_and_get_code(compiled, *args)
                out_s = compiled(*((xs, w) if kind == "fused" else (xs,)))
                torch.cuda.synchronize(); us = graph_bench(lambda: compiled(*args))
            key = (kind, label)
            if qname == "clamp": refs[key] = (out, out_s); eq = "-"
            else: eq = f"exact={all(same(a, b) for a, b in zip(out, refs[key][0]))} specials_exact={all(same(a, b) for a, b in zip(out_s, refs[key][1]))}"
            print(f"{qname:8s} {kind:10s} {label:14s}: {us:6.2f}us {eq}", flush=True)
