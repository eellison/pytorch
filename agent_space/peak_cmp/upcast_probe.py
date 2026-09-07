import sys, torch, contextlib, statistics, re
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
def mxfp8_quant_recip(normed):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // 32, 32)
    amax = g.abs().float().amax(dim=-1)
    raw = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw)
    q = (g.float() * recip_ue8m0(scale).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
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
refs = {}
for fmt, quant in (("mxfp8", mxfp8_quant_recip), ("nvfp4", QUANT["nvfp4"])):
    fn = lambda x, w, q=quant: q(rms(x, w))
    for upcast in (True, False):
        for label, patches, persistent in (("persistent", {}, True), ("persistent_cd", {"coordinate_descent_tuning": True}, True), ("looped_cd", {"coordinate_descent_tuning": True}, False)):
            torch._dynamo.reset(); metrics.reset()
            with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, "triton.codegen_upcast_to_fp32": upcast, **patches}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
                us = graph_bench(lambda: compiled(x, w))
            k = [s for s in srcs if "@triton_heuristics" in s][0]; body = k[k.index("@triton.jit"):]
            loads = [l.strip()[:90] for l in body.splitlines() if "tl.load(in_ptr0" in l]
            key = (fmt, label)
            if upcast: refs[key] = out; eq = "-"
            else: eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, refs[key]))
            print(f"{fmt} upcast={upcast!s:5s} {label:14s}: {us:6.2f}us exact_vs_upcast={eq} | {loads[0] if loads else ''}", flush=True)
