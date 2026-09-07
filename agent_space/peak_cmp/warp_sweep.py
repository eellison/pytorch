import sys, re, os, statistics, importlib.util, torch, contextlib
fmt = os.environ.get("FMT", "mxfp8")
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
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
for K in (2048, 4096, 8192):
    M = 32 * 1024 * 1024 // K
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    fn = lambda x, w: QUANT[fmt](rms(x, w))
    torch._dynamo.reset()
    with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}), V.set_choices_handler(_P()):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False); ref, srcs = run_and_get_code(compiled, x, w)
        default_us = graph_bench(lambda: compiled(x, w))
    src = [s for s in srcs if "@triton_heuristics" in s][0]
    body = src[src.index("@triton.jit"):src.index("''', device_str")]
    name = re.search(r"def (triton_\w+)\(", body).group(1)
    modpath = os.path.abspath(f"agent_space/peak_cmp/_gen_{fmt}_{K}.py")
    open(modpath, "w").write("import triton\nimport triton.language as tl\nfrom torch._inductor.runtime import triton_helpers\nfrom torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n" + body)
    spec = importlib.util.spec_from_file_location(f"_gen_{fmt}_{K}", modpath); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); kern = getattr(mod, name)
    sig = re.search(r"def triton_\w+\((.*?)\):", body).group(1).split(",")
    nout = sum(1 for a in sig if a.strip().startswith("out_ptr"))
    print(f"{fmt} {M}x{K}: default-autotuned persistent {default_us:6.2f}us", flush=True)
    import itertools
    order = None
    for nw in (1, 2, 4, 8, 16):
        done = False
        for perm in ([order] if order else itertools.permutations(range(len(ref)))):
            outs = [torch.empty_like(ref[i]) for i in perm]
            try:
                ck = kern[(M,)](x, w, *outs, M, K, XBLOCK=1, num_warps=nw); torch.cuda.synchronize()
            except Exception as ex:
                err = str(ex); continue
            order = perm
            us = graph_bench(lambda: kern[(M,)](x, w, *outs, M, K, XBLOCK=1, num_warps=nw))
            exact = all(torch.equal(o.view(torch.uint8), ref[i].view(torch.uint8)) for o, i in zip(outs, perm))
            print(f"   nw={nw:2d}: {us:6.2f}us regs={ck.n_regs} spills={ck.n_spills} exact_vs_default={exact}", flush=True)
            done = True; break
        if not done: print(f"   nw={nw:2d}: ERROR {err[:100]}", flush=True)
