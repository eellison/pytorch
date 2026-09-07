import re, os, sys, statistics, importlib.util, torch
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
src = open("agent_space/peak_cmp/_gen_mxfp8_kernel.py").read()
name = re.search(r"def (triton_\w+)\(", src).group(1)
variants = {"fp32_resident": src,
            "bf16_load_single_upcast": src.replace("tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), None, eviction_policy='evict_first').to(tl.float32)", "tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), None, eviction_policy='evict_first')")}
assert variants["bf16_load_single_upcast"] != src
def graph_bench(f, warmup=20, samples=50, calls=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
ref = {}
for vname, text in variants.items():
    path = os.path.abspath(f"agent_space/peak_cmp/_gen_var_{vname}.py"); open(path, "w").write(text)
    spec = importlib.util.spec_from_file_location(f"_gen_var_{vname}", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    kern = getattr(mod, name)
    for nw in (2, 4, 8):
        q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sc = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
        ck = kern[(M,)](x, w, sc, q, M, K, XBLOCK=1, num_warps=nw); torch.cuda.synchronize()
        us = graph_bench(lambda: kern[(M,)](x, w, sc, q, M, K, XBLOCK=1, num_warps=nw))
        key = nw
        if vname == "fp32_resident": ref[key] = (q.clone(), sc.clone()); eq = "-"
        else: eq = torch.equal(q, ref[key][0]) and torch.equal(sc, ref[key][1])
        print(f"{vname:24s} nw={nw}: {us:6.2f}us regs={ck.n_regs} spills={ck.n_spills} exact_vs_fp32={eq}", flush=True)
