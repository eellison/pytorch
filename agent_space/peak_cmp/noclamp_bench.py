import statistics, torch, importlib.util, os
spec = importlib.util.spec_from_file_location("hk", os.path.abspath("agent_space/peak_cmp/_hk_kernels.py")); hk = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk)
def graph_bench(fn, warmup=20, samples=50, calls=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): fn()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
# include large-magnitude and special values to exercise saturation paths
x[0, :64] = 3.0e38; x[1, :32] = float("inf"); x[2, :8] = float("nan"); x[3] = 0.0
ref = {}
for name, kern in (("with_clamp", hk.k_fp32_resident), ("no_clamp", hk.k_noclamp)):
    for nw in (2, 4):
        q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
        ck = kern[(M,)](x, w, sf, q, K=K, XBLOCK=1, R0_BLOCK=K, num_warps=nw); torch.cuda.synchronize()
        us = graph_bench(lambda: kern[(M,)](x, w, sf, q, K=K, XBLOCK=1, R0_BLOCK=K, num_warps=nw))
        if name == "with_clamp": ref[nw] = (q.clone(), sf.clone()); eq = "-"
        else:
            qa, qb = q.view(torch.uint8), ref[nw][0].view(torch.uint8)
            nan_both = torch.isnan(q.float()) & torch.isnan(ref[nw][0].float())
            eq = f"bytes equal (ignoring NaN-vs-NaN)={torch.equal(qa[~nan_both], qb[~nan_both])}, differing={int((qa != qb).sum())}, sf equal={torch.equal(sf, ref[nw][1])}"
        print(f"{name:10s} nw={nw}: {us:6.2f}us regs={ck.n_regs} {eq}", flush=True)
