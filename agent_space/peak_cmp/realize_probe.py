import sys, torch, contextlib, re, statistics
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fn = lambda x, w: QUANT["nvfp4"](rms(x, w))
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
for label, extra, persistent in (("baseline persistent", {}, True), ("realize_opcount=1 persistent", {"realize_opcount_threshold": 1}, True), ("realize_opcount=1 persistent cd", {"realize_opcount_threshold": 1, "coordinate_descent_tuning": True}, True), ("realize_opcount=1 looped cd", {"realize_opcount_threshold": 1, "coordinate_descent_tuning": True}, False)):
    torch._dynamo.reset(); metrics.reset()
    with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **extra}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w); torch.cuda.synchronize()
        us = graph_bench(lambda: compiled(x, w))
    k = [s for s in srcs if "@triton_heuristics" in s][0]
    body = k[k.index("@triton.jit"):]
    wrapper = [s for s in srcs if "def call(" in s or "class Runner" in s][0]
    allocs = len(re.findall(r"empty_strided_cuda\(", wrapper))
    if ref is None: ref = out
    eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, ref))
    print(f"== {label}: {us:.2f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} allocs={allocs} exact_vs_baseline={eq} stores={body.count('tl.store(')}")
    for line in body.splitlines():
        l = line.strip()
        if "tl.split(" in l or "tl.store(" in l or "broadcast_to(" in l: print("     ", l[:150])
