import sys, re, statistics, torch, contextlib, triton, triton.language as tl
sys.argv = ["x", "rmsnorm_mxfp8", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
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
fn = lambda x, w: mxfp8_quant_recip(rms(x, w))
torch._dynamo.reset()
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}), V.set_choices_handler(_P()):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False); ref, srcs = run_and_get_code(compiled, x, w)
src = [s for s in srcs if "@triton_heuristics" in s][0]
body = src[src.index("@triton.jit"):src.index("''', device_str")]
name = re.search(r"def (triton_\w+)\(", body).group(1)
import importlib.util, os
modpath = os.path.abspath("agent_space/peak_cmp/_gen_mxfp8_kernel.py")
open(modpath, "w").write("import triton\nimport triton.language as tl\nfrom torch._inductor.runtime import triton_helpers\nfrom torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n" + body)
spec = importlib.util.spec_from_file_location("_gen_mxfp8_kernel", modpath); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
kern = getattr(mod, name)
sig = re.search(r"def triton_\w+\((.*?)\):", body).group(1)
print("signature:", sig[:120])
def graph_bench(f, warmup=20, samples=50, calls=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
for XBLOCK in (1, 2, 4):
    for nw in (2, 4, 8, 16):
        q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sc = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
        grid = (M // XBLOCK,)
        try:
            ck = kern[grid](x, w, sc, q, M, K, XBLOCK=XBLOCK, num_warps=nw); torch.cuda.synchronize()
            us = graph_bench(lambda: kern[grid](x, w, sc, q, M, K, XBLOCK=XBLOCK, num_warps=nw))
            ok = torch.equal(q.view(torch.uint8), ref[0].view(torch.uint8)) and torch.equal(sc, ref[1].view(torch.uint8).reshape(-1))
            print(f"XBLOCK={XBLOCK} nw={nw:2d}: {us:6.2f}us regs={ck.n_regs} spills={ck.n_spills} smem={ck.metadata.shared} exact={ok}", flush=True)
        except Exception as ex:
            print(f"XBLOCK={XBLOCK} nw={nw:2d}: ERROR {str(ex)[:100]}")
