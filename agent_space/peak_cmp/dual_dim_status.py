import sys, torch, contextlib, statistics
import torch.nn.functional as F
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch._inductor.choices import InductorChoices
FP8 = torch.float8_e4m3fn
PACK = "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
def nvfp4_rowwise(x):
    M, K = x.shape
    xg = x.view(M, K // 16, 16); amax = xg.float().abs().amax(dim=-1)
    scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(FP8)
    xp = xg.view(M, K // 16, 8, 2); inv = scale.float().reciprocal().unsqueeze(-1)
    packed = inline_asm_elementwise(xp[..., 0].float() * inv, xp[..., 1].float() * inv, asm_str=PACK, constraints="=r,f,f", dtype=torch.int32, is_pure=True, pack=1)
    return packed.to(torch.uint8).view(M, K // 2), scale
def nvfp4_colwise(x):
    M, K = x.shape
    xg = x.view(M // 16, 16, K); amax = xg.float().abs().amax(dim=1)
    scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(FP8)
    xp = x.view(M // 16, 8, 2, K); inv = scale.float().reciprocal().unsqueeze(1)
    packed = inline_asm_elementwise(xp[:, :, 0, :].float() * inv, xp[:, :, 1, :].float() * inv, asm_str=PACK, constraints="=r,f,f", dtype=torch.int32, is_pure=True, pack=1)
    return packed.to(torch.uint8).view(M // 2, K), scale
def rms(x, w): return F.rms_norm(x, (x.shape[-1],), w, eps=1e-6)
def graph_bench(f, warmup=10, samples=30, calls=50):
    for _ in range(warmup): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
class _P(InductorChoices):
    @staticmethod
    def should_use_persistent_reduction(*a, **k): return True
cases = {"rms+rowwise": (lambda x, w: nvfp4_rowwise(rms(x, w)), 2), "rms+colwise": (lambda x, w: nvfp4_colwise(rms(x, w)), 2),
         "rms+dual": (lambda x, w: (*nvfp4_rowwise(rms(x, w)), *nvfp4_colwise(rms(x, w))), 2), "dual (no rms)": (lambda x: (*nvfp4_rowwise(x), *nvfp4_colwise(x)), 1)}
for M, K in ((8192, 4096), (16384, 7168)):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    for name, (fn, nargs) in cases.items():
        args = (x, w)[:nargs]
        for label, patches, persistent in (("default", {}, False), ("cd", {"coordinate_descent_tuning": True}, False)):
            torch._dynamo.reset(); metrics.reset()
            with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **patches}), (V.set_choices_handler(_P()) if persistent else contextlib.nullcontext()):
                compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, *args); torch.cuda.synchronize()
                us = graph_bench(lambda: compiled(*args))
            names = [l[4:l.index("(")][:18] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
            print(f"{M}x{K} {name:14s} {label:8s}: {us:7.1f}us kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} {names}", flush=True)
