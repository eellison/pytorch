import sys, torch, statistics
import torch.nn.functional as F
from torch._inductor import config, metrics, inductor_prims
from torch._inductor.utils import run_and_get_code
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
RECIP = "{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}"
def recip_ue8m0(s): return inline_asm_elementwise(s.to(torch.int32), asm_str=RECIP, constraints="=f,r", dtype=torch.float32, is_pure=True, pack=1)
def mxfp8_colwise(x):
    M, K = x.shape
    g = x.view(M // 32, 32, K)
    amax = g.abs().float().amax(dim=1)                      # (M/32, K)
    scale = inductor_prims.cvt_e8m0_rceil((amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny))
    q = (g.float() * recip_ue8m0(scale).unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(M, K)
    return q, scale
def rms(x, w): return F.rms_norm(x, (x.shape[-1],), w, eps=1e-6)
M, K = int(sys.argv[1]), int(sys.argv[2]); which = sys.argv[3]
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fn, args = {"colwise": (mxfp8_colwise, (x,)), "rms+colwise": (lambda x, w: mxfp8_colwise(rms(x, w)), (x, w))}[which]
torch._dynamo.reset(); metrics.reset()
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), *args)
names = [l[4:l.index("(")][:40] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
print(f"RESULT {which} {M}x{K}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} {names}")
