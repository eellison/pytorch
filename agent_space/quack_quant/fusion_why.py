import sys, torch
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz
M, K, BS = 2048, 3072, 32
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
def scale_and_hp(x):
    data_hp = x.reshape(M, K // BS, BS)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1).to(torch.float32)
    sb = qz._compute_e8m0_scale_floor(max_abs, qz.F4_E2M1_MAX_POW2)
    scale_fp32 = torch.clamp((sb.to(torch.int32) << qz.MBITS_F32).view(torch.float32), min=qz.F32_MIN_NORMAL)
    return data_hp.to(torch.float32) / scale_fp32, sb.view(torch.float8_e8m0fnu).squeeze(-1)
def v4(x):
    data_lp, scale = scale_and_hp(x)
    codes = data_lp.reshape(M, K).clamp(0, 15).to(torch.uint8)
    c = codes.view(M, K // 2, 2)
    return c[..., 0] | (c[..., 1] << 4), scale
def v5(x):
    data_lp, scale = scale_and_hp(x)
    c = data_lp.view(M, K // BS, BS // 2, 2).clamp(0, 15).to(torch.uint8)
    return (c[..., 0] | (c[..., 1] << 4)).view(M, K // 2), scale
fn = {"v4": v4, "v5": v5}[sys.argv[1]]
torch._dynamo.reset(); metrics.reset()
with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
print(f"RESULT kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
