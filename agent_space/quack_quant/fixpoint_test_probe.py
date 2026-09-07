import torch, sys
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_code
M, K, G = int(__import__("os").environ.get("PM", 2048)), int(__import__("os").environ.get("PK", 3072)), 32
def f_test(x):
    blocks = x.view(M, K // G, G)
    amax = blocks.abs().amax(dim=-1).unsqueeze(-1)
    scale = torch.clamp(amax.float() / 6.0, min=1e-12)
    codes = (blocks.float() / scale).reshape(M, K).clamp(0, 15).to(torch.uint8)
    pairs = codes.view(M, K // 2, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4), scale.squeeze(-1)
from quack.blockscaled import quantize as qz
def f_v4(x):
    data_hp = x.reshape(M, K // G, G)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1).to(torch.float32)
    sb = qz._compute_e8m0_scale_floor(max_abs, qz.F4_E2M1_MAX_POW2)
    scale_fp32 = torch.clamp((sb.to(torch.int32) << qz.MBITS_F32).view(torch.float32), min=qz.F32_MIN_NORMAL)
    data_lp = data_hp.to(torch.float32) / scale_fp32
    codes = data_lp.reshape(M, K).clamp(0, 15).to(torch.uint8)
    c = codes.view(M, K // 2, 2)
    return c[..., 0] | (c[..., 1] << 4), sb.view(torch.float8_e8m0fnu).squeeze(-1)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
for name, fn in (("f_test", f_test), ("f_v4", f_v4)):
    torch._dynamo.reset(); metrics.reset()
    with config.patch({"triton.nested_reduction": True, "split_reductions": False, "loop_ordering_after_fusion": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        out, srcs = run_and_get_code(torch.compile(fn), x)
    names = [l[4:l.index("(")][:30] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
    print(f"{name}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} {names}", flush=True)
