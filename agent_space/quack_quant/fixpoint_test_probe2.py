import torch, os
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_code
from quack.blockscaled import quantize as qz
M, K, G = 2048, 3072, 32
def f_flat1d(x):
    blocks = x.view(M, K // G, G)
    amax = blocks.abs().amax(dim=-1).unsqueeze(-1)
    scale = torch.clamp(amax.float() / 6.0, min=1e-12)
    codes = (blocks.float() / scale).reshape(M, K).clamp(0, 15).to(torch.uint8)
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(M, K // 2), scale.squeeze(-1)
def f_flat1d_e8m0(x):
    blocks = x.view(M, K // G, G)
    amax = blocks.abs().amax(dim=-1).unsqueeze(-1).float()
    sb = qz._compute_e8m0_scale_floor(amax, 2)
    scale = torch.clamp((sb.to(torch.int32) << 23).view(torch.float32), min=2.0 ** -126)
    codes = (blocks.float() / scale).reshape(M, K).clamp(0, 15).to(torch.uint8)
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(M, K // 2), sb.squeeze(-1)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
for cfg_name, patches in (("test-cfg", {"split_reductions": False, "loop_ordering_after_fusion": True}), ("plain", {})):
    for name, fn in (("f_flat1d", f_flat1d), ("f_flat1d_e8m0", f_flat1d_e8m0), ("quack to_mxfp4", qz.to_mxfp4)):
        torch._dynamo.reset(); metrics.reset()
        with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, **patches}):
            out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
        print(f"{cfg_name:8s} {name:16s}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}", flush=True)
