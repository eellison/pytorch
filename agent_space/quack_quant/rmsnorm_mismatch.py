import torch, torch.nn.functional as F
from torch._inductor import config
from quack.blockscaled import quantize as qz
M, K = 2048, 3072
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
ref = F.rms_norm(x, (K,), w)
for label, patches in (("default", {}), ("emulate_precision_casts", {"emulate_precision_casts": True})):
    torch._dynamo.reset()
    with config.patch({"triton.cudagraphs": False, "fx_graph_cache": False, **patches}):
        y = torch.compile(lambda x, w: F.rms_norm(x, (K,), w), fullgraph=True)(x, w)
    diff = (y.float() - ref.float()).abs(); n = (y != ref).sum().item()
    ulp = ((y.view(torch.int16).int() - ref.view(torch.int16).int()).abs()).max().item()
    q_ref = qz.to_mx(ref)[0]; q_y = qz.to_mx(y)[0]
    print(f"rmsnorm compiled[{label}] vs eager: {n}/{y.numel()} bf16 elems differ ({100*n/y.numel():.3f}%), max|diff|={diff.max().item():.3e}, max ulp={ulp}; mxfp8 codes differ: {(q_ref.view(torch.uint8) != q_y.view(torch.uint8)).sum().item()}")
