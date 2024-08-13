import torch
import torch._inductor
import triton

from torch._inductor import config

config.max_autotune_gemm_backends = "TRITON"

# @torch.compile(mode="max-autotune")
# def fn(x, y, index):
#     return (x[index] + 1).to(torch.float) @ y


@torch.compile(mode="max-autotune")
def fn(x, y, index):
    return (x + 1).to(torch.float) @ y


x = torch.rand([4096, 4096], dtype=torch.bfloat16, device="cuda")
y = torch.rand([4096, 4096], device="cuda")
index = torch.randperm(4096, device="cuda")

print(triton.testing.do_bench(lambda: fn(x, y, index)))

# with torch._inductor.utils.fresh_inductor_cache():
#     print(triton.testing.do_bench(lambda: fn(x, y, index)))


M = K = N = 2048
