import torch
import torch._inductor
import triton

from torch._inductor import config

config.max_autotune_gemm_backends = "TRITON"

# @torch.compile(mode="max-autotune")
# def fn(x, y, index):
#     return (x[index] + 1).to(torch.float) @ y


def fn(x, y):
    return x.to(dtype=torch.float16) @ y

fn_c = torch.compile(mode="max-autotune")(fn)
x = torch.rand([4096, 4096], dtype=float, device="cuda")
y = torch.rand([4096, 4096], device="cuda", dtype=torch.float16)

torch.testing.assert_close(fn_c(x, y), fn(x, y))

print(triton.testing.do_bench(lambda: fn(x, y)))

torch._dynamo.reset()

from torch._inductor import config
config.prologue_fusion = False
print(triton.testing.do_bench(lambda: fn(x, y)))


M = K = N = 2048
