
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    buf0 = empty_strided((2048, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_2, as_strided(primals_3, (2048, 768), (768, 1)), as_strided(primals_1, (768, 768), (1, 768)), beta=1, alpha=1, out=buf0)
    del primals_2
    return (as_strided(buf0, (4, 512, 768), (393216, 768, 1)), as_strided(primals_3, (2048, 768), (768, 1)), as_strided(primals_1, (768, 768), (768, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3]))
