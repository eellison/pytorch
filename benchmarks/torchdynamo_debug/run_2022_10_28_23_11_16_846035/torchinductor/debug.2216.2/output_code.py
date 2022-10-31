
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
seed_cuda_0 = None  # 12bf87036c8e625335a9db42dcf50de0c1ec952294785adced537424d5733e17


kernel0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.9
    tmp6 = tmp4 < tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp10 = (tmp9 != 0)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    torch.randint(2**31, size=(), dtype=torch.int64, out=seed_cuda_0)
    buf1 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.bool)
    stream0 = get_cuda_stream(0)
    kernel0.run(seed_cuda_0, buf1, 1572864, grid=grid(1572864), stream=stream0)
    return (buf1, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    seed_cuda_0 = rand_strided((), (), device='cuda', dtype=torch.int64)
    arg0_1 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))
