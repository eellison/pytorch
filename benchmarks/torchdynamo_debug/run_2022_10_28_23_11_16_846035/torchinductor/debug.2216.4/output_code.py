
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


kernel0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*u8', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = (xindex // 262144)
    x3 = (xindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x2)), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.uint8)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp3, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf0 = empty_strided((4, 1, 512, 512), (262144, 1048576, 512, 1), device='cuda', dtype=torch.uint8)
    stream0 = get_cuda_stream(0)
    kernel0.run(arg0_1, buf0, 1048576, grid=grid(1048576), stream=stream0)
    del arg0_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 512), (512, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))
