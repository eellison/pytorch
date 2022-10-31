
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

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 0.0
    tmp3 = tl.where(tmp0, tmp1, tmp2)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(arg1_1, arg0_1, buf0, 12582912, grid=grid(12582912), stream=stream0)
    del arg0_1
    del arg1_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float32)
    arg1_1 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.bool)
    print_performance(lambda: call([arg0_1, arg1_1]))
