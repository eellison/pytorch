
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

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    buf0 = empty_strided((4, 512), (512, 1), device='cuda', dtype=torch.int64)
    stream0 = get_cuda_stream(0)
    kernel0.run(buf0, 2048, grid=grid(2048), stream=stream0)
    buf1 = empty_strided((4, 512), (512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf1, 2048, grid=grid(2048), stream=stream0)
    return (buf0, buf1, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    print_performance(lambda: call([]))
