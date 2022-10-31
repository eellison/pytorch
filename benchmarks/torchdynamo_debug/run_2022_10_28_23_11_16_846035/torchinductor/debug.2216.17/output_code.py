
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

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*x2) + (2304*x1) + (1179648*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 64.0
    tmp4 = 1
    tmp5 = tmp3 * tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = tmp2 / tmp6
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp7, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4096, 512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 3072
    ynumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    y3 = yindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + (192*x1) + (2304*y3) + (1179648*x2)), xmask & ymask)
    tl.store(out_ptr0 + (y3 + (512*x4) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


kernel2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex % 768
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (192*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x5 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    buf0 = empty_strided((2048, 2304), (2304, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(primals_4, (2048, 768), (768, 1)), as_strided(primals_3, (768, 2304), (1, 768)), out=buf0)
    buf1 = empty_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(buf0, primals_1, buf1, 1572864, grid=grid(1572864), stream=stream0)
    del primals_1
    buf2 = empty_strided((4, 12, 64, 512), (393216, 32768, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf0, buf2, 3072, 512, grid=grid(3072, 512), stream=stream0)
    buf3 = empty_strided((48, 512, 512), (262144, 512, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf1, (48, 512, 64), (32768, 64, 1)), as_strided(buf2, (48, 64, 512), (32768, 512, 1)), out=buf3)
    buf4 = empty_strided((4, 12, 512, 64), (393216, 64, 768, 1), device='cuda', dtype=torch.float32)
    kernel2.run(buf0, primals_2, buf4, 1572864, grid=grid(1572864), stream=stream0)
    del buf0
    del primals_2
    return (as_strided(buf3, (4, 12, 512, 512), (3145728, 262144, 512, 1)), buf4, as_strided(primals_4, (2048, 768), (768, 1)), as_strided(buf1, (48, 64, 512), (32768, 1, 64)), as_strided(buf2, (48, 512, 64), (32768, 1, 512)), as_strided(primals_3, (2304, 768), (768, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((2304, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_4 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4]))
