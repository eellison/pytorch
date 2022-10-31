
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 512],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*u8', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x2 = (xindex // 6144)
    x4 = xindex
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x0) + (262144*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr1 + (r3 + (512*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = (tmp0 != 0)
        tmp2 = tmp1 == 0
        tmp3 = -3.4028234663852886e+38
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        _tmp6 = tl.where(xmask & rmask & (_tmp6 < tmp5), tmp5, _tmp6)
    tmp6 = tl.reshape(tl.max(_tmp6, 1), [XBLOCK, 1])
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp7 = tl.load(in_ptr0 + (r3 + (512*x0) + (262144*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r3 + (512*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp8 = (tmp7 != 0)
        tmp9 = tmp8 == 0
        tmp10 = -3.4028234663852886e+38
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
    tmp15 = tl.reshape(tl.sum(_tmp15, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp16 = tl.load(in_ptr0 + (r3 + (512*x0) + (262144*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp21 = tl.load(in_ptr1 + (r3 + (512*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp17 = (tmp16 != 0)
        tmp18 = tmp17 == 0
        tmp19 = 0.0
        tmp20 = -3.4028234663852886e+38
        tmp22 = tl.where(tmp18, tmp20, tmp21)
        tmp23 = tmp22 - tmp6
        tmp24 = tl.exp(tmp23)
        tmp25 = tmp24 / tmp15
        tmp26 = tl.where(tmp18, tmp19, tmp25)
        tl.store(out_ptr2 + (r3 + (512*x4) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, xmask & rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf2 = empty_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(arg1_1, arg0_1, buf2, 24576, 512, grid=grid(24576), stream=stream0)
    del arg0_1
    del arg1_1
    return (buf2, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda', dtype=torch.float32)
    arg1_1 = rand_strided((4, 1, 512, 512), (262144, 1048576, 512, 1), device='cuda', dtype=torch.uint8)
    print_performance(lambda: call([arg0_1, arg1_1]))
