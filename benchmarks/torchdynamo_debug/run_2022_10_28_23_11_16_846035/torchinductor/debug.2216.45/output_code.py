
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

@reduction(size_hints=[2048, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp6 = tmp4 + tmp5
        tmp7 = 768
        tmp8 = tmp3 / tmp7
        tmp9 = tmp6 - tmp8
        tl.store(out_ptr1 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, xmask & rmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(out_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = tmp10 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp14 = tl.load(out_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp15 = 768
        tmp16 = tmp12 / tmp15
        tmp17 = 1e-07
        tmp18 = tmp16 + tmp17
        tmp19 = tl.sqrt(tmp18)
        tmp20 = tmp14 / tmp19
        tmp21 = tmp13 * tmp20
        tmp23 = tmp21 + tmp22
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp23, xmask & rmask)
    tmp24 = 768
    tmp25 = tmp12 / tmp24
    tmp26 = 1e-07
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp28, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    buf1 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    buf3 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    buf4 = empty_strided((4, 512, 1), (512, 1, 2048), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_3, primals_4, primals_1, primals_2, buf1, buf3, buf4, 2048, 768, grid=grid(2048), stream=stream0)
    del primals_2
    del primals_3
    del primals_4
    return (buf3, primals_1, buf1, buf4, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    primals_4 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4]))
