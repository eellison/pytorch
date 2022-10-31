
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

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.load(in_ptr1 + (r1 + (768*tmp0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp1, xmask & rmask)
    x2 = xindex % 512
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp2 = tl.load(out_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp4 = tmp2 + tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(out_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (r1 + (768*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp8 = tmp6 + tmp7
        tmp9 = 768
        tmp10 = tmp5 / tmp9
        tmp11 = tmp8 - tmp10
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.reshape(tl.sum(_tmp13, 1), [XBLOCK, 1])
    tmp29 = tl.load(in_ptr5 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp15 = tl.load(out_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr2 + (r1 + (768*x2)), xmask & rmask, eviction_policy='evict_last')
        tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp17 = tmp15 + tmp16
        tmp18 = 768
        tmp19 = tmp5 / tmp18
        tmp20 = tmp17 - tmp19
        tmp21 = tmp13 / tmp18
        tmp22 = 1e-07
        tmp23 = tmp21 + tmp22
        tmp24 = tl.sqrt(tmp23)
        tmp25 = tmp20 / tmp24
        tmp26 = tmp14 * tmp25
        tmp28 = tmp26 + tmp27
        tmp30 = tmp28 * tmp29
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp30, xmask & rmask)
    tmp31 = 768
    tmp32 = tmp5 / tmp31
    tmp33 = tmp13 / tmp31
    tmp34 = 1e-07
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tl.store(out_ptr4 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp32, xmask)
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp36, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    buf1 = empty_strided((1, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_5, primals_4, buf1, 393216, grid=grid(393216), stream=stream0)
    del primals_4
    buf0 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    buf4 = empty_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    buf5 = empty_strided((4, 512, 1), (512, 1, 2048), device='cuda', dtype=torch.float32)
    buf6 = empty_strided((4, 512, 1), (512, 1, 2048), device='cuda', dtype=torch.float32)
    kernel1.run(primals_6, primals_3, buf1, primals_1, primals_2, primals_7, buf0, buf4, buf5, buf6, 2048, 768, grid=grid(2048), stream=stream0)
    del primals_2
    del primals_3
    return (buf4, primals_1, primals_7, buf0, buf1, buf5, buf6, as_strided(primals_5, (512, ), (1, )), as_strided(primals_6, (2048, ), (1, )), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((50265, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_4 = rand_strided((512, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_5 = rand_strided((1, 512), (512, 1), device='cuda', dtype=torch.int64)
    primals_6 = rand_strided((4, 512), (512, 1), device='cuda', dtype=torch.int64)
    primals_7 = rand_strided((4, 512), (512, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7]))
