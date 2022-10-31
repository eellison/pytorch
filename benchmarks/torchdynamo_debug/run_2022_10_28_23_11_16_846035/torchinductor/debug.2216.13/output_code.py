
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp4 < 0, -1, 1)
    tmp6 = tl.where(tmp4 == 0, 0, tmp5)
    tmp7 = 1.0
    tmp8 = tl.abs(tmp4)
    tmp9 = 0.3275911
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10 + tmp7
    tmp12 = 1 / tmp11
    tmp13 = tmp12 * tmp7
    tmp14 = 1.061405429
    tmp15 = tmp13 * tmp14
    tmp16 = -1.453152027
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 * tmp13
    tmp19 = 1.421413741
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20 * tmp13
    tmp22 = -0.284496736
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 * tmp13
    tmp25 = 0.254829592
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 * tmp13
    tmp28 = -tmp8
    tmp29 = tmp28 * tmp8
    tmp30 = tl.exp(tmp29)
    tmp31 = tmp27 * tmp30
    tmp32 = tmp7 - tmp31
    tmp33 = tmp6 * tmp32
    tmp34 = 1
    tmp35 = tmp33 + tmp34
    tmp36 = tmp2 * tmp35
    tmp37 = tmp35 * tmp1
    tmp38 = tmp0 * tmp0
    tmp39 = -0.5
    tmp40 = tmp38 * tmp39
    tmp41 = tl.exp(tmp40)
    tmp42 = 0.3989422804014327
    tmp43 = tmp41 * tmp42
    tmp44 = tmp0 * tmp43
    tmp45 = tmp37 + tmp44
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp36, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp45, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    buf0 = empty_strided((2048, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_2, as_strided(primals_3, (2048, 768), (768, 1)), as_strided(primals_1, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf0)
    del primals_2
    buf1 = empty_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda', dtype=torch.float32)
    buf2 = empty_strided((4, 512, 3072), (1572864, 3072, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(buf0, buf1, buf2, 6291456, grid=grid(6291456), stream=stream0)
    return (buf1, as_strided(primals_3, (2048, 768), (768, 1)), buf2, as_strided(primals_1, (3072, 768), (768, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3]))
