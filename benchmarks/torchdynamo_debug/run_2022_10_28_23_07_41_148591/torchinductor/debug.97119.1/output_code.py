
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.1111111111111112
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp0 * tmp8
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


kernel2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64028672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = -1.0
    tl.store(out_ptr0 + (tmp0 + (250112*x0) + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


kernel5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 262144],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 250112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (250112*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = 256.0
        tmp3 = tmp1 / tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp5, xmask)
''')


kernel6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64028672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 250112)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr3 + (x2), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp3 = 256.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 * tmp4
    tmp7 = tl.exp(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp5 - tmp9
    tmp11 = tmp0 + tmp10
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


kernel7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr3 + (r2 + (128*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp8 = tmp6 * tmp7
        tmp9 = tmp5 * tmp8
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp9, _tmp10)
    tmp10 = tl.reshape(tl.sum(_tmp10, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp10, xmask)
''')


kernel8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[512, 2],
              reduction_hint=ReductionHint.OUTER_TINY,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


kernel9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr3 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 * tmp8
        _tmp10 = tl.where(xmask & rmask, _tmp10 + tmp9, _tmp10)
    tmp10 = tl.reshape(tl.sum(_tmp10, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp10, xmask)
''')


kernel10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp10 = tl.load(in_ptr4 + (x1), xmask)
    tmp18 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = -0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp8 * tmp8
    tmp14 = tmp13 * tmp8
    tmp15 = tmp12 * tmp14
    tmp16 = 512
    tmp17 = tmp15 / tmp16
    tmp19 = 2.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 * tmp20
    tmp22 = tmp9 + tmp21
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp22, xmask)
''')


kernel11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tmp5 * tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask)
    tmp8 = tl.load(in_ptr3 + (x0), xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = 1.0
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = 0.7978845608028654
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp18, xmask)
''')


kernel14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp10 = tl.load(in_ptr3 + (x0), xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask)
    tmp1 = 0.044715
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = 3.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = 0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp8 + tmp22
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


kernel15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr3 + (r2 + (128*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.reshape(tl.sum(_tmp7, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp7, xmask)
''')


kernel16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.reshape(tl.sum(_tmp7, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp7, xmask)
''')


kernel17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp9 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 * tmp6
    tmp13 = tmp12 * tmp6
    tmp14 = tmp11 * tmp13
    tmp15 = 512
    tmp16 = tmp14 / tmp15
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp8 + tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 6
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (49152*x3)), xmask)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 4521984 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 4521984 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 128
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (49152*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2) + (8192*x1) + (49152*x3)), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


kernel22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256, 512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 256
    ynumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    y2 = yindex % 64
    y3 = (yindex // 64)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (y2 + (64*x0) + (8192*y3) + (49152*x1)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*y4) + (49152*x1)), xmask & ymask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y4 + (384*x5) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp2, xmask & ymask)
''')


kernel23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x2 = (xindex // 384) % 128
    x3 = (xindex // 49152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (49152*x3)), xmask)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp5, xmask)
''')


kernel25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp5, xmask)
''')


kernel26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = -0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 * tmp4
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 * tmp11
    tmp13 = 512
    tmp14 = tmp12 / tmp13
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tmp6 + tmp18
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


kernel27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 4325376 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 4325376 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr4 + (r2 + (128*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
    tmp9 = tl.reshape(tl.sum(_tmp9, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp9, xmask)
''')


kernel30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
    tmp9 = tl.reshape(tl.sum(_tmp9, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp9, xmask)
''')


kernel31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 * tmp8
    tmp15 = tmp14 * tmp8
    tmp16 = tmp13 * tmp15
    tmp17 = 512
    tmp18 = tmp16 / tmp17
    tmp20 = 2.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp10 + tmp22
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


kernel32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 4128768 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 4128768 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 3932160 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 3932160 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp9 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 * tmp6
    tmp13 = tmp12 * tmp6
    tmp14 = tmp11 * tmp13
    tmp15 = 512
    tmp16 = tmp14 / tmp15
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp8 + tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 3735552 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 3735552 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 3538944 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 3538944 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr3 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 * tmp8
    tmp15 = tmp14 * tmp8
    tmp16 = tmp13 * tmp15
    tmp17 = 512
    tmp18 = tmp16 / tmp17
    tmp20 = 2.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp10 + tmp22
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


kernel42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp10 = tl.load(in_ptr3 + (x0), xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask)
    tmp1 = 0.044715
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = 3.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = 0.5
    tmp22 = tmp20 * tmp21
    tmp23 = tmp8 + tmp22
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


kernel43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 3342336 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 3342336 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    tmp9 = tl.load(in_ptr5 + (x0), xmask)
    tmp11 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tl.load(in_ptr7 + (x0), xmask)
    tmp15 = tl.load(in_ptr8 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


kernel46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp4 = tl.load(in_ptr3 + (x1), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = -0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp4 * tmp4
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 * tmp11
    tmp13 = 512
    tmp14 = tmp12 / tmp13
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tmp6 + tmp18
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


kernel47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 3145728 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp21 = tl.load(in_ptr5 + (x2), xmask)
    tmp24 = tl.load(in_ptr6 + (x2), xmask)
    tmp26 = tl.load(in_ptr7 + (x1), xmask)
    tmp34 = tl.load(in_ptr8 + (x2), xmask)
    tmp37 = tl.load(in_ptr9 + (x2), xmask)
    tmp39 = tl.load(in_ptr10 + (x1), xmask)
    tmp2 = 3932160 + x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = 3538944 + x2
    tmp18 = tl.rand(tmp1, tmp17)
    tmp19 = tmp18 > tmp4
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp9
    tmp25 = tmp23 * tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp25 - tmp27
    tmp29 = tmp16 + tmp28
    tmp30 = 3145728 + x2
    tmp31 = tl.rand(tmp1, tmp30)
    tmp32 = tmp31 > tmp4
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp9
    tmp38 = tmp36 * tmp37
    tmp40 = tmp37 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp29 + tmp41
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp42, xmask)
''')


kernel49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 3145728 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask)
    tmp9 = tl.load(in_ptr5 + (x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 * tmp6
    tmp13 = tmp12 * tmp6
    tmp14 = tmp11 * tmp13
    tmp15 = 512
    tmp16 = tmp14 / tmp15
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp8 + tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 2949120 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 2949120 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 2752512 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 2752512 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 * tmp8
    tmp15 = tmp14 * tmp8
    tmp16 = tmp13 * tmp15
    tmp17 = 512
    tmp18 = tmp16 / tmp17
    tmp20 = 2.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp10 + tmp22
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


kernel56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 2555904 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 2555904 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 2359296 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 2359296 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 2162688 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 2162688 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 1966080 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp21 = tl.load(in_ptr5 + (x2), xmask)
    tmp24 = tl.load(in_ptr6 + (x2), xmask)
    tmp26 = tl.load(in_ptr7 + (x1), xmask)
    tmp34 = tl.load(in_ptr8 + (x2), xmask)
    tmp37 = tl.load(in_ptr9 + (x2), xmask)
    tmp39 = tl.load(in_ptr10 + (x1), xmask)
    tmp2 = 2752512 + x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = 2359296 + x2
    tmp18 = tl.rand(tmp1, tmp17)
    tmp19 = tmp18 > tmp4
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp9
    tmp25 = tmp23 * tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp25 - tmp27
    tmp29 = tmp16 + tmp28
    tmp30 = 1966080 + x2
    tmp31 = tl.rand(tmp1, tmp30)
    tmp32 = tmp31 > tmp4
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp9
    tmp38 = tmp36 * tmp37
    tmp40 = tmp37 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp29 + tmp41
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp42, xmask)
''')


kernel64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 1966080 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 1769472 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 1769472 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr7 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_ptr3 + (x0), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    tmp9 = tl.load(in_ptr5 + (x0), xmask)
    tmp11 = tl.load(in_ptr6 + (x0), xmask)
    tmp13 = tl.load(in_ptr7 + (x0), xmask)
    tmp15 = tl.load(in_ptr8 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


kernel68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 1572864 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr0 + (98304 + x2), xmask)
    tmp22 = tl.load(in_ptr2 + (98304 + x2), xmask)
    tmp25 = tl.load(in_ptr3 + (98304 + x2), xmask)
    tmp27 = tl.load(in_ptr4 + (768 + x1), xmask)
    tmp2 = 1572864 + x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp18 = 1671168 + x2
    tmp19 = tl.rand(tmp1, tmp18)
    tmp20 = tmp19 > tmp4
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 * tmp9
    tmp26 = tmp24 * tmp25
    tmp28 = tmp25 * tmp27
    tmp29 = tmp26 - tmp28
    tmp30 = tmp17 + tmp29
    tmp31 = tmp16 + tmp30
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


kernel70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x2 = xindex
    x1 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = -1
    tmp2 = tmp0 != tmp1
    tmp4 = 0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.atomic_add(out_ptr0 + (x1 + (6*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 1572864 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr5 + (r2 + (128*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = 1.1111111111111112
        tmp10 = tmp8 * tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp14, xmask)
''')


kernel74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr5 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = 1.1111111111111112
        tmp12 = tmp10 * tmp11
        tmp13 = tmp6 * tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x2), xmask)
    tmp21 = tl.load(in_ptr8 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 * tmp8
    tmp15 = tmp14 * tmp8
    tmp16 = tmp13 * tmp15
    tmp17 = 512
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tmp10 + tmp27
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp28, xmask)
''')


kernel76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128057344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x2), xmask)
    tmp1 = -1
    tmp2 = tmp0 != tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 * tmp7
    tmp9 = 0
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.atomic_add(out_ptr0 + (x0 + (512*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


kernel78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 1376256 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 1376256 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256, 512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 256
    ynumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*y2) + (49152*x1)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (384*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


kernel81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 1179648 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 1179648 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 983040 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 983040 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 786432 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp21 = tl.load(in_ptr5 + (x2), xmask)
    tmp24 = tl.load(in_ptr6 + (x2), xmask)
    tmp26 = tl.load(in_ptr7 + (x1), xmask)
    tmp34 = tl.load(in_ptr8 + (x2), xmask)
    tmp37 = tl.load(in_ptr9 + (x2), xmask)
    tmp39 = tl.load(in_ptr10 + (x1), xmask)
    tmp2 = 1179648 + x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = 983040 + x2
    tmp18 = tl.rand(tmp1, tmp17)
    tmp19 = tmp18 > tmp4
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp9
    tmp25 = tmp23 * tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp25 - tmp27
    tmp29 = tmp16 + tmp28
    tmp30 = 786432 + x2
    tmp31 = tl.rand(tmp1, tmp30)
    tmp32 = tmp31 > tmp4
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp9
    tmp38 = tmp36 * tmp37
    tmp40 = tmp37 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp29 + tmp41
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp42, xmask)
''')


kernel87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 786432 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel88 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 589824 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel89 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 589824 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel90 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 393216 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel91 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 393216 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel92 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 196608 + r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel93 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr5 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp21 = tl.load(in_ptr5 + (x2), xmask)
    tmp24 = tl.load(in_ptr6 + (x2), xmask)
    tmp26 = tl.load(in_ptr7 + (x1), xmask)
    tmp34 = tl.load(in_ptr8 + (x2), xmask)
    tmp37 = tl.load(in_ptr9 + (x2), xmask)
    tmp39 = tl.load(in_ptr10 + (x1), xmask)
    tmp2 = 589824 + x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp17 = 393216 + x2
    tmp18 = tl.rand(tmp1, tmp17)
    tmp19 = tmp18 > tmp4
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 * tmp9
    tmp25 = tmp23 * tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp25 - tmp27
    tmp29 = tmp16 + tmp28
    tmp30 = 196608 + x2
    tmp31 = tl.rand(tmp1, tmp30)
    tmp32 = tmp31 > tmp4
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35 * tmp9
    tmp38 = tmp36 * tmp37
    tmp40 = tmp37 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp29 + tmp41
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp42, xmask)
''')


kernel94 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 196608 + x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel95 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = r1 + (128*x0)
        tmp2 = tl.rand(tmp0, tmp1)
        tmp3 = 0.1
        tmp4 = tmp2 > tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = 1.1111111111111112
        tmp9 = tmp7 * tmp8
        tmp11 = tmp9 * tmp10
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel96 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp7 = tl.load(in_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_ptr3 + (x2), xmask)
    tmp13 = tl.load(in_ptr4 + (x1), xmask)
    tmp17 = tl.load(in_ptr0 + (98304 + x2), xmask)
    tmp22 = tl.load(in_ptr2 + (98304 + x2), xmask)
    tmp25 = tl.load(in_ptr3 + (98304 + x2), xmask)
    tmp27 = tl.load(in_ptr4 + (768 + x1), xmask)
    tmp2 = x2
    tmp3 = tl.rand(tmp1, tmp2)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 - tmp14
    tmp16 = tmp0 + tmp15
    tmp18 = 98304 + x2
    tmp19 = tl.rand(tmp1, tmp18)
    tmp20 = tmp19 > tmp4
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 * tmp9
    tmp26 = tmp24 * tmp25
    tmp28 = tmp25 * tmp27
    tmp29 = tmp26 - tmp28
    tmp30 = tmp17 + tmp29
    tmp31 = tmp16 + tmp30
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp31, xmask)
''')


kernel97 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp10 = tl.load(in_ptr2 + (x2), xmask)
    tmp12 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = x2
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp11 - tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel98 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x0), xmask)
    tmp8 = tl.load(in_ptr5 + (x1), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x2), xmask)
    tmp21 = tl.load(in_ptr8 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 + tmp9
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 * tmp8
    tmp15 = tmp14 * tmp8
    tmp16 = tmp13 * tmp15
    tmp17 = 512
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 * tmp26
    tmp28 = tmp10 + tmp27
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp28, xmask)
''')


kernel99 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128057344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, embedding, gt, reciprocal, div_2, philox_seed_like, view_9, gt_3, add_6, reciprocal_1, mm_4, sub_3, mm_5, gt_4, view_12, gt_5, add_11, reciprocal_3, div_3, view_21, gt_7, add_14, reciprocal_4, mm_11, sub_5, mm_12, gt_8, view_24, gt_9, add_19, reciprocal_6, div_4, view_33, gt_11, add_22, reciprocal_7, mm_18, sub_7, mm_19, gt_12, view_36, gt_13, add_27, reciprocal_9, div_5, view_45, gt_15, add_30, reciprocal_10, mm_25, sub_9, mm_26, gt_16, view_48, gt_17, add_35, reciprocal_12, div_6, view_57, gt_19, add_38, reciprocal_13, mm_32, sub_11, mm_33, gt_20, view_60, gt_21, add_43, reciprocal_15, div_7, view_69, gt_23, add_46, reciprocal_16, mm_39, sub_13, mm_40, gt_24, view_72, gt_25, add_51, reciprocal_18, div_8, view_81, gt_27, add_54, reciprocal_19, mm_46, sub_15, mm_47, gt_28, view_84, gt_29, add_59, reciprocal_21, div_9, view_93, gt_31, add_62, reciprocal_22, mm_53, sub_17, mm_54, gt_32, view_96, gt_33, add_67, reciprocal_24, gt_34, embedding_2, gt_35, reciprocal_25, div_12, view_106, gt_37, add_74, reciprocal_26, div_13, view_115, gt_39, add_78, reciprocal_27, mm_64, sub_23, mm_65, gt_40, view_118, gt_41, add_83, reciprocal_29, div_14, view_127, gt_43, add_86, reciprocal_30, div_15, view_136, gt_45, add_89, reciprocal_31, mm_75, sub_26, mm_76, gt_46, view_139, gt_47, add_94, reciprocal_33, div_16, view_148, gt_49, add_97, reciprocal_34, div_17, view_157, gt_51, add_100, reciprocal_35, mm_86, sub_29, mm_87, gt_52, view_160, gt_53, add_105, reciprocal_37, div_18, view_169, gt_55, add_108, reciprocal_38, div_19, view_178, gt_57, add_111, reciprocal_39, mm_97, sub_32, mm_98, gt_58, view_181, gt_59, add_116, reciprocal_41, div_20, view_190, gt_61, add_119, reciprocal_42, div_21, view_199, gt_63, add_122, reciprocal_43, mm_108, sub_35, mm_109, gt_64, view_202, gt_65, add_127, reciprocal_45, div_22, view_211, gt_67, add_130, reciprocal_46, div_23, view_220, gt_69, add_133, reciprocal_47, mm_119, sub_38, mm_120, gt_70, view_223, gt_71, add_138, reciprocal_49, div_24, view_232, gt_73, add_141, reciprocal_50, div_25, view_241, gt_75, add_144, reciprocal_51, mm_130, sub_41, mm_131, gt_76, view_244, gt_77, add_149, reciprocal_53, div_26, view_253, gt_79, add_152, reciprocal_54, div_27, view_262, gt_81, add_155, reciprocal_55, mm_141, sub_44, mm_142, gt_82, view_265, gt_83, add_160, reciprocal_57, gt_84, view_266, sub_46, unsqueeze_17, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, view_560, permute_750, permute_751, permute_756, permute_761, permute_766, view_572, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, view_741, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, view_753, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35 = args
    args.clear()
    buf0 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_1, gt, embedding, reciprocal, buf0, 131072, grid=grid(131072), stream=stream0)
    buf1 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_2, add_6, reciprocal_1, buf1, 131072, grid=grid(131072), stream=stream0)
    buf2 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_3, add_11, reciprocal_3, buf2, 131072, grid=grid(131072), stream=stream0)
    buf3 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_4, add_14, reciprocal_4, buf3, 131072, grid=grid(131072), stream=stream0)
    buf4 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_5, add_19, reciprocal_6, buf4, 131072, grid=grid(131072), stream=stream0)
    buf5 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_6, add_22, reciprocal_7, buf5, 131072, grid=grid(131072), stream=stream0)
    buf6 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_7, add_27, reciprocal_9, buf6, 131072, grid=grid(131072), stream=stream0)
    buf7 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_8, add_30, reciprocal_10, buf7, 131072, grid=grid(131072), stream=stream0)
    buf8 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_9, add_35, reciprocal_12, buf8, 131072, grid=grid(131072), stream=stream0)
    buf9 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_10, add_38, reciprocal_13, buf9, 131072, grid=grid(131072), stream=stream0)
    buf10 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_11, add_43, reciprocal_15, buf10, 131072, grid=grid(131072), stream=stream0)
    buf11 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_12, add_46, reciprocal_16, buf11, 131072, grid=grid(131072), stream=stream0)
    buf12 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_13, add_51, reciprocal_18, buf12, 131072, grid=grid(131072), stream=stream0)
    buf13 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_14, add_54, reciprocal_19, buf13, 131072, grid=grid(131072), stream=stream0)
    buf14 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_15, add_59, reciprocal_21, buf14, 131072, grid=grid(131072), stream=stream0)
    buf15 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_16, add_62, reciprocal_22, buf15, 131072, grid=grid(131072), stream=stream0)
    buf16 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel0.run(primals_18, gt_35, embedding_2, reciprocal_25, buf16, 131072, grid=grid(131072), stream=stream0)
    buf17 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel2.run(gt_34, primals_17, add_67, reciprocal_24, buf17, 131072, grid=grid(131072), stream=stream0)
    buf18 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_20, add_78, reciprocal_27, buf18, 131072, grid=grid(131072), stream=stream0)
    buf19 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_21, add_83, reciprocal_29, buf19, 131072, grid=grid(131072), stream=stream0)
    buf20 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_23, add_89, reciprocal_31, buf20, 131072, grid=grid(131072), stream=stream0)
    buf21 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_24, add_94, reciprocal_33, buf21, 131072, grid=grid(131072), stream=stream0)
    buf22 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_26, add_100, reciprocal_35, buf22, 131072, grid=grid(131072), stream=stream0)
    buf23 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_27, add_105, reciprocal_37, buf23, 131072, grid=grid(131072), stream=stream0)
    buf24 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_29, add_111, reciprocal_39, buf24, 131072, grid=grid(131072), stream=stream0)
    buf25 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_30, add_116, reciprocal_41, buf25, 131072, grid=grid(131072), stream=stream0)
    buf26 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_32, add_122, reciprocal_43, buf26, 131072, grid=grid(131072), stream=stream0)
    buf27 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_33, add_127, reciprocal_45, buf27, 131072, grid=grid(131072), stream=stream0)
    buf28 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_35, add_133, reciprocal_47, buf28, 131072, grid=grid(131072), stream=stream0)
    buf29 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_36, add_138, reciprocal_49, buf29, 131072, grid=grid(131072), stream=stream0)
    buf30 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_38, add_144, reciprocal_51, buf30, 131072, grid=grid(131072), stream=stream0)
    buf31 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_39, add_149, reciprocal_53, buf31, 131072, grid=grid(131072), stream=stream0)
    buf32 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_41, add_155, reciprocal_55, buf32, 131072, grid=grid(131072), stream=stream0)
    buf33 = empty_strided((256, 250112), (250112, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf33, 64028672, grid=grid(64028672), stream=stream0)
    kernel4.run(unsqueeze_17, buf33, 256, grid=grid(256), stream=stream0)
    del unsqueeze_17
    buf35 = empty_strided((256, 1), (1, 256), device='cuda', dtype=torch.float32)
    kernel5.run(buf33, tangents_1, buf35, 256, 250112, grid=grid(256), stream=stream0)
    buf36 = empty_strided((2, 128, 250112), (32014336, 250112, 1), device='cuda', dtype=torch.float32)
    kernel6.run(tangents_2, buf33, tangents_1, sub_46, buf35, buf36, 64028672, grid=grid(64028672), stream=stream0)
    del buf33
    del sub_46
    del tangents_1
    del tangents_2
    buf37 = empty_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf36, (250112, 256), (1, 250112)), view_266, out=buf37)
    del view_266
    buf38 = empty_strided((256, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf36, (256, 250112), (250112, 1)), permute_269, out=buf38)
    del buf36
    del permute_269
    buf39 = empty_strided((1, 1, 512, 2), (1024, 1024, 1, 512), device='cuda', dtype=torch.float32)
    kernel7.run(buf38, gt_84, add_160, reciprocal_57, buf39, 1024, 128, grid=grid(1024), stream=stream0)
    buf40 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf39, buf40, 512, 2, grid=grid(512), stream=stream0)
    buf41 = as_strided(buf35, (2, 128, 1), (128, 1, 256)); del buf35  # reuse
    kernel9.run(buf38, gt_84, primals_42, add_160, buf41, 256, 512, grid=grid(256), stream=stream0)
    buf42 = as_strided(buf38, (2, 128, 512), (65536, 512, 1)); del buf38  # reuse
    kernel10.run(buf42, gt_84, primals_42, reciprocal_57, buf41, add_160, 131072, grid=grid(131072), stream=stream0)
    del add_160
    del gt_84
    del primals_42
    del reciprocal_57
    buf43 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel11.run(buf42, gt_83, buf43, 131072, grid=grid(131072), stream=stream0)
    del gt_83
    buf44 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf43, (512, 256), (1, 512)), view_265, out=buf44)
    del view_265
    buf45 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf43, (256, 512), (512, 1)), permute_273, out=buf45)
    del permute_273
    buf46 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel12.run(buf45, gt_82, mm_141, sub_44, buf46, 262144, grid=grid(262144), stream=stream0)
    buf47 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf46, (1024, 256), (1, 1024)), as_strided(buf32, (256, 512), (512, 1)), out=buf47)
    buf48 = as_strided(buf43, (256, 512), (512, 1)); del buf43  # reuse
    aten.mm.out(as_strided(buf46, (256, 1024), (1024, 1)), permute_277, out=buf48)
    del permute_277
    buf49 = buf46; del buf46  # reuse
    kernel13.run(buf45, gt_82, mm_142, mm_141, sub_44, buf49, 262144, grid=grid(262144), stream=stream0)
    buf50 = buf49; del buf49  # reuse
    kernel14.run(buf50, mm_141, buf45, gt_82, mm_142, sub_44, 262144, grid=grid(262144), stream=stream0)
    del gt_82
    del mm_141
    del mm_142
    del sub_44
    buf51 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf50, (1024, 256), (1, 1024)), as_strided(buf32, (256, 512), (512, 1)), out=buf51)
    buf52 = as_strided(buf32, (256, 512), (512, 1)); del buf32  # reuse
    aten.mm.out(as_strided(buf50, (256, 1024), (1024, 1)), permute_281, out=buf52)
    del permute_281
    buf53 = buf39; del buf39  # reuse
    kernel15.run(buf48, buf52, add_155, reciprocal_55, buf53, 1024, 128, grid=grid(1024), stream=stream0)
    buf54 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf53, buf54, 512, 2, grid=grid(512), stream=stream0)
    buf55 = buf41; del buf41  # reuse
    kernel16.run(buf48, buf52, primals_41, add_155, buf55, 256, 512, grid=grid(256), stream=stream0)
    buf56 = as_strided(buf52, (2, 128, 512), (65536, 512, 1)); del buf52  # reuse
    kernel17.run(buf56, buf42, buf48, primals_41, reciprocal_55, buf55, add_155, 131072, grid=grid(131072), stream=stream0)
    del add_155
    del primals_41
    del reciprocal_55
    buf57 = as_strided(buf48, (2, 128, 512), (65536, 512, 1)); del buf48  # reuse
    kernel11.run(buf56, gt_81, buf57, 131072, grid=grid(131072), stream=stream0)
    del gt_81
    buf58 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf57, (512, 256), (1, 512)), view_262, out=buf58)
    del view_262
    buf59 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf57, (256, 512), (512, 1)), permute_285, out=buf59)
    del permute_285
    buf60 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel18.run(buf59, buf60, 98304, grid=grid(98304), stream=stream0)
    buf61 = as_strided(buf59, (12, 128, 64), (8192, 64, 1)); del buf59  # reuse
    aten.bmm.out(permute_288, as_strided(buf60, (12, 128, 64), (8192, 64, 1)), out=buf61)
    del permute_288
    buf62 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf60, (12, 128, 64), (8192, 64, 1)), permute_289, out=buf62)
    del permute_289
    buf63 = empty_strided((2, 6, 128, 1), (768, 128, 1, 1536), device='cuda', dtype=torch.float32)
    kernel19.run(philox_seed_like, buf62, div_27, buf63, 1536, 128, grid=grid(1536), stream=stream0)
    buf64 = as_strided(buf62, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf62  # reuse
    kernel20.run(buf64, philox_seed_like, div_27, buf63, 196608, grid=grid(196608), stream=stream0)
    del div_27
    buf65 = as_strided(buf60, (12, 64, 128), (8192, 128, 1)); del buf60  # reuse
    aten.bmm.out(permute_290, as_strided(buf64, (12, 128, 128), (16384, 128, 1)), out=buf65)
    del permute_290
    buf66 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf64, (12, 128, 128), (16384, 128, 1)), permute_291, out=buf66)
    del permute_291
    buf67 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel21.run(tangents_34, buf61, buf67, 98304, grid=grid(98304), stream=stream0)
    del tangents_34
    buf68 = as_strided(buf64, (384, 512), (512, 1)); del buf64  # reuse
    aten.mm.out(as_strided(buf67, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf68)
    buf69 = as_strided(buf57, (256, 512), (512, 1)); del buf57  # reuse
    aten.mm.out(as_strided(buf67, (256, 384), (384, 1)), permute_296, out=buf69)
    del permute_296
    buf70 = buf67; del buf67  # reuse
    kernel22.run(tangents_33, buf65, buf70, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_33
    buf71 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf70, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf71)
    buf72 = as_strided(buf42, (256, 512), (512, 1)); del buf42  # reuse
    aten.mm.out(as_strided(buf70, (256, 384), (384, 1)), permute_301, out=buf72)
    del permute_301
    buf73 = buf70; del buf70  # reuse
    kernel23.run(buf66, buf73, 98304, grid=grid(98304), stream=stream0)
    buf74 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel1.run(primals_40, add_152, reciprocal_54, buf74, 131072, grid=grid(131072), stream=stream0)
    buf75 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf73, (384, 256), (1, 384)), as_strided(buf74, (256, 512), (512, 1)), out=buf75)
    buf76 = as_strided(buf74, (256, 512), (512, 1)); del buf74  # reuse
    aten.mm.out(as_strided(buf73, (256, 384), (384, 1)), permute_306, out=buf76)
    del permute_306
    buf77 = buf53; del buf53  # reuse
    kernel24.run(buf76, add_152, reciprocal_54, buf77, 1024, 128, grid=grid(1024), stream=stream0)
    buf78 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf77, buf78, 512, 2, grid=grid(512), stream=stream0)
    buf79 = buf55; del buf55  # reuse
    kernel25.run(buf76, primals_40, add_152, buf79, 256, 512, grid=grid(256), stream=stream0)
    buf80 = buf56; del buf56  # reuse
    kernel26.run(buf80, buf76, primals_40, reciprocal_54, buf79, add_152, 131072, grid=grid(131072), stream=stream0)
    del add_152
    del primals_40
    del reciprocal_54
    buf81 = as_strided(buf76, (2, 128, 512), (65536, 512, 1)); del buf76  # reuse
    kernel11.run(buf80, gt_79, buf81, 131072, grid=grid(131072), stream=stream0)
    del gt_79
    buf82 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf81, (512, 256), (1, 512)), view_253, out=buf82)
    del view_253
    buf83 = as_strided(buf73, (256, 384), (384, 1)); del buf73  # reuse
    aten.mm.out(as_strided(buf81, (256, 512), (512, 1)), permute_310, out=buf83)
    del permute_310
    buf84 = as_strided(buf66, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf66  # reuse
    kernel18.run(buf83, buf84, 98304, grid=grid(98304), stream=stream0)
    buf85 = as_strided(buf83, (12, 128, 64), (8192, 64, 1)); del buf83  # reuse
    aten.bmm.out(permute_313, as_strided(buf84, (12, 128, 64), (8192, 64, 1)), out=buf85)
    del permute_313
    buf86 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf84, (12, 128, 64), (8192, 64, 1)), permute_314, out=buf86)
    del permute_314
    buf87 = buf63; del buf63  # reuse
    kernel27.run(philox_seed_like, buf86, div_26, buf87, 1536, 128, grid=grid(1536), stream=stream0)
    buf88 = as_strided(buf86, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf86  # reuse
    kernel28.run(buf88, philox_seed_like, div_26, buf87, 196608, grid=grid(196608), stream=stream0)
    del div_26
    buf89 = as_strided(buf84, (12, 64, 128), (8192, 128, 1)); del buf84  # reuse
    aten.bmm.out(permute_315, as_strided(buf88, (12, 128, 128), (16384, 128, 1)), out=buf89)
    del permute_315
    buf90 = as_strided(buf65, (12, 128, 64), (8192, 64, 1)); del buf65  # reuse
    aten.bmm.out(as_strided(buf88, (12, 128, 128), (16384, 128, 1)), permute_316, out=buf90)
    del permute_316
    buf91 = as_strided(buf61, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf61  # reuse
    kernel21.run(tangents_32, buf85, buf91, 98304, grid=grid(98304), stream=stream0)
    del tangents_32
    buf92 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf91, (384, 256), (1, 384)), as_strided(buf31, (256, 512), (512, 1)), out=buf92)
    buf93 = as_strided(buf81, (256, 512), (512, 1)); del buf81  # reuse
    aten.mm.out(as_strided(buf91, (256, 384), (384, 1)), permute_321, out=buf93)
    del permute_321
    buf94 = buf91; del buf91  # reuse
    kernel22.run(tangents_31, buf89, buf94, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_31
    buf95 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf94, (384, 256), (1, 384)), as_strided(buf31, (256, 512), (512, 1)), out=buf95)
    buf96 = empty_strided((256, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf94, (256, 384), (384, 1)), permute_326, out=buf96)
    del permute_326
    buf97 = buf94; del buf94  # reuse
    kernel23.run(buf90, buf97, 98304, grid=grid(98304), stream=stream0)
    buf98 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf97, (384, 256), (1, 384)), as_strided(buf31, (256, 512), (512, 1)), out=buf98)
    buf99 = as_strided(buf31, (256, 512), (512, 1)); del buf31  # reuse
    aten.mm.out(as_strided(buf97, (256, 384), (384, 1)), permute_331, out=buf99)
    del permute_331
    buf100 = buf77; del buf77  # reuse
    kernel29.run(buf93, buf96, buf99, add_149, reciprocal_53, buf100, 1024, 128, grid=grid(1024), stream=stream0)
    buf101 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf100, buf101, 512, 2, grid=grid(512), stream=stream0)
    buf102 = buf79; del buf79  # reuse
    kernel30.run(buf93, buf96, buf99, primals_39, add_149, buf102, 256, 512, grid=grid(256), stream=stream0)
    buf103 = buf80; del buf80  # reuse
    kernel31.run(buf103, buf93, buf96, buf99, primals_39, reciprocal_53, buf102, add_149, 131072, grid=grid(131072), stream=stream0)
    del add_149
    del primals_39
    del reciprocal_53
    buf104 = as_strided(buf99, (2, 128, 512), (65536, 512, 1)); del buf99  # reuse
    kernel11.run(buf103, gt_77, buf104, 131072, grid=grid(131072), stream=stream0)
    del gt_77
    buf105 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf104, (512, 256), (1, 512)), view_244, out=buf105)
    del view_244
    buf106 = as_strided(buf50, (256, 1024), (1024, 1)); del buf50  # reuse
    aten.mm.out(as_strided(buf104, (256, 512), (512, 1)), permute_335, out=buf106)
    del permute_335
    buf107 = as_strided(buf45, (2, 128, 1024), (131072, 1024, 1)); del buf45  # reuse
    kernel12.run(buf106, gt_76, mm_130, sub_41, buf107, 262144, grid=grid(262144), stream=stream0)
    buf108 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf107, (1024, 256), (1, 1024)), as_strided(buf30, (256, 512), (512, 1)), out=buf108)
    buf109 = as_strided(buf104, (256, 512), (512, 1)); del buf104  # reuse
    aten.mm.out(as_strided(buf107, (256, 1024), (1024, 1)), permute_339, out=buf109)
    del permute_339
    buf110 = buf107; del buf107  # reuse
    kernel13.run(buf106, gt_76, mm_131, mm_130, sub_41, buf110, 262144, grid=grid(262144), stream=stream0)
    buf111 = buf110; del buf110  # reuse
    kernel14.run(buf111, mm_130, buf106, gt_76, mm_131, sub_41, 262144, grid=grid(262144), stream=stream0)
    del gt_76
    del mm_130
    del mm_131
    del sub_41
    buf112 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf111, (1024, 256), (1, 1024)), as_strided(buf30, (256, 512), (512, 1)), out=buf112)
    buf113 = as_strided(buf30, (256, 512), (512, 1)); del buf30  # reuse
    aten.mm.out(as_strided(buf111, (256, 1024), (1024, 1)), permute_343, out=buf113)
    del permute_343
    buf114 = buf100; del buf100  # reuse
    kernel15.run(buf109, buf113, add_144, reciprocal_51, buf114, 1024, 128, grid=grid(1024), stream=stream0)
    buf115 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf114, buf115, 512, 2, grid=grid(512), stream=stream0)
    buf116 = buf102; del buf102  # reuse
    kernel16.run(buf109, buf113, primals_38, add_144, buf116, 256, 512, grid=grid(256), stream=stream0)
    buf117 = as_strided(buf113, (2, 128, 512), (65536, 512, 1)); del buf113  # reuse
    kernel17.run(buf117, buf103, buf109, primals_38, reciprocal_51, buf116, add_144, 131072, grid=grid(131072), stream=stream0)
    del add_144
    del primals_38
    del reciprocal_51
    buf118 = as_strided(buf109, (2, 128, 512), (65536, 512, 1)); del buf109  # reuse
    kernel11.run(buf117, gt_75, buf118, 131072, grid=grid(131072), stream=stream0)
    del gt_75
    buf119 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf118, (512, 256), (1, 512)), view_241, out=buf119)
    del view_241
    buf120 = as_strided(buf97, (256, 384), (384, 1)); del buf97  # reuse
    aten.mm.out(as_strided(buf118, (256, 512), (512, 1)), permute_347, out=buf120)
    del permute_347
    buf121 = as_strided(buf90, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf90  # reuse
    kernel18.run(buf120, buf121, 98304, grid=grid(98304), stream=stream0)
    buf122 = as_strided(buf120, (12, 128, 64), (8192, 64, 1)); del buf120  # reuse
    aten.bmm.out(permute_350, as_strided(buf121, (12, 128, 64), (8192, 64, 1)), out=buf122)
    del permute_350
    buf123 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf121, (12, 128, 64), (8192, 64, 1)), permute_351, out=buf123)
    del permute_351
    buf124 = buf87; del buf87  # reuse
    kernel32.run(philox_seed_like, buf123, div_25, buf124, 1536, 128, grid=grid(1536), stream=stream0)
    buf125 = as_strided(buf123, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf123  # reuse
    kernel33.run(buf125, philox_seed_like, div_25, buf124, 196608, grid=grid(196608), stream=stream0)
    del div_25
    buf126 = as_strided(buf121, (12, 64, 128), (8192, 128, 1)); del buf121  # reuse
    aten.bmm.out(permute_352, as_strided(buf125, (12, 128, 128), (16384, 128, 1)), out=buf126)
    del permute_352
    buf127 = as_strided(buf89, (12, 128, 64), (8192, 64, 1)); del buf89  # reuse
    aten.bmm.out(as_strided(buf125, (12, 128, 128), (16384, 128, 1)), permute_353, out=buf127)
    del permute_353
    buf128 = as_strided(buf85, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf85  # reuse
    kernel21.run(tangents_30, buf122, buf128, 98304, grid=grid(98304), stream=stream0)
    del tangents_30
    buf129 = as_strided(buf125, (384, 512), (512, 1)); del buf125  # reuse
    aten.mm.out(as_strided(buf128, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf129)
    buf130 = as_strided(buf118, (256, 512), (512, 1)); del buf118  # reuse
    aten.mm.out(as_strided(buf128, (256, 384), (384, 1)), permute_358, out=buf130)
    del permute_358
    buf131 = buf128; del buf128  # reuse
    kernel22.run(tangents_29, buf126, buf131, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_29
    buf132 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf131, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf132)
    buf133 = as_strided(buf103, (256, 512), (512, 1)); del buf103  # reuse
    aten.mm.out(as_strided(buf131, (256, 384), (384, 1)), permute_363, out=buf133)
    del permute_363
    buf134 = buf131; del buf131  # reuse
    kernel23.run(buf127, buf134, 98304, grid=grid(98304), stream=stream0)
    buf135 = as_strided(buf96, (2, 128, 512), (65536, 512, 1)); del buf96  # reuse
    kernel1.run(primals_37, add_141, reciprocal_50, buf135, 131072, grid=grid(131072), stream=stream0)
    buf136 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf134, (384, 256), (1, 384)), as_strided(buf135, (256, 512), (512, 1)), out=buf136)
    buf137 = as_strided(buf135, (256, 512), (512, 1)); del buf135  # reuse
    aten.mm.out(as_strided(buf134, (256, 384), (384, 1)), permute_368, out=buf137)
    del permute_368
    buf138 = buf114; del buf114  # reuse
    kernel24.run(buf137, add_141, reciprocal_50, buf138, 1024, 128, grid=grid(1024), stream=stream0)
    buf139 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf138, buf139, 512, 2, grid=grid(512), stream=stream0)
    buf140 = buf116; del buf116  # reuse
    kernel25.run(buf137, primals_37, add_141, buf140, 256, 512, grid=grid(256), stream=stream0)
    buf141 = buf117; del buf117  # reuse
    kernel26.run(buf141, buf137, primals_37, reciprocal_50, buf140, add_141, 131072, grid=grid(131072), stream=stream0)
    del add_141
    del primals_37
    del reciprocal_50
    buf142 = as_strided(buf137, (2, 128, 512), (65536, 512, 1)); del buf137  # reuse
    kernel11.run(buf141, gt_73, buf142, 131072, grid=grid(131072), stream=stream0)
    del gt_73
    buf143 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf142, (512, 256), (1, 512)), view_232, out=buf143)
    del view_232
    buf144 = as_strided(buf134, (256, 384), (384, 1)); del buf134  # reuse
    aten.mm.out(as_strided(buf142, (256, 512), (512, 1)), permute_372, out=buf144)
    del permute_372
    buf145 = as_strided(buf127, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf127  # reuse
    kernel18.run(buf144, buf145, 98304, grid=grid(98304), stream=stream0)
    buf146 = as_strided(buf144, (12, 128, 64), (8192, 64, 1)); del buf144  # reuse
    aten.bmm.out(permute_375, as_strided(buf145, (12, 128, 64), (8192, 64, 1)), out=buf146)
    del permute_375
    buf147 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf145, (12, 128, 64), (8192, 64, 1)), permute_376, out=buf147)
    del permute_376
    buf148 = buf124; del buf124  # reuse
    kernel34.run(philox_seed_like, buf147, div_24, buf148, 1536, 128, grid=grid(1536), stream=stream0)
    buf149 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel35.run(philox_seed_like, buf147, div_24, buf148, buf149, 196608, grid=grid(196608), stream=stream0)
    buf150 = as_strided(buf145, (12, 64, 128), (8192, 128, 1)); del buf145  # reuse
    aten.bmm.out(permute_377, as_strided(buf149, (12, 128, 128), (16384, 128, 1)), out=buf150)
    del permute_377
    buf151 = as_strided(buf126, (12, 128, 64), (8192, 64, 1)); del buf126  # reuse
    aten.bmm.out(as_strided(buf149, (12, 128, 128), (16384, 128, 1)), permute_378, out=buf151)
    del permute_378
    buf152 = as_strided(buf122, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf122  # reuse
    kernel21.run(tangents_28, buf146, buf152, 98304, grid=grid(98304), stream=stream0)
    del tangents_28
    buf153 = as_strided(buf149, (384, 512), (512, 1)); del buf149  # reuse
    aten.mm.out(as_strided(buf152, (384, 256), (1, 384)), as_strided(buf29, (256, 512), (512, 1)), out=buf153)
    buf154 = as_strided(buf142, (256, 512), (512, 1)); del buf142  # reuse
    aten.mm.out(as_strided(buf152, (256, 384), (384, 1)), permute_383, out=buf154)
    del permute_383
    buf155 = buf152; del buf152  # reuse
    kernel22.run(tangents_27, buf150, buf155, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_27
    buf156 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf155, (384, 256), (1, 384)), as_strided(buf29, (256, 512), (512, 1)), out=buf156)
    buf157 = buf93; del buf93  # reuse
    aten.mm.out(as_strided(buf155, (256, 384), (384, 1)), permute_388, out=buf157)
    del permute_388
    buf158 = buf155; del buf155  # reuse
    kernel23.run(buf151, buf158, 98304, grid=grid(98304), stream=stream0)
    buf159 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf158, (384, 256), (1, 384)), as_strided(buf29, (256, 512), (512, 1)), out=buf159)
    buf160 = as_strided(buf29, (256, 512), (512, 1)); del buf29  # reuse
    aten.mm.out(as_strided(buf158, (256, 384), (384, 1)), permute_393, out=buf160)
    del permute_393
    buf161 = buf138; del buf138  # reuse
    kernel29.run(buf154, buf157, buf160, add_138, reciprocal_49, buf161, 1024, 128, grid=grid(1024), stream=stream0)
    buf162 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf161, buf162, 512, 2, grid=grid(512), stream=stream0)
    buf163 = buf140; del buf140  # reuse
    kernel30.run(buf154, buf157, buf160, primals_36, add_138, buf163, 256, 512, grid=grid(256), stream=stream0)
    buf164 = buf141; del buf141  # reuse
    kernel31.run(buf164, buf154, buf157, buf160, primals_36, reciprocal_49, buf163, add_138, 131072, grid=grid(131072), stream=stream0)
    del add_138
    del primals_36
    del reciprocal_49
    buf165 = as_strided(buf160, (2, 128, 512), (65536, 512, 1)); del buf160  # reuse
    kernel11.run(buf164, gt_71, buf165, 131072, grid=grid(131072), stream=stream0)
    del gt_71
    buf166 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf165, (512, 256), (1, 512)), view_223, out=buf166)
    del view_223
    buf167 = as_strided(buf111, (256, 1024), (1024, 1)); del buf111  # reuse
    aten.mm.out(as_strided(buf165, (256, 512), (512, 1)), permute_397, out=buf167)
    del permute_397
    buf168 = as_strided(buf106, (2, 128, 1024), (131072, 1024, 1)); del buf106  # reuse
    kernel12.run(buf167, gt_70, mm_119, sub_38, buf168, 262144, grid=grid(262144), stream=stream0)
    buf169 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf168, (1024, 256), (1, 1024)), as_strided(buf28, (256, 512), (512, 1)), out=buf169)
    buf170 = as_strided(buf165, (256, 512), (512, 1)); del buf165  # reuse
    aten.mm.out(as_strided(buf168, (256, 1024), (1024, 1)), permute_401, out=buf170)
    del permute_401
    buf171 = buf168; del buf168  # reuse
    kernel13.run(buf167, gt_70, mm_120, mm_119, sub_38, buf171, 262144, grid=grid(262144), stream=stream0)
    buf172 = buf171; del buf171  # reuse
    kernel14.run(buf172, mm_119, buf167, gt_70, mm_120, sub_38, 262144, grid=grid(262144), stream=stream0)
    del gt_70
    del mm_119
    del mm_120
    del sub_38
    buf173 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf172, (1024, 256), (1, 1024)), as_strided(buf28, (256, 512), (512, 1)), out=buf173)
    buf174 = as_strided(buf28, (256, 512), (512, 1)); del buf28  # reuse
    aten.mm.out(as_strided(buf172, (256, 1024), (1024, 1)), permute_405, out=buf174)
    del permute_405
    buf175 = buf161; del buf161  # reuse
    kernel15.run(buf170, buf174, add_133, reciprocal_47, buf175, 1024, 128, grid=grid(1024), stream=stream0)
    buf176 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf175, buf176, 512, 2, grid=grid(512), stream=stream0)
    buf177 = buf163; del buf163  # reuse
    kernel16.run(buf170, buf174, primals_35, add_133, buf177, 256, 512, grid=grid(256), stream=stream0)
    buf178 = buf164; del buf164  # reuse
    kernel36.run(buf178, buf170, buf174, primals_35, reciprocal_47, buf177, add_133, 131072, grid=grid(131072), stream=stream0)
    del add_133
    del primals_35
    del reciprocal_47
    buf179 = as_strided(buf174, (2, 128, 512), (65536, 512, 1)); del buf174  # reuse
    kernel11.run(buf178, gt_69, buf179, 131072, grid=grid(131072), stream=stream0)
    del gt_69
    buf180 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf179, (512, 256), (1, 512)), view_220, out=buf180)
    del view_220
    buf181 = as_strided(buf158, (256, 384), (384, 1)); del buf158  # reuse
    aten.mm.out(as_strided(buf179, (256, 512), (512, 1)), permute_409, out=buf181)
    del permute_409
    buf182 = as_strided(buf151, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf151  # reuse
    kernel18.run(buf181, buf182, 98304, grid=grid(98304), stream=stream0)
    buf183 = as_strided(buf181, (12, 128, 64), (8192, 64, 1)); del buf181  # reuse
    aten.bmm.out(permute_412, as_strided(buf182, (12, 128, 64), (8192, 64, 1)), out=buf183)
    del permute_412
    buf184 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf182, (12, 128, 64), (8192, 64, 1)), permute_413, out=buf184)
    del permute_413
    buf185 = empty_strided((2, 6, 128, 1), (768, 128, 1, 1536), device='cuda', dtype=torch.float32)
    kernel37.run(philox_seed_like, buf184, div_23, buf185, 1536, 128, grid=grid(1536), stream=stream0)
    buf186 = as_strided(buf184, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf184  # reuse
    kernel38.run(buf186, philox_seed_like, div_23, buf185, 196608, grid=grid(196608), stream=stream0)
    del div_23
    buf187 = as_strided(buf182, (12, 64, 128), (8192, 128, 1)); del buf182  # reuse
    aten.bmm.out(permute_414, as_strided(buf186, (12, 128, 128), (16384, 128, 1)), out=buf187)
    del permute_414
    buf188 = as_strided(buf150, (12, 128, 64), (8192, 64, 1)); del buf150  # reuse
    aten.bmm.out(as_strided(buf186, (12, 128, 128), (16384, 128, 1)), permute_415, out=buf188)
    del permute_415
    buf189 = as_strided(buf146, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf146  # reuse
    kernel21.run(tangents_26, buf183, buf189, 98304, grid=grid(98304), stream=stream0)
    del tangents_26
    buf190 = as_strided(buf186, (384, 512), (512, 1)); del buf186  # reuse
    aten.mm.out(as_strided(buf189, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf190)
    buf191 = as_strided(buf179, (256, 512), (512, 1)); del buf179  # reuse
    aten.mm.out(as_strided(buf189, (256, 384), (384, 1)), permute_420, out=buf191)
    del permute_420
    buf192 = buf189; del buf189  # reuse
    kernel22.run(tangents_25, buf187, buf192, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_25
    buf193 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf192, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf193)
    buf194 = buf170; del buf170  # reuse
    aten.mm.out(as_strided(buf192, (256, 384), (384, 1)), permute_425, out=buf194)
    del permute_425
    buf195 = buf192; del buf192  # reuse
    kernel23.run(buf188, buf195, 98304, grid=grid(98304), stream=stream0)
    buf196 = as_strided(buf157, (2, 128, 512), (65536, 512, 1)); del buf157  # reuse
    kernel1.run(primals_34, add_130, reciprocal_46, buf196, 131072, grid=grid(131072), stream=stream0)
    buf197 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf195, (384, 256), (1, 384)), as_strided(buf196, (256, 512), (512, 1)), out=buf197)
    buf198 = as_strided(buf196, (256, 512), (512, 1)); del buf196  # reuse
    aten.mm.out(as_strided(buf195, (256, 384), (384, 1)), permute_430, out=buf198)
    del permute_430
    buf199 = buf175; del buf175  # reuse
    kernel24.run(buf198, add_130, reciprocal_46, buf199, 1024, 128, grid=grid(1024), stream=stream0)
    buf200 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf199, buf200, 512, 2, grid=grid(512), stream=stream0)
    buf201 = buf177; del buf177  # reuse
    kernel25.run(buf198, primals_34, add_130, buf201, 256, 512, grid=grid(256), stream=stream0)
    buf202 = buf178; del buf178  # reuse
    kernel26.run(buf202, buf198, primals_34, reciprocal_46, buf201, add_130, 131072, grid=grid(131072), stream=stream0)
    del add_130
    del primals_34
    del reciprocal_46
    buf203 = as_strided(buf198, (2, 128, 512), (65536, 512, 1)); del buf198  # reuse
    kernel11.run(buf202, gt_67, buf203, 131072, grid=grid(131072), stream=stream0)
    del gt_67
    buf204 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf203, (512, 256), (1, 512)), view_211, out=buf204)
    del view_211
    buf205 = as_strided(buf195, (256, 384), (384, 1)); del buf195  # reuse
    aten.mm.out(as_strided(buf203, (256, 512), (512, 1)), permute_434, out=buf205)
    del permute_434
    buf206 = as_strided(buf188, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf188  # reuse
    kernel18.run(buf205, buf206, 98304, grid=grid(98304), stream=stream0)
    buf207 = as_strided(buf205, (12, 128, 64), (8192, 64, 1)); del buf205  # reuse
    aten.bmm.out(permute_437, as_strided(buf206, (12, 128, 64), (8192, 64, 1)), out=buf207)
    del permute_437
    buf208 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf206, (12, 128, 64), (8192, 64, 1)), permute_438, out=buf208)
    del permute_438
    buf209 = buf185; del buf185  # reuse
    kernel39.run(philox_seed_like, buf208, div_22, buf209, 1536, 128, grid=grid(1536), stream=stream0)
    buf210 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel40.run(philox_seed_like, buf208, div_22, buf209, buf210, 196608, grid=grid(196608), stream=stream0)
    buf211 = as_strided(buf206, (12, 64, 128), (8192, 128, 1)); del buf206  # reuse
    aten.bmm.out(permute_439, as_strided(buf210, (12, 128, 128), (16384, 128, 1)), out=buf211)
    del permute_439
    buf212 = as_strided(buf187, (12, 128, 64), (8192, 64, 1)); del buf187  # reuse
    aten.bmm.out(as_strided(buf210, (12, 128, 128), (16384, 128, 1)), permute_440, out=buf212)
    del permute_440
    buf213 = as_strided(buf183, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf183  # reuse
    kernel21.run(tangents_24, buf207, buf213, 98304, grid=grid(98304), stream=stream0)
    del tangents_24
    buf214 = as_strided(buf210, (384, 512), (512, 1)); del buf210  # reuse
    aten.mm.out(as_strided(buf213, (384, 256), (1, 384)), as_strided(buf27, (256, 512), (512, 1)), out=buf214)
    buf215 = as_strided(buf203, (256, 512), (512, 1)); del buf203  # reuse
    aten.mm.out(as_strided(buf213, (256, 384), (384, 1)), permute_445, out=buf215)
    del permute_445
    buf216 = buf213; del buf213  # reuse
    kernel22.run(tangents_23, buf211, buf216, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_23
    buf217 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf216, (384, 256), (1, 384)), as_strided(buf27, (256, 512), (512, 1)), out=buf217)
    buf218 = buf154; del buf154  # reuse
    aten.mm.out(as_strided(buf216, (256, 384), (384, 1)), permute_450, out=buf218)
    del permute_450
    buf219 = buf216; del buf216  # reuse
    kernel23.run(buf212, buf219, 98304, grid=grid(98304), stream=stream0)
    buf220 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf219, (384, 256), (1, 384)), as_strided(buf27, (256, 512), (512, 1)), out=buf220)
    buf221 = as_strided(buf27, (256, 512), (512, 1)); del buf27  # reuse
    aten.mm.out(as_strided(buf219, (256, 384), (384, 1)), permute_455, out=buf221)
    del permute_455
    buf222 = buf199; del buf199  # reuse
    kernel29.run(buf215, buf218, buf221, add_127, reciprocal_45, buf222, 1024, 128, grid=grid(1024), stream=stream0)
    buf223 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf222, buf223, 512, 2, grid=grid(512), stream=stream0)
    buf224 = buf201; del buf201  # reuse
    kernel30.run(buf215, buf218, buf221, primals_33, add_127, buf224, 256, 512, grid=grid(256), stream=stream0)
    buf225 = as_strided(buf221, (2, 128, 512), (65536, 512, 1)); del buf221  # reuse
    kernel41.run(buf225, buf202, buf215, buf218, primals_33, reciprocal_45, buf224, add_127, 131072, grid=grid(131072), stream=stream0)
    del add_127
    del buf202
    del buf215
    del primals_33
    del reciprocal_45
    buf226 = as_strided(buf218, (2, 128, 512), (65536, 512, 1)); del buf218  # reuse
    kernel11.run(buf225, gt_65, buf226, 131072, grid=grid(131072), stream=stream0)
    del gt_65
    buf227 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf226, (512, 256), (1, 512)), view_202, out=buf227)
    del view_202
    buf228 = as_strided(buf172, (256, 1024), (1024, 1)); del buf172  # reuse
    aten.mm.out(as_strided(buf226, (256, 512), (512, 1)), permute_459, out=buf228)
    del permute_459
    buf229 = as_strided(buf167, (2, 128, 1024), (131072, 1024, 1)); del buf167  # reuse
    kernel12.run(buf228, gt_64, mm_108, sub_35, buf229, 262144, grid=grid(262144), stream=stream0)
    buf230 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf229, (1024, 256), (1, 1024)), as_strided(buf26, (256, 512), (512, 1)), out=buf230)
    buf231 = as_strided(buf226, (256, 512), (512, 1)); del buf226  # reuse
    aten.mm.out(as_strided(buf229, (256, 1024), (1024, 1)), permute_463, out=buf231)
    del permute_463
    buf232 = buf229; del buf229  # reuse
    kernel13.run(buf228, gt_64, mm_109, mm_108, sub_35, buf232, 262144, grid=grid(262144), stream=stream0)
    buf233 = as_strided(buf228, (2, 128, 1024), (131072, 1024, 1)); del buf228  # reuse
    kernel42.run(buf233, buf232, mm_108, gt_64, mm_109, sub_35, 262144, grid=grid(262144), stream=stream0)
    del gt_64
    del mm_108
    del mm_109
    del sub_35
    buf234 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf233, (1024, 256), (1, 1024)), as_strided(buf26, (256, 512), (512, 1)), out=buf234)
    buf235 = as_strided(buf26, (256, 512), (512, 1)); del buf26  # reuse
    aten.mm.out(as_strided(buf233, (256, 1024), (1024, 1)), permute_467, out=buf235)
    del permute_467
    buf236 = buf222; del buf222  # reuse
    kernel15.run(buf231, buf235, add_122, reciprocal_43, buf236, 1024, 128, grid=grid(1024), stream=stream0)
    buf237 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf236, buf237, 512, 2, grid=grid(512), stream=stream0)
    buf238 = buf224; del buf224  # reuse
    kernel16.run(buf231, buf235, primals_32, add_122, buf238, 256, 512, grid=grid(256), stream=stream0)
    buf239 = buf225; del buf225  # reuse
    kernel36.run(buf239, buf231, buf235, primals_32, reciprocal_43, buf238, add_122, 131072, grid=grid(131072), stream=stream0)
    del add_122
    del primals_32
    del reciprocal_43
    buf240 = as_strided(buf235, (2, 128, 512), (65536, 512, 1)); del buf235  # reuse
    kernel11.run(buf239, gt_63, buf240, 131072, grid=grid(131072), stream=stream0)
    del gt_63
    buf241 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf240, (512, 256), (1, 512)), view_199, out=buf241)
    del view_199
    buf242 = as_strided(buf219, (256, 384), (384, 1)); del buf219  # reuse
    aten.mm.out(as_strided(buf240, (256, 512), (512, 1)), permute_471, out=buf242)
    del permute_471
    buf243 = as_strided(buf212, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf212  # reuse
    kernel18.run(buf242, buf243, 98304, grid=grid(98304), stream=stream0)
    buf244 = as_strided(buf242, (12, 128, 64), (8192, 64, 1)); del buf242  # reuse
    aten.bmm.out(permute_474, as_strided(buf243, (12, 128, 64), (8192, 64, 1)), out=buf244)
    del permute_474
    buf245 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf243, (12, 128, 64), (8192, 64, 1)), permute_475, out=buf245)
    del permute_475
    buf246 = empty_strided((2, 6, 128, 1), (768, 128, 1, 1536), device='cuda', dtype=torch.float32)
    kernel43.run(philox_seed_like, buf245, div_21, buf246, 1536, 128, grid=grid(1536), stream=stream0)
    buf247 = as_strided(buf245, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf245  # reuse
    kernel44.run(buf247, philox_seed_like, div_21, buf246, 196608, grid=grid(196608), stream=stream0)
    del div_21
    buf248 = as_strided(buf243, (12, 64, 128), (8192, 128, 1)); del buf243  # reuse
    aten.bmm.out(permute_476, as_strided(buf247, (12, 128, 128), (16384, 128, 1)), out=buf248)
    del permute_476
    buf249 = as_strided(buf211, (12, 128, 64), (8192, 64, 1)); del buf211  # reuse
    aten.bmm.out(as_strided(buf247, (12, 128, 128), (16384, 128, 1)), permute_477, out=buf249)
    del permute_477
    buf250 = as_strided(buf207, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf207  # reuse
    kernel21.run(tangents_22, buf244, buf250, 98304, grid=grid(98304), stream=stream0)
    del tangents_22
    buf251 = as_strided(buf247, (384, 512), (512, 1)); del buf247  # reuse
    aten.mm.out(as_strided(buf250, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf251)
    buf252 = as_strided(buf240, (256, 512), (512, 1)); del buf240  # reuse
    aten.mm.out(as_strided(buf250, (256, 384), (384, 1)), permute_482, out=buf252)
    del permute_482
    buf253 = buf250; del buf250  # reuse
    kernel22.run(tangents_21, buf248, buf253, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_21
    buf254 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf253, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf254)
    buf255 = buf231; del buf231  # reuse
    aten.mm.out(as_strided(buf253, (256, 384), (384, 1)), permute_487, out=buf255)
    del permute_487
    buf256 = as_strided(buf69, (2, 128, 512), (65536, 512, 1)); del buf69  # reuse
    kernel45.run(buf256, tangents_35, buf72, buf130, buf133, buf191, buf194, buf252, buf255, 131072, grid=grid(131072), stream=stream0)
    del buf130
    del buf133
    del buf191
    del buf194
    del buf252
    del tangents_35
    buf257 = buf253; del buf253  # reuse
    kernel23.run(buf249, buf257, 98304, grid=grid(98304), stream=stream0)
    buf258 = as_strided(buf72, (2, 128, 512), (65536, 512, 1)); del buf72  # reuse
    kernel1.run(primals_31, add_119, reciprocal_42, buf258, 131072, grid=grid(131072), stream=stream0)
    buf259 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf257, (384, 256), (1, 384)), as_strided(buf258, (256, 512), (512, 1)), out=buf259)
    buf260 = as_strided(buf258, (256, 512), (512, 1)); del buf258  # reuse
    aten.mm.out(as_strided(buf257, (256, 384), (384, 1)), permute_492, out=buf260)
    del permute_492
    buf261 = buf236; del buf236  # reuse
    kernel24.run(buf260, add_119, reciprocal_42, buf261, 1024, 128, grid=grid(1024), stream=stream0)
    buf262 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf261, buf262, 512, 2, grid=grid(512), stream=stream0)
    buf263 = buf238; del buf238  # reuse
    kernel25.run(buf260, primals_31, add_119, buf263, 256, 512, grid=grid(256), stream=stream0)
    buf264 = as_strided(buf260, (2, 128, 512), (65536, 512, 1)); del buf260  # reuse
    kernel46.run(buf264, buf239, primals_31, reciprocal_42, buf263, add_119, 131072, grid=grid(131072), stream=stream0)
    del add_119
    del primals_31
    del reciprocal_42
    buf265 = buf239; del buf239  # reuse
    kernel11.run(buf264, gt_61, buf265, 131072, grid=grid(131072), stream=stream0)
    del gt_61
    buf266 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf265, (512, 256), (1, 512)), view_190, out=buf266)
    del view_190
    buf267 = as_strided(buf257, (256, 384), (384, 1)); del buf257  # reuse
    aten.mm.out(as_strided(buf265, (256, 512), (512, 1)), permute_496, out=buf267)
    del permute_496
    buf268 = as_strided(buf249, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf249  # reuse
    kernel18.run(buf267, buf268, 98304, grid=grid(98304), stream=stream0)
    buf269 = as_strided(buf267, (12, 128, 64), (8192, 64, 1)); del buf267  # reuse
    aten.bmm.out(permute_499, as_strided(buf268, (12, 128, 64), (8192, 64, 1)), out=buf269)
    del permute_499
    buf270 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf268, (12, 128, 64), (8192, 64, 1)), permute_500, out=buf270)
    del permute_500
    buf271 = buf246; del buf246  # reuse
    kernel47.run(philox_seed_like, buf270, div_20, buf271, 1536, 128, grid=grid(1536), stream=stream0)
    buf272 = buf88; del buf88  # reuse
    kernel48.run(buf272, philox_seed_like, buf147, div_24, buf148, buf208, div_22, buf209, buf270, div_20, buf271, 196608, grid=grid(196608), stream=stream0)
    del div_22
    del div_24
    buf273 = as_strided(buf270, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf270  # reuse
    kernel49.run(buf273, philox_seed_like, div_20, buf271, 196608, grid=grid(196608), stream=stream0)
    del div_20
    buf274 = as_strided(buf268, (12, 64, 128), (8192, 128, 1)); del buf268  # reuse
    aten.bmm.out(permute_501, as_strided(buf273, (12, 128, 128), (16384, 128, 1)), out=buf274)
    del permute_501
    buf275 = as_strided(buf248, (12, 128, 64), (8192, 64, 1)); del buf248  # reuse
    aten.bmm.out(as_strided(buf273, (12, 128, 128), (16384, 128, 1)), permute_502, out=buf275)
    del permute_502
    buf276 = as_strided(buf244, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf244  # reuse
    kernel21.run(tangents_20, buf269, buf276, 98304, grid=grid(98304), stream=stream0)
    del tangents_20
    buf277 = as_strided(buf273, (384, 512), (512, 1)); del buf273  # reuse
    aten.mm.out(as_strided(buf276, (384, 256), (1, 384)), as_strided(buf25, (256, 512), (512, 1)), out=buf277)
    buf278 = as_strided(buf265, (256, 512), (512, 1)); del buf265  # reuse
    aten.mm.out(as_strided(buf276, (256, 384), (384, 1)), permute_507, out=buf278)
    del permute_507
    buf279 = buf276; del buf276  # reuse
    kernel22.run(tangents_19, buf274, buf279, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_19
    buf280 = as_strided(buf208, (384, 512), (512, 1)); del buf208  # reuse
    aten.mm.out(as_strided(buf279, (384, 256), (1, 384)), as_strided(buf25, (256, 512), (512, 1)), out=buf280)
    buf281 = buf255; del buf255  # reuse
    aten.mm.out(as_strided(buf279, (256, 384), (384, 1)), permute_512, out=buf281)
    del permute_512
    buf282 = buf279; del buf279  # reuse
    kernel23.run(buf275, buf282, 98304, grid=grid(98304), stream=stream0)
    buf283 = as_strided(buf147, (384, 512), (512, 1)); del buf147  # reuse
    aten.mm.out(as_strided(buf282, (384, 256), (1, 384)), as_strided(buf25, (256, 512), (512, 1)), out=buf283)
    buf284 = as_strided(buf25, (256, 512), (512, 1)); del buf25  # reuse
    aten.mm.out(as_strided(buf282, (256, 384), (384, 1)), permute_517, out=buf284)
    del permute_517
    buf285 = buf261; del buf261  # reuse
    kernel29.run(buf278, buf281, buf284, add_116, reciprocal_41, buf285, 1024, 128, grid=grid(1024), stream=stream0)
    buf286 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf285, buf286, 512, 2, grid=grid(512), stream=stream0)
    buf287 = buf263; del buf263  # reuse
    kernel30.run(buf278, buf281, buf284, primals_30, add_116, buf287, 256, 512, grid=grid(256), stream=stream0)
    buf288 = as_strided(buf284, (2, 128, 512), (65536, 512, 1)); del buf284  # reuse
    kernel41.run(buf288, buf264, buf278, buf281, primals_30, reciprocal_41, buf287, add_116, 131072, grid=grid(131072), stream=stream0)
    del add_116
    del primals_30
    del reciprocal_41
    buf289 = as_strided(buf281, (2, 128, 512), (65536, 512, 1)); del buf281  # reuse
    kernel11.run(buf288, gt_59, buf289, 131072, grid=grid(131072), stream=stream0)
    del gt_59
    buf290 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf289, (512, 256), (1, 512)), view_181, out=buf290)
    del view_181
    buf291 = as_strided(buf233, (256, 1024), (1024, 1)); del buf233  # reuse
    aten.mm.out(as_strided(buf289, (256, 512), (512, 1)), permute_521, out=buf291)
    del permute_521
    buf292 = buf232; del buf232  # reuse
    kernel12.run(buf291, gt_58, mm_97, sub_32, buf292, 262144, grid=grid(262144), stream=stream0)
    buf293 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf292, (1024, 256), (1, 1024)), as_strided(buf24, (256, 512), (512, 1)), out=buf293)
    buf294 = as_strided(buf289, (256, 512), (512, 1)); del buf289  # reuse
    aten.mm.out(as_strided(buf292, (256, 1024), (1024, 1)), permute_525, out=buf294)
    del permute_525
    buf295 = buf292; del buf292  # reuse
    kernel13.run(buf291, gt_58, mm_98, mm_97, sub_32, buf295, 262144, grid=grid(262144), stream=stream0)
    buf296 = buf295; del buf295  # reuse
    kernel14.run(buf296, mm_97, buf291, gt_58, mm_98, sub_32, 262144, grid=grid(262144), stream=stream0)
    del gt_58
    del mm_97
    del mm_98
    del sub_32
    buf297 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf296, (1024, 256), (1, 1024)), as_strided(buf24, (256, 512), (512, 1)), out=buf297)
    buf298 = as_strided(buf24, (256, 512), (512, 1)); del buf24  # reuse
    aten.mm.out(as_strided(buf296, (256, 1024), (1024, 1)), permute_529, out=buf298)
    del permute_529
    buf299 = buf285; del buf285  # reuse
    kernel15.run(buf294, buf298, add_111, reciprocal_39, buf299, 1024, 128, grid=grid(1024), stream=stream0)
    buf300 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf299, buf300, 512, 2, grid=grid(512), stream=stream0)
    buf301 = buf287; del buf287  # reuse
    kernel16.run(buf294, buf298, primals_29, add_111, buf301, 256, 512, grid=grid(256), stream=stream0)
    buf302 = as_strided(buf294, (2, 128, 512), (65536, 512, 1)); del buf294  # reuse
    kernel50.run(buf302, buf288, buf298, primals_29, reciprocal_39, buf301, add_111, 131072, grid=grid(131072), stream=stream0)
    del add_111
    del primals_29
    del reciprocal_39
    buf303 = as_strided(buf298, (2, 128, 512), (65536, 512, 1)); del buf298  # reuse
    kernel11.run(buf302, gt_57, buf303, 131072, grid=grid(131072), stream=stream0)
    del gt_57
    buf304 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf303, (512, 256), (1, 512)), view_178, out=buf304)
    del view_178
    buf305 = as_strided(buf282, (256, 384), (384, 1)); del buf282  # reuse
    aten.mm.out(as_strided(buf303, (256, 512), (512, 1)), permute_533, out=buf305)
    del permute_533
    buf306 = as_strided(buf275, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf275  # reuse
    kernel18.run(buf305, buf306, 98304, grid=grid(98304), stream=stream0)
    buf307 = as_strided(buf305, (12, 128, 64), (8192, 64, 1)); del buf305  # reuse
    aten.bmm.out(permute_536, as_strided(buf306, (12, 128, 64), (8192, 64, 1)), out=buf307)
    del permute_536
    buf308 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf306, (12, 128, 64), (8192, 64, 1)), permute_537, out=buf308)
    del permute_537
    buf309 = buf271; del buf271  # reuse
    kernel51.run(philox_seed_like, buf308, div_19, buf309, 1536, 128, grid=grid(1536), stream=stream0)
    buf310 = as_strided(buf308, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf308  # reuse
    kernel52.run(buf310, philox_seed_like, div_19, buf309, 196608, grid=grid(196608), stream=stream0)
    del div_19
    buf311 = as_strided(buf306, (12, 64, 128), (8192, 128, 1)); del buf306  # reuse
    aten.bmm.out(permute_538, as_strided(buf310, (12, 128, 128), (16384, 128, 1)), out=buf311)
    del permute_538
    buf312 = as_strided(buf274, (12, 128, 64), (8192, 64, 1)); del buf274  # reuse
    aten.bmm.out(as_strided(buf310, (12, 128, 128), (16384, 128, 1)), permute_539, out=buf312)
    del permute_539
    buf313 = as_strided(buf269, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf269  # reuse
    kernel21.run(tangents_18, buf307, buf313, 98304, grid=grid(98304), stream=stream0)
    del tangents_18
    buf314 = as_strided(buf310, (384, 512), (512, 1)); del buf310  # reuse
    aten.mm.out(as_strided(buf313, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf314)
    buf315 = as_strided(buf303, (256, 512), (512, 1)); del buf303  # reuse
    aten.mm.out(as_strided(buf313, (256, 384), (384, 1)), permute_544, out=buf315)
    del permute_544
    buf316 = buf313; del buf313  # reuse
    kernel22.run(tangents_17, buf311, buf316, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_17
    buf317 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf316, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf317)
    buf318 = as_strided(buf288, (256, 512), (512, 1)); del buf288  # reuse
    aten.mm.out(as_strided(buf316, (256, 384), (384, 1)), permute_549, out=buf318)
    del permute_549
    buf319 = buf316; del buf316  # reuse
    kernel23.run(buf312, buf319, 98304, grid=grid(98304), stream=stream0)
    buf320 = as_strided(buf278, (2, 128, 512), (65536, 512, 1)); del buf278  # reuse
    kernel1.run(primals_28, add_108, reciprocal_38, buf320, 131072, grid=grid(131072), stream=stream0)
    buf321 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf319, (384, 256), (1, 384)), as_strided(buf320, (256, 512), (512, 1)), out=buf321)
    buf322 = as_strided(buf320, (256, 512), (512, 1)); del buf320  # reuse
    aten.mm.out(as_strided(buf319, (256, 384), (384, 1)), permute_554, out=buf322)
    del permute_554
    buf323 = buf299; del buf299  # reuse
    kernel24.run(buf322, add_108, reciprocal_38, buf323, 1024, 128, grid=grid(1024), stream=stream0)
    buf324 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf323, buf324, 512, 2, grid=grid(512), stream=stream0)
    buf325 = buf301; del buf301  # reuse
    kernel25.run(buf322, primals_28, add_108, buf325, 256, 512, grid=grid(256), stream=stream0)
    buf326 = as_strided(buf322, (2, 128, 512), (65536, 512, 1)); del buf322  # reuse
    kernel46.run(buf326, buf302, primals_28, reciprocal_38, buf325, add_108, 131072, grid=grid(131072), stream=stream0)
    del add_108
    del primals_28
    del reciprocal_38
    buf327 = buf302; del buf302  # reuse
    kernel11.run(buf326, gt_55, buf327, 131072, grid=grid(131072), stream=stream0)
    del gt_55
    buf328 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf327, (512, 256), (1, 512)), view_169, out=buf328)
    del view_169
    buf329 = as_strided(buf319, (256, 384), (384, 1)); del buf319  # reuse
    aten.mm.out(as_strided(buf327, (256, 512), (512, 1)), permute_558, out=buf329)
    del permute_558
    buf330 = as_strided(buf312, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf312  # reuse
    kernel18.run(buf329, buf330, 98304, grid=grid(98304), stream=stream0)
    buf331 = as_strided(buf329, (12, 128, 64), (8192, 64, 1)); del buf329  # reuse
    aten.bmm.out(permute_561, as_strided(buf330, (12, 128, 64), (8192, 64, 1)), out=buf331)
    del permute_561
    buf332 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf330, (12, 128, 64), (8192, 64, 1)), permute_562, out=buf332)
    del permute_562
    buf333 = buf309; del buf309  # reuse
    kernel53.run(philox_seed_like, buf332, div_18, buf333, 1536, 128, grid=grid(1536), stream=stream0)
    buf334 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel54.run(philox_seed_like, buf332, div_18, buf333, buf334, 196608, grid=grid(196608), stream=stream0)
    buf335 = as_strided(buf330, (12, 64, 128), (8192, 128, 1)); del buf330  # reuse
    aten.bmm.out(permute_563, as_strided(buf334, (12, 128, 128), (16384, 128, 1)), out=buf335)
    del permute_563
    buf336 = as_strided(buf311, (12, 128, 64), (8192, 64, 1)); del buf311  # reuse
    aten.bmm.out(as_strided(buf334, (12, 128, 128), (16384, 128, 1)), permute_564, out=buf336)
    del permute_564
    buf337 = as_strided(buf307, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf307  # reuse
    kernel21.run(tangents_16, buf331, buf337, 98304, grid=grid(98304), stream=stream0)
    del tangents_16
    buf338 = as_strided(buf334, (384, 512), (512, 1)); del buf334  # reuse
    aten.mm.out(as_strided(buf337, (384, 256), (1, 384)), as_strided(buf23, (256, 512), (512, 1)), out=buf338)
    buf339 = as_strided(buf327, (256, 512), (512, 1)); del buf327  # reuse
    aten.mm.out(as_strided(buf337, (256, 384), (384, 1)), permute_569, out=buf339)
    del permute_569
    buf340 = buf337; del buf337  # reuse
    kernel22.run(tangents_15, buf335, buf340, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_15
    buf341 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf340, (384, 256), (1, 384)), as_strided(buf23, (256, 512), (512, 1)), out=buf341)
    buf342 = as_strided(buf264, (256, 512), (512, 1)); del buf264  # reuse
    aten.mm.out(as_strided(buf340, (256, 384), (384, 1)), permute_574, out=buf342)
    del permute_574
    buf343 = buf340; del buf340  # reuse
    kernel23.run(buf336, buf343, 98304, grid=grid(98304), stream=stream0)
    buf344 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf343, (384, 256), (1, 384)), as_strided(buf23, (256, 512), (512, 1)), out=buf344)
    buf345 = as_strided(buf23, (256, 512), (512, 1)); del buf23  # reuse
    aten.mm.out(as_strided(buf343, (256, 384), (384, 1)), permute_579, out=buf345)
    del permute_579
    buf346 = buf323; del buf323  # reuse
    kernel29.run(buf339, buf342, buf345, add_105, reciprocal_37, buf346, 1024, 128, grid=grid(1024), stream=stream0)
    buf347 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf346, buf347, 512, 2, grid=grid(512), stream=stream0)
    buf348 = buf325; del buf325  # reuse
    kernel30.run(buf339, buf342, buf345, primals_27, add_105, buf348, 256, 512, grid=grid(256), stream=stream0)
    buf349 = as_strided(buf339, (2, 128, 512), (65536, 512, 1)); del buf339  # reuse
    kernel55.run(buf349, buf326, buf342, buf345, primals_27, reciprocal_37, buf348, add_105, 131072, grid=grid(131072), stream=stream0)
    del add_105
    del primals_27
    del reciprocal_37
    buf350 = as_strided(buf345, (2, 128, 512), (65536, 512, 1)); del buf345  # reuse
    kernel11.run(buf349, gt_53, buf350, 131072, grid=grid(131072), stream=stream0)
    del gt_53
    buf351 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf350, (512, 256), (1, 512)), view_160, out=buf351)
    del view_160
    buf352 = as_strided(buf296, (256, 1024), (1024, 1)); del buf296  # reuse
    aten.mm.out(as_strided(buf350, (256, 512), (512, 1)), permute_583, out=buf352)
    del permute_583
    buf353 = as_strided(buf291, (2, 128, 1024), (131072, 1024, 1)); del buf291  # reuse
    kernel12.run(buf352, gt_52, mm_86, sub_29, buf353, 262144, grid=grid(262144), stream=stream0)
    buf354 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf353, (1024, 256), (1, 1024)), as_strided(buf22, (256, 512), (512, 1)), out=buf354)
    buf355 = as_strided(buf350, (256, 512), (512, 1)); del buf350  # reuse
    aten.mm.out(as_strided(buf353, (256, 1024), (1024, 1)), permute_587, out=buf355)
    del permute_587
    buf356 = buf353; del buf353  # reuse
    kernel13.run(buf352, gt_52, mm_87, mm_86, sub_29, buf356, 262144, grid=grid(262144), stream=stream0)
    buf357 = as_strided(buf352, (2, 128, 1024), (131072, 1024, 1)); del buf352  # reuse
    kernel42.run(buf357, buf356, mm_86, gt_52, mm_87, sub_29, 262144, grid=grid(262144), stream=stream0)
    del gt_52
    del mm_86
    del mm_87
    del sub_29
    buf358 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf357, (1024, 256), (1, 1024)), as_strided(buf22, (256, 512), (512, 1)), out=buf358)
    buf359 = as_strided(buf22, (256, 512), (512, 1)); del buf22  # reuse
    aten.mm.out(as_strided(buf357, (256, 1024), (1024, 1)), permute_591, out=buf359)
    del permute_591
    buf360 = buf346; del buf346  # reuse
    kernel15.run(buf355, buf359, add_100, reciprocal_35, buf360, 1024, 128, grid=grid(1024), stream=stream0)
    buf361 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf360, buf361, 512, 2, grid=grid(512), stream=stream0)
    buf362 = buf348; del buf348  # reuse
    kernel16.run(buf355, buf359, primals_26, add_100, buf362, 256, 512, grid=grid(256), stream=stream0)
    buf363 = as_strided(buf359, (2, 128, 512), (65536, 512, 1)); del buf359  # reuse
    kernel17.run(buf363, buf349, buf355, primals_26, reciprocal_35, buf362, add_100, 131072, grid=grid(131072), stream=stream0)
    del add_100
    del primals_26
    del reciprocal_35
    buf364 = as_strided(buf355, (2, 128, 512), (65536, 512, 1)); del buf355  # reuse
    kernel11.run(buf363, gt_51, buf364, 131072, grid=grid(131072), stream=stream0)
    del gt_51
    buf365 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf364, (512, 256), (1, 512)), view_157, out=buf365)
    del view_157
    buf366 = as_strided(buf343, (256, 384), (384, 1)); del buf343  # reuse
    aten.mm.out(as_strided(buf364, (256, 512), (512, 1)), permute_595, out=buf366)
    del permute_595
    buf367 = as_strided(buf336, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf336  # reuse
    kernel18.run(buf366, buf367, 98304, grid=grid(98304), stream=stream0)
    buf368 = as_strided(buf366, (12, 128, 64), (8192, 64, 1)); del buf366  # reuse
    aten.bmm.out(permute_598, as_strided(buf367, (12, 128, 64), (8192, 64, 1)), out=buf368)
    del permute_598
    buf369 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf367, (12, 128, 64), (8192, 64, 1)), permute_599, out=buf369)
    del permute_599
    buf370 = buf209; del buf209  # reuse
    kernel56.run(philox_seed_like, buf369, div_17, buf370, 1536, 128, grid=grid(1536), stream=stream0)
    buf371 = as_strided(buf369, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf369  # reuse
    kernel57.run(buf371, philox_seed_like, div_17, buf370, 196608, grid=grid(196608), stream=stream0)
    del div_17
    buf372 = as_strided(buf367, (12, 64, 128), (8192, 128, 1)); del buf367  # reuse
    aten.bmm.out(permute_600, as_strided(buf371, (12, 128, 128), (16384, 128, 1)), out=buf372)
    del permute_600
    buf373 = as_strided(buf335, (12, 128, 64), (8192, 64, 1)); del buf335  # reuse
    aten.bmm.out(as_strided(buf371, (12, 128, 128), (16384, 128, 1)), permute_601, out=buf373)
    del permute_601
    buf374 = as_strided(buf331, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf331  # reuse
    kernel21.run(tangents_14, buf368, buf374, 98304, grid=grid(98304), stream=stream0)
    del tangents_14
    buf375 = as_strided(buf371, (384, 512), (512, 1)); del buf371  # reuse
    aten.mm.out(as_strided(buf374, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf375)
    buf376 = as_strided(buf364, (256, 512), (512, 1)); del buf364  # reuse
    aten.mm.out(as_strided(buf374, (256, 384), (384, 1)), permute_606, out=buf376)
    del permute_606
    buf377 = buf374; del buf374  # reuse
    kernel22.run(tangents_13, buf372, buf377, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_13
    buf378 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf377, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf378)
    buf379 = as_strided(buf349, (256, 512), (512, 1)); del buf349  # reuse
    aten.mm.out(as_strided(buf377, (256, 384), (384, 1)), permute_611, out=buf379)
    del permute_611
    buf380 = buf377; del buf377  # reuse
    kernel23.run(buf373, buf380, 98304, grid=grid(98304), stream=stream0)
    buf381 = as_strided(buf342, (2, 128, 512), (65536, 512, 1)); del buf342  # reuse
    kernel1.run(primals_25, add_97, reciprocal_34, buf381, 131072, grid=grid(131072), stream=stream0)
    buf382 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf380, (384, 256), (1, 384)), as_strided(buf381, (256, 512), (512, 1)), out=buf382)
    buf383 = as_strided(buf381, (256, 512), (512, 1)); del buf381  # reuse
    aten.mm.out(as_strided(buf380, (256, 384), (384, 1)), permute_616, out=buf383)
    del permute_616
    buf384 = buf360; del buf360  # reuse
    kernel24.run(buf383, add_97, reciprocal_34, buf384, 1024, 128, grid=grid(1024), stream=stream0)
    buf385 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf384, buf385, 512, 2, grid=grid(512), stream=stream0)
    buf386 = buf362; del buf362  # reuse
    kernel25.run(buf383, primals_25, add_97, buf386, 256, 512, grid=grid(256), stream=stream0)
    buf387 = buf363; del buf363  # reuse
    kernel26.run(buf387, buf383, primals_25, reciprocal_34, buf386, add_97, 131072, grid=grid(131072), stream=stream0)
    del add_97
    del primals_25
    del reciprocal_34
    buf388 = as_strided(buf383, (2, 128, 512), (65536, 512, 1)); del buf383  # reuse
    kernel11.run(buf387, gt_49, buf388, 131072, grid=grid(131072), stream=stream0)
    del gt_49
    buf389 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf388, (512, 256), (1, 512)), view_148, out=buf389)
    del view_148
    buf390 = as_strided(buf380, (256, 384), (384, 1)); del buf380  # reuse
    aten.mm.out(as_strided(buf388, (256, 512), (512, 1)), permute_620, out=buf390)
    del permute_620
    buf391 = as_strided(buf373, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf373  # reuse
    kernel18.run(buf390, buf391, 98304, grid=grid(98304), stream=stream0)
    buf392 = as_strided(buf390, (12, 128, 64), (8192, 64, 1)); del buf390  # reuse
    aten.bmm.out(permute_623, as_strided(buf391, (12, 128, 64), (8192, 64, 1)), out=buf392)
    del permute_623
    buf393 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf391, (12, 128, 64), (8192, 64, 1)), permute_624, out=buf393)
    del permute_624
    buf394 = buf370; del buf370  # reuse
    kernel58.run(philox_seed_like, buf393, div_16, buf394, 1536, 128, grid=grid(1536), stream=stream0)
    buf395 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel59.run(philox_seed_like, buf393, div_16, buf394, buf395, 196608, grid=grid(196608), stream=stream0)
    buf396 = as_strided(buf391, (12, 64, 128), (8192, 128, 1)); del buf391  # reuse
    aten.bmm.out(permute_625, as_strided(buf395, (12, 128, 128), (16384, 128, 1)), out=buf396)
    del permute_625
    buf397 = as_strided(buf372, (12, 128, 64), (8192, 64, 1)); del buf372  # reuse
    aten.bmm.out(as_strided(buf395, (12, 128, 128), (16384, 128, 1)), permute_626, out=buf397)
    del permute_626
    buf398 = as_strided(buf368, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf368  # reuse
    kernel21.run(tangents_12, buf392, buf398, 98304, grid=grid(98304), stream=stream0)
    del tangents_12
    buf399 = as_strided(buf395, (384, 512), (512, 1)); del buf395  # reuse
    aten.mm.out(as_strided(buf398, (384, 256), (1, 384)), as_strided(buf21, (256, 512), (512, 1)), out=buf399)
    buf400 = as_strided(buf388, (256, 512), (512, 1)); del buf388  # reuse
    aten.mm.out(as_strided(buf398, (256, 384), (384, 1)), permute_631, out=buf400)
    del permute_631
    buf401 = buf398; del buf398  # reuse
    kernel22.run(tangents_11, buf396, buf401, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_11
    buf402 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf401, (384, 256), (1, 384)), as_strided(buf21, (256, 512), (512, 1)), out=buf402)
    buf403 = as_strided(buf326, (256, 512), (512, 1)); del buf326  # reuse
    aten.mm.out(as_strided(buf401, (256, 384), (384, 1)), permute_636, out=buf403)
    del permute_636
    buf404 = buf401; del buf401  # reuse
    kernel23.run(buf397, buf404, 98304, grid=grid(98304), stream=stream0)
    buf405 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf404, (384, 256), (1, 384)), as_strided(buf21, (256, 512), (512, 1)), out=buf405)
    buf406 = as_strided(buf21, (256, 512), (512, 1)); del buf21  # reuse
    aten.mm.out(as_strided(buf404, (256, 384), (384, 1)), permute_641, out=buf406)
    del permute_641
    buf407 = buf384; del buf384  # reuse
    kernel29.run(buf400, buf403, buf406, add_94, reciprocal_33, buf407, 1024, 128, grid=grid(1024), stream=stream0)
    buf408 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf407, buf408, 512, 2, grid=grid(512), stream=stream0)
    buf409 = buf386; del buf386  # reuse
    kernel30.run(buf400, buf403, buf406, primals_24, add_94, buf409, 256, 512, grid=grid(256), stream=stream0)
    buf410 = as_strided(buf406, (2, 128, 512), (65536, 512, 1)); del buf406  # reuse
    kernel41.run(buf410, buf387, buf400, buf403, primals_24, reciprocal_33, buf409, add_94, 131072, grid=grid(131072), stream=stream0)
    del add_94
    del primals_24
    del reciprocal_33
    buf411 = as_strided(buf403, (2, 128, 512), (65536, 512, 1)); del buf403  # reuse
    kernel11.run(buf410, gt_47, buf411, 131072, grid=grid(131072), stream=stream0)
    del gt_47
    buf412 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf411, (512, 256), (1, 512)), view_139, out=buf412)
    del view_139
    buf413 = as_strided(buf357, (256, 1024), (1024, 1)); del buf357  # reuse
    aten.mm.out(as_strided(buf411, (256, 512), (512, 1)), permute_645, out=buf413)
    del permute_645
    buf414 = buf356; del buf356  # reuse
    kernel12.run(buf413, gt_46, mm_75, sub_26, buf414, 262144, grid=grid(262144), stream=stream0)
    buf415 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf414, (1024, 256), (1, 1024)), as_strided(buf20, (256, 512), (512, 1)), out=buf415)
    buf416 = as_strided(buf411, (256, 512), (512, 1)); del buf411  # reuse
    aten.mm.out(as_strided(buf414, (256, 1024), (1024, 1)), permute_649, out=buf416)
    del permute_649
    buf417 = buf414; del buf414  # reuse
    kernel13.run(buf413, gt_46, mm_76, mm_75, sub_26, buf417, 262144, grid=grid(262144), stream=stream0)
    buf418 = as_strided(buf413, (2, 128, 1024), (131072, 1024, 1)); del buf413  # reuse
    kernel42.run(buf418, buf417, mm_75, gt_46, mm_76, sub_26, 262144, grid=grid(262144), stream=stream0)
    del gt_46
    del mm_75
    del mm_76
    del sub_26
    buf419 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf418, (1024, 256), (1, 1024)), as_strided(buf20, (256, 512), (512, 1)), out=buf419)
    buf420 = as_strided(buf20, (256, 512), (512, 1)); del buf20  # reuse
    aten.mm.out(as_strided(buf418, (256, 1024), (1024, 1)), permute_653, out=buf420)
    del permute_653
    buf421 = buf407; del buf407  # reuse
    kernel15.run(buf416, buf420, add_89, reciprocal_31, buf421, 1024, 128, grid=grid(1024), stream=stream0)
    buf422 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf421, buf422, 512, 2, grid=grid(512), stream=stream0)
    buf423 = buf409; del buf409  # reuse
    kernel16.run(buf416, buf420, primals_23, add_89, buf423, 256, 512, grid=grid(256), stream=stream0)
    buf424 = as_strided(buf416, (2, 128, 512), (65536, 512, 1)); del buf416  # reuse
    kernel50.run(buf424, buf410, buf420, primals_23, reciprocal_31, buf423, add_89, 131072, grid=grid(131072), stream=stream0)
    del add_89
    del primals_23
    del reciprocal_31
    buf425 = as_strided(buf420, (2, 128, 512), (65536, 512, 1)); del buf420  # reuse
    kernel11.run(buf424, gt_45, buf425, 131072, grid=grid(131072), stream=stream0)
    del gt_45
    buf426 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf425, (512, 256), (1, 512)), view_136, out=buf426)
    del view_136
    buf427 = as_strided(buf404, (256, 384), (384, 1)); del buf404  # reuse
    aten.mm.out(as_strided(buf425, (256, 512), (512, 1)), permute_657, out=buf427)
    del permute_657
    buf428 = as_strided(buf397, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf397  # reuse
    kernel18.run(buf427, buf428, 98304, grid=grid(98304), stream=stream0)
    buf429 = as_strided(buf427, (12, 128, 64), (8192, 64, 1)); del buf427  # reuse
    aten.bmm.out(permute_660, as_strided(buf428, (12, 128, 64), (8192, 64, 1)), out=buf429)
    del permute_660
    buf430 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf428, (12, 128, 64), (8192, 64, 1)), permute_661, out=buf430)
    del permute_661
    buf431 = buf148; del buf148  # reuse
    kernel60.run(philox_seed_like, buf430, div_15, buf431, 1536, 128, grid=grid(1536), stream=stream0)
    buf432 = as_strided(buf430, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf430  # reuse
    kernel61.run(buf432, philox_seed_like, div_15, buf431, 196608, grid=grid(196608), stream=stream0)
    del div_15
    buf433 = as_strided(buf428, (12, 64, 128), (8192, 128, 1)); del buf428  # reuse
    aten.bmm.out(permute_662, as_strided(buf432, (12, 128, 128), (16384, 128, 1)), out=buf433)
    del permute_662
    buf434 = as_strided(buf396, (12, 128, 64), (8192, 64, 1)); del buf396  # reuse
    aten.bmm.out(as_strided(buf432, (12, 128, 128), (16384, 128, 1)), permute_663, out=buf434)
    del permute_663
    buf435 = as_strided(buf392, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf392  # reuse
    kernel21.run(tangents_10, buf429, buf435, 98304, grid=grid(98304), stream=stream0)
    del tangents_10
    buf436 = as_strided(buf432, (384, 512), (512, 1)); del buf432  # reuse
    aten.mm.out(as_strided(buf435, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf436)
    buf437 = as_strided(buf425, (256, 512), (512, 1)); del buf425  # reuse
    aten.mm.out(as_strided(buf435, (256, 384), (384, 1)), permute_668, out=buf437)
    del permute_668
    buf438 = buf435; del buf435  # reuse
    kernel22.run(tangents_9, buf433, buf438, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_9
    buf439 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf438, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf439)
    buf440 = as_strided(buf410, (256, 512), (512, 1)); del buf410  # reuse
    aten.mm.out(as_strided(buf438, (256, 384), (384, 1)), permute_673, out=buf440)
    del permute_673
    buf441 = buf438; del buf438  # reuse
    kernel23.run(buf434, buf441, 98304, grid=grid(98304), stream=stream0)
    buf442 = as_strided(buf400, (2, 128, 512), (65536, 512, 1)); del buf400  # reuse
    kernel1.run(primals_22, add_86, reciprocal_30, buf442, 131072, grid=grid(131072), stream=stream0)
    buf443 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf441, (384, 256), (1, 384)), as_strided(buf442, (256, 512), (512, 1)), out=buf443)
    buf444 = as_strided(buf442, (256, 512), (512, 1)); del buf442  # reuse
    aten.mm.out(as_strided(buf441, (256, 384), (384, 1)), permute_678, out=buf444)
    del permute_678
    buf445 = buf421; del buf421  # reuse
    kernel24.run(buf444, add_86, reciprocal_30, buf445, 1024, 128, grid=grid(1024), stream=stream0)
    buf446 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf445, buf446, 512, 2, grid=grid(512), stream=stream0)
    buf447 = buf423; del buf423  # reuse
    kernel25.run(buf444, primals_22, add_86, buf447, 256, 512, grid=grid(256), stream=stream0)
    buf448 = buf424; del buf424  # reuse
    kernel26.run(buf448, buf444, primals_22, reciprocal_30, buf447, add_86, 131072, grid=grid(131072), stream=stream0)
    del add_86
    del primals_22
    del reciprocal_30
    buf449 = as_strided(buf444, (2, 128, 512), (65536, 512, 1)); del buf444  # reuse
    kernel11.run(buf448, gt_43, buf449, 131072, grid=grid(131072), stream=stream0)
    del gt_43
    buf450 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf449, (512, 256), (1, 512)), view_127, out=buf450)
    del view_127
    buf451 = as_strided(buf441, (256, 384), (384, 1)); del buf441  # reuse
    aten.mm.out(as_strided(buf449, (256, 512), (512, 1)), permute_682, out=buf451)
    del permute_682
    buf452 = as_strided(buf434, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf434  # reuse
    kernel18.run(buf451, buf452, 98304, grid=grid(98304), stream=stream0)
    buf453 = as_strided(buf451, (12, 128, 64), (8192, 64, 1)); del buf451  # reuse
    aten.bmm.out(permute_685, as_strided(buf452, (12, 128, 64), (8192, 64, 1)), out=buf453)
    del permute_685
    buf454 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf452, (12, 128, 64), (8192, 64, 1)), permute_686, out=buf454)
    del permute_686
    buf455 = buf431; del buf431  # reuse
    kernel62.run(philox_seed_like, buf454, div_14, buf455, 1536, 128, grid=grid(1536), stream=stream0)
    buf456 = buf272; del buf272  # reuse
    kernel63.run(buf456, philox_seed_like, buf332, div_18, buf333, buf393, div_16, buf394, buf454, div_14, buf455, 196608, grid=grid(196608), stream=stream0)
    del div_16
    del div_18
    buf457 = as_strided(buf454, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf454  # reuse
    kernel64.run(buf457, philox_seed_like, div_14, buf455, 196608, grid=grid(196608), stream=stream0)
    del div_14
    buf458 = as_strided(buf452, (12, 64, 128), (8192, 128, 1)); del buf452  # reuse
    aten.bmm.out(permute_687, as_strided(buf457, (12, 128, 128), (16384, 128, 1)), out=buf458)
    del permute_687
    buf459 = as_strided(buf433, (12, 128, 64), (8192, 64, 1)); del buf433  # reuse
    aten.bmm.out(as_strided(buf457, (12, 128, 128), (16384, 128, 1)), permute_688, out=buf459)
    del permute_688
    buf460 = as_strided(buf429, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf429  # reuse
    kernel21.run(tangents_8, buf453, buf460, 98304, grid=grid(98304), stream=stream0)
    del tangents_8
    buf461 = as_strided(buf457, (384, 512), (512, 1)); del buf457  # reuse
    aten.mm.out(as_strided(buf460, (384, 256), (1, 384)), as_strided(buf19, (256, 512), (512, 1)), out=buf461)
    buf462 = as_strided(buf449, (256, 512), (512, 1)); del buf449  # reuse
    aten.mm.out(as_strided(buf460, (256, 384), (384, 1)), permute_693, out=buf462)
    del permute_693
    buf463 = buf460; del buf460  # reuse
    kernel22.run(tangents_7, buf458, buf463, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_7
    buf464 = as_strided(buf393, (384, 512), (512, 1)); del buf393  # reuse
    aten.mm.out(as_strided(buf463, (384, 256), (1, 384)), as_strided(buf19, (256, 512), (512, 1)), out=buf464)
    buf465 = as_strided(buf387, (256, 512), (512, 1)); del buf387  # reuse
    aten.mm.out(as_strided(buf463, (256, 384), (384, 1)), permute_698, out=buf465)
    del permute_698
    buf466 = buf463; del buf463  # reuse
    kernel23.run(buf459, buf466, 98304, grid=grid(98304), stream=stream0)
    buf467 = as_strided(buf332, (384, 512), (512, 1)); del buf332  # reuse
    aten.mm.out(as_strided(buf466, (384, 256), (1, 384)), as_strided(buf19, (256, 512), (512, 1)), out=buf467)
    buf468 = as_strided(buf19, (256, 512), (512, 1)); del buf19  # reuse
    aten.mm.out(as_strided(buf466, (256, 384), (384, 1)), permute_703, out=buf468)
    del permute_703
    buf469 = buf445; del buf445  # reuse
    kernel29.run(buf462, buf465, buf468, add_83, reciprocal_29, buf469, 1024, 128, grid=grid(1024), stream=stream0)
    buf470 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf469, buf470, 512, 2, grid=grid(512), stream=stream0)
    buf471 = buf447; del buf447  # reuse
    kernel30.run(buf462, buf465, buf468, primals_21, add_83, buf471, 256, 512, grid=grid(256), stream=stream0)
    buf472 = buf448; del buf448  # reuse
    kernel31.run(buf472, buf462, buf465, buf468, primals_21, reciprocal_29, buf471, add_83, 131072, grid=grid(131072), stream=stream0)
    del add_83
    del buf462
    del buf465
    del primals_21
    del reciprocal_29
    buf473 = as_strided(buf468, (2, 128, 512), (65536, 512, 1)); del buf468  # reuse
    kernel11.run(buf472, gt_41, buf473, 131072, grid=grid(131072), stream=stream0)
    del gt_41
    buf474 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf473, (512, 256), (1, 512)), view_118, out=buf474)
    del view_118
    buf475 = as_strided(buf418, (256, 1024), (1024, 1)); del buf418  # reuse
    aten.mm.out(as_strided(buf473, (256, 512), (512, 1)), permute_707, out=buf475)
    del permute_707
    buf476 = buf417; del buf417  # reuse
    kernel12.run(buf475, gt_40, mm_64, sub_23, buf476, 262144, grid=grid(262144), stream=stream0)
    buf477 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf476, (1024, 256), (1, 1024)), as_strided(buf18, (256, 512), (512, 1)), out=buf477)
    buf478 = as_strided(buf473, (256, 512), (512, 1)); del buf473  # reuse
    aten.mm.out(as_strided(buf476, (256, 1024), (1024, 1)), permute_711, out=buf478)
    del permute_711
    buf479 = buf476; del buf476  # reuse
    kernel13.run(buf475, gt_40, mm_65, mm_64, sub_23, buf479, 262144, grid=grid(262144), stream=stream0)
    buf480 = buf479; del buf479  # reuse
    kernel14.run(buf480, mm_64, buf475, gt_40, mm_65, sub_23, 262144, grid=grid(262144), stream=stream0)
    del gt_40
    del mm_64
    del mm_65
    del sub_23
    buf481 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf480, (1024, 256), (1, 1024)), as_strided(buf18, (256, 512), (512, 1)), out=buf481)
    buf482 = as_strided(buf18, (256, 512), (512, 1)); del buf18  # reuse
    aten.mm.out(as_strided(buf480, (256, 1024), (1024, 1)), permute_715, out=buf482)
    del permute_715
    buf483 = buf469; del buf469  # reuse
    kernel15.run(buf478, buf482, add_78, reciprocal_27, buf483, 1024, 128, grid=grid(1024), stream=stream0)
    buf484 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf483, buf484, 512, 2, grid=grid(512), stream=stream0)
    buf485 = buf471; del buf471  # reuse
    kernel16.run(buf478, buf482, primals_20, add_78, buf485, 256, 512, grid=grid(256), stream=stream0)
    buf486 = as_strided(buf482, (2, 128, 512), (65536, 512, 1)); del buf482  # reuse
    kernel17.run(buf486, buf472, buf478, primals_20, reciprocal_27, buf485, add_78, 131072, grid=grid(131072), stream=stream0)
    del add_78
    del buf472
    del primals_20
    del reciprocal_27
    buf487 = as_strided(buf478, (2, 128, 512), (65536, 512, 1)); del buf478  # reuse
    kernel11.run(buf486, gt_39, buf487, 131072, grid=grid(131072), stream=stream0)
    del gt_39
    buf488 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf487, (512, 256), (1, 512)), view_115, out=buf488)
    del view_115
    buf489 = as_strided(buf466, (256, 384), (384, 1)); del buf466  # reuse
    aten.mm.out(as_strided(buf487, (256, 512), (512, 1)), permute_719, out=buf489)
    del permute_719
    buf490 = as_strided(buf459, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf459  # reuse
    kernel18.run(buf489, buf490, 98304, grid=grid(98304), stream=stream0)
    buf491 = as_strided(buf489, (12, 128, 64), (8192, 64, 1)); del buf489  # reuse
    aten.bmm.out(permute_722, as_strided(buf490, (12, 128, 64), (8192, 64, 1)), out=buf491)
    del permute_722
    buf492 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf490, (12, 128, 64), (8192, 64, 1)), permute_723, out=buf492)
    del permute_723
    buf493 = buf455; del buf455  # reuse
    kernel65.run(philox_seed_like, buf492, div_13, buf493, 1536, 128, grid=grid(1536), stream=stream0)
    buf494 = as_strided(buf492, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf492  # reuse
    kernel66.run(buf494, philox_seed_like, div_13, buf493, 196608, grid=grid(196608), stream=stream0)
    del div_13
    buf495 = as_strided(buf490, (12, 64, 128), (8192, 128, 1)); del buf490  # reuse
    aten.bmm.out(permute_724, as_strided(buf494, (12, 128, 128), (16384, 128, 1)), out=buf495)
    del permute_724
    buf496 = as_strided(buf458, (12, 128, 64), (8192, 64, 1)); del buf458  # reuse
    aten.bmm.out(as_strided(buf494, (12, 128, 128), (16384, 128, 1)), permute_725, out=buf496)
    del permute_725
    buf497 = as_strided(buf453, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf453  # reuse
    kernel21.run(tangents_6, buf491, buf497, 98304, grid=grid(98304), stream=stream0)
    del tangents_6
    buf498 = as_strided(buf494, (384, 512), (512, 1)); del buf494  # reuse
    aten.mm.out(as_strided(buf497, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf498)
    buf499 = as_strided(buf487, (256, 512), (512, 1)); del buf487  # reuse
    aten.mm.out(as_strided(buf497, (256, 384), (384, 1)), permute_730, out=buf499)
    del permute_730
    buf500 = buf497; del buf497  # reuse
    kernel22.run(tangents_5, buf495, buf500, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_5
    buf501 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf500, (384, 256), (1, 384)), as_strided(buf17, (256, 512), (512, 1)), out=buf501)
    buf502 = as_strided(buf17, (256, 512), (512, 1)); del buf17  # reuse
    aten.mm.out(as_strided(buf500, (256, 384), (384, 1)), permute_735, out=buf502)
    del permute_735
    buf503 = as_strided(buf499, (2, 128, 512), (65536, 512, 1)); del buf499  # reuse
    kernel67.run(buf503, buf256, buf315, buf318, buf376, buf379, buf437, buf440, buf502, 131072, grid=grid(131072), stream=stream0)
    del buf256
    del buf315
    del buf318
    del buf376
    del buf379
    del buf437
    buf504 = buf500; del buf500  # reuse
    kernel23.run(buf496, buf504, 98304, grid=grid(98304), stream=stream0)
    buf505 = as_strided(buf502, (2, 128, 512), (65536, 512, 1)); del buf502  # reuse
    kernel1.run(primals_19, add_74, reciprocal_26, buf505, 131072, grid=grid(131072), stream=stream0)
    buf506 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf504, (384, 256), (1, 384)), as_strided(buf505, (256, 512), (512, 1)), out=buf506)
    buf507 = as_strided(buf505, (256, 512), (512, 1)); del buf505  # reuse
    aten.mm.out(as_strided(buf504, (256, 384), (384, 1)), permute_740, out=buf507)
    del permute_740
    buf508 = buf483; del buf483  # reuse
    kernel24.run(buf507, add_74, reciprocal_26, buf508, 1024, 128, grid=grid(1024), stream=stream0)
    buf509 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf508, buf509, 512, 2, grid=grid(512), stream=stream0)
    buf510 = buf485; del buf485  # reuse
    kernel25.run(buf507, primals_19, add_74, buf510, 256, 512, grid=grid(256), stream=stream0)
    buf511 = as_strided(buf507, (2, 128, 512), (65536, 512, 1)); del buf507  # reuse
    kernel46.run(buf511, buf486, primals_19, reciprocal_26, buf510, add_74, 131072, grid=grid(131072), stream=stream0)
    del add_74
    del primals_19
    del reciprocal_26
    buf512 = buf486; del buf486  # reuse
    kernel11.run(buf511, gt_37, buf512, 131072, grid=grid(131072), stream=stream0)
    del gt_37
    buf513 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf512, (512, 256), (1, 512)), view_106, out=buf513)
    del view_106
    buf514 = as_strided(buf504, (256, 384), (384, 1)); del buf504  # reuse
    aten.mm.out(as_strided(buf512, (256, 512), (512, 1)), permute_744, out=buf514)
    del permute_744
    buf515 = as_strided(buf496, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf496  # reuse
    kernel18.run(buf514, buf515, 98304, grid=grid(98304), stream=stream0)
    buf516 = as_strided(buf514, (12, 128, 64), (8192, 64, 1)); del buf514  # reuse
    aten.bmm.out(permute_747, as_strided(buf515, (12, 128, 64), (8192, 64, 1)), out=buf516)
    del permute_747
    buf517 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf515, (12, 128, 64), (8192, 64, 1)), permute_748, out=buf517)
    del permute_748
    buf518 = buf493; del buf493  # reuse
    kernel68.run(philox_seed_like, buf517, div_12, buf518, 1536, 128, grid=grid(1536), stream=stream0)
    buf519 = as_strided(buf515, (1, 6, 128, 128), (98304, 16384, 128, 1)); del buf515  # reuse
    kernel69.run(buf456, philox_seed_like, buf517, div_12, buf518, buf519, 98304, grid=grid(98304), stream=stream0)
    buf520 = empty_strided((32, 6), (6, 1), device='cuda', dtype=torch.float32)
    kernel70.run(buf520, 192, grid=grid(192), stream=stream0)
    kernel71.run(view_560, buf519, buf520, 98304, grid=grid(98304), stream=stream0)
    del view_560
    buf522 = as_strided(buf517, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf517  # reuse
    kernel72.run(buf522, philox_seed_like, div_12, buf518, 196608, grid=grid(196608), stream=stream0)
    del div_12
    buf523 = as_strided(buf519, (12, 64, 128), (8192, 128, 1)); del buf519  # reuse
    aten.bmm.out(permute_750, as_strided(buf522, (12, 128, 128), (16384, 128, 1)), out=buf523)
    del permute_750
    buf524 = as_strided(buf495, (12, 128, 64), (8192, 64, 1)); del buf495  # reuse
    aten.bmm.out(as_strided(buf522, (12, 128, 128), (16384, 128, 1)), permute_751, out=buf524)
    del permute_751
    buf525 = as_strided(buf491, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf491  # reuse
    kernel21.run(tangents_4, buf516, buf525, 98304, grid=grid(98304), stream=stream0)
    del tangents_4
    buf526 = as_strided(buf522, (384, 512), (512, 1)); del buf522  # reuse
    aten.mm.out(as_strided(buf525, (384, 256), (1, 384)), as_strided(buf16, (256, 512), (512, 1)), out=buf526)
    buf527 = as_strided(buf512, (256, 512), (512, 1)); del buf512  # reuse
    aten.mm.out(as_strided(buf525, (256, 384), (384, 1)), permute_756, out=buf527)
    del permute_756
    buf528 = buf525; del buf525  # reuse
    kernel22.run(tangents_3, buf523, buf528, 256, 384, grid=grid(256, 384), stream=stream0)
    del tangents_3
    buf529 = as_strided(buf456, (384, 512), (512, 1)); del buf456  # reuse
    aten.mm.out(as_strided(buf528, (384, 256), (1, 384)), as_strided(buf16, (256, 512), (512, 1)), out=buf529)
    buf530 = buf440; del buf440  # reuse
    aten.mm.out(as_strided(buf528, (256, 384), (384, 1)), permute_761, out=buf530)
    del permute_761
    buf531 = buf528; del buf528  # reuse
    kernel23.run(buf524, buf531, 98304, grid=grid(98304), stream=stream0)
    buf532 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf531, (384, 256), (1, 384)), as_strided(buf16, (256, 512), (512, 1)), out=buf532)
    buf533 = as_strided(buf16, (256, 512), (512, 1)); del buf16  # reuse
    aten.mm.out(as_strided(buf531, (256, 384), (384, 1)), permute_766, out=buf533)
    del permute_766
    buf534 = buf508; del buf508  # reuse
    kernel73.run(buf527, buf530, buf533, gt_35, embedding_2, reciprocal_25, buf534, 1024, 128, grid=grid(1024), stream=stream0)
    buf535 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf534, buf535, 512, 2, grid=grid(512), stream=stream0)
    buf536 = buf510; del buf510  # reuse
    kernel74.run(buf527, buf530, buf533, primals_18, gt_35, embedding_2, buf536, 256, 512, grid=grid(256), stream=stream0)
    buf537 = buf511; del buf511  # reuse
    kernel75.run(buf537, buf527, buf530, buf533, primals_18, reciprocal_25, buf536, gt_35, embedding_2, 131072, grid=grid(131072), stream=stream0)
    del buf527
    del buf530
    del buf533
    del embedding_2
    del primals_18
    del reciprocal_25
    buf538 = empty_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    kernel76.run(buf538, 128057344, grid=grid(128057344), stream=stream0)
    kernel77.run(view_572, buf537, gt_35, buf538, 131072, grid=grid(131072), stream=stream0)
    del gt_35
    del view_572
    buf540 = buf534; del buf534  # reuse
    kernel7.run(buf503, gt_34, add_67, reciprocal_24, buf540, 1024, 128, grid=grid(1024), stream=stream0)
    buf541 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf540, buf541, 512, 2, grid=grid(512), stream=stream0)
    buf542 = buf536; del buf536  # reuse
    kernel9.run(buf503, gt_34, primals_17, add_67, buf542, 256, 512, grid=grid(256), stream=stream0)
    buf543 = buf503; del buf503  # reuse
    kernel10.run(buf543, gt_34, primals_17, reciprocal_24, buf542, add_67, 131072, grid=grid(131072), stream=stream0)
    del add_67
    del gt_34
    del primals_17
    del reciprocal_24
    buf544 = buf537; del buf537  # reuse
    kernel11.run(buf543, gt_33, buf544, 131072, grid=grid(131072), stream=stream0)
    del gt_33
    buf545 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf544, (512, 256), (1, 512)), view_96, out=buf545)
    del view_96
    buf546 = as_strided(buf480, (256, 1024), (1024, 1)); del buf480  # reuse
    aten.mm.out(as_strided(buf544, (256, 512), (512, 1)), permute_770, out=buf546)
    del permute_770
    buf547 = as_strided(buf475, (2, 128, 1024), (131072, 1024, 1)); del buf475  # reuse
    kernel12.run(buf546, gt_32, mm_53, sub_17, buf547, 262144, grid=grid(262144), stream=stream0)
    buf548 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf547, (1024, 256), (1, 1024)), as_strided(buf15, (256, 512), (512, 1)), out=buf548)
    buf549 = as_strided(buf544, (256, 512), (512, 1)); del buf544  # reuse
    aten.mm.out(as_strided(buf547, (256, 1024), (1024, 1)), permute_774, out=buf549)
    del permute_774
    buf550 = buf547; del buf547  # reuse
    kernel13.run(buf546, gt_32, mm_54, mm_53, sub_17, buf550, 262144, grid=grid(262144), stream=stream0)
    buf551 = as_strided(buf546, (2, 128, 1024), (131072, 1024, 1)); del buf546  # reuse
    kernel42.run(buf551, buf550, mm_53, gt_32, mm_54, sub_17, 262144, grid=grid(262144), stream=stream0)
    del gt_32
    del mm_53
    del mm_54
    del sub_17
    buf552 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf551, (1024, 256), (1, 1024)), as_strided(buf15, (256, 512), (512, 1)), out=buf552)
    buf553 = as_strided(buf15, (256, 512), (512, 1)); del buf15  # reuse
    aten.mm.out(as_strided(buf551, (256, 1024), (1024, 1)), permute_778, out=buf553)
    del permute_778
    buf554 = buf540; del buf540  # reuse
    kernel15.run(buf549, buf553, add_62, reciprocal_22, buf554, 1024, 128, grid=grid(1024), stream=stream0)
    buf555 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf554, buf555, 512, 2, grid=grid(512), stream=stream0)
    buf556 = buf542; del buf542  # reuse
    kernel16.run(buf549, buf553, primals_16, add_62, buf556, 256, 512, grid=grid(256), stream=stream0)
    buf557 = as_strided(buf553, (2, 128, 512), (65536, 512, 1)); del buf553  # reuse
    kernel17.run(buf557, buf543, buf549, primals_16, reciprocal_22, buf556, add_62, 131072, grid=grid(131072), stream=stream0)
    del add_62
    del primals_16
    del reciprocal_22
    buf558 = as_strided(buf549, (2, 128, 512), (65536, 512, 1)); del buf549  # reuse
    kernel11.run(buf557, gt_31, buf558, 131072, grid=grid(131072), stream=stream0)
    del gt_31
    buf559 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf558, (512, 256), (1, 512)), view_93, out=buf559)
    del view_93
    buf560 = as_strided(buf531, (256, 384), (384, 1)); del buf531  # reuse
    aten.mm.out(as_strided(buf558, (256, 512), (512, 1)), permute_782, out=buf560)
    del permute_782
    buf561 = as_strided(buf524, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf524  # reuse
    kernel18.run(buf560, buf561, 98304, grid=grid(98304), stream=stream0)
    buf562 = as_strided(buf560, (12, 128, 64), (8192, 64, 1)); del buf560  # reuse
    aten.bmm.out(permute_785, as_strided(buf561, (12, 128, 64), (8192, 64, 1)), out=buf562)
    del permute_785
    buf563 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf561, (12, 128, 64), (8192, 64, 1)), permute_786, out=buf563)
    del permute_786
    buf564 = buf518; del buf518  # reuse
    kernel78.run(philox_seed_like, buf563, div_9, buf564, 1536, 128, grid=grid(1536), stream=stream0)
    buf565 = as_strided(buf563, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf563  # reuse
    kernel79.run(buf565, philox_seed_like, div_9, buf564, 196608, grid=grid(196608), stream=stream0)
    del div_9
    buf566 = as_strided(buf561, (12, 64, 128), (8192, 128, 1)); del buf561  # reuse
    aten.bmm.out(permute_787, as_strided(buf565, (12, 128, 128), (16384, 128, 1)), out=buf566)
    del permute_787
    buf567 = as_strided(buf523, (12, 128, 64), (8192, 64, 1)); del buf523  # reuse
    aten.bmm.out(as_strided(buf565, (12, 128, 128), (16384, 128, 1)), permute_788, out=buf567)
    del permute_788
    buf568 = as_strided(buf516, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf516  # reuse
    kernel23.run(buf562, buf568, 98304, grid=grid(98304), stream=stream0)
    buf569 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf568, (384, 256), (1, 384)), as_strided(buf14, (256, 512), (512, 1)), out=buf569)
    buf570 = as_strided(buf558, (256, 512), (512, 1)); del buf558  # reuse
    aten.mm.out(as_strided(buf568, (256, 384), (384, 1)), permute_793, out=buf570)
    del permute_793
    buf571 = as_strided(buf568, (2, 128, 384), (49152, 384, 1)); del buf568  # reuse
    kernel80.run(buf566, buf571, 256, 384, grid=grid(256, 384), stream=stream0)
    buf572 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf571, (384, 256), (1, 384)), as_strided(buf14, (256, 512), (512, 1)), out=buf572)
    buf573 = as_strided(buf543, (256, 512), (512, 1)); del buf543  # reuse
    aten.mm.out(as_strided(buf571, (256, 384), (384, 1)), permute_798, out=buf573)
    del permute_798
    buf574 = as_strided(buf571, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf571  # reuse
    kernel23.run(buf567, buf574, 98304, grid=grid(98304), stream=stream0)
    buf575 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf574, (384, 256), (1, 384)), as_strided(buf14, (256, 512), (512, 1)), out=buf575)
    buf576 = as_strided(buf14, (256, 512), (512, 1)); del buf14  # reuse
    aten.mm.out(as_strided(buf574, (256, 384), (384, 1)), permute_803, out=buf576)
    del permute_803
    buf577 = buf554; del buf554  # reuse
    kernel29.run(buf570, buf573, buf576, add_59, reciprocal_21, buf577, 1024, 128, grid=grid(1024), stream=stream0)
    buf578 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf577, buf578, 512, 2, grid=grid(512), stream=stream0)
    buf579 = buf556; del buf556  # reuse
    kernel30.run(buf570, buf573, buf576, primals_15, add_59, buf579, 256, 512, grid=grid(256), stream=stream0)
    buf580 = as_strided(buf570, (2, 128, 512), (65536, 512, 1)); del buf570  # reuse
    kernel55.run(buf580, buf557, buf573, buf576, primals_15, reciprocal_21, buf579, add_59, 131072, grid=grid(131072), stream=stream0)
    del add_59
    del buf557
    del buf573
    del primals_15
    del reciprocal_21
    buf581 = as_strided(buf576, (2, 128, 512), (65536, 512, 1)); del buf576  # reuse
    kernel11.run(buf580, gt_29, buf581, 131072, grid=grid(131072), stream=stream0)
    del gt_29
    buf582 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf581, (512, 256), (1, 512)), view_84, out=buf582)
    del view_84
    buf583 = as_strided(buf551, (256, 1024), (1024, 1)); del buf551  # reuse
    aten.mm.out(as_strided(buf581, (256, 512), (512, 1)), permute_807, out=buf583)
    del permute_807
    buf584 = buf550; del buf550  # reuse
    kernel12.run(buf583, gt_28, mm_46, sub_15, buf584, 262144, grid=grid(262144), stream=stream0)
    buf585 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf584, (1024, 256), (1, 1024)), as_strided(buf13, (256, 512), (512, 1)), out=buf585)
    buf586 = as_strided(buf581, (256, 512), (512, 1)); del buf581  # reuse
    aten.mm.out(as_strided(buf584, (256, 1024), (1024, 1)), permute_811, out=buf586)
    del permute_811
    buf587 = buf584; del buf584  # reuse
    kernel13.run(buf583, gt_28, mm_47, mm_46, sub_15, buf587, 262144, grid=grid(262144), stream=stream0)
    buf588 = buf587; del buf587  # reuse
    kernel14.run(buf588, mm_46, buf583, gt_28, mm_47, sub_15, 262144, grid=grid(262144), stream=stream0)
    del gt_28
    del mm_46
    del mm_47
    del sub_15
    buf589 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf588, (1024, 256), (1, 1024)), as_strided(buf13, (256, 512), (512, 1)), out=buf589)
    buf590 = as_strided(buf13, (256, 512), (512, 1)); del buf13  # reuse
    aten.mm.out(as_strided(buf588, (256, 1024), (1024, 1)), permute_815, out=buf590)
    del permute_815
    buf591 = buf577; del buf577  # reuse
    kernel15.run(buf586, buf590, add_54, reciprocal_19, buf591, 1024, 128, grid=grid(1024), stream=stream0)
    buf592 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf591, buf592, 512, 2, grid=grid(512), stream=stream0)
    buf593 = buf579; del buf579  # reuse
    kernel16.run(buf586, buf590, primals_14, add_54, buf593, 256, 512, grid=grid(256), stream=stream0)
    buf594 = as_strided(buf586, (2, 128, 512), (65536, 512, 1)); del buf586  # reuse
    kernel50.run(buf594, buf580, buf590, primals_14, reciprocal_19, buf593, add_54, 131072, grid=grid(131072), stream=stream0)
    del add_54
    del primals_14
    del reciprocal_19
    buf595 = as_strided(buf590, (2, 128, 512), (65536, 512, 1)); del buf590  # reuse
    kernel11.run(buf594, gt_27, buf595, 131072, grid=grid(131072), stream=stream0)
    del gt_27
    buf596 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf595, (512, 256), (1, 512)), view_81, out=buf596)
    del view_81
    buf597 = as_strided(buf574, (256, 384), (384, 1)); del buf574  # reuse
    aten.mm.out(as_strided(buf595, (256, 512), (512, 1)), permute_819, out=buf597)
    del permute_819
    buf598 = as_strided(buf567, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf567  # reuse
    kernel18.run(buf597, buf598, 98304, grid=grid(98304), stream=stream0)
    buf599 = as_strided(buf597, (12, 128, 64), (8192, 64, 1)); del buf597  # reuse
    aten.bmm.out(permute_822, as_strided(buf598, (12, 128, 64), (8192, 64, 1)), out=buf599)
    del permute_822
    buf600 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf598, (12, 128, 64), (8192, 64, 1)), permute_823, out=buf600)
    del permute_823
    buf601 = buf564; del buf564  # reuse
    kernel81.run(philox_seed_like, buf600, div_8, buf601, 1536, 128, grid=grid(1536), stream=stream0)
    buf602 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel82.run(philox_seed_like, buf600, div_8, buf601, buf602, 196608, grid=grid(196608), stream=stream0)
    buf603 = as_strided(buf598, (12, 64, 128), (8192, 128, 1)); del buf598  # reuse
    aten.bmm.out(permute_824, as_strided(buf602, (12, 128, 128), (16384, 128, 1)), out=buf603)
    del permute_824
    buf604 = as_strided(buf566, (12, 128, 64), (8192, 64, 1)); del buf566  # reuse
    aten.bmm.out(as_strided(buf602, (12, 128, 128), (16384, 128, 1)), permute_825, out=buf604)
    del permute_825
    buf605 = as_strided(buf562, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf562  # reuse
    kernel23.run(buf599, buf605, 98304, grid=grid(98304), stream=stream0)
    buf606 = as_strided(buf602, (384, 512), (512, 1)); del buf602  # reuse
    aten.mm.out(as_strided(buf605, (384, 256), (1, 384)), as_strided(buf12, (256, 512), (512, 1)), out=buf606)
    buf607 = as_strided(buf595, (256, 512), (512, 1)); del buf595  # reuse
    aten.mm.out(as_strided(buf605, (256, 384), (384, 1)), permute_830, out=buf607)
    del permute_830
    buf608 = as_strided(buf605, (2, 128, 384), (49152, 384, 1)); del buf605  # reuse
    kernel80.run(buf603, buf608, 256, 384, grid=grid(256, 384), stream=stream0)
    buf609 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf608, (384, 256), (1, 384)), as_strided(buf12, (256, 512), (512, 1)), out=buf609)
    buf610 = as_strided(buf580, (256, 512), (512, 1)); del buf580  # reuse
    aten.mm.out(as_strided(buf608, (256, 384), (384, 1)), permute_835, out=buf610)
    del permute_835
    buf611 = as_strided(buf608, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf608  # reuse
    kernel23.run(buf604, buf611, 98304, grid=grid(98304), stream=stream0)
    buf612 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf611, (384, 256), (1, 384)), as_strided(buf12, (256, 512), (512, 1)), out=buf612)
    buf613 = as_strided(buf12, (256, 512), (512, 1)); del buf12  # reuse
    aten.mm.out(as_strided(buf611, (256, 384), (384, 1)), permute_840, out=buf613)
    del permute_840
    buf614 = buf591; del buf591  # reuse
    kernel29.run(buf607, buf610, buf613, add_51, reciprocal_18, buf614, 1024, 128, grid=grid(1024), stream=stream0)
    buf615 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf614, buf615, 512, 2, grid=grid(512), stream=stream0)
    buf616 = buf593; del buf593  # reuse
    kernel30.run(buf607, buf610, buf613, primals_13, add_51, buf616, 256, 512, grid=grid(256), stream=stream0)
    buf617 = buf594; del buf594  # reuse
    kernel31.run(buf617, buf607, buf610, buf613, primals_13, reciprocal_18, buf616, add_51, 131072, grid=grid(131072), stream=stream0)
    del add_51
    del buf607
    del buf610
    del primals_13
    del reciprocal_18
    buf618 = as_strided(buf613, (2, 128, 512), (65536, 512, 1)); del buf613  # reuse
    kernel11.run(buf617, gt_25, buf618, 131072, grid=grid(131072), stream=stream0)
    del gt_25
    buf619 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf618, (512, 256), (1, 512)), view_72, out=buf619)
    del view_72
    buf620 = as_strided(buf588, (256, 1024), (1024, 1)); del buf588  # reuse
    aten.mm.out(as_strided(buf618, (256, 512), (512, 1)), permute_844, out=buf620)
    del permute_844
    buf621 = as_strided(buf583, (2, 128, 1024), (131072, 1024, 1)); del buf583  # reuse
    kernel12.run(buf620, gt_24, mm_39, sub_13, buf621, 262144, grid=grid(262144), stream=stream0)
    buf622 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf621, (1024, 256), (1, 1024)), as_strided(buf11, (256, 512), (512, 1)), out=buf622)
    buf623 = as_strided(buf618, (256, 512), (512, 1)); del buf618  # reuse
    aten.mm.out(as_strided(buf621, (256, 1024), (1024, 1)), permute_848, out=buf623)
    del permute_848
    buf624 = buf621; del buf621  # reuse
    kernel13.run(buf620, gt_24, mm_40, mm_39, sub_13, buf624, 262144, grid=grid(262144), stream=stream0)
    buf625 = buf624; del buf624  # reuse
    kernel14.run(buf625, mm_39, buf620, gt_24, mm_40, sub_13, 262144, grid=grid(262144), stream=stream0)
    del gt_24
    del mm_39
    del mm_40
    del sub_13
    buf626 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf625, (1024, 256), (1, 1024)), as_strided(buf11, (256, 512), (512, 1)), out=buf626)
    buf627 = as_strided(buf11, (256, 512), (512, 1)); del buf11  # reuse
    aten.mm.out(as_strided(buf625, (256, 1024), (1024, 1)), permute_852, out=buf627)
    del permute_852
    buf628 = buf614; del buf614  # reuse
    kernel15.run(buf623, buf627, add_46, reciprocal_16, buf628, 1024, 128, grid=grid(1024), stream=stream0)
    buf629 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf628, buf629, 512, 2, grid=grid(512), stream=stream0)
    buf630 = buf616; del buf616  # reuse
    kernel16.run(buf623, buf627, primals_12, add_46, buf630, 256, 512, grid=grid(256), stream=stream0)
    buf631 = as_strided(buf623, (2, 128, 512), (65536, 512, 1)); del buf623  # reuse
    kernel50.run(buf631, buf617, buf627, primals_12, reciprocal_16, buf630, add_46, 131072, grid=grid(131072), stream=stream0)
    del add_46
    del primals_12
    del reciprocal_16
    buf632 = as_strided(buf627, (2, 128, 512), (65536, 512, 1)); del buf627  # reuse
    kernel11.run(buf631, gt_23, buf632, 131072, grid=grid(131072), stream=stream0)
    del gt_23
    buf633 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf632, (512, 256), (1, 512)), view_69, out=buf633)
    del view_69
    buf634 = as_strided(buf611, (256, 384), (384, 1)); del buf611  # reuse
    aten.mm.out(as_strided(buf632, (256, 512), (512, 1)), permute_856, out=buf634)
    del permute_856
    buf635 = as_strided(buf604, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf604  # reuse
    kernel18.run(buf634, buf635, 98304, grid=grid(98304), stream=stream0)
    buf636 = as_strided(buf634, (12, 128, 64), (8192, 64, 1)); del buf634  # reuse
    aten.bmm.out(permute_859, as_strided(buf635, (12, 128, 64), (8192, 64, 1)), out=buf636)
    del permute_859
    buf637 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf635, (12, 128, 64), (8192, 64, 1)), permute_860, out=buf637)
    del permute_860
    buf638 = buf394; del buf394  # reuse
    kernel83.run(philox_seed_like, buf637, div_7, buf638, 1536, 128, grid=grid(1536), stream=stream0)
    buf639 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel84.run(philox_seed_like, buf637, div_7, buf638, buf639, 196608, grid=grid(196608), stream=stream0)
    buf640 = as_strided(buf635, (12, 64, 128), (8192, 128, 1)); del buf635  # reuse
    aten.bmm.out(permute_861, as_strided(buf639, (12, 128, 128), (16384, 128, 1)), out=buf640)
    del permute_861
    buf641 = as_strided(buf603, (12, 128, 64), (8192, 64, 1)); del buf603  # reuse
    aten.bmm.out(as_strided(buf639, (12, 128, 128), (16384, 128, 1)), permute_862, out=buf641)
    del permute_862
    buf642 = as_strided(buf599, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf599  # reuse
    kernel23.run(buf636, buf642, 98304, grid=grid(98304), stream=stream0)
    buf643 = as_strided(buf639, (384, 512), (512, 1)); del buf639  # reuse
    aten.mm.out(as_strided(buf642, (384, 256), (1, 384)), as_strided(buf10, (256, 512), (512, 1)), out=buf643)
    buf644 = as_strided(buf632, (256, 512), (512, 1)); del buf632  # reuse
    aten.mm.out(as_strided(buf642, (256, 384), (384, 1)), permute_867, out=buf644)
    del permute_867
    buf645 = as_strided(buf642, (2, 128, 384), (49152, 384, 1)); del buf642  # reuse
    kernel80.run(buf640, buf645, 256, 384, grid=grid(256, 384), stream=stream0)
    buf646 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf645, (384, 256), (1, 384)), as_strided(buf10, (256, 512), (512, 1)), out=buf646)
    buf647 = as_strided(buf617, (256, 512), (512, 1)); del buf617  # reuse
    aten.mm.out(as_strided(buf645, (256, 384), (384, 1)), permute_872, out=buf647)
    del permute_872
    buf648 = as_strided(buf645, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf645  # reuse
    kernel23.run(buf641, buf648, 98304, grid=grid(98304), stream=stream0)
    buf649 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf648, (384, 256), (1, 384)), as_strided(buf10, (256, 512), (512, 1)), out=buf649)
    buf650 = as_strided(buf10, (256, 512), (512, 1)); del buf10  # reuse
    aten.mm.out(as_strided(buf648, (256, 384), (384, 1)), permute_877, out=buf650)
    del permute_877
    buf651 = buf628; del buf628  # reuse
    kernel29.run(buf644, buf647, buf650, add_43, reciprocal_15, buf651, 1024, 128, grid=grid(1024), stream=stream0)
    buf652 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf651, buf652, 512, 2, grid=grid(512), stream=stream0)
    buf653 = buf630; del buf630  # reuse
    kernel30.run(buf644, buf647, buf650, primals_11, add_43, buf653, 256, 512, grid=grid(256), stream=stream0)
    buf654 = buf631; del buf631  # reuse
    kernel31.run(buf654, buf644, buf647, buf650, primals_11, reciprocal_15, buf653, add_43, 131072, grid=grid(131072), stream=stream0)
    del add_43
    del buf644
    del buf647
    del primals_11
    del reciprocal_15
    buf655 = as_strided(buf650, (2, 128, 512), (65536, 512, 1)); del buf650  # reuse
    kernel11.run(buf654, gt_21, buf655, 131072, grid=grid(131072), stream=stream0)
    del gt_21
    buf656 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf655, (512, 256), (1, 512)), view_60, out=buf656)
    del view_60
    buf657 = as_strided(buf625, (256, 1024), (1024, 1)); del buf625  # reuse
    aten.mm.out(as_strided(buf655, (256, 512), (512, 1)), permute_881, out=buf657)
    del permute_881
    buf658 = as_strided(buf620, (2, 128, 1024), (131072, 1024, 1)); del buf620  # reuse
    kernel12.run(buf657, gt_20, mm_32, sub_11, buf658, 262144, grid=grid(262144), stream=stream0)
    buf659 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf658, (1024, 256), (1, 1024)), as_strided(buf9, (256, 512), (512, 1)), out=buf659)
    buf660 = as_strided(buf655, (256, 512), (512, 1)); del buf655  # reuse
    aten.mm.out(as_strided(buf658, (256, 1024), (1024, 1)), permute_885, out=buf660)
    del permute_885
    buf661 = buf658; del buf658  # reuse
    kernel13.run(buf657, gt_20, mm_33, mm_32, sub_11, buf661, 262144, grid=grid(262144), stream=stream0)
    buf662 = as_strided(buf657, (2, 128, 1024), (131072, 1024, 1)); del buf657  # reuse
    kernel42.run(buf662, buf661, mm_32, gt_20, mm_33, sub_11, 262144, grid=grid(262144), stream=stream0)
    del gt_20
    del mm_32
    del mm_33
    del sub_11
    buf663 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf662, (1024, 256), (1, 1024)), as_strided(buf9, (256, 512), (512, 1)), out=buf663)
    buf664 = as_strided(buf9, (256, 512), (512, 1)); del buf9  # reuse
    aten.mm.out(as_strided(buf662, (256, 1024), (1024, 1)), permute_889, out=buf664)
    del permute_889
    buf665 = buf651; del buf651  # reuse
    kernel15.run(buf660, buf664, add_38, reciprocal_13, buf665, 1024, 128, grid=grid(1024), stream=stream0)
    buf666 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf665, buf666, 512, 2, grid=grid(512), stream=stream0)
    buf667 = buf653; del buf653  # reuse
    kernel16.run(buf660, buf664, primals_10, add_38, buf667, 256, 512, grid=grid(256), stream=stream0)
    buf668 = as_strided(buf660, (2, 128, 512), (65536, 512, 1)); del buf660  # reuse
    kernel50.run(buf668, buf654, buf664, primals_10, reciprocal_13, buf667, add_38, 131072, grid=grid(131072), stream=stream0)
    del add_38
    del primals_10
    del reciprocal_13
    buf669 = as_strided(buf664, (2, 128, 512), (65536, 512, 1)); del buf664  # reuse
    kernel11.run(buf668, gt_19, buf669, 131072, grid=grid(131072), stream=stream0)
    del gt_19
    buf670 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf669, (512, 256), (1, 512)), view_57, out=buf670)
    del view_57
    buf671 = as_strided(buf648, (256, 384), (384, 1)); del buf648  # reuse
    aten.mm.out(as_strided(buf669, (256, 512), (512, 1)), permute_893, out=buf671)
    del permute_893
    buf672 = as_strided(buf641, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf641  # reuse
    kernel18.run(buf671, buf672, 98304, grid=grid(98304), stream=stream0)
    buf673 = as_strided(buf671, (12, 128, 64), (8192, 64, 1)); del buf671  # reuse
    aten.bmm.out(permute_896, as_strided(buf672, (12, 128, 64), (8192, 64, 1)), out=buf673)
    del permute_896
    buf674 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf672, (12, 128, 64), (8192, 64, 1)), permute_897, out=buf674)
    del permute_897
    buf675 = buf333; del buf333  # reuse
    kernel85.run(philox_seed_like, buf674, div_6, buf675, 1536, 128, grid=grid(1536), stream=stream0)
    buf676 = as_strided(buf600, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf600  # reuse
    kernel86.run(buf676, buf565, philox_seed_like, div_8, buf601, buf637, div_7, buf638, buf674, div_6, buf675, 196608, grid=grid(196608), stream=stream0)
    del div_7
    del div_8
    buf677 = as_strided(buf674, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf674  # reuse
    kernel87.run(buf677, philox_seed_like, div_6, buf675, 196608, grid=grid(196608), stream=stream0)
    del div_6
    buf678 = as_strided(buf672, (12, 64, 128), (8192, 128, 1)); del buf672  # reuse
    aten.bmm.out(permute_898, as_strided(buf677, (12, 128, 128), (16384, 128, 1)), out=buf678)
    del permute_898
    buf679 = as_strided(buf640, (12, 128, 64), (8192, 64, 1)); del buf640  # reuse
    aten.bmm.out(as_strided(buf677, (12, 128, 128), (16384, 128, 1)), permute_899, out=buf679)
    del permute_899
    buf680 = as_strided(buf636, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf636  # reuse
    kernel23.run(buf673, buf680, 98304, grid=grid(98304), stream=stream0)
    buf681 = as_strided(buf677, (384, 512), (512, 1)); del buf677  # reuse
    aten.mm.out(as_strided(buf680, (384, 256), (1, 384)), as_strided(buf8, (256, 512), (512, 1)), out=buf681)
    buf682 = as_strided(buf669, (256, 512), (512, 1)); del buf669  # reuse
    aten.mm.out(as_strided(buf680, (256, 384), (384, 1)), permute_904, out=buf682)
    del permute_904
    buf683 = as_strided(buf680, (2, 128, 384), (49152, 384, 1)); del buf680  # reuse
    kernel80.run(buf678, buf683, 256, 384, grid=grid(256, 384), stream=stream0)
    buf684 = as_strided(buf637, (384, 512), (512, 1)); del buf637  # reuse
    aten.mm.out(as_strided(buf683, (384, 256), (1, 384)), as_strided(buf8, (256, 512), (512, 1)), out=buf684)
    buf685 = as_strided(buf654, (256, 512), (512, 1)); del buf654  # reuse
    aten.mm.out(as_strided(buf683, (256, 384), (384, 1)), permute_909, out=buf685)
    del permute_909
    buf686 = as_strided(buf683, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf683  # reuse
    kernel23.run(buf679, buf686, 98304, grid=grid(98304), stream=stream0)
    buf687 = as_strided(buf565, (384, 512), (512, 1)); del buf565  # reuse
    aten.mm.out(as_strided(buf686, (384, 256), (1, 384)), as_strided(buf8, (256, 512), (512, 1)), out=buf687)
    buf688 = as_strided(buf8, (256, 512), (512, 1)); del buf8  # reuse
    aten.mm.out(as_strided(buf686, (256, 384), (384, 1)), permute_914, out=buf688)
    del permute_914
    buf689 = buf665; del buf665  # reuse
    kernel29.run(buf682, buf685, buf688, add_35, reciprocal_12, buf689, 1024, 128, grid=grid(1024), stream=stream0)
    buf690 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf689, buf690, 512, 2, grid=grid(512), stream=stream0)
    buf691 = buf667; del buf667  # reuse
    kernel30.run(buf682, buf685, buf688, primals_9, add_35, buf691, 256, 512, grid=grid(256), stream=stream0)
    buf692 = buf668; del buf668  # reuse
    kernel31.run(buf692, buf682, buf685, buf688, primals_9, reciprocal_12, buf691, add_35, 131072, grid=grid(131072), stream=stream0)
    del add_35
    del buf682
    del buf685
    del primals_9
    del reciprocal_12
    buf693 = as_strided(buf688, (2, 128, 512), (65536, 512, 1)); del buf688  # reuse
    kernel11.run(buf692, gt_17, buf693, 131072, grid=grid(131072), stream=stream0)
    del gt_17
    buf694 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf693, (512, 256), (1, 512)), view_48, out=buf694)
    del view_48
    buf695 = as_strided(buf662, (256, 1024), (1024, 1)); del buf662  # reuse
    aten.mm.out(as_strided(buf693, (256, 512), (512, 1)), permute_918, out=buf695)
    del permute_918
    buf696 = buf661; del buf661  # reuse
    kernel12.run(buf695, gt_16, mm_25, sub_9, buf696, 262144, grid=grid(262144), stream=stream0)
    buf697 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf696, (1024, 256), (1, 1024)), as_strided(buf7, (256, 512), (512, 1)), out=buf697)
    buf698 = as_strided(buf693, (256, 512), (512, 1)); del buf693  # reuse
    aten.mm.out(as_strided(buf696, (256, 1024), (1024, 1)), permute_922, out=buf698)
    del permute_922
    buf699 = buf696; del buf696  # reuse
    kernel13.run(buf695, gt_16, mm_26, mm_25, sub_9, buf699, 262144, grid=grid(262144), stream=stream0)
    buf700 = buf699; del buf699  # reuse
    kernel14.run(buf700, mm_25, buf695, gt_16, mm_26, sub_9, 262144, grid=grid(262144), stream=stream0)
    del gt_16
    del mm_25
    del mm_26
    del sub_9
    buf701 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf700, (1024, 256), (1, 1024)), as_strided(buf7, (256, 512), (512, 1)), out=buf701)
    buf702 = as_strided(buf7, (256, 512), (512, 1)); del buf7  # reuse
    aten.mm.out(as_strided(buf700, (256, 1024), (1024, 1)), permute_926, out=buf702)
    del permute_926
    buf703 = buf689; del buf689  # reuse
    kernel15.run(buf698, buf702, add_30, reciprocal_10, buf703, 1024, 128, grid=grid(1024), stream=stream0)
    buf704 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf703, buf704, 512, 2, grid=grid(512), stream=stream0)
    buf705 = buf691; del buf691  # reuse
    kernel16.run(buf698, buf702, primals_8, add_30, buf705, 256, 512, grid=grid(256), stream=stream0)
    buf706 = as_strided(buf702, (2, 128, 512), (65536, 512, 1)); del buf702  # reuse
    kernel17.run(buf706, buf692, buf698, primals_8, reciprocal_10, buf705, add_30, 131072, grid=grid(131072), stream=stream0)
    del add_30
    del primals_8
    del reciprocal_10
    buf707 = as_strided(buf698, (2, 128, 512), (65536, 512, 1)); del buf698  # reuse
    kernel11.run(buf706, gt_15, buf707, 131072, grid=grid(131072), stream=stream0)
    del gt_15
    buf708 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf707, (512, 256), (1, 512)), view_45, out=buf708)
    del view_45
    buf709 = as_strided(buf686, (256, 384), (384, 1)); del buf686  # reuse
    aten.mm.out(as_strided(buf707, (256, 512), (512, 1)), permute_930, out=buf709)
    del permute_930
    buf710 = as_strided(buf679, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf679  # reuse
    kernel18.run(buf709, buf710, 98304, grid=grid(98304), stream=stream0)
    buf711 = as_strided(buf709, (12, 128, 64), (8192, 64, 1)); del buf709  # reuse
    aten.bmm.out(permute_933, as_strided(buf710, (12, 128, 64), (8192, 64, 1)), out=buf711)
    del permute_933
    buf712 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf710, (12, 128, 64), (8192, 64, 1)), permute_934, out=buf712)
    del permute_934
    buf713 = buf675; del buf675  # reuse
    kernel88.run(philox_seed_like, buf712, div_5, buf713, 1536, 128, grid=grid(1536), stream=stream0)
    buf714 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel89.run(philox_seed_like, buf712, div_5, buf713, buf714, 196608, grid=grid(196608), stream=stream0)
    buf715 = as_strided(buf710, (12, 64, 128), (8192, 128, 1)); del buf710  # reuse
    aten.bmm.out(permute_935, as_strided(buf714, (12, 128, 128), (16384, 128, 1)), out=buf715)
    del permute_935
    buf716 = as_strided(buf678, (12, 128, 64), (8192, 64, 1)); del buf678  # reuse
    aten.bmm.out(as_strided(buf714, (12, 128, 128), (16384, 128, 1)), permute_936, out=buf716)
    del permute_936
    buf717 = as_strided(buf673, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf673  # reuse
    kernel23.run(buf711, buf717, 98304, grid=grid(98304), stream=stream0)
    buf718 = as_strided(buf714, (384, 512), (512, 1)); del buf714  # reuse
    aten.mm.out(as_strided(buf717, (384, 256), (1, 384)), as_strided(buf6, (256, 512), (512, 1)), out=buf718)
    buf719 = as_strided(buf707, (256, 512), (512, 1)); del buf707  # reuse
    aten.mm.out(as_strided(buf717, (256, 384), (384, 1)), permute_941, out=buf719)
    del permute_941
    buf720 = as_strided(buf717, (2, 128, 384), (49152, 384, 1)); del buf717  # reuse
    kernel80.run(buf715, buf720, 256, 384, grid=grid(256, 384), stream=stream0)
    buf721 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf720, (384, 256), (1, 384)), as_strided(buf6, (256, 512), (512, 1)), out=buf721)
    buf722 = as_strided(buf692, (256, 512), (512, 1)); del buf692  # reuse
    aten.mm.out(as_strided(buf720, (256, 384), (384, 1)), permute_946, out=buf722)
    del permute_946
    buf723 = as_strided(buf720, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf720  # reuse
    kernel23.run(buf716, buf723, 98304, grid=grid(98304), stream=stream0)
    buf724 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf723, (384, 256), (1, 384)), as_strided(buf6, (256, 512), (512, 1)), out=buf724)
    buf725 = as_strided(buf6, (256, 512), (512, 1)); del buf6  # reuse
    aten.mm.out(as_strided(buf723, (256, 384), (384, 1)), permute_951, out=buf725)
    del permute_951
    buf726 = buf703; del buf703  # reuse
    kernel29.run(buf719, buf722, buf725, add_27, reciprocal_9, buf726, 1024, 128, grid=grid(1024), stream=stream0)
    buf727 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf726, buf727, 512, 2, grid=grid(512), stream=stream0)
    buf728 = buf705; del buf705  # reuse
    kernel30.run(buf719, buf722, buf725, primals_7, add_27, buf728, 256, 512, grid=grid(256), stream=stream0)
    buf729 = as_strided(buf725, (2, 128, 512), (65536, 512, 1)); del buf725  # reuse
    kernel41.run(buf729, buf706, buf719, buf722, primals_7, reciprocal_9, buf728, add_27, 131072, grid=grid(131072), stream=stream0)
    del add_27
    del buf706
    del buf719
    del primals_7
    del reciprocal_9
    buf730 = as_strided(buf722, (2, 128, 512), (65536, 512, 1)); del buf722  # reuse
    kernel11.run(buf729, gt_13, buf730, 131072, grid=grid(131072), stream=stream0)
    del gt_13
    buf731 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf730, (512, 256), (1, 512)), view_36, out=buf731)
    del view_36
    buf732 = as_strided(buf700, (256, 1024), (1024, 1)); del buf700  # reuse
    aten.mm.out(as_strided(buf730, (256, 512), (512, 1)), permute_955, out=buf732)
    del permute_955
    buf733 = as_strided(buf695, (2, 128, 1024), (131072, 1024, 1)); del buf695  # reuse
    kernel12.run(buf732, gt_12, mm_18, sub_7, buf733, 262144, grid=grid(262144), stream=stream0)
    buf734 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf733, (1024, 256), (1, 1024)), as_strided(buf5, (256, 512), (512, 1)), out=buf734)
    buf735 = as_strided(buf730, (256, 512), (512, 1)); del buf730  # reuse
    aten.mm.out(as_strided(buf733, (256, 1024), (1024, 1)), permute_959, out=buf735)
    del permute_959
    buf736 = buf733; del buf733  # reuse
    kernel13.run(buf732, gt_12, mm_19, mm_18, sub_7, buf736, 262144, grid=grid(262144), stream=stream0)
    buf737 = as_strided(buf732, (2, 128, 1024), (131072, 1024, 1)); del buf732  # reuse
    kernel42.run(buf737, buf736, mm_18, gt_12, mm_19, sub_7, 262144, grid=grid(262144), stream=stream0)
    del gt_12
    del mm_18
    del mm_19
    del sub_7
    buf738 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf737, (1024, 256), (1, 1024)), as_strided(buf5, (256, 512), (512, 1)), out=buf738)
    buf739 = as_strided(buf5, (256, 512), (512, 1)); del buf5  # reuse
    aten.mm.out(as_strided(buf737, (256, 1024), (1024, 1)), permute_963, out=buf739)
    del permute_963
    buf740 = buf726; del buf726  # reuse
    kernel15.run(buf735, buf739, add_22, reciprocal_7, buf740, 1024, 128, grid=grid(1024), stream=stream0)
    buf741 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf740, buf741, 512, 2, grid=grid(512), stream=stream0)
    buf742 = buf728; del buf728  # reuse
    kernel16.run(buf735, buf739, primals_6, add_22, buf742, 256, 512, grid=grid(256), stream=stream0)
    buf743 = as_strided(buf739, (2, 128, 512), (65536, 512, 1)); del buf739  # reuse
    kernel17.run(buf743, buf729, buf735, primals_6, reciprocal_7, buf742, add_22, 131072, grid=grid(131072), stream=stream0)
    del add_22
    del primals_6
    del reciprocal_7
    buf744 = as_strided(buf735, (2, 128, 512), (65536, 512, 1)); del buf735  # reuse
    kernel11.run(buf743, gt_11, buf744, 131072, grid=grid(131072), stream=stream0)
    del gt_11
    buf745 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf744, (512, 256), (1, 512)), view_33, out=buf745)
    del view_33
    buf746 = as_strided(buf723, (256, 384), (384, 1)); del buf723  # reuse
    aten.mm.out(as_strided(buf744, (256, 512), (512, 1)), permute_967, out=buf746)
    del permute_967
    buf747 = as_strided(buf716, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf716  # reuse
    kernel18.run(buf746, buf747, 98304, grid=grid(98304), stream=stream0)
    buf748 = as_strided(buf746, (12, 128, 64), (8192, 64, 1)); del buf746  # reuse
    aten.bmm.out(permute_970, as_strided(buf747, (12, 128, 64), (8192, 64, 1)), out=buf748)
    del permute_970
    buf749 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf747, (12, 128, 64), (8192, 64, 1)), permute_971, out=buf749)
    del permute_971
    buf750 = buf638; del buf638  # reuse
    kernel90.run(philox_seed_like, buf749, div_4, buf750, 1536, 128, grid=grid(1536), stream=stream0)
    buf751 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel91.run(philox_seed_like, buf749, div_4, buf750, buf751, 196608, grid=grid(196608), stream=stream0)
    buf752 = as_strided(buf747, (12, 64, 128), (8192, 128, 1)); del buf747  # reuse
    aten.bmm.out(permute_972, as_strided(buf751, (12, 128, 128), (16384, 128, 1)), out=buf752)
    del permute_972
    buf753 = as_strided(buf715, (12, 128, 64), (8192, 64, 1)); del buf715  # reuse
    aten.bmm.out(as_strided(buf751, (12, 128, 128), (16384, 128, 1)), permute_973, out=buf753)
    del permute_973
    buf754 = as_strided(buf711, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf711  # reuse
    kernel23.run(buf748, buf754, 98304, grid=grid(98304), stream=stream0)
    buf755 = as_strided(buf751, (384, 512), (512, 1)); del buf751  # reuse
    aten.mm.out(as_strided(buf754, (384, 256), (1, 384)), as_strided(buf4, (256, 512), (512, 1)), out=buf755)
    buf756 = as_strided(buf744, (256, 512), (512, 1)); del buf744  # reuse
    aten.mm.out(as_strided(buf754, (256, 384), (384, 1)), permute_978, out=buf756)
    del permute_978
    buf757 = as_strided(buf754, (2, 128, 384), (49152, 384, 1)); del buf754  # reuse
    kernel80.run(buf752, buf757, 256, 384, grid=grid(256, 384), stream=stream0)
    buf758 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf757, (384, 256), (1, 384)), as_strided(buf4, (256, 512), (512, 1)), out=buf758)
    buf759 = as_strided(buf729, (256, 512), (512, 1)); del buf729  # reuse
    aten.mm.out(as_strided(buf757, (256, 384), (384, 1)), permute_983, out=buf759)
    del permute_983
    buf760 = as_strided(buf757, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf757  # reuse
    kernel23.run(buf753, buf760, 98304, grid=grid(98304), stream=stream0)
    buf761 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf760, (384, 256), (1, 384)), as_strided(buf4, (256, 512), (512, 1)), out=buf761)
    buf762 = as_strided(buf4, (256, 512), (512, 1)); del buf4  # reuse
    aten.mm.out(as_strided(buf760, (256, 384), (384, 1)), permute_988, out=buf762)
    del permute_988
    buf763 = buf740; del buf740  # reuse
    kernel29.run(buf756, buf759, buf762, add_19, reciprocal_6, buf763, 1024, 128, grid=grid(1024), stream=stream0)
    buf764 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf763, buf764, 512, 2, grid=grid(512), stream=stream0)
    buf765 = buf742; del buf742  # reuse
    kernel30.run(buf756, buf759, buf762, primals_5, add_19, buf765, 256, 512, grid=grid(256), stream=stream0)
    buf766 = as_strided(buf762, (2, 128, 512), (65536, 512, 1)); del buf762  # reuse
    kernel41.run(buf766, buf743, buf756, buf759, primals_5, reciprocal_6, buf765, add_19, 131072, grid=grid(131072), stream=stream0)
    del add_19
    del buf743
    del buf756
    del primals_5
    del reciprocal_6
    buf767 = as_strided(buf759, (2, 128, 512), (65536, 512, 1)); del buf759  # reuse
    kernel11.run(buf766, gt_9, buf767, 131072, grid=grid(131072), stream=stream0)
    del gt_9
    buf768 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf767, (512, 256), (1, 512)), view_24, out=buf768)
    del view_24
    buf769 = as_strided(buf737, (256, 1024), (1024, 1)); del buf737  # reuse
    aten.mm.out(as_strided(buf767, (256, 512), (512, 1)), permute_992, out=buf769)
    del permute_992
    buf770 = buf736; del buf736  # reuse
    kernel12.run(buf769, gt_8, mm_11, sub_5, buf770, 262144, grid=grid(262144), stream=stream0)
    buf771 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf770, (1024, 256), (1, 1024)), as_strided(buf3, (256, 512), (512, 1)), out=buf771)
    buf772 = as_strided(buf767, (256, 512), (512, 1)); del buf767  # reuse
    aten.mm.out(as_strided(buf770, (256, 1024), (1024, 1)), permute_996, out=buf772)
    del permute_996
    buf773 = buf770; del buf770  # reuse
    kernel13.run(buf769, gt_8, mm_12, mm_11, sub_5, buf773, 262144, grid=grid(262144), stream=stream0)
    buf774 = buf773; del buf773  # reuse
    kernel14.run(buf774, mm_11, buf769, gt_8, mm_12, sub_5, 262144, grid=grid(262144), stream=stream0)
    del gt_8
    del mm_11
    del mm_12
    del sub_5
    buf775 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf774, (1024, 256), (1, 1024)), as_strided(buf3, (256, 512), (512, 1)), out=buf775)
    buf776 = as_strided(buf3, (256, 512), (512, 1)); del buf3  # reuse
    aten.mm.out(as_strided(buf774, (256, 1024), (1024, 1)), permute_1000, out=buf776)
    del permute_1000
    buf777 = buf763; del buf763  # reuse
    kernel15.run(buf772, buf776, add_14, reciprocal_4, buf777, 1024, 128, grid=grid(1024), stream=stream0)
    buf778 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf777, buf778, 512, 2, grid=grid(512), stream=stream0)
    buf779 = buf765; del buf765  # reuse
    kernel16.run(buf772, buf776, primals_4, add_14, buf779, 256, 512, grid=grid(256), stream=stream0)
    buf780 = buf766; del buf766  # reuse
    kernel36.run(buf780, buf772, buf776, primals_4, reciprocal_4, buf779, add_14, 131072, grid=grid(131072), stream=stream0)
    del add_14
    del primals_4
    del reciprocal_4
    buf781 = as_strided(buf776, (2, 128, 512), (65536, 512, 1)); del buf776  # reuse
    kernel11.run(buf780, gt_7, buf781, 131072, grid=grid(131072), stream=stream0)
    del gt_7
    buf782 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf781, (512, 256), (1, 512)), view_21, out=buf782)
    del view_21
    buf783 = as_strided(buf760, (256, 384), (384, 1)); del buf760  # reuse
    aten.mm.out(as_strided(buf781, (256, 512), (512, 1)), permute_1004, out=buf783)
    del permute_1004
    buf784 = as_strided(buf753, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf753  # reuse
    kernel18.run(buf783, buf784, 98304, grid=grid(98304), stream=stream0)
    buf785 = as_strided(buf783, (12, 128, 64), (8192, 64, 1)); del buf783  # reuse
    aten.bmm.out(permute_1007, as_strided(buf784, (12, 128, 64), (8192, 64, 1)), out=buf785)
    del permute_1007
    buf786 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf784, (12, 128, 64), (8192, 64, 1)), permute_1008, out=buf786)
    del permute_1008
    buf787 = buf601; del buf601  # reuse
    kernel92.run(philox_seed_like, buf786, div_3, buf787, 1536, 128, grid=grid(1536), stream=stream0)
    buf788 = as_strided(buf749, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf749  # reuse
    kernel93.run(buf788, buf676, philox_seed_like, buf712, div_5, buf713, div_4, buf750, buf786, div_3, buf787, 196608, grid=grid(196608), stream=stream0)
    del buf713
    del buf750
    del div_4
    del div_5
    buf789 = as_strided(buf786, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf786  # reuse
    kernel94.run(buf789, philox_seed_like, div_3, buf787, 196608, grid=grid(196608), stream=stream0)
    del div_3
    buf790 = as_strided(buf784, (12, 64, 128), (8192, 128, 1)); del buf784  # reuse
    aten.bmm.out(permute_1009, as_strided(buf789, (12, 128, 128), (16384, 128, 1)), out=buf790)
    del permute_1009
    buf791 = as_strided(buf752, (12, 128, 64), (8192, 64, 1)); del buf752  # reuse
    aten.bmm.out(as_strided(buf789, (12, 128, 128), (16384, 128, 1)), permute_1010, out=buf791)
    del permute_1010
    buf792 = as_strided(buf748, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf748  # reuse
    kernel23.run(buf785, buf792, 98304, grid=grid(98304), stream=stream0)
    buf793 = as_strided(buf789, (384, 512), (512, 1)); del buf789  # reuse
    aten.mm.out(as_strided(buf792, (384, 256), (1, 384)), as_strided(buf2, (256, 512), (512, 1)), out=buf793)
    buf794 = as_strided(buf781, (256, 512), (512, 1)); del buf781  # reuse
    aten.mm.out(as_strided(buf792, (256, 384), (384, 1)), permute_1015, out=buf794)
    del permute_1015
    buf795 = as_strided(buf792, (2, 128, 384), (49152, 384, 1)); del buf792  # reuse
    kernel80.run(buf790, buf795, 256, 384, grid=grid(256, 384), stream=stream0)
    buf796 = as_strided(buf712, (384, 512), (512, 1)); del buf712  # reuse
    aten.mm.out(as_strided(buf795, (384, 256), (1, 384)), as_strided(buf2, (256, 512), (512, 1)), out=buf796)
    buf797 = buf772; del buf772  # reuse
    aten.mm.out(as_strided(buf795, (256, 384), (384, 1)), permute_1020, out=buf797)
    del permute_1020
    buf798 = as_strided(buf795, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf795  # reuse
    kernel23.run(buf791, buf798, 98304, grid=grid(98304), stream=stream0)
    buf799 = as_strided(buf676, (384, 512), (512, 1)); del buf676  # reuse
    aten.mm.out(as_strided(buf798, (384, 256), (1, 384)), as_strided(buf2, (256, 512), (512, 1)), out=buf799)
    buf800 = as_strided(buf2, (256, 512), (512, 1)); del buf2  # reuse
    aten.mm.out(as_strided(buf798, (256, 384), (384, 1)), permute_1025, out=buf800)
    del permute_1025
    buf801 = buf777; del buf777  # reuse
    kernel29.run(buf794, buf797, buf800, add_11, reciprocal_3, buf801, 1024, 128, grid=grid(1024), stream=stream0)
    buf802 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf801, buf802, 512, 2, grid=grid(512), stream=stream0)
    buf803 = buf779; del buf779  # reuse
    kernel30.run(buf794, buf797, buf800, primals_3, add_11, buf803, 256, 512, grid=grid(256), stream=stream0)
    buf804 = as_strided(buf794, (2, 128, 512), (65536, 512, 1)); del buf794  # reuse
    kernel55.run(buf804, buf780, buf797, buf800, primals_3, reciprocal_3, buf803, add_11, 131072, grid=grid(131072), stream=stream0)
    del add_11
    del buf780
    del buf797
    del primals_3
    del reciprocal_3
    buf805 = as_strided(buf800, (2, 128, 512), (65536, 512, 1)); del buf800  # reuse
    kernel11.run(buf804, gt_5, buf805, 131072, grid=grid(131072), stream=stream0)
    del gt_5
    buf806 = empty_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf805, (512, 256), (1, 512)), view_12, out=buf806)
    del view_12
    buf807 = as_strided(buf774, (256, 1024), (1024, 1)); del buf774  # reuse
    aten.mm.out(as_strided(buf805, (256, 512), (512, 1)), permute_1029, out=buf807)
    del permute_1029
    buf808 = as_strided(buf769, (2, 128, 1024), (131072, 1024, 1)); del buf769  # reuse
    kernel12.run(buf807, gt_4, mm_4, sub_3, buf808, 262144, grid=grid(262144), stream=stream0)
    buf809 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf808, (1024, 256), (1, 1024)), as_strided(buf1, (256, 512), (512, 1)), out=buf809)
    buf810 = as_strided(buf805, (256, 512), (512, 1)); del buf805  # reuse
    aten.mm.out(as_strided(buf808, (256, 1024), (1024, 1)), permute_1033, out=buf810)
    del permute_1033
    buf811 = buf808; del buf808  # reuse
    kernel13.run(buf807, gt_4, mm_5, mm_4, sub_3, buf811, 262144, grid=grid(262144), stream=stream0)
    buf812 = buf811; del buf811  # reuse
    kernel14.run(buf812, mm_4, buf807, gt_4, mm_5, sub_3, 262144, grid=grid(262144), stream=stream0)
    del buf807
    del gt_4
    del mm_4
    del mm_5
    del sub_3
    buf813 = empty_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf812, (1024, 256), (1, 1024)), as_strided(buf1, (256, 512), (512, 1)), out=buf813)
    buf814 = as_strided(buf1, (256, 512), (512, 1)); del buf1  # reuse
    aten.mm.out(as_strided(buf812, (256, 1024), (1024, 1)), permute_1037, out=buf814)
    del buf812
    del permute_1037
    buf815 = buf801; del buf801  # reuse
    kernel15.run(buf810, buf814, add_6, reciprocal_1, buf815, 1024, 128, grid=grid(1024), stream=stream0)
    buf816 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf815, buf816, 512, 2, grid=grid(512), stream=stream0)
    buf817 = buf803; del buf803  # reuse
    kernel16.run(buf810, buf814, primals_2, add_6, buf817, 256, 512, grid=grid(256), stream=stream0)
    buf818 = as_strided(buf810, (2, 128, 512), (65536, 512, 1)); del buf810  # reuse
    kernel50.run(buf818, buf804, buf814, primals_2, reciprocal_1, buf817, add_6, 131072, grid=grid(131072), stream=stream0)
    del add_6
    del primals_2
    del reciprocal_1
    buf819 = as_strided(buf814, (2, 128, 512), (65536, 512, 1)); del buf814  # reuse
    kernel11.run(buf818, gt_3, buf819, 131072, grid=grid(131072), stream=stream0)
    del gt_3
    buf820 = empty_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf819, (512, 256), (1, 512)), view_9, out=buf820)
    del view_9
    buf821 = as_strided(buf798, (256, 384), (384, 1)); del buf798  # reuse
    aten.mm.out(as_strided(buf819, (256, 512), (512, 1)), permute_1041, out=buf821)
    del permute_1041
    buf822 = as_strided(buf791, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf791  # reuse
    kernel18.run(buf821, buf822, 98304, grid=grid(98304), stream=stream0)
    buf823 = as_strided(buf821, (12, 128, 64), (8192, 64, 1)); del buf821  # reuse
    aten.bmm.out(permute_1044, as_strided(buf822, (12, 128, 64), (8192, 64, 1)), out=buf823)
    del permute_1044
    buf824 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf822, (12, 128, 64), (8192, 64, 1)), permute_1045, out=buf824)
    del permute_1045
    buf825 = buf787; del buf787  # reuse
    kernel95.run(philox_seed_like, buf824, div_2, buf825, 1536, 128, grid=grid(1536), stream=stream0)
    buf826 = as_strided(buf822, (1, 6, 128, 128), (98304, 16384, 128, 1)); del buf822  # reuse
    kernel96.run(buf788, philox_seed_like, buf824, div_2, buf825, buf826, 98304, grid=grid(98304), stream=stream0)
    buf827 = empty_strided((32, 6), (6, 1), device='cuda', dtype=torch.float32)
    kernel70.run(buf827, 192, grid=grid(192), stream=stream0)
    kernel71.run(view_741, buf826, buf827, 98304, grid=grid(98304), stream=stream0)
    del view_741
    buf829 = as_strided(buf824, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf824  # reuse
    kernel97.run(buf829, philox_seed_like, div_2, buf825, 196608, grid=grid(196608), stream=stream0)
    del buf825
    del div_2
    del philox_seed_like
    buf830 = as_strided(buf826, (12, 64, 128), (8192, 128, 1)); del buf826  # reuse
    aten.bmm.out(permute_1047, as_strided(buf829, (12, 128, 128), (16384, 128, 1)), out=buf830)
    del permute_1047
    buf831 = as_strided(buf790, (12, 128, 64), (8192, 64, 1)); del buf790  # reuse
    aten.bmm.out(as_strided(buf829, (12, 128, 128), (16384, 128, 1)), permute_1048, out=buf831)
    del permute_1048
    buf832 = as_strided(buf785, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf785  # reuse
    kernel23.run(buf823, buf832, 98304, grid=grid(98304), stream=stream0)
    del buf823
    buf833 = as_strided(buf829, (384, 512), (512, 1)); del buf829  # reuse
    aten.mm.out(as_strided(buf832, (384, 256), (1, 384)), as_strided(buf0, (256, 512), (512, 1)), out=buf833)
    buf834 = as_strided(buf819, (256, 512), (512, 1)); del buf819  # reuse
    aten.mm.out(as_strided(buf832, (256, 384), (384, 1)), permute_1053, out=buf834)
    del permute_1053
    buf835 = as_strided(buf832, (2, 128, 384), (49152, 384, 1)); del buf832  # reuse
    kernel80.run(buf830, buf835, 256, 384, grid=grid(256, 384), stream=stream0)
    del buf830
    buf836 = as_strided(buf788, (384, 512), (512, 1)); del buf788  # reuse
    aten.mm.out(as_strided(buf835, (384, 256), (1, 384)), as_strided(buf0, (256, 512), (512, 1)), out=buf836)
    buf837 = as_strided(buf804, (256, 512), (512, 1)); del buf804  # reuse
    aten.mm.out(as_strided(buf835, (256, 384), (384, 1)), permute_1058, out=buf837)
    del permute_1058
    buf838 = as_strided(buf835, (2, 128, 6, 64), (49152, 384, 64, 1)); del buf835  # reuse
    kernel23.run(buf831, buf838, 98304, grid=grid(98304), stream=stream0)
    del buf831
    buf839 = empty_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf838, (384, 256), (1, 384)), as_strided(buf0, (256, 512), (512, 1)), out=buf839)
    buf840 = as_strided(buf0, (256, 512), (512, 1)); del buf0  # reuse
    aten.mm.out(as_strided(buf838, (256, 384), (384, 1)), permute_1063, out=buf840)
    del buf838
    del permute_1063
    buf841 = buf815; del buf815  # reuse
    kernel73.run(buf834, buf837, buf840, gt, embedding, reciprocal, buf841, 1024, 128, grid=grid(1024), stream=stream0)
    buf842 = empty_strided((1, 1, 512), (512, 512, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf841, buf842, 512, 2, grid=grid(512), stream=stream0)
    del buf841
    buf843 = buf817; del buf817  # reuse
    kernel74.run(buf834, buf837, buf840, primals_1, gt, embedding, buf843, 256, 512, grid=grid(256), stream=stream0)
    buf844 = as_strided(buf834, (2, 128, 512), (65536, 512, 1)); del buf834  # reuse
    kernel98.run(buf844, buf818, buf837, buf840, primals_1, reciprocal, buf843, gt, embedding, 131072, grid=grid(131072), stream=stream0)
    del buf818
    del buf837
    del buf840
    del buf843
    del embedding
    del primals_1
    del reciprocal
    buf845 = empty_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    kernel76.run(buf845, 128057344, grid=grid(128057344), stream=stream0)
    kernel77.run(view_753, buf844, gt, buf845, 131072, grid=grid(131072), stream=stream0)
    del buf844
    del gt
    del view_753
    buf847 = empty_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    kernel99.run(buf538, buf845, buf847, 128057344, grid=grid(128057344), stream=stream0)
    return (as_strided(buf842, (512, ), (1, )), as_strided(buf816, (512, ), (1, )), as_strided(buf802, (512, ), (1, )), as_strided(buf778, (512, ), (1, )), as_strided(buf764, (512, ), (1, )), as_strided(buf741, (512, ), (1, )), as_strided(buf727, (512, ), (1, )), as_strided(buf704, (512, ), (1, )), as_strided(buf690, (512, ), (1, )), as_strided(buf666, (512, ), (1, )), as_strided(buf652, (512, ), (1, )), as_strided(buf629, (512, ), (1, )), as_strided(buf615, (512, ), (1, )), as_strided(buf592, (512, ), (1, )), as_strided(buf578, (512, ), (1, )), as_strided(buf555, (512, ), (1, )), as_strided(buf541, (512, ), (1, )), as_strided(buf535, (512, ), (1, )), as_strided(buf509, (512, ), (1, )), as_strided(buf484, (512, ), (1, )), as_strided(buf470, (512, ), (1, )), as_strided(buf446, (512, ), (1, )), as_strided(buf422, (512, ), (1, )), as_strided(buf408, (512, ), (1, )), as_strided(buf385, (512, ), (1, )), as_strided(buf361, (512, ), (1, )), as_strided(buf347, (512, ), (1, )), as_strided(buf324, (512, ), (1, )), as_strided(buf300, (512, ), (1, )), as_strided(buf286, (512, ), (1, )), as_strided(buf262, (512, ), (1, )), as_strided(buf237, (512, ), (1, )), as_strided(buf223, (512, ), (1, )), as_strided(buf200, (512, ), (1, )), as_strided(buf176, (512, ), (1, )), as_strided(buf162, (512, ), (1, )), as_strided(buf139, (512, ), (1, )), as_strided(buf115, (512, ), (1, )), as_strided(buf101, (512, ), (1, )), as_strided(buf78, (512, ), (1, )), as_strided(buf54, (512, ), (1, )), as_strided(buf40, (512, ), (1, )), buf847, as_strided(buf839, (384, 512), (512, 1)), as_strided(buf836, (384, 512), (512, 1)), as_strided(buf833, (384, 512), (512, 1)), buf827, as_strided(buf820, (512, 384), (384, 1)), as_strided(buf813, (1024, 512), (512, 1)), as_strided(buf809, (1024, 512), (512, 1)), as_strided(buf806, (512, 1024), (1024, 1)), as_strided(buf799, (384, 512), (512, 1)), as_strided(buf796, (384, 512), (512, 1)), as_strided(buf793, (384, 512), (512, 1)), as_strided(buf782, (512, 384), (384, 1)), as_strided(buf775, (1024, 512), (512, 1)), as_strided(buf771, (1024, 512), (512, 1)), as_strided(buf768, (512, 1024), (1024, 1)), as_strided(buf761, (384, 512), (512, 1)), as_strided(buf758, (384, 512), (512, 1)), as_strided(buf755, (384, 512), (512, 1)), as_strided(buf745, (512, 384), (384, 1)), as_strided(buf738, (1024, 512), (512, 1)), as_strided(buf734, (1024, 512), (512, 1)), as_strided(buf731, (512, 1024), (1024, 1)), as_strided(buf724, (384, 512), (512, 1)), as_strided(buf721, (384, 512), (512, 1)), as_strided(buf718, (384, 512), (512, 1)), as_strided(buf708, (512, 384), (384, 1)), as_strided(buf701, (1024, 512), (512, 1)), as_strided(buf697, (1024, 512), (512, 1)), as_strided(buf694, (512, 1024), (1024, 1)), as_strided(buf687, (384, 512), (512, 1)), as_strided(buf684, (384, 512), (512, 1)), as_strided(buf681, (384, 512), (512, 1)), as_strided(buf670, (512, 384), (384, 1)), as_strided(buf663, (1024, 512), (512, 1)), as_strided(buf659, (1024, 512), (512, 1)), as_strided(buf656, (512, 1024), (1024, 1)), as_strided(buf649, (384, 512), (512, 1)), as_strided(buf646, (384, 512), (512, 1)), as_strided(buf643, (384, 512), (512, 1)), as_strided(buf633, (512, 384), (384, 1)), as_strided(buf626, (1024, 512), (512, 1)), as_strided(buf622, (1024, 512), (512, 1)), as_strided(buf619, (512, 1024), (1024, 1)), as_strided(buf612, (384, 512), (512, 1)), as_strided(buf609, (384, 512), (512, 1)), as_strided(buf606, (384, 512), (512, 1)), as_strided(buf596, (512, 384), (384, 1)), as_strided(buf589, (1024, 512), (512, 1)), as_strided(buf585, (1024, 512), (512, 1)), as_strided(buf582, (512, 1024), (1024, 1)), as_strided(buf575, (384, 512), (512, 1)), as_strided(buf572, (384, 512), (512, 1)), as_strided(buf569, (384, 512), (512, 1)), as_strided(buf559, (512, 384), (384, 1)), as_strided(buf552, (1024, 512), (512, 1)), as_strided(buf548, (1024, 512), (512, 1)), as_strided(buf545, (512, 1024), (1024, 1)), as_strided(buf532, (384, 512), (512, 1)), as_strided(buf529, (384, 512), (512, 1)), as_strided(buf526, (384, 512), (512, 1)), buf520, as_strided(buf513, (512, 384), (384, 1)), as_strided(buf506, (384, 512), (512, 1)), as_strided(buf501, (384, 512), (512, 1)), as_strided(buf498, (384, 512), (512, 1)), as_strided(buf488, (512, 384), (384, 1)), as_strided(buf481, (1024, 512), (512, 1)), as_strided(buf477, (1024, 512), (512, 1)), as_strided(buf474, (512, 1024), (1024, 1)), as_strided(buf467, (384, 512), (512, 1)), as_strided(buf464, (384, 512), (512, 1)), as_strided(buf461, (384, 512), (512, 1)), as_strided(buf450, (512, 384), (384, 1)), as_strided(buf443, (384, 512), (512, 1)), as_strided(buf439, (384, 512), (512, 1)), as_strided(buf436, (384, 512), (512, 1)), as_strided(buf426, (512, 384), (384, 1)), as_strided(buf419, (1024, 512), (512, 1)), as_strided(buf415, (1024, 512), (512, 1)), as_strided(buf412, (512, 1024), (1024, 1)), as_strided(buf405, (384, 512), (512, 1)), as_strided(buf402, (384, 512), (512, 1)), as_strided(buf399, (384, 512), (512, 1)), as_strided(buf389, (512, 384), (384, 1)), as_strided(buf382, (384, 512), (512, 1)), as_strided(buf378, (384, 512), (512, 1)), as_strided(buf375, (384, 512), (512, 1)), as_strided(buf365, (512, 384), (384, 1)), as_strided(buf358, (1024, 512), (512, 1)), as_strided(buf354, (1024, 512), (512, 1)), as_strided(buf351, (512, 1024), (1024, 1)), as_strided(buf344, (384, 512), (512, 1)), as_strided(buf341, (384, 512), (512, 1)), as_strided(buf338, (384, 512), (512, 1)), as_strided(buf328, (512, 384), (384, 1)), as_strided(buf321, (384, 512), (512, 1)), as_strided(buf317, (384, 512), (512, 1)), as_strided(buf314, (384, 512), (512, 1)), as_strided(buf304, (512, 384), (384, 1)), as_strided(buf297, (1024, 512), (512, 1)), as_strided(buf293, (1024, 512), (512, 1)), as_strided(buf290, (512, 1024), (1024, 1)), as_strided(buf283, (384, 512), (512, 1)), as_strided(buf280, (384, 512), (512, 1)), as_strided(buf277, (384, 512), (512, 1)), as_strided(buf266, (512, 384), (384, 1)), as_strided(buf259, (384, 512), (512, 1)), as_strided(buf254, (384, 512), (512, 1)), as_strided(buf251, (384, 512), (512, 1)), as_strided(buf241, (512, 384), (384, 1)), as_strided(buf234, (1024, 512), (512, 1)), as_strided(buf230, (1024, 512), (512, 1)), as_strided(buf227, (512, 1024), (1024, 1)), as_strided(buf220, (384, 512), (512, 1)), as_strided(buf217, (384, 512), (512, 1)), as_strided(buf214, (384, 512), (512, 1)), as_strided(buf204, (512, 384), (384, 1)), as_strided(buf197, (384, 512), (512, 1)), as_strided(buf193, (384, 512), (512, 1)), as_strided(buf190, (384, 512), (512, 1)), as_strided(buf180, (512, 384), (384, 1)), as_strided(buf173, (1024, 512), (512, 1)), as_strided(buf169, (1024, 512), (512, 1)), as_strided(buf166, (512, 1024), (1024, 1)), as_strided(buf159, (384, 512), (512, 1)), as_strided(buf156, (384, 512), (512, 1)), as_strided(buf153, (384, 512), (512, 1)), as_strided(buf143, (512, 384), (384, 1)), as_strided(buf136, (384, 512), (512, 1)), as_strided(buf132, (384, 512), (512, 1)), as_strided(buf129, (384, 512), (512, 1)), as_strided(buf119, (512, 384), (384, 1)), as_strided(buf112, (1024, 512), (512, 1)), as_strided(buf108, (1024, 512), (512, 1)), as_strided(buf105, (512, 1024), (1024, 1)), as_strided(buf98, (384, 512), (512, 1)), as_strided(buf95, (384, 512), (512, 1)), as_strided(buf92, (384, 512), (512, 1)), as_strided(buf82, (512, 384), (384, 1)), as_strided(buf75, (384, 512), (512, 1)), as_strided(buf71, (384, 512), (512, 1)), as_strided(buf68, (384, 512), (512, 1)), as_strided(buf58, (512, 384), (384, 1)), as_strided(buf51, (1024, 512), (512, 1)), as_strided(buf47, (1024, 512), (512, 1)), as_strided(buf44, (512, 1024), (1024, 1)), as_strided(buf37, (250112, 512), (512, 1)), None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
    embedding = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    gt = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    reciprocal = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_2 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    philox_seed_like = rand_strided((), (), device='cuda', dtype=torch.int64)
    view_9 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_3 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_6 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_1 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_4 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_3 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_5 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_4 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_12 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_5 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_11 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_3 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_3 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_21 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_7 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_14 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_4 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_11 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_5 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_12 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_8 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_24 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_9 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_19 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_6 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_4 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_33 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_11 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_22 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_7 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_18 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_7 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_19 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_12 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_36 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_13 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_27 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_9 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_5 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_45 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_15 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_30 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_10 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_25 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_9 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_26 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_16 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_48 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_17 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_35 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_12 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_6 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_57 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_19 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_38 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_13 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_32 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_11 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_33 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_20 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_60 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_21 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_43 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_15 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_7 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_69 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_23 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_46 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_16 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_39 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_13 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_40 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_24 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_72 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_25 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_51 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_18 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_8 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_81 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_27 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_54 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_19 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_46 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_15 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_47 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_28 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_84 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_29 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_59 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_21 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_9 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_93 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_31 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_62 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_22 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_53 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_17 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_54 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_32 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_96 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_33 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_67 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_24 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    gt_34 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    embedding_2 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    gt_35 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    reciprocal_25 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_12 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_106 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_37 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_74 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_26 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_13 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_115 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_39 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_78 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_27 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_64 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_23 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_65 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_40 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_118 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_41 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_83 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_29 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_14 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_127 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_43 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_86 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_30 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_15 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_136 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_45 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_89 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_31 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_75 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_26 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_76 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_46 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_139 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_47 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_94 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_33 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_16 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_148 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_49 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_97 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_34 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_17 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_157 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_51 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_100 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_35 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_86 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_29 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_87 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_52 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_160 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_53 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_105 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_37 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_18 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_169 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_55 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_108 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_38 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_19 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_178 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_57 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_111 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_39 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_97 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_32 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_98 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_58 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_181 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_59 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_116 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_41 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_20 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_190 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_61 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_119 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_42 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_21 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_199 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_63 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_122 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_43 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_108 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_35 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_109 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_64 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_202 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_65 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_127 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_45 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_22 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_211 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_67 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_130 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_46 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_23 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_220 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_69 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_133 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_47 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_119 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_38 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_120 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_70 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_223 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_71 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_138 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_49 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_24 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_232 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_73 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_141 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_50 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_25 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_241 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_75 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_144 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_51 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_130 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_41 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_131 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_76 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_244 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_77 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_149 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_53 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_26 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_253 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_79 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_152 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_54 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    div_27 = rand_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    view_262 = rand_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    gt_81 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_155 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_55 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    mm_141 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    sub_44 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    mm_142 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_82 = rand_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    view_265 = rand_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    gt_83 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    add_160 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    reciprocal_57 = rand_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    gt_84 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    view_266 = rand_strided((256, 512), (512, 1), device='cuda', dtype=torch.float32)
    sub_46 = rand_strided((256, 250112), (250112, 1), device='cuda', dtype=torch.float32)
    unsqueeze_17 = rand_strided((256, 1), (1, 1), device='cuda', dtype=torch.int64)
    permute_269 = rand_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_273 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_277 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_281 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_285 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_288 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_289 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_290 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_291 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_296 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_301 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_306 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_310 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_313 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_314 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_315 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_316 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_321 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_326 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_331 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_335 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_339 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_343 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_347 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_350 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_351 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_352 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_353 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_358 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_363 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_368 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_372 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_375 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_376 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_377 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_378 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_383 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_388 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_393 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_397 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_401 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_405 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_409 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_412 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_413 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_414 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_415 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_420 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_425 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_430 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_434 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_437 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_438 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_439 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_440 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_445 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_450 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_455 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_459 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_463 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_467 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_471 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_474 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_475 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_476 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_477 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_482 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_487 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_492 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_496 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_499 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_500 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_501 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_502 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_507 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_512 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_517 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_521 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_525 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_529 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_533 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_536 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_537 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_538 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_539 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_544 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_549 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_554 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_558 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_561 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_562 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_563 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_564 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_569 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_574 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_579 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_583 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_587 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_591 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_595 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_598 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_599 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_600 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_601 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_606 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_611 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_616 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_620 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_623 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_624 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_625 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_626 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_631 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_636 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_641 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_645 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_649 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_653 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_657 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_660 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_661 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_662 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_663 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_668 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_673 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_678 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_682 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_685 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_686 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_687 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_688 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_693 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_698 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_703 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_707 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_711 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_715 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_719 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_722 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_723 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_724 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_725 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_730 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_735 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_740 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_744 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_747 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_748 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    view_560 = rand_strided((16384, ), (1, ), device='cuda', dtype=torch.int64)
    permute_750 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_751 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_756 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_761 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_766 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    view_572 = rand_strided((256, ), (1, ), device='cuda', dtype=torch.int64)
    permute_770 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_774 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_778 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_782 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_785 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_786 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_787 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_788 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_793 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_798 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_803 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_807 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_811 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_815 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_819 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_822 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_823 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_824 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_825 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_830 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_835 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_840 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_844 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_848 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_852 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_856 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_859 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_860 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_861 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_862 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_867 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_872 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_877 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_881 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_885 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_889 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_893 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_896 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_897 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_898 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_899 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_904 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_909 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_914 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_918 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_922 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_926 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_930 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_933 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_934 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_935 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_936 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_941 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_946 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_951 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_955 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_959 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_963 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_967 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_970 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_971 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_972 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_973 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_978 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_983 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_988 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_992 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_996 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1000 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1004 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_1007 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_1008 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_1009 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_1010 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_1015 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1020 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1025 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1029 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    permute_1033 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1037 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1041 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    permute_1044 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_1045 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    view_741 = rand_strided((16384, ), (1, ), device='cuda', dtype=torch.int64)
    permute_1047 = rand_strided((12, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_1048 = rand_strided((12, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_1053 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1058 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    permute_1063 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    view_753 = rand_strided((256, ), (1, ), device='cuda', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda', dtype=torch.float32)
    tangents_2 = rand_strided((2, 128, 250112), (32014336, 250112, 1), device='cuda', dtype=torch.float32)
    tangents_3 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_4 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_5 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_6 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_7 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_8 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_9 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_10 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_11 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_12 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_13 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_14 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_15 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_16 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_17 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_18 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_19 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_20 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_21 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_22 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_23 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_24 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_25 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_26 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_27 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_28 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_29 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_30 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_31 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_32 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_33 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_34 = rand_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    tangents_35 = rand_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, embedding, gt, reciprocal, div_2, philox_seed_like, view_9, gt_3, add_6, reciprocal_1, mm_4, sub_3, mm_5, gt_4, view_12, gt_5, add_11, reciprocal_3, div_3, view_21, gt_7, add_14, reciprocal_4, mm_11, sub_5, mm_12, gt_8, view_24, gt_9, add_19, reciprocal_6, div_4, view_33, gt_11, add_22, reciprocal_7, mm_18, sub_7, mm_19, gt_12, view_36, gt_13, add_27, reciprocal_9, div_5, view_45, gt_15, add_30, reciprocal_10, mm_25, sub_9, mm_26, gt_16, view_48, gt_17, add_35, reciprocal_12, div_6, view_57, gt_19, add_38, reciprocal_13, mm_32, sub_11, mm_33, gt_20, view_60, gt_21, add_43, reciprocal_15, div_7, view_69, gt_23, add_46, reciprocal_16, mm_39, sub_13, mm_40, gt_24, view_72, gt_25, add_51, reciprocal_18, div_8, view_81, gt_27, add_54, reciprocal_19, mm_46, sub_15, mm_47, gt_28, view_84, gt_29, add_59, reciprocal_21, div_9, view_93, gt_31, add_62, reciprocal_22, mm_53, sub_17, mm_54, gt_32, view_96, gt_33, add_67, reciprocal_24, gt_34, embedding_2, gt_35, reciprocal_25, div_12, view_106, gt_37, add_74, reciprocal_26, div_13, view_115, gt_39, add_78, reciprocal_27, mm_64, sub_23, mm_65, gt_40, view_118, gt_41, add_83, reciprocal_29, div_14, view_127, gt_43, add_86, reciprocal_30, div_15, view_136, gt_45, add_89, reciprocal_31, mm_75, sub_26, mm_76, gt_46, view_139, gt_47, add_94, reciprocal_33, div_16, view_148, gt_49, add_97, reciprocal_34, div_17, view_157, gt_51, add_100, reciprocal_35, mm_86, sub_29, mm_87, gt_52, view_160, gt_53, add_105, reciprocal_37, div_18, view_169, gt_55, add_108, reciprocal_38, div_19, view_178, gt_57, add_111, reciprocal_39, mm_97, sub_32, mm_98, gt_58, view_181, gt_59, add_116, reciprocal_41, div_20, view_190, gt_61, add_119, reciprocal_42, div_21, view_199, gt_63, add_122, reciprocal_43, mm_108, sub_35, mm_109, gt_64, view_202, gt_65, add_127, reciprocal_45, div_22, view_211, gt_67, add_130, reciprocal_46, div_23, view_220, gt_69, add_133, reciprocal_47, mm_119, sub_38, mm_120, gt_70, view_223, gt_71, add_138, reciprocal_49, div_24, view_232, gt_73, add_141, reciprocal_50, div_25, view_241, gt_75, add_144, reciprocal_51, mm_130, sub_41, mm_131, gt_76, view_244, gt_77, add_149, reciprocal_53, div_26, view_253, gt_79, add_152, reciprocal_54, div_27, view_262, gt_81, add_155, reciprocal_55, mm_141, sub_44, mm_142, gt_82, view_265, gt_83, add_160, reciprocal_57, gt_84, view_266, sub_46, unsqueeze_17, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, view_560, permute_750, permute_751, permute_756, permute_761, permute_766, view_572, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, view_741, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, view_753, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35]))
