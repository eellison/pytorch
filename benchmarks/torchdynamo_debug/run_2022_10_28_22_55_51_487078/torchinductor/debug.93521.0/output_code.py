
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x3 = (xindex // 768)
    x0 = xindex % 768
    x1 = (xindex // 768) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask)
    tmp5 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tmp3 = tl.load(in_ptr3 + (x0 + (768*tmp2) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tmp4 = tmp1 + tmp3
    tmp6 = tl.load(in_ptr5 + (x0 + (768*tmp5) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tmp7 = tmp4 + tmp6
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp7, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


kernel2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    _tmp6 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = 768
        tmp3 = tmp1 / tmp2
        tmp4 = tmp0 - tmp3
        tmp5 = tmp4 * tmp4
        _tmp6 = tl.where(xmask & rmask, _tmp6 + tmp5, _tmp6)
    tmp6 = tl.reshape(tl.sum(_tmp6, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp6, xmask)
''')


kernel3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask)
    tmp2 = 768
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp5 / tmp2
    tmp7 = 1e-12
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sqrt(tmp8)
    tmp10 = 1 / tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


kernel4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp7 * tmp12
    tmp14 = 1.1111111111111112
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp15, xmask)
''')


kernel5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 12
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (98304*x3)), xmask)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536, 128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 49152
    ynumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*y2) + (98304*x1)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (128*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


kernel7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = 1
        tmp5 = tmp3 - tmp4
        tmp6 = -3.4028234663852886e+38
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        _tmp9 = tl.where(xmask & rmask & (_tmp9 < tmp8), tmp8, _tmp9)
    tmp9 = tl.reshape(tl.max(_tmp9, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp9, xmask)
''')


kernel8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp9 = tl.load(in_ptr1 + (x0), xmask)
    _tmp12 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 8.0
        tmp2 = tmp0 / tmp1
        tmp3 = 1.0
        tmp4 = 1
        tmp5 = tmp3 - tmp4
        tmp6 = -3.4028234663852886e+38
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = tl.exp(tmp10)
        _tmp12 = tl.where(xmask & rmask, _tmp12 + tmp11, _tmp12)
    tmp12 = tl.reshape(tl.sum(_tmp12, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp12, xmask)
''')


kernel9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6291456 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 128
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (98304*x3)), xmask)
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 18874368 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 18874368 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 18874368 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


kernel15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
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
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp36, xmask)
''')


kernel16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 25165824 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 25165824 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 25165824 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 31457280 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 44040192 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 44040192 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 44040192 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 50331648 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 50331648 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 50331648 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 56623104 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 69206016 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 69206016 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 69206016 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 75497472 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 75497472 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 75497472 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 81788928 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 94371840 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 94371840 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 94371840 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 100663296 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 100663296 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 100663296 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 106954752 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 119537664 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 119537664 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 119537664 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 125829120 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 125829120 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 125829120 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel47 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 132120576 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 144703488 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 144703488 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 144703488 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel51 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 150994944 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 150994944 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 150994944 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 157286400 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 169869312 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 169869312 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 169869312 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 176160768 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 176160768 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 176160768 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 182452224 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 195035136 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 195035136 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 195035136 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 201326592 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 201326592 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 201326592 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 207618048 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 220200960 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 220200960 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 220200960 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 226492416 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 226492416 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 226492416 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 232783872 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 245366784 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 245366784 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 245366784 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 251658240 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 251658240 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 251658240 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 257949696 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 270532608 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 270532608 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 270532608 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 276824064 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 276824064 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel88 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 276824064 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel89 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp17 = tl.load(in_ptr2 + (x1), xmask)
    tmp20 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 283115520 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.0
    tmp12 = 1
    tmp13 = tmp11 - tmp12
    tmp14 = -3.4028234663852886e+38
    tmp15 = tmp13 * tmp14
    tmp16 = tmp10 + tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp7 * tmp21
    tmp23 = 1.1111111111111112
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel90 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 295698432 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel91 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 295698432 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel92 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 295698432 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel93 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp14 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 301989888 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        _tmp14 = tl.where(xmask & rmask, _tmp14 + tmp13, _tmp14)
    tmp14 = tl.reshape(tl.sum(_tmp14, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp14, xmask)
''')


kernel94 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    tmp14 = tl.load(in_ptr3 + (x0), xmask)
    _tmp19 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 301989888 + r1 + (768*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp15 = 768
        tmp16 = tmp14 / tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tmp17 * tmp17
        _tmp19 = tl.where(xmask & rmask, _tmp19 + tmp18, _tmp19)
    tmp19 = tl.reshape(tl.sum(_tmp19, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp19, xmask)
''')


kernel95 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp12 = tl.load(in_ptr2 + (x2), xmask)
    tmp14 = tl.load(in_ptr3 + (x1), xmask)
    tmp18 = tl.load(in_ptr4 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 301989888 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 768
    tmp16 = tmp14 / tmp15
    tmp17 = tmp13 - tmp16
    tmp19 = tmp18 / tmp15
    tmp20 = 1e-12
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = 1 / tmp22
    tmp24 = tmp17 * tmp23
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp24, xmask)
''')


kernel96 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp37 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
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
        _tmp37 = tl.where(xmask & rmask, _tmp37 + tmp36, _tmp37)
    tmp37 = tl.reshape(tl.sum(_tmp37, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp37, xmask)
''')


kernel97 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp37 = tl.load(in_ptr1 + (x0), xmask)
    _tmp42 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
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
        tmp38 = 768
        tmp39 = tmp37 / tmp38
        tmp40 = tmp36 - tmp39
        tmp41 = tmp40 * tmp40
        _tmp42 = tl.where(xmask & rmask, _tmp42 + tmp41, _tmp42)
    tmp42 = tl.reshape(tl.sum(_tmp42, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp42, xmask)
''')


kernel98 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp37 = tl.load(in_ptr1 + (x1), xmask)
    tmp41 = tl.load(in_ptr2 + (x1), xmask)
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
    tmp38 = 768
    tmp39 = tmp37 / tmp38
    tmp40 = tmp36 - tmp39
    tmp42 = tmp41 / tmp38
    tmp43 = 1e-12
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sqrt(tmp44)
    tmp46 = 1 / tmp45
    tmp47 = tmp40 * tmp46
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp47, xmask)
''')


kernel99 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.reshape(tl.max(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


kernel100 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 32768],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        _tmp4 = tl.where(xmask & rmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.reshape(tl.sum(_tmp4, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp4, xmask)
''')


kernel101 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250036224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 30522)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.log(tmp3)
    tmp5 = tmp2 - tmp4
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel102 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (tmp0 + (30522*r0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = -tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp3, None)
''')


kernel103 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 8192
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp2, None)
''')


kernel104 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel105 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6291456 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel106 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 18874368 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel107 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 25165824 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel108 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 31457280 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel109 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 44040192 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel110 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 50331648 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel111 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 56623104 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel112 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 69206016 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel113 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 75497472 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel114 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 81788928 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel115 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 94371840 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel116 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 100663296 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel117 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 106954752 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel118 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 119537664 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel119 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 125829120 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel120 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 132120576 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel121 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 144703488 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel122 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 150994944 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel123 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 157286400 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel124 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 169869312 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel125 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 176160768 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel126 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 182452224 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel127 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 195035136 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel128 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 201326592 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel129 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 207618048 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel130 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 220200960 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel131 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 226492416 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel132 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 232783872 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel133 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 245366784 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel134 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 251658240 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel135 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 257949696 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel136 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 270532608 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel137 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 276824064 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel138 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 283115520 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel139 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 295698432 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel140 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 301989888 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel141 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 768
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-12
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sqrt(tmp4)
    tmp6 = 1 / tmp5
    tmp7 = tmp6 / tmp1
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp7, xmask)
''')


kernel142 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.7071067811865476
    tmp2 = tmp0 * tmp1
    tmp3 = tl.where(tmp2 < 0, -1, 1)
    tmp4 = tl.where(tmp2 == 0, 0, tmp3)
    tmp5 = 1.0
    tmp6 = tl.abs(tmp2)
    tmp7 = 0.3275911
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp5
    tmp10 = 1 / tmp9
    tmp11 = tmp10 * tmp5
    tmp12 = 1.061405429
    tmp13 = tmp11 * tmp12
    tmp14 = -1.453152027
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = 1.421413741
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 * tmp11
    tmp20 = -0.284496736
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 * tmp11
    tmp23 = 0.254829592
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = -tmp6
    tmp27 = tmp26 * tmp6
    tmp28 = tl.exp(tmp27)
    tmp29 = tmp25 * tmp28
    tmp30 = tmp5 - tmp29
    tmp31 = tmp4 * tmp30
    tmp32 = 1
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp0 * tmp0
    tmp37 = -0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tl.exp(tmp38)
    tmp40 = 0.3989422804014327
    tmp41 = tmp39 * tmp40
    tmp42 = tmp0 * tmp41
    tmp43 = tmp35 + tmp42
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp43, xmask)
''')


kernel143 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.7071067811865476
    tmp2 = tmp0 * tmp1
    tmp3 = tl.where(tmp2 < 0, -1, 1)
    tmp4 = tl.where(tmp2 == 0, 0, tmp3)
    tmp5 = 1.0
    tmp6 = tl.abs(tmp2)
    tmp7 = 0.3275911
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp5
    tmp10 = 1 / tmp9
    tmp11 = tmp10 * tmp5
    tmp12 = 1.061405429
    tmp13 = tmp11 * tmp12
    tmp14 = -1.453152027
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = 1.421413741
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 * tmp11
    tmp20 = -0.284496736
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21 * tmp11
    tmp23 = 0.254829592
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = -tmp6
    tmp27 = tmp26 * tmp6
    tmp28 = tl.exp(tmp27)
    tmp29 = tmp25 * tmp28
    tmp30 = tmp5 - tmp29
    tmp31 = tmp4 * tmp30
    tmp32 = 1
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp0 * tmp0
    tmp37 = -0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tl.exp(tmp38)
    tmp40 = 0.3989422804014327
    tmp41 = tmp39 * tmp40
    tmp42 = tmp0 * tmp41
    tmp43 = tmp35 + tmp42
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp43, xmask)
''')


kernel144 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp9 = tl.load(in_ptr1 + (x1), xmask)
    tmp12 = tl.load(in_ptr2 + (x1), xmask)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1.0
    tmp4 = 1
    tmp5 = tmp3 - tmp4
    tmp6 = -3.4028234663852886e+38
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp8 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp13 = tmp11 / tmp12
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    torch.randint(2**31, size=(), dtype=torch.int64, out=seed_cuda_0)
    buf0 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_205, primals_1, primals_203, primals_2, primals_204, primals_3, buf0, 6291456, grid=grid(6291456), stream=stream0)
    del primals_2
    del primals_3
    buf1 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel1.run(buf0, buf1, 8192, 768, grid=grid(8192), stream=stream0)
    buf2 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel2.run(buf0, buf1, buf2, 8192, 768, grid=grid(8192), stream=stream0)
    buf3 = buf1; del buf1  # reuse
    kernel1.run(buf0, buf3, 8192, 768, grid=grid(8192), stream=stream0)
    buf4 = buf0; del buf0  # reuse
    kernel3.run(buf4, buf3, buf2, 6291456, grid=grid(6291456), stream=stream0)
    buf5 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel4.run(seed_cuda_0, buf4, primals_4, primals_5, buf5, 6291456, grid=grid(6291456), stream=stream0)
    del primals_5
    buf6 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_7, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_6, (768, 768), (1, 768)), beta=1, alpha=1, out=buf6)
    del primals_7
    buf7 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_9, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_8, (768, 768), (1, 768)), beta=1, alpha=1, out=buf7)
    del primals_9
    buf8 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_11, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_10, (768, 768), (1, 768)), beta=1, alpha=1, out=buf8)
    del primals_11
    buf9 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf6, buf9, 6291456, grid=grid(6291456), stream=stream0)
    buf10 = as_strided(buf6, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf6  # reuse
    kernel6.run(buf7, buf10, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf11 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf9, (768, 128, 64), (8192, 64, 1)), as_strided(buf10, (768, 64, 128), (8192, 128, 1)), out=buf11)
    buf12 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf11, buf12, 98304, 128, grid=grid(98304), stream=stream0)
    buf13 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf11, buf12, buf13, 98304, 128, grid=grid(98304), stream=stream0)
    buf14 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel9.run(seed_cuda_0, buf11, buf12, buf13, buf14, 12582912, grid=grid(12582912), stream=stream0)
    buf15 = as_strided(buf7, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf7  # reuse
    kernel5.run(buf8, buf15, 6291456, grid=grid(6291456), stream=stream0)
    buf16 = as_strided(buf8, (768, 128, 64), (8192, 64, 1)); del buf8  # reuse
    aten.bmm.out(as_strided(buf14, (768, 128, 128), (16384, 128, 1)), as_strided(buf15, (768, 128, 64), (8192, 64, 1)), out=buf16)
    buf17 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf16, buf17, 6291456, grid=grid(6291456), stream=stream0)
    buf18 = as_strided(buf16, (8192, 768), (768, 1)); del buf16  # reuse
    aten.addmm.out(primals_13, as_strided(buf17, (8192, 768), (768, 1)), as_strided(primals_12, (768, 768), (1, 768)), beta=1, alpha=1, out=buf18)
    del primals_13
    buf19 = buf3; del buf3  # reuse
    kernel11.run(seed_cuda_0, buf18, buf5, buf19, 8192, 768, grid=grid(8192), stream=stream0)
    buf20 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel12.run(seed_cuda_0, buf18, buf5, buf19, buf20, 8192, 768, grid=grid(8192), stream=stream0)
    buf21 = buf19; del buf19  # reuse
    kernel11.run(seed_cuda_0, buf18, buf5, buf21, 8192, 768, grid=grid(8192), stream=stream0)
    buf22 = as_strided(buf18, (64, 128, 768), (98304, 768, 1)); del buf18  # reuse
    kernel13.run(buf22, seed_cuda_0, buf5, buf21, buf20, 6291456, grid=grid(6291456), stream=stream0)
    buf23 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf22, primals_14, primals_15, buf23, 6291456, grid=grid(6291456), stream=stream0)
    del primals_15
    buf24 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_17, as_strided(buf23, (8192, 768), (768, 1)), as_strided(primals_16, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf24)
    del primals_17
    buf25 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf24, buf25, 25165824, grid=grid(25165824), stream=stream0)
    buf26 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_19, as_strided(buf25, (8192, 3072), (3072, 1)), as_strided(primals_18, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf26)
    del primals_19
    buf27 = buf21; del buf21  # reuse
    kernel16.run(seed_cuda_0, buf26, buf23, buf27, 8192, 768, grid=grid(8192), stream=stream0)
    buf28 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel17.run(seed_cuda_0, buf26, buf23, buf27, buf28, 8192, 768, grid=grid(8192), stream=stream0)
    buf29 = buf27; del buf27  # reuse
    kernel16.run(seed_cuda_0, buf26, buf23, buf29, 8192, 768, grid=grid(8192), stream=stream0)
    buf30 = as_strided(buf26, (64, 128, 768), (98304, 768, 1)); del buf26  # reuse
    kernel18.run(buf30, seed_cuda_0, buf23, buf29, buf28, 6291456, grid=grid(6291456), stream=stream0)
    buf31 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf30, primals_20, primals_21, buf31, 6291456, grid=grid(6291456), stream=stream0)
    del primals_21
    buf32 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_23, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_22, (768, 768), (1, 768)), beta=1, alpha=1, out=buf32)
    del primals_23
    buf33 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_25, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_24, (768, 768), (1, 768)), beta=1, alpha=1, out=buf33)
    del primals_25
    buf34 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_27, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_26, (768, 768), (1, 768)), beta=1, alpha=1, out=buf34)
    del primals_27
    buf35 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf32, buf35, 6291456, grid=grid(6291456), stream=stream0)
    buf36 = as_strided(buf32, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf32  # reuse
    kernel6.run(buf33, buf36, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf37 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf35, (768, 128, 64), (8192, 64, 1)), as_strided(buf36, (768, 64, 128), (8192, 128, 1)), out=buf37)
    buf38 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf37, buf38, 98304, 128, grid=grid(98304), stream=stream0)
    buf39 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf37, buf38, buf39, 98304, 128, grid=grid(98304), stream=stream0)
    buf40 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel19.run(seed_cuda_0, buf37, buf38, buf39, buf40, 12582912, grid=grid(12582912), stream=stream0)
    buf41 = as_strided(buf33, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf33  # reuse
    kernel5.run(buf34, buf41, 6291456, grid=grid(6291456), stream=stream0)
    buf42 = as_strided(buf34, (768, 128, 64), (8192, 64, 1)); del buf34  # reuse
    aten.bmm.out(as_strided(buf40, (768, 128, 128), (16384, 128, 1)), as_strided(buf41, (768, 128, 64), (8192, 64, 1)), out=buf42)
    buf43 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf42, buf43, 6291456, grid=grid(6291456), stream=stream0)
    buf44 = as_strided(buf42, (8192, 768), (768, 1)); del buf42  # reuse
    aten.addmm.out(primals_29, as_strided(buf43, (8192, 768), (768, 1)), as_strided(primals_28, (768, 768), (1, 768)), beta=1, alpha=1, out=buf44)
    del primals_29
    buf45 = buf29; del buf29  # reuse
    kernel20.run(seed_cuda_0, buf44, buf31, buf45, 8192, 768, grid=grid(8192), stream=stream0)
    buf46 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel21.run(seed_cuda_0, buf44, buf31, buf45, buf46, 8192, 768, grid=grid(8192), stream=stream0)
    buf47 = buf45; del buf45  # reuse
    kernel20.run(seed_cuda_0, buf44, buf31, buf47, 8192, 768, grid=grid(8192), stream=stream0)
    buf48 = as_strided(buf44, (64, 128, 768), (98304, 768, 1)); del buf44  # reuse
    kernel22.run(buf48, seed_cuda_0, buf31, buf47, buf46, 6291456, grid=grid(6291456), stream=stream0)
    buf49 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf48, primals_30, primals_31, buf49, 6291456, grid=grid(6291456), stream=stream0)
    del primals_31
    buf50 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_33, as_strided(buf49, (8192, 768), (768, 1)), as_strided(primals_32, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf50)
    del primals_33
    buf51 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf50, buf51, 25165824, grid=grid(25165824), stream=stream0)
    buf52 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_35, as_strided(buf51, (8192, 3072), (3072, 1)), as_strided(primals_34, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf52)
    del primals_35
    buf53 = buf47; del buf47  # reuse
    kernel23.run(seed_cuda_0, buf52, buf49, buf53, 8192, 768, grid=grid(8192), stream=stream0)
    buf54 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel24.run(seed_cuda_0, buf52, buf49, buf53, buf54, 8192, 768, grid=grid(8192), stream=stream0)
    buf55 = buf53; del buf53  # reuse
    kernel23.run(seed_cuda_0, buf52, buf49, buf55, 8192, 768, grid=grid(8192), stream=stream0)
    buf56 = as_strided(buf52, (64, 128, 768), (98304, 768, 1)); del buf52  # reuse
    kernel25.run(buf56, seed_cuda_0, buf49, buf55, buf54, 6291456, grid=grid(6291456), stream=stream0)
    buf57 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf56, primals_36, primals_37, buf57, 6291456, grid=grid(6291456), stream=stream0)
    del primals_37
    buf58 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_39, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_38, (768, 768), (1, 768)), beta=1, alpha=1, out=buf58)
    del primals_39
    buf59 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_41, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_40, (768, 768), (1, 768)), beta=1, alpha=1, out=buf59)
    del primals_41
    buf60 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_43, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_42, (768, 768), (1, 768)), beta=1, alpha=1, out=buf60)
    del primals_43
    buf61 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf58, buf61, 6291456, grid=grid(6291456), stream=stream0)
    buf62 = as_strided(buf58, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf58  # reuse
    kernel6.run(buf59, buf62, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf63 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf61, (768, 128, 64), (8192, 64, 1)), as_strided(buf62, (768, 64, 128), (8192, 128, 1)), out=buf63)
    buf64 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf63, buf64, 98304, 128, grid=grid(98304), stream=stream0)
    buf65 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf63, buf64, buf65, 98304, 128, grid=grid(98304), stream=stream0)
    buf66 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel26.run(seed_cuda_0, buf63, buf64, buf65, buf66, 12582912, grid=grid(12582912), stream=stream0)
    buf67 = as_strided(buf59, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf59  # reuse
    kernel5.run(buf60, buf67, 6291456, grid=grid(6291456), stream=stream0)
    buf68 = as_strided(buf60, (768, 128, 64), (8192, 64, 1)); del buf60  # reuse
    aten.bmm.out(as_strided(buf66, (768, 128, 128), (16384, 128, 1)), as_strided(buf67, (768, 128, 64), (8192, 64, 1)), out=buf68)
    buf69 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf68, buf69, 6291456, grid=grid(6291456), stream=stream0)
    buf70 = as_strided(buf68, (8192, 768), (768, 1)); del buf68  # reuse
    aten.addmm.out(primals_45, as_strided(buf69, (8192, 768), (768, 1)), as_strided(primals_44, (768, 768), (1, 768)), beta=1, alpha=1, out=buf70)
    del primals_45
    buf71 = buf55; del buf55  # reuse
    kernel27.run(seed_cuda_0, buf70, buf57, buf71, 8192, 768, grid=grid(8192), stream=stream0)
    buf72 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel28.run(seed_cuda_0, buf70, buf57, buf71, buf72, 8192, 768, grid=grid(8192), stream=stream0)
    buf73 = buf71; del buf71  # reuse
    kernel27.run(seed_cuda_0, buf70, buf57, buf73, 8192, 768, grid=grid(8192), stream=stream0)
    buf74 = as_strided(buf70, (64, 128, 768), (98304, 768, 1)); del buf70  # reuse
    kernel29.run(buf74, seed_cuda_0, buf57, buf73, buf72, 6291456, grid=grid(6291456), stream=stream0)
    buf75 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf74, primals_46, primals_47, buf75, 6291456, grid=grid(6291456), stream=stream0)
    del primals_47
    buf76 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_49, as_strided(buf75, (8192, 768), (768, 1)), as_strided(primals_48, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf76)
    del primals_49
    buf77 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf76, buf77, 25165824, grid=grid(25165824), stream=stream0)
    buf78 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_51, as_strided(buf77, (8192, 3072), (3072, 1)), as_strided(primals_50, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf78)
    del primals_51
    buf79 = buf73; del buf73  # reuse
    kernel30.run(seed_cuda_0, buf78, buf75, buf79, 8192, 768, grid=grid(8192), stream=stream0)
    buf80 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel31.run(seed_cuda_0, buf78, buf75, buf79, buf80, 8192, 768, grid=grid(8192), stream=stream0)
    buf81 = buf79; del buf79  # reuse
    kernel30.run(seed_cuda_0, buf78, buf75, buf81, 8192, 768, grid=grid(8192), stream=stream0)
    buf82 = as_strided(buf78, (64, 128, 768), (98304, 768, 1)); del buf78  # reuse
    kernel32.run(buf82, seed_cuda_0, buf75, buf81, buf80, 6291456, grid=grid(6291456), stream=stream0)
    buf83 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf82, primals_52, primals_53, buf83, 6291456, grid=grid(6291456), stream=stream0)
    del primals_53
    buf84 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_55, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_54, (768, 768), (1, 768)), beta=1, alpha=1, out=buf84)
    del primals_55
    buf85 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_57, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_56, (768, 768), (1, 768)), beta=1, alpha=1, out=buf85)
    del primals_57
    buf86 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_59, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_58, (768, 768), (1, 768)), beta=1, alpha=1, out=buf86)
    del primals_59
    buf87 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf84, buf87, 6291456, grid=grid(6291456), stream=stream0)
    buf88 = as_strided(buf84, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf84  # reuse
    kernel6.run(buf85, buf88, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf89 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf87, (768, 128, 64), (8192, 64, 1)), as_strided(buf88, (768, 64, 128), (8192, 128, 1)), out=buf89)
    buf90 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf89, buf90, 98304, 128, grid=grid(98304), stream=stream0)
    buf91 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf89, buf90, buf91, 98304, 128, grid=grid(98304), stream=stream0)
    buf92 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel33.run(seed_cuda_0, buf89, buf90, buf91, buf92, 12582912, grid=grid(12582912), stream=stream0)
    buf93 = as_strided(buf85, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf85  # reuse
    kernel5.run(buf86, buf93, 6291456, grid=grid(6291456), stream=stream0)
    buf94 = as_strided(buf86, (768, 128, 64), (8192, 64, 1)); del buf86  # reuse
    aten.bmm.out(as_strided(buf92, (768, 128, 128), (16384, 128, 1)), as_strided(buf93, (768, 128, 64), (8192, 64, 1)), out=buf94)
    buf95 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf94, buf95, 6291456, grid=grid(6291456), stream=stream0)
    buf96 = as_strided(buf94, (8192, 768), (768, 1)); del buf94  # reuse
    aten.addmm.out(primals_61, as_strided(buf95, (8192, 768), (768, 1)), as_strided(primals_60, (768, 768), (1, 768)), beta=1, alpha=1, out=buf96)
    del primals_61
    buf97 = buf81; del buf81  # reuse
    kernel34.run(seed_cuda_0, buf96, buf83, buf97, 8192, 768, grid=grid(8192), stream=stream0)
    buf98 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel35.run(seed_cuda_0, buf96, buf83, buf97, buf98, 8192, 768, grid=grid(8192), stream=stream0)
    buf99 = buf97; del buf97  # reuse
    kernel34.run(seed_cuda_0, buf96, buf83, buf99, 8192, 768, grid=grid(8192), stream=stream0)
    buf100 = as_strided(buf96, (64, 128, 768), (98304, 768, 1)); del buf96  # reuse
    kernel36.run(buf100, seed_cuda_0, buf83, buf99, buf98, 6291456, grid=grid(6291456), stream=stream0)
    buf101 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf100, primals_62, primals_63, buf101, 6291456, grid=grid(6291456), stream=stream0)
    del primals_63
    buf102 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_65, as_strided(buf101, (8192, 768), (768, 1)), as_strided(primals_64, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf102)
    del primals_65
    buf103 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf102, buf103, 25165824, grid=grid(25165824), stream=stream0)
    buf104 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_67, as_strided(buf103, (8192, 3072), (3072, 1)), as_strided(primals_66, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf104)
    del primals_67
    buf105 = buf99; del buf99  # reuse
    kernel37.run(seed_cuda_0, buf104, buf101, buf105, 8192, 768, grid=grid(8192), stream=stream0)
    buf106 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel38.run(seed_cuda_0, buf104, buf101, buf105, buf106, 8192, 768, grid=grid(8192), stream=stream0)
    buf107 = buf105; del buf105  # reuse
    kernel37.run(seed_cuda_0, buf104, buf101, buf107, 8192, 768, grid=grid(8192), stream=stream0)
    buf108 = as_strided(buf104, (64, 128, 768), (98304, 768, 1)); del buf104  # reuse
    kernel39.run(buf108, seed_cuda_0, buf101, buf107, buf106, 6291456, grid=grid(6291456), stream=stream0)
    buf109 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf108, primals_68, primals_69, buf109, 6291456, grid=grid(6291456), stream=stream0)
    del primals_69
    buf110 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_71, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_70, (768, 768), (1, 768)), beta=1, alpha=1, out=buf110)
    del primals_71
    buf111 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_73, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_72, (768, 768), (1, 768)), beta=1, alpha=1, out=buf111)
    del primals_73
    buf112 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_75, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_74, (768, 768), (1, 768)), beta=1, alpha=1, out=buf112)
    del primals_75
    buf113 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf110, buf113, 6291456, grid=grid(6291456), stream=stream0)
    buf114 = as_strided(buf110, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf110  # reuse
    kernel6.run(buf111, buf114, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf115 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf113, (768, 128, 64), (8192, 64, 1)), as_strided(buf114, (768, 64, 128), (8192, 128, 1)), out=buf115)
    buf116 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf115, buf116, 98304, 128, grid=grid(98304), stream=stream0)
    buf117 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf115, buf116, buf117, 98304, 128, grid=grid(98304), stream=stream0)
    buf118 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel40.run(seed_cuda_0, buf115, buf116, buf117, buf118, 12582912, grid=grid(12582912), stream=stream0)
    buf119 = as_strided(buf111, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf111  # reuse
    kernel5.run(buf112, buf119, 6291456, grid=grid(6291456), stream=stream0)
    buf120 = as_strided(buf112, (768, 128, 64), (8192, 64, 1)); del buf112  # reuse
    aten.bmm.out(as_strided(buf118, (768, 128, 128), (16384, 128, 1)), as_strided(buf119, (768, 128, 64), (8192, 64, 1)), out=buf120)
    buf121 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf120, buf121, 6291456, grid=grid(6291456), stream=stream0)
    buf122 = as_strided(buf120, (8192, 768), (768, 1)); del buf120  # reuse
    aten.addmm.out(primals_77, as_strided(buf121, (8192, 768), (768, 1)), as_strided(primals_76, (768, 768), (1, 768)), beta=1, alpha=1, out=buf122)
    del primals_77
    buf123 = buf107; del buf107  # reuse
    kernel41.run(seed_cuda_0, buf122, buf109, buf123, 8192, 768, grid=grid(8192), stream=stream0)
    buf124 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel42.run(seed_cuda_0, buf122, buf109, buf123, buf124, 8192, 768, grid=grid(8192), stream=stream0)
    buf125 = buf123; del buf123  # reuse
    kernel41.run(seed_cuda_0, buf122, buf109, buf125, 8192, 768, grid=grid(8192), stream=stream0)
    buf126 = as_strided(buf122, (64, 128, 768), (98304, 768, 1)); del buf122  # reuse
    kernel43.run(buf126, seed_cuda_0, buf109, buf125, buf124, 6291456, grid=grid(6291456), stream=stream0)
    buf127 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf126, primals_78, primals_79, buf127, 6291456, grid=grid(6291456), stream=stream0)
    del primals_79
    buf128 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_81, as_strided(buf127, (8192, 768), (768, 1)), as_strided(primals_80, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf128)
    del primals_81
    buf129 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf128, buf129, 25165824, grid=grid(25165824), stream=stream0)
    buf130 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_83, as_strided(buf129, (8192, 3072), (3072, 1)), as_strided(primals_82, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf130)
    del primals_83
    buf131 = buf125; del buf125  # reuse
    kernel44.run(seed_cuda_0, buf130, buf127, buf131, 8192, 768, grid=grid(8192), stream=stream0)
    buf132 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel45.run(seed_cuda_0, buf130, buf127, buf131, buf132, 8192, 768, grid=grid(8192), stream=stream0)
    buf133 = buf131; del buf131  # reuse
    kernel44.run(seed_cuda_0, buf130, buf127, buf133, 8192, 768, grid=grid(8192), stream=stream0)
    buf134 = as_strided(buf130, (64, 128, 768), (98304, 768, 1)); del buf130  # reuse
    kernel46.run(buf134, seed_cuda_0, buf127, buf133, buf132, 6291456, grid=grid(6291456), stream=stream0)
    buf135 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf134, primals_84, primals_85, buf135, 6291456, grid=grid(6291456), stream=stream0)
    del primals_85
    buf136 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_87, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_86, (768, 768), (1, 768)), beta=1, alpha=1, out=buf136)
    del primals_87
    buf137 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_89, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_88, (768, 768), (1, 768)), beta=1, alpha=1, out=buf137)
    del primals_89
    buf138 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_91, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_90, (768, 768), (1, 768)), beta=1, alpha=1, out=buf138)
    del primals_91
    buf139 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf136, buf139, 6291456, grid=grid(6291456), stream=stream0)
    buf140 = as_strided(buf136, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf136  # reuse
    kernel6.run(buf137, buf140, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf141 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf139, (768, 128, 64), (8192, 64, 1)), as_strided(buf140, (768, 64, 128), (8192, 128, 1)), out=buf141)
    buf142 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf141, buf142, 98304, 128, grid=grid(98304), stream=stream0)
    buf143 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf141, buf142, buf143, 98304, 128, grid=grid(98304), stream=stream0)
    buf144 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel47.run(seed_cuda_0, buf141, buf142, buf143, buf144, 12582912, grid=grid(12582912), stream=stream0)
    buf145 = as_strided(buf137, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf137  # reuse
    kernel5.run(buf138, buf145, 6291456, grid=grid(6291456), stream=stream0)
    buf146 = as_strided(buf138, (768, 128, 64), (8192, 64, 1)); del buf138  # reuse
    aten.bmm.out(as_strided(buf144, (768, 128, 128), (16384, 128, 1)), as_strided(buf145, (768, 128, 64), (8192, 64, 1)), out=buf146)
    buf147 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf146, buf147, 6291456, grid=grid(6291456), stream=stream0)
    buf148 = as_strided(buf146, (8192, 768), (768, 1)); del buf146  # reuse
    aten.addmm.out(primals_93, as_strided(buf147, (8192, 768), (768, 1)), as_strided(primals_92, (768, 768), (1, 768)), beta=1, alpha=1, out=buf148)
    del primals_93
    buf149 = buf133; del buf133  # reuse
    kernel48.run(seed_cuda_0, buf148, buf135, buf149, 8192, 768, grid=grid(8192), stream=stream0)
    buf150 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel49.run(seed_cuda_0, buf148, buf135, buf149, buf150, 8192, 768, grid=grid(8192), stream=stream0)
    buf151 = buf149; del buf149  # reuse
    kernel48.run(seed_cuda_0, buf148, buf135, buf151, 8192, 768, grid=grid(8192), stream=stream0)
    buf152 = as_strided(buf148, (64, 128, 768), (98304, 768, 1)); del buf148  # reuse
    kernel50.run(buf152, seed_cuda_0, buf135, buf151, buf150, 6291456, grid=grid(6291456), stream=stream0)
    buf153 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf152, primals_94, primals_95, buf153, 6291456, grid=grid(6291456), stream=stream0)
    del primals_95
    buf154 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_97, as_strided(buf153, (8192, 768), (768, 1)), as_strided(primals_96, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf154)
    del primals_97
    buf155 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf154, buf155, 25165824, grid=grid(25165824), stream=stream0)
    buf156 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_99, as_strided(buf155, (8192, 3072), (3072, 1)), as_strided(primals_98, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf156)
    del primals_99
    buf157 = buf151; del buf151  # reuse
    kernel51.run(seed_cuda_0, buf156, buf153, buf157, 8192, 768, grid=grid(8192), stream=stream0)
    buf158 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel52.run(seed_cuda_0, buf156, buf153, buf157, buf158, 8192, 768, grid=grid(8192), stream=stream0)
    buf159 = buf157; del buf157  # reuse
    kernel51.run(seed_cuda_0, buf156, buf153, buf159, 8192, 768, grid=grid(8192), stream=stream0)
    buf160 = as_strided(buf156, (64, 128, 768), (98304, 768, 1)); del buf156  # reuse
    kernel53.run(buf160, seed_cuda_0, buf153, buf159, buf158, 6291456, grid=grid(6291456), stream=stream0)
    buf161 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf160, primals_100, primals_101, buf161, 6291456, grid=grid(6291456), stream=stream0)
    del primals_101
    buf162 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_103, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_102, (768, 768), (1, 768)), beta=1, alpha=1, out=buf162)
    del primals_103
    buf163 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_105, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_104, (768, 768), (1, 768)), beta=1, alpha=1, out=buf163)
    del primals_105
    buf164 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_107, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_106, (768, 768), (1, 768)), beta=1, alpha=1, out=buf164)
    del primals_107
    buf165 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf162, buf165, 6291456, grid=grid(6291456), stream=stream0)
    buf166 = as_strided(buf162, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf162  # reuse
    kernel6.run(buf163, buf166, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf167 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf165, (768, 128, 64), (8192, 64, 1)), as_strided(buf166, (768, 64, 128), (8192, 128, 1)), out=buf167)
    buf168 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf167, buf168, 98304, 128, grid=grid(98304), stream=stream0)
    buf169 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf167, buf168, buf169, 98304, 128, grid=grid(98304), stream=stream0)
    buf170 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel54.run(seed_cuda_0, buf167, buf168, buf169, buf170, 12582912, grid=grid(12582912), stream=stream0)
    buf171 = as_strided(buf163, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf163  # reuse
    kernel5.run(buf164, buf171, 6291456, grid=grid(6291456), stream=stream0)
    buf172 = as_strided(buf164, (768, 128, 64), (8192, 64, 1)); del buf164  # reuse
    aten.bmm.out(as_strided(buf170, (768, 128, 128), (16384, 128, 1)), as_strided(buf171, (768, 128, 64), (8192, 64, 1)), out=buf172)
    buf173 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf172, buf173, 6291456, grid=grid(6291456), stream=stream0)
    buf174 = as_strided(buf172, (8192, 768), (768, 1)); del buf172  # reuse
    aten.addmm.out(primals_109, as_strided(buf173, (8192, 768), (768, 1)), as_strided(primals_108, (768, 768), (1, 768)), beta=1, alpha=1, out=buf174)
    del primals_109
    buf175 = buf159; del buf159  # reuse
    kernel55.run(seed_cuda_0, buf174, buf161, buf175, 8192, 768, grid=grid(8192), stream=stream0)
    buf176 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel56.run(seed_cuda_0, buf174, buf161, buf175, buf176, 8192, 768, grid=grid(8192), stream=stream0)
    buf177 = buf175; del buf175  # reuse
    kernel55.run(seed_cuda_0, buf174, buf161, buf177, 8192, 768, grid=grid(8192), stream=stream0)
    buf178 = as_strided(buf174, (64, 128, 768), (98304, 768, 1)); del buf174  # reuse
    kernel57.run(buf178, seed_cuda_0, buf161, buf177, buf176, 6291456, grid=grid(6291456), stream=stream0)
    buf179 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf178, primals_110, primals_111, buf179, 6291456, grid=grid(6291456), stream=stream0)
    del primals_111
    buf180 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_113, as_strided(buf179, (8192, 768), (768, 1)), as_strided(primals_112, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf180)
    del primals_113
    buf181 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf180, buf181, 25165824, grid=grid(25165824), stream=stream0)
    buf182 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_115, as_strided(buf181, (8192, 3072), (3072, 1)), as_strided(primals_114, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf182)
    del primals_115
    buf183 = buf177; del buf177  # reuse
    kernel58.run(seed_cuda_0, buf182, buf179, buf183, 8192, 768, grid=grid(8192), stream=stream0)
    buf184 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel59.run(seed_cuda_0, buf182, buf179, buf183, buf184, 8192, 768, grid=grid(8192), stream=stream0)
    buf185 = buf183; del buf183  # reuse
    kernel58.run(seed_cuda_0, buf182, buf179, buf185, 8192, 768, grid=grid(8192), stream=stream0)
    buf186 = as_strided(buf182, (64, 128, 768), (98304, 768, 1)); del buf182  # reuse
    kernel60.run(buf186, seed_cuda_0, buf179, buf185, buf184, 6291456, grid=grid(6291456), stream=stream0)
    buf187 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf186, primals_116, primals_117, buf187, 6291456, grid=grid(6291456), stream=stream0)
    del primals_117
    buf188 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_119, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_118, (768, 768), (1, 768)), beta=1, alpha=1, out=buf188)
    del primals_119
    buf189 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_121, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_120, (768, 768), (1, 768)), beta=1, alpha=1, out=buf189)
    del primals_121
    buf190 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_123, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_122, (768, 768), (1, 768)), beta=1, alpha=1, out=buf190)
    del primals_123
    buf191 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf188, buf191, 6291456, grid=grid(6291456), stream=stream0)
    buf192 = as_strided(buf188, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf188  # reuse
    kernel6.run(buf189, buf192, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf193 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf191, (768, 128, 64), (8192, 64, 1)), as_strided(buf192, (768, 64, 128), (8192, 128, 1)), out=buf193)
    buf194 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf193, buf194, 98304, 128, grid=grid(98304), stream=stream0)
    buf195 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf193, buf194, buf195, 98304, 128, grid=grid(98304), stream=stream0)
    buf196 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel61.run(seed_cuda_0, buf193, buf194, buf195, buf196, 12582912, grid=grid(12582912), stream=stream0)
    buf197 = as_strided(buf189, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf189  # reuse
    kernel5.run(buf190, buf197, 6291456, grid=grid(6291456), stream=stream0)
    buf198 = as_strided(buf190, (768, 128, 64), (8192, 64, 1)); del buf190  # reuse
    aten.bmm.out(as_strided(buf196, (768, 128, 128), (16384, 128, 1)), as_strided(buf197, (768, 128, 64), (8192, 64, 1)), out=buf198)
    buf199 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf198, buf199, 6291456, grid=grid(6291456), stream=stream0)
    buf200 = as_strided(buf198, (8192, 768), (768, 1)); del buf198  # reuse
    aten.addmm.out(primals_125, as_strided(buf199, (8192, 768), (768, 1)), as_strided(primals_124, (768, 768), (1, 768)), beta=1, alpha=1, out=buf200)
    del primals_125
    buf201 = buf185; del buf185  # reuse
    kernel62.run(seed_cuda_0, buf200, buf187, buf201, 8192, 768, grid=grid(8192), stream=stream0)
    buf202 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel63.run(seed_cuda_0, buf200, buf187, buf201, buf202, 8192, 768, grid=grid(8192), stream=stream0)
    buf203 = buf201; del buf201  # reuse
    kernel62.run(seed_cuda_0, buf200, buf187, buf203, 8192, 768, grid=grid(8192), stream=stream0)
    buf204 = as_strided(buf200, (64, 128, 768), (98304, 768, 1)); del buf200  # reuse
    kernel64.run(buf204, seed_cuda_0, buf187, buf203, buf202, 6291456, grid=grid(6291456), stream=stream0)
    buf205 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf204, primals_126, primals_127, buf205, 6291456, grid=grid(6291456), stream=stream0)
    del primals_127
    buf206 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_129, as_strided(buf205, (8192, 768), (768, 1)), as_strided(primals_128, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf206)
    del primals_129
    buf207 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf206, buf207, 25165824, grid=grid(25165824), stream=stream0)
    buf208 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_131, as_strided(buf207, (8192, 3072), (3072, 1)), as_strided(primals_130, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf208)
    del primals_131
    buf209 = buf203; del buf203  # reuse
    kernel65.run(seed_cuda_0, buf208, buf205, buf209, 8192, 768, grid=grid(8192), stream=stream0)
    buf210 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel66.run(seed_cuda_0, buf208, buf205, buf209, buf210, 8192, 768, grid=grid(8192), stream=stream0)
    buf211 = buf209; del buf209  # reuse
    kernel65.run(seed_cuda_0, buf208, buf205, buf211, 8192, 768, grid=grid(8192), stream=stream0)
    buf212 = as_strided(buf208, (64, 128, 768), (98304, 768, 1)); del buf208  # reuse
    kernel67.run(buf212, seed_cuda_0, buf205, buf211, buf210, 6291456, grid=grid(6291456), stream=stream0)
    buf213 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf212, primals_132, primals_133, buf213, 6291456, grid=grid(6291456), stream=stream0)
    del primals_133
    buf214 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_135, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_134, (768, 768), (1, 768)), beta=1, alpha=1, out=buf214)
    del primals_135
    buf215 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_137, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_136, (768, 768), (1, 768)), beta=1, alpha=1, out=buf215)
    del primals_137
    buf216 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_139, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_138, (768, 768), (1, 768)), beta=1, alpha=1, out=buf216)
    del primals_139
    buf217 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf214, buf217, 6291456, grid=grid(6291456), stream=stream0)
    buf218 = as_strided(buf214, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf214  # reuse
    kernel6.run(buf215, buf218, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf219 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf217, (768, 128, 64), (8192, 64, 1)), as_strided(buf218, (768, 64, 128), (8192, 128, 1)), out=buf219)
    buf220 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf219, buf220, 98304, 128, grid=grid(98304), stream=stream0)
    buf221 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf219, buf220, buf221, 98304, 128, grid=grid(98304), stream=stream0)
    buf222 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel68.run(seed_cuda_0, buf219, buf220, buf221, buf222, 12582912, grid=grid(12582912), stream=stream0)
    buf223 = as_strided(buf215, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf215  # reuse
    kernel5.run(buf216, buf223, 6291456, grid=grid(6291456), stream=stream0)
    buf224 = as_strided(buf216, (768, 128, 64), (8192, 64, 1)); del buf216  # reuse
    aten.bmm.out(as_strided(buf222, (768, 128, 128), (16384, 128, 1)), as_strided(buf223, (768, 128, 64), (8192, 64, 1)), out=buf224)
    buf225 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf224, buf225, 6291456, grid=grid(6291456), stream=stream0)
    buf226 = as_strided(buf224, (8192, 768), (768, 1)); del buf224  # reuse
    aten.addmm.out(primals_141, as_strided(buf225, (8192, 768), (768, 1)), as_strided(primals_140, (768, 768), (1, 768)), beta=1, alpha=1, out=buf226)
    del primals_141
    buf227 = buf211; del buf211  # reuse
    kernel69.run(seed_cuda_0, buf226, buf213, buf227, 8192, 768, grid=grid(8192), stream=stream0)
    buf228 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel70.run(seed_cuda_0, buf226, buf213, buf227, buf228, 8192, 768, grid=grid(8192), stream=stream0)
    buf229 = buf227; del buf227  # reuse
    kernel69.run(seed_cuda_0, buf226, buf213, buf229, 8192, 768, grid=grid(8192), stream=stream0)
    buf230 = as_strided(buf226, (64, 128, 768), (98304, 768, 1)); del buf226  # reuse
    kernel71.run(buf230, seed_cuda_0, buf213, buf229, buf228, 6291456, grid=grid(6291456), stream=stream0)
    buf231 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf230, primals_142, primals_143, buf231, 6291456, grid=grid(6291456), stream=stream0)
    del primals_143
    buf232 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_145, as_strided(buf231, (8192, 768), (768, 1)), as_strided(primals_144, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf232)
    del primals_145
    buf233 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf232, buf233, 25165824, grid=grid(25165824), stream=stream0)
    buf234 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_147, as_strided(buf233, (8192, 3072), (3072, 1)), as_strided(primals_146, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf234)
    del primals_147
    buf235 = buf229; del buf229  # reuse
    kernel72.run(seed_cuda_0, buf234, buf231, buf235, 8192, 768, grid=grid(8192), stream=stream0)
    buf236 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel73.run(seed_cuda_0, buf234, buf231, buf235, buf236, 8192, 768, grid=grid(8192), stream=stream0)
    buf237 = buf235; del buf235  # reuse
    kernel72.run(seed_cuda_0, buf234, buf231, buf237, 8192, 768, grid=grid(8192), stream=stream0)
    buf238 = as_strided(buf234, (64, 128, 768), (98304, 768, 1)); del buf234  # reuse
    kernel74.run(buf238, seed_cuda_0, buf231, buf237, buf236, 6291456, grid=grid(6291456), stream=stream0)
    buf239 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf238, primals_148, primals_149, buf239, 6291456, grid=grid(6291456), stream=stream0)
    del primals_149
    buf240 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_151, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_150, (768, 768), (1, 768)), beta=1, alpha=1, out=buf240)
    del primals_151
    buf241 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_153, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_152, (768, 768), (1, 768)), beta=1, alpha=1, out=buf241)
    del primals_153
    buf242 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_155, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_154, (768, 768), (1, 768)), beta=1, alpha=1, out=buf242)
    del primals_155
    buf243 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf240, buf243, 6291456, grid=grid(6291456), stream=stream0)
    buf244 = as_strided(buf240, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf240  # reuse
    kernel6.run(buf241, buf244, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf245 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf243, (768, 128, 64), (8192, 64, 1)), as_strided(buf244, (768, 64, 128), (8192, 128, 1)), out=buf245)
    buf246 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf245, buf246, 98304, 128, grid=grid(98304), stream=stream0)
    buf247 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf245, buf246, buf247, 98304, 128, grid=grid(98304), stream=stream0)
    buf248 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel75.run(seed_cuda_0, buf245, buf246, buf247, buf248, 12582912, grid=grid(12582912), stream=stream0)
    buf249 = as_strided(buf241, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf241  # reuse
    kernel5.run(buf242, buf249, 6291456, grid=grid(6291456), stream=stream0)
    buf250 = as_strided(buf242, (768, 128, 64), (8192, 64, 1)); del buf242  # reuse
    aten.bmm.out(as_strided(buf248, (768, 128, 128), (16384, 128, 1)), as_strided(buf249, (768, 128, 64), (8192, 64, 1)), out=buf250)
    buf251 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf250, buf251, 6291456, grid=grid(6291456), stream=stream0)
    buf252 = as_strided(buf250, (8192, 768), (768, 1)); del buf250  # reuse
    aten.addmm.out(primals_157, as_strided(buf251, (8192, 768), (768, 1)), as_strided(primals_156, (768, 768), (1, 768)), beta=1, alpha=1, out=buf252)
    del primals_157
    buf253 = buf237; del buf237  # reuse
    kernel76.run(seed_cuda_0, buf252, buf239, buf253, 8192, 768, grid=grid(8192), stream=stream0)
    buf254 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel77.run(seed_cuda_0, buf252, buf239, buf253, buf254, 8192, 768, grid=grid(8192), stream=stream0)
    buf255 = buf253; del buf253  # reuse
    kernel76.run(seed_cuda_0, buf252, buf239, buf255, 8192, 768, grid=grid(8192), stream=stream0)
    buf256 = as_strided(buf252, (64, 128, 768), (98304, 768, 1)); del buf252  # reuse
    kernel78.run(buf256, seed_cuda_0, buf239, buf255, buf254, 6291456, grid=grid(6291456), stream=stream0)
    buf257 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf256, primals_158, primals_159, buf257, 6291456, grid=grid(6291456), stream=stream0)
    del primals_159
    buf258 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_161, as_strided(buf257, (8192, 768), (768, 1)), as_strided(primals_160, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf258)
    del primals_161
    buf259 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf258, buf259, 25165824, grid=grid(25165824), stream=stream0)
    buf260 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_163, as_strided(buf259, (8192, 3072), (3072, 1)), as_strided(primals_162, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf260)
    del primals_163
    buf261 = buf255; del buf255  # reuse
    kernel79.run(seed_cuda_0, buf260, buf257, buf261, 8192, 768, grid=grid(8192), stream=stream0)
    buf262 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel80.run(seed_cuda_0, buf260, buf257, buf261, buf262, 8192, 768, grid=grid(8192), stream=stream0)
    buf263 = buf261; del buf261  # reuse
    kernel79.run(seed_cuda_0, buf260, buf257, buf263, 8192, 768, grid=grid(8192), stream=stream0)
    buf264 = as_strided(buf260, (64, 128, 768), (98304, 768, 1)); del buf260  # reuse
    kernel81.run(buf264, seed_cuda_0, buf257, buf263, buf262, 6291456, grid=grid(6291456), stream=stream0)
    buf265 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf264, primals_164, primals_165, buf265, 6291456, grid=grid(6291456), stream=stream0)
    del primals_165
    buf266 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_167, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_166, (768, 768), (1, 768)), beta=1, alpha=1, out=buf266)
    del primals_167
    buf267 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_169, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_168, (768, 768), (1, 768)), beta=1, alpha=1, out=buf267)
    del primals_169
    buf268 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_171, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_170, (768, 768), (1, 768)), beta=1, alpha=1, out=buf268)
    del primals_171
    buf269 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf266, buf269, 6291456, grid=grid(6291456), stream=stream0)
    buf270 = as_strided(buf266, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf266  # reuse
    kernel6.run(buf267, buf270, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf271 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf269, (768, 128, 64), (8192, 64, 1)), as_strided(buf270, (768, 64, 128), (8192, 128, 1)), out=buf271)
    buf272 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf271, buf272, 98304, 128, grid=grid(98304), stream=stream0)
    buf273 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf271, buf272, buf273, 98304, 128, grid=grid(98304), stream=stream0)
    buf274 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel82.run(seed_cuda_0, buf271, buf272, buf273, buf274, 12582912, grid=grid(12582912), stream=stream0)
    buf275 = as_strided(buf267, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf267  # reuse
    kernel5.run(buf268, buf275, 6291456, grid=grid(6291456), stream=stream0)
    buf276 = as_strided(buf268, (768, 128, 64), (8192, 64, 1)); del buf268  # reuse
    aten.bmm.out(as_strided(buf274, (768, 128, 128), (16384, 128, 1)), as_strided(buf275, (768, 128, 64), (8192, 64, 1)), out=buf276)
    buf277 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf276, buf277, 6291456, grid=grid(6291456), stream=stream0)
    buf278 = as_strided(buf276, (8192, 768), (768, 1)); del buf276  # reuse
    aten.addmm.out(primals_173, as_strided(buf277, (8192, 768), (768, 1)), as_strided(primals_172, (768, 768), (1, 768)), beta=1, alpha=1, out=buf278)
    del primals_173
    buf279 = buf263; del buf263  # reuse
    kernel83.run(seed_cuda_0, buf278, buf265, buf279, 8192, 768, grid=grid(8192), stream=stream0)
    buf280 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel84.run(seed_cuda_0, buf278, buf265, buf279, buf280, 8192, 768, grid=grid(8192), stream=stream0)
    buf281 = buf279; del buf279  # reuse
    kernel83.run(seed_cuda_0, buf278, buf265, buf281, 8192, 768, grid=grid(8192), stream=stream0)
    buf282 = as_strided(buf278, (64, 128, 768), (98304, 768, 1)); del buf278  # reuse
    kernel85.run(buf282, seed_cuda_0, buf265, buf281, buf280, 6291456, grid=grid(6291456), stream=stream0)
    buf283 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf282, primals_174, primals_175, buf283, 6291456, grid=grid(6291456), stream=stream0)
    del primals_175
    buf284 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_177, as_strided(buf283, (8192, 768), (768, 1)), as_strided(primals_176, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf284)
    del primals_177
    buf285 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf284, buf285, 25165824, grid=grid(25165824), stream=stream0)
    buf286 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_179, as_strided(buf285, (8192, 3072), (3072, 1)), as_strided(primals_178, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf286)
    del primals_179
    buf287 = buf281; del buf281  # reuse
    kernel86.run(seed_cuda_0, buf286, buf283, buf287, 8192, 768, grid=grid(8192), stream=stream0)
    buf288 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel87.run(seed_cuda_0, buf286, buf283, buf287, buf288, 8192, 768, grid=grid(8192), stream=stream0)
    buf289 = buf287; del buf287  # reuse
    kernel86.run(seed_cuda_0, buf286, buf283, buf289, 8192, 768, grid=grid(8192), stream=stream0)
    buf290 = as_strided(buf286, (64, 128, 768), (98304, 768, 1)); del buf286  # reuse
    kernel88.run(buf290, seed_cuda_0, buf283, buf289, buf288, 6291456, grid=grid(6291456), stream=stream0)
    buf291 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf290, primals_180, primals_181, buf291, 6291456, grid=grid(6291456), stream=stream0)
    del primals_181
    buf292 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_183, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_182, (768, 768), (1, 768)), beta=1, alpha=1, out=buf292)
    del primals_183
    buf293 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_185, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_184, (768, 768), (1, 768)), beta=1, alpha=1, out=buf293)
    del primals_185
    buf294 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_187, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_186, (768, 768), (1, 768)), beta=1, alpha=1, out=buf294)
    del primals_187
    buf295 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel5.run(buf292, buf295, 6291456, grid=grid(6291456), stream=stream0)
    buf296 = as_strided(buf292, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf292  # reuse
    kernel6.run(buf293, buf296, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf297 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf295, (768, 128, 64), (8192, 64, 1)), as_strided(buf296, (768, 64, 128), (8192, 128, 1)), out=buf297)
    buf298 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel7.run(buf297, buf298, 98304, 128, grid=grid(98304), stream=stream0)
    buf299 = empty_strided((64, 12, 128, 1), (1536, 128, 1, 98304), device='cuda', dtype=torch.float32)
    kernel8.run(buf297, buf298, buf299, 98304, 128, grid=grid(98304), stream=stream0)
    buf300 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel89.run(seed_cuda_0, buf297, buf298, buf299, buf300, 12582912, grid=grid(12582912), stream=stream0)
    buf301 = as_strided(buf293, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf293  # reuse
    kernel5.run(buf294, buf301, 6291456, grid=grid(6291456), stream=stream0)
    buf302 = as_strided(buf294, (768, 128, 64), (8192, 64, 1)); del buf294  # reuse
    aten.bmm.out(as_strided(buf300, (768, 128, 128), (16384, 128, 1)), as_strided(buf301, (768, 128, 64), (8192, 64, 1)), out=buf302)
    buf303 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel10.run(buf302, buf303, 6291456, grid=grid(6291456), stream=stream0)
    buf304 = as_strided(buf302, (8192, 768), (768, 1)); del buf302  # reuse
    aten.addmm.out(primals_189, as_strided(buf303, (8192, 768), (768, 1)), as_strided(primals_188, (768, 768), (1, 768)), beta=1, alpha=1, out=buf304)
    del primals_189
    buf305 = buf289; del buf289  # reuse
    kernel90.run(seed_cuda_0, buf304, buf291, buf305, 8192, 768, grid=grid(8192), stream=stream0)
    buf306 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel91.run(seed_cuda_0, buf304, buf291, buf305, buf306, 8192, 768, grid=grid(8192), stream=stream0)
    buf307 = buf305; del buf305  # reuse
    kernel90.run(seed_cuda_0, buf304, buf291, buf307, 8192, 768, grid=grid(8192), stream=stream0)
    buf308 = as_strided(buf304, (64, 128, 768), (98304, 768, 1)); del buf304  # reuse
    kernel92.run(buf308, seed_cuda_0, buf291, buf307, buf306, 6291456, grid=grid(6291456), stream=stream0)
    buf309 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf308, primals_190, primals_191, buf309, 6291456, grid=grid(6291456), stream=stream0)
    del primals_191
    buf310 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_193, as_strided(buf309, (8192, 768), (768, 1)), as_strided(primals_192, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf310)
    del primals_193
    buf311 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf310, buf311, 25165824, grid=grid(25165824), stream=stream0)
    buf312 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_195, as_strided(buf311, (8192, 3072), (3072, 1)), as_strided(primals_194, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf312)
    del primals_195
    buf313 = buf307; del buf307  # reuse
    kernel93.run(seed_cuda_0, buf312, buf309, buf313, 8192, 768, grid=grid(8192), stream=stream0)
    buf314 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel94.run(seed_cuda_0, buf312, buf309, buf313, buf314, 8192, 768, grid=grid(8192), stream=stream0)
    buf315 = buf313; del buf313  # reuse
    kernel93.run(seed_cuda_0, buf312, buf309, buf315, 8192, 768, grid=grid(8192), stream=stream0)
    buf316 = as_strided(buf312, (64, 128, 768), (98304, 768, 1)); del buf312  # reuse
    kernel95.run(buf316, seed_cuda_0, buf309, buf315, buf314, 6291456, grid=grid(6291456), stream=stream0)
    buf317 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf316, primals_196, primals_197, buf317, 6291456, grid=grid(6291456), stream=stream0)
    del primals_197
    buf318 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_199, as_strided(buf317, (8192, 768), (768, 1)), as_strided(primals_198, (768, 768), (1, 768)), beta=1, alpha=1, out=buf318)
    del primals_199
    buf319 = buf315; del buf315  # reuse
    kernel96.run(buf318, buf319, 8192, 768, grid=grid(8192), stream=stream0)
    buf320 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel97.run(buf318, buf319, buf320, 8192, 768, grid=grid(8192), stream=stream0)
    buf321 = buf319; del buf319  # reuse
    kernel96.run(buf318, buf321, 8192, 768, grid=grid(8192), stream=stream0)
    buf322 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel98.run(buf318, buf321, buf320, buf322, 6291456, grid=grid(6291456), stream=stream0)
    buf323 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf322, primals_200, primals_201, buf323, 6291456, grid=grid(6291456), stream=stream0)
    del primals_201
    buf324 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_202, as_strided(buf323, (8192, 768), (768, 1)), as_strided(primals_1, (768, 30522), (1, 768)), beta=1, alpha=1, out=buf324)
    del primals_202
    buf325 = as_strided(buf321, (8192, 1), (1, 8192)); del buf321  # reuse
    kernel99.run(buf324, buf325, 8192, 30522, grid=grid(8192), stream=stream0)
    buf326 = empty_strided((8192, 1), (1, 8192), device='cuda', dtype=torch.float32)
    kernel100.run(buf324, buf325, buf326, 8192, 30522, grid=grid(8192), stream=stream0)
    buf327 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    kernel101.run(buf324, buf325, buf326, buf327, 250036224, grid=grid(250036224), stream=stream0)
    del buf325
    del buf326
    buf328 = empty_strided((), (), device='cuda', dtype=torch.float32)
    kernel102.run(primals_206, buf327, buf328, 1, 8192, grid=grid(1), stream=stream0)
    buf329 = buf328; del buf328  # reuse
    kernel103.run(buf329, 1, grid=grid(1), stream=stream0)
    buf330 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel104.run(seed_cuda_0, buf330, 6291456, grid=grid(6291456), stream=stream0)
    buf331 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel105.run(seed_cuda_0, buf331, 12582912, grid=grid(12582912), stream=stream0)
    buf332 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel106.run(seed_cuda_0, buf332, 6291456, grid=grid(6291456), stream=stream0)
    buf333 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel107.run(seed_cuda_0, buf333, 6291456, grid=grid(6291456), stream=stream0)
    buf334 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel108.run(seed_cuda_0, buf334, 12582912, grid=grid(12582912), stream=stream0)
    buf335 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel109.run(seed_cuda_0, buf335, 6291456, grid=grid(6291456), stream=stream0)
    buf336 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel110.run(seed_cuda_0, buf336, 6291456, grid=grid(6291456), stream=stream0)
    buf337 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel111.run(seed_cuda_0, buf337, 12582912, grid=grid(12582912), stream=stream0)
    buf338 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel112.run(seed_cuda_0, buf338, 6291456, grid=grid(6291456), stream=stream0)
    buf339 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel113.run(seed_cuda_0, buf339, 6291456, grid=grid(6291456), stream=stream0)
    buf340 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel114.run(seed_cuda_0, buf340, 12582912, grid=grid(12582912), stream=stream0)
    buf341 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel115.run(seed_cuda_0, buf341, 6291456, grid=grid(6291456), stream=stream0)
    buf342 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel116.run(seed_cuda_0, buf342, 6291456, grid=grid(6291456), stream=stream0)
    buf343 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel117.run(seed_cuda_0, buf343, 12582912, grid=grid(12582912), stream=stream0)
    buf344 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel118.run(seed_cuda_0, buf344, 6291456, grid=grid(6291456), stream=stream0)
    buf345 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel119.run(seed_cuda_0, buf345, 6291456, grid=grid(6291456), stream=stream0)
    buf346 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel120.run(seed_cuda_0, buf346, 12582912, grid=grid(12582912), stream=stream0)
    buf347 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel121.run(seed_cuda_0, buf347, 6291456, grid=grid(6291456), stream=stream0)
    buf348 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel122.run(seed_cuda_0, buf348, 6291456, grid=grid(6291456), stream=stream0)
    buf349 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel123.run(seed_cuda_0, buf349, 12582912, grid=grid(12582912), stream=stream0)
    buf350 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel124.run(seed_cuda_0, buf350, 6291456, grid=grid(6291456), stream=stream0)
    buf351 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel125.run(seed_cuda_0, buf351, 6291456, grid=grid(6291456), stream=stream0)
    buf352 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel126.run(seed_cuda_0, buf352, 12582912, grid=grid(12582912), stream=stream0)
    buf353 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel127.run(seed_cuda_0, buf353, 6291456, grid=grid(6291456), stream=stream0)
    buf354 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel128.run(seed_cuda_0, buf354, 6291456, grid=grid(6291456), stream=stream0)
    buf355 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel129.run(seed_cuda_0, buf355, 12582912, grid=grid(12582912), stream=stream0)
    buf356 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel130.run(seed_cuda_0, buf356, 6291456, grid=grid(6291456), stream=stream0)
    buf357 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel131.run(seed_cuda_0, buf357, 6291456, grid=grid(6291456), stream=stream0)
    buf358 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel132.run(seed_cuda_0, buf358, 12582912, grid=grid(12582912), stream=stream0)
    buf359 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel133.run(seed_cuda_0, buf359, 6291456, grid=grid(6291456), stream=stream0)
    buf360 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel134.run(seed_cuda_0, buf360, 6291456, grid=grid(6291456), stream=stream0)
    buf361 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel135.run(seed_cuda_0, buf361, 12582912, grid=grid(12582912), stream=stream0)
    buf362 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel136.run(seed_cuda_0, buf362, 6291456, grid=grid(6291456), stream=stream0)
    buf363 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel137.run(seed_cuda_0, buf363, 6291456, grid=grid(6291456), stream=stream0)
    buf364 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel138.run(seed_cuda_0, buf364, 12582912, grid=grid(12582912), stream=stream0)
    buf365 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel139.run(seed_cuda_0, buf365, 6291456, grid=grid(6291456), stream=stream0)
    buf366 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    kernel140.run(seed_cuda_0, buf366, 6291456, grid=grid(6291456), stream=stream0)
    buf367 = buf320; del buf320  # reuse
    kernel141.run(buf367, 8192, grid=grid(8192), stream=stream0)
    buf368 = as_strided(buf318, (64, 128, 768), (98304, 768, 1)); del buf318  # reuse
    kernel142.run(buf368, 6291456, grid=grid(6291456), stream=stream0)
    buf369 = buf314; del buf314  # reuse
    kernel141.run(buf369, 8192, grid=grid(8192), stream=stream0)
    buf370 = as_strided(buf310, (64, 128, 3072), (393216, 3072, 1)); del buf310  # reuse
    kernel143.run(buf370, 25165824, grid=grid(25165824), stream=stream0)
    buf371 = buf306; del buf306  # reuse
    kernel141.run(buf371, 8192, grid=grid(8192), stream=stream0)
    buf372 = as_strided(buf297, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf297  # reuse
    kernel144.run(buf372, buf298, buf299, 12582912, grid=grid(12582912), stream=stream0)
    del buf298
    del buf299
    buf373 = buf288; del buf288  # reuse
    kernel141.run(buf373, 8192, grid=grid(8192), stream=stream0)
    buf374 = as_strided(buf284, (64, 128, 3072), (393216, 3072, 1)); del buf284  # reuse
    kernel143.run(buf374, 25165824, grid=grid(25165824), stream=stream0)
    buf375 = buf280; del buf280  # reuse
    kernel141.run(buf375, 8192, grid=grid(8192), stream=stream0)
    buf376 = as_strided(buf271, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf271  # reuse
    kernel144.run(buf376, buf272, buf273, 12582912, grid=grid(12582912), stream=stream0)
    del buf272
    del buf273
    buf377 = buf262; del buf262  # reuse
    kernel141.run(buf377, 8192, grid=grid(8192), stream=stream0)
    buf378 = as_strided(buf258, (64, 128, 3072), (393216, 3072, 1)); del buf258  # reuse
    kernel143.run(buf378, 25165824, grid=grid(25165824), stream=stream0)
    buf379 = buf254; del buf254  # reuse
    kernel141.run(buf379, 8192, grid=grid(8192), stream=stream0)
    buf380 = as_strided(buf245, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf245  # reuse
    kernel144.run(buf380, buf246, buf247, 12582912, grid=grid(12582912), stream=stream0)
    del buf246
    del buf247
    buf381 = buf236; del buf236  # reuse
    kernel141.run(buf381, 8192, grid=grid(8192), stream=stream0)
    buf382 = as_strided(buf232, (64, 128, 3072), (393216, 3072, 1)); del buf232  # reuse
    kernel143.run(buf382, 25165824, grid=grid(25165824), stream=stream0)
    buf383 = buf228; del buf228  # reuse
    kernel141.run(buf383, 8192, grid=grid(8192), stream=stream0)
    buf384 = as_strided(buf219, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf219  # reuse
    kernel144.run(buf384, buf220, buf221, 12582912, grid=grid(12582912), stream=stream0)
    del buf220
    del buf221
    buf385 = buf210; del buf210  # reuse
    kernel141.run(buf385, 8192, grid=grid(8192), stream=stream0)
    buf386 = as_strided(buf206, (64, 128, 3072), (393216, 3072, 1)); del buf206  # reuse
    kernel143.run(buf386, 25165824, grid=grid(25165824), stream=stream0)
    buf387 = buf202; del buf202  # reuse
    kernel141.run(buf387, 8192, grid=grid(8192), stream=stream0)
    buf388 = as_strided(buf193, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf193  # reuse
    kernel144.run(buf388, buf194, buf195, 12582912, grid=grid(12582912), stream=stream0)
    del buf194
    del buf195
    buf389 = buf184; del buf184  # reuse
    kernel141.run(buf389, 8192, grid=grid(8192), stream=stream0)
    buf390 = as_strided(buf180, (64, 128, 3072), (393216, 3072, 1)); del buf180  # reuse
    kernel143.run(buf390, 25165824, grid=grid(25165824), stream=stream0)
    buf391 = buf176; del buf176  # reuse
    kernel141.run(buf391, 8192, grid=grid(8192), stream=stream0)
    buf392 = as_strided(buf167, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf167  # reuse
    kernel144.run(buf392, buf168, buf169, 12582912, grid=grid(12582912), stream=stream0)
    del buf168
    del buf169
    buf393 = buf158; del buf158  # reuse
    kernel141.run(buf393, 8192, grid=grid(8192), stream=stream0)
    buf394 = as_strided(buf154, (64, 128, 3072), (393216, 3072, 1)); del buf154  # reuse
    kernel143.run(buf394, 25165824, grid=grid(25165824), stream=stream0)
    buf395 = buf150; del buf150  # reuse
    kernel141.run(buf395, 8192, grid=grid(8192), stream=stream0)
    buf396 = as_strided(buf141, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf141  # reuse
    kernel144.run(buf396, buf142, buf143, 12582912, grid=grid(12582912), stream=stream0)
    del buf142
    del buf143
    buf397 = buf132; del buf132  # reuse
    kernel141.run(buf397, 8192, grid=grid(8192), stream=stream0)
    buf398 = as_strided(buf128, (64, 128, 3072), (393216, 3072, 1)); del buf128  # reuse
    kernel143.run(buf398, 25165824, grid=grid(25165824), stream=stream0)
    buf399 = buf124; del buf124  # reuse
    kernel141.run(buf399, 8192, grid=grid(8192), stream=stream0)
    buf400 = as_strided(buf115, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf115  # reuse
    kernel144.run(buf400, buf116, buf117, 12582912, grid=grid(12582912), stream=stream0)
    del buf116
    del buf117
    buf401 = buf106; del buf106  # reuse
    kernel141.run(buf401, 8192, grid=grid(8192), stream=stream0)
    buf402 = as_strided(buf102, (64, 128, 3072), (393216, 3072, 1)); del buf102  # reuse
    kernel143.run(buf402, 25165824, grid=grid(25165824), stream=stream0)
    buf403 = buf98; del buf98  # reuse
    kernel141.run(buf403, 8192, grid=grid(8192), stream=stream0)
    buf404 = as_strided(buf89, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf89  # reuse
    kernel144.run(buf404, buf90, buf91, 12582912, grid=grid(12582912), stream=stream0)
    del buf90
    del buf91
    buf405 = buf80; del buf80  # reuse
    kernel141.run(buf405, 8192, grid=grid(8192), stream=stream0)
    buf406 = as_strided(buf76, (64, 128, 3072), (393216, 3072, 1)); del buf76  # reuse
    kernel143.run(buf406, 25165824, grid=grid(25165824), stream=stream0)
    buf407 = buf72; del buf72  # reuse
    kernel141.run(buf407, 8192, grid=grid(8192), stream=stream0)
    buf408 = as_strided(buf63, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf63  # reuse
    kernel144.run(buf408, buf64, buf65, 12582912, grid=grid(12582912), stream=stream0)
    del buf64
    del buf65
    buf409 = buf54; del buf54  # reuse
    kernel141.run(buf409, 8192, grid=grid(8192), stream=stream0)
    buf410 = as_strided(buf50, (64, 128, 3072), (393216, 3072, 1)); del buf50  # reuse
    kernel143.run(buf410, 25165824, grid=grid(25165824), stream=stream0)
    buf411 = buf46; del buf46  # reuse
    kernel141.run(buf411, 8192, grid=grid(8192), stream=stream0)
    buf412 = as_strided(buf37, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf37  # reuse
    kernel144.run(buf412, buf38, buf39, 12582912, grid=grid(12582912), stream=stream0)
    del buf38
    del buf39
    buf413 = buf28; del buf28  # reuse
    kernel141.run(buf413, 8192, grid=grid(8192), stream=stream0)
    buf414 = as_strided(buf24, (64, 128, 3072), (393216, 3072, 1)); del buf24  # reuse
    kernel143.run(buf414, 25165824, grid=grid(25165824), stream=stream0)
    buf415 = buf20; del buf20  # reuse
    kernel141.run(buf415, 8192, grid=grid(8192), stream=stream0)
    buf416 = as_strided(buf11, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf11  # reuse
    kernel144.run(buf416, buf12, buf13, 12582912, grid=grid(12582912), stream=stream0)
    del buf12
    del buf13
    buf417 = buf2; del buf2  # reuse
    kernel141.run(buf417, 8192, grid=grid(8192), stream=stream0)
    return (buf329, as_strided(buf324, (64, 128, 30522), (3906816, 30522, 1)), primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, as_strided(primals_203, (1, 128), (512, 1)), buf4, buf330, as_strided(buf5, (8192, 768), (768, 1)), buf331, as_strided(buf17, (8192, 768), (768, 1)), buf332, buf22, as_strided(buf23, (8192, 768), (768, 1)), as_strided(buf25, (8192, 3072), (3072, 1)), buf333, buf30, as_strided(buf31, (8192, 768), (768, 1)), buf334, as_strided(buf43, (8192, 768), (768, 1)), buf335, buf48, as_strided(buf49, (8192, 768), (768, 1)), as_strided(buf51, (8192, 3072), (3072, 1)), buf336, buf56, as_strided(buf57, (8192, 768), (768, 1)), buf337, as_strided(buf69, (8192, 768), (768, 1)), buf338, buf74, as_strided(buf75, (8192, 768), (768, 1)), as_strided(buf77, (8192, 3072), (3072, 1)), buf339, buf82, as_strided(buf83, (8192, 768), (768, 1)), buf340, as_strided(buf95, (8192, 768), (768, 1)), buf341, buf100, as_strided(buf101, (8192, 768), (768, 1)), as_strided(buf103, (8192, 3072), (3072, 1)), buf342, buf108, as_strided(buf109, (8192, 768), (768, 1)), buf343, as_strided(buf121, (8192, 768), (768, 1)), buf344, buf126, as_strided(buf127, (8192, 768), (768, 1)), as_strided(buf129, (8192, 3072), (3072, 1)), buf345, buf134, as_strided(buf135, (8192, 768), (768, 1)), buf346, as_strided(buf147, (8192, 768), (768, 1)), buf347, buf152, as_strided(buf153, (8192, 768), (768, 1)), as_strided(buf155, (8192, 3072), (3072, 1)), buf348, buf160, as_strided(buf161, (8192, 768), (768, 1)), buf349, as_strided(buf173, (8192, 768), (768, 1)), buf350, buf178, as_strided(buf179, (8192, 768), (768, 1)), as_strided(buf181, (8192, 3072), (3072, 1)), buf351, buf186, as_strided(buf187, (8192, 768), (768, 1)), buf352, as_strided(buf199, (8192, 768), (768, 1)), buf353, buf204, as_strided(buf205, (8192, 768), (768, 1)), as_strided(buf207, (8192, 3072), (3072, 1)), buf354, buf212, as_strided(buf213, (8192, 768), (768, 1)), buf355, as_strided(buf225, (8192, 768), (768, 1)), buf356, buf230, as_strided(buf231, (8192, 768), (768, 1)), as_strided(buf233, (8192, 3072), (3072, 1)), buf357, buf238, as_strided(buf239, (8192, 768), (768, 1)), buf358, as_strided(buf251, (8192, 768), (768, 1)), buf359, buf256, as_strided(buf257, (8192, 768), (768, 1)), as_strided(buf259, (8192, 3072), (3072, 1)), buf360, buf264, as_strided(buf265, (8192, 768), (768, 1)), buf361, as_strided(buf277, (8192, 768), (768, 1)), buf362, buf282, as_strided(buf283, (8192, 768), (768, 1)), as_strided(buf285, (8192, 3072), (3072, 1)), buf363, buf290, as_strided(buf291, (8192, 768), (768, 1)), buf364, as_strided(buf303, (8192, 768), (768, 1)), buf365, buf308, as_strided(buf309, (8192, 768), (768, 1)), as_strided(buf311, (8192, 3072), (3072, 1)), buf366, buf316, as_strided(buf317, (8192, 768), (768, 1)), buf322, as_strided(buf323, (8192, 768), (768, 1)), buf327, as_strided(primals_206, (8192, 1), (1, 1)), as_strided(primals_1, (30522, 768), (768, 1)), buf367, buf368, as_strided(primals_198, (768, 768), (768, 1)), buf369, as_strided(primals_194, (768, 3072), (3072, 1)), buf370, as_strided(primals_192, (3072, 768), (768, 1)), buf371, as_strided(primals_188, (768, 768), (768, 1)), as_strided(buf300, (768, 128, 128), (16384, 1, 128)), as_strided(buf301, (768, 64, 128), (8192, 1, 64)), buf372, as_strided(buf295, (768, 64, 128), (8192, 1, 64)), as_strided(buf296, (768, 128, 64), (8192, 1, 128)), as_strided(primals_186, (768, 768), (768, 1)), as_strided(primals_184, (768, 768), (768, 1)), as_strided(primals_182, (768, 768), (768, 1)), buf373, as_strided(primals_178, (768, 3072), (3072, 1)), buf374, as_strided(primals_176, (3072, 768), (768, 1)), buf375, as_strided(primals_172, (768, 768), (768, 1)), as_strided(buf274, (768, 128, 128), (16384, 1, 128)), as_strided(buf275, (768, 64, 128), (8192, 1, 64)), buf376, as_strided(buf269, (768, 64, 128), (8192, 1, 64)), as_strided(buf270, (768, 128, 64), (8192, 1, 128)), as_strided(primals_170, (768, 768), (768, 1)), as_strided(primals_168, (768, 768), (768, 1)), as_strided(primals_166, (768, 768), (768, 1)), buf377, as_strided(primals_162, (768, 3072), (3072, 1)), buf378, as_strided(primals_160, (3072, 768), (768, 1)), buf379, as_strided(primals_156, (768, 768), (768, 1)), as_strided(buf248, (768, 128, 128), (16384, 1, 128)), as_strided(buf249, (768, 64, 128), (8192, 1, 64)), buf380, as_strided(buf243, (768, 64, 128), (8192, 1, 64)), as_strided(buf244, (768, 128, 64), (8192, 1, 128)), as_strided(primals_154, (768, 768), (768, 1)), as_strided(primals_152, (768, 768), (768, 1)), as_strided(primals_150, (768, 768), (768, 1)), buf381, as_strided(primals_146, (768, 3072), (3072, 1)), buf382, as_strided(primals_144, (3072, 768), (768, 1)), buf383, as_strided(primals_140, (768, 768), (768, 1)), as_strided(buf222, (768, 128, 128), (16384, 1, 128)), as_strided(buf223, (768, 64, 128), (8192, 1, 64)), buf384, as_strided(buf217, (768, 64, 128), (8192, 1, 64)), as_strided(buf218, (768, 128, 64), (8192, 1, 128)), as_strided(primals_138, (768, 768), (768, 1)), as_strided(primals_136, (768, 768), (768, 1)), as_strided(primals_134, (768, 768), (768, 1)), buf385, as_strided(primals_130, (768, 3072), (3072, 1)), buf386, as_strided(primals_128, (3072, 768), (768, 1)), buf387, as_strided(primals_124, (768, 768), (768, 1)), as_strided(buf196, (768, 128, 128), (16384, 1, 128)), as_strided(buf197, (768, 64, 128), (8192, 1, 64)), buf388, as_strided(buf191, (768, 64, 128), (8192, 1, 64)), as_strided(buf192, (768, 128, 64), (8192, 1, 128)), as_strided(primals_122, (768, 768), (768, 1)), as_strided(primals_120, (768, 768), (768, 1)), as_strided(primals_118, (768, 768), (768, 1)), buf389, as_strided(primals_114, (768, 3072), (3072, 1)), buf390, as_strided(primals_112, (3072, 768), (768, 1)), buf391, as_strided(primals_108, (768, 768), (768, 1)), as_strided(buf170, (768, 128, 128), (16384, 1, 128)), as_strided(buf171, (768, 64, 128), (8192, 1, 64)), buf392, as_strided(buf165, (768, 64, 128), (8192, 1, 64)), as_strided(buf166, (768, 128, 64), (8192, 1, 128)), as_strided(primals_106, (768, 768), (768, 1)), as_strided(primals_104, (768, 768), (768, 1)), as_strided(primals_102, (768, 768), (768, 1)), buf393, as_strided(primals_98, (768, 3072), (3072, 1)), buf394, as_strided(primals_96, (3072, 768), (768, 1)), buf395, as_strided(primals_92, (768, 768), (768, 1)), as_strided(buf144, (768, 128, 128), (16384, 1, 128)), as_strided(buf145, (768, 64, 128), (8192, 1, 64)), buf396, as_strided(buf139, (768, 64, 128), (8192, 1, 64)), as_strided(buf140, (768, 128, 64), (8192, 1, 128)), as_strided(primals_90, (768, 768), (768, 1)), as_strided(primals_88, (768, 768), (768, 1)), as_strided(primals_86, (768, 768), (768, 1)), buf397, as_strided(primals_82, (768, 3072), (3072, 1)), buf398, as_strided(primals_80, (3072, 768), (768, 1)), buf399, as_strided(primals_76, (768, 768), (768, 1)), as_strided(buf118, (768, 128, 128), (16384, 1, 128)), as_strided(buf119, (768, 64, 128), (8192, 1, 64)), buf400, as_strided(buf113, (768, 64, 128), (8192, 1, 64)), as_strided(buf114, (768, 128, 64), (8192, 1, 128)), as_strided(primals_74, (768, 768), (768, 1)), as_strided(primals_72, (768, 768), (768, 1)), as_strided(primals_70, (768, 768), (768, 1)), buf401, as_strided(primals_66, (768, 3072), (3072, 1)), buf402, as_strided(primals_64, (3072, 768), (768, 1)), buf403, as_strided(primals_60, (768, 768), (768, 1)), as_strided(buf92, (768, 128, 128), (16384, 1, 128)), as_strided(buf93, (768, 64, 128), (8192, 1, 64)), buf404, as_strided(buf87, (768, 64, 128), (8192, 1, 64)), as_strided(buf88, (768, 128, 64), (8192, 1, 128)), as_strided(primals_58, (768, 768), (768, 1)), as_strided(primals_56, (768, 768), (768, 1)), as_strided(primals_54, (768, 768), (768, 1)), buf405, as_strided(primals_50, (768, 3072), (3072, 1)), buf406, as_strided(primals_48, (3072, 768), (768, 1)), buf407, as_strided(primals_44, (768, 768), (768, 1)), as_strided(buf66, (768, 128, 128), (16384, 1, 128)), as_strided(buf67, (768, 64, 128), (8192, 1, 64)), buf408, as_strided(buf61, (768, 64, 128), (8192, 1, 64)), as_strided(buf62, (768, 128, 64), (8192, 1, 128)), as_strided(primals_42, (768, 768), (768, 1)), as_strided(primals_40, (768, 768), (768, 1)), as_strided(primals_38, (768, 768), (768, 1)), buf409, as_strided(primals_34, (768, 3072), (3072, 1)), buf410, as_strided(primals_32, (3072, 768), (768, 1)), buf411, as_strided(primals_28, (768, 768), (768, 1)), as_strided(buf40, (768, 128, 128), (16384, 1, 128)), as_strided(buf41, (768, 64, 128), (8192, 1, 64)), buf412, as_strided(buf35, (768, 64, 128), (8192, 1, 64)), as_strided(buf36, (768, 128, 64), (8192, 1, 128)), as_strided(primals_26, (768, 768), (768, 1)), as_strided(primals_24, (768, 768), (768, 1)), as_strided(primals_22, (768, 768), (768, 1)), buf413, as_strided(primals_18, (768, 3072), (3072, 1)), buf414, as_strided(primals_16, (3072, 768), (768, 1)), buf415, as_strided(primals_12, (768, 768), (768, 1)), as_strided(buf14, (768, 128, 128), (16384, 1, 128)), as_strided(buf15, (768, 64, 128), (8192, 1, 64)), buf416, as_strided(buf9, (768, 64, 128), (8192, 1, 64)), as_strided(buf10, (768, 128, 64), (8192, 1, 128)), as_strided(primals_10, (768, 768), (768, 1)), as_strided(primals_8, (768, 768), (768, 1)), as_strided(primals_6, (768, 768), (768, 1)), buf417, as_strided(primals_204, (128, ), (1, )), as_strided(primals_205, (8192, ), (1, )), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    seed_cuda_0 = rand_strided((), (), device='cuda', dtype=torch.int64)
    primals_1 = rand_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_2 = rand_strided((2, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_3 = rand_strided((512, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_16 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_17 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_18 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_24 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_32 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_33 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_34 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_38 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_40 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_48 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_50 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_54 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_56 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_64 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_65 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_66 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_72 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_80 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_82 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_96 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_97 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_98 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_102 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_104 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_112 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_113 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_114 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_120 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_122 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_124 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_128 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_129 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_130 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_134 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_136 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_145 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_150 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_152 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_154 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_160 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_161 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_162 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_166 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_168 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_170 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_172 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_176 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_177 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_178 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_182 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_184 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_188 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_192 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_193 = rand_strided((3072, ), (1, ), device='cuda', dtype=torch.float32)
    primals_194 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_198 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_202 = rand_strided((30522, ), (1, ), device='cuda', dtype=torch.float32)
    primals_203 = rand_strided((1, 512), (512, 1), device='cuda', dtype=torch.int64)
    primals_204 = rand_strided((1, 512), (512, 1), device='cuda', dtype=torch.int64)
    primals_205 = rand_strided((64, 128), (128, 1), device='cuda', dtype=torch.int64)
    primals_206 = rand_strided((64, 128), (128, 1), device='cuda', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]))
