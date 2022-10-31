
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 512)
    x0 = xindex % 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*tmp0) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


kernel1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = r1 + (512*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.reshape(tl.sum(_tmp13, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp13, xmask)
''')


kernel2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x2), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = x2
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp14 = 512
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = tl.sqrt(tmp17)
    tmp19 = 1 / tmp18
    tmp20 = tmp12 * tmp19
    tmp21 = tmp0 * tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel3 = async_compile.triton('''
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


kernel4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024, 128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 768
    ynumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.reshape(tl.arange(0, YBLOCK), [1, YBLOCK])
    ymask = yindex < ynumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*y2) + (49152*x1)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (128*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


kernel5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = r3
        tmp2 = x0
        tmp3 = tmp1 - tmp2
        tmp4 = 0
        tmp5 = tmp3 > tmp4
        tmp6 = tmp5.to(tl.int64)
        tmp7 = 16
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8 + tmp4
        tmp10 = tl.abs(tmp3)
        tmp11 = 8
        tmp12 = tmp10 < tmp11
        tmp13 = tmp10.to(tl.float32)
        tmp14 = tmp13 / tmp11
        tmp15 = tl.log(tmp14)
        tmp16 = 2.772588722239781
        tmp17 = tmp15 / tmp16
        tmp18 = tmp17 * tmp11
        tmp19 = tmp18.to(tl.int64)
        tmp20 = tmp19 + tmp11
        tmp21 = 15
        tmp22 = tl.minimum(tmp20, tmp21)
        tmp23 = tl.where(tmp12, tmp10, tmp22)
        tmp24 = tmp9 + tmp23
        tmp25 = tl.load(in_ptr1 + (x1 + (6*tmp24) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = 1.0
        tmp27 = 1
        tmp28 = tmp26 - tmp27
        tmp29 = -3.4028234663852886e+38
        tmp30 = tmp28 * tmp29
        tmp31 = tmp25 + tmp30
        tmp32 = tmp0 + tmp31
        _tmp33 = tl.where(xmask & rmask & (_tmp33 < tmp32), tmp32, _tmp33)
    tmp33 = tl.reshape(tl.max(_tmp33, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x4, tmp33, xmask)
''')


kernel6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    tmp33 = tl.load(in_ptr2 + (x4), xmask)
    _tmp36 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = r3
        tmp2 = x0
        tmp3 = tmp1 - tmp2
        tmp4 = 0
        tmp5 = tmp3 > tmp4
        tmp6 = tmp5.to(tl.int64)
        tmp7 = 16
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8 + tmp4
        tmp10 = tl.abs(tmp3)
        tmp11 = 8
        tmp12 = tmp10 < tmp11
        tmp13 = tmp10.to(tl.float32)
        tmp14 = tmp13 / tmp11
        tmp15 = tl.log(tmp14)
        tmp16 = 2.772588722239781
        tmp17 = tmp15 / tmp16
        tmp18 = tmp17 * tmp11
        tmp19 = tmp18.to(tl.int64)
        tmp20 = tmp19 + tmp11
        tmp21 = 15
        tmp22 = tl.minimum(tmp20, tmp21)
        tmp23 = tl.where(tmp12, tmp10, tmp22)
        tmp24 = tmp9 + tmp23
        tmp25 = tl.load(in_ptr1 + (x1 + (6*tmp24) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = 1.0
        tmp27 = 1
        tmp28 = tmp26 - tmp27
        tmp29 = -3.4028234663852886e+38
        tmp30 = tmp28 * tmp29
        tmp31 = tmp25 + tmp30
        tmp32 = tmp0 + tmp31
        tmp34 = tmp32 - tmp33
        tmp35 = tl.exp(tmp34)
        _tmp36 = tl.where(xmask & rmask, _tmp36 + tmp35, _tmp36)
    tmp36 = tl.reshape(tl.sum(_tmp36, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x4, tmp36, xmask)
''')


kernel7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x2 = (xindex // 16384) % 6
    x6 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp33 = tl.load(in_ptr2 + (x6), xmask)
    tmp36 = tl.load(in_ptr3 + (x6), xmask)
    tmp1 = x0
    tmp2 = x1
    tmp3 = tmp1 - tmp2
    tmp4 = 0
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.int64)
    tmp7 = 16
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 + tmp4
    tmp10 = tl.abs(tmp3)
    tmp11 = 8
    tmp12 = tmp10 < tmp11
    tmp13 = tmp10.to(tl.float32)
    tmp14 = tmp13 / tmp11
    tmp15 = tl.log(tmp14)
    tmp16 = 2.772588722239781
    tmp17 = tmp15 / tmp16
    tmp18 = tmp17 * tmp11
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tmp19 + tmp11
    tmp21 = 15
    tmp22 = tl.minimum(tmp20, tmp21)
    tmp23 = tl.where(tmp12, tmp10, tmp22)
    tmp24 = tmp9 + tmp23
    tmp25 = tl.load(in_ptr1 + (x2 + (6*tmp24) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tmp26 = 1.0
    tmp27 = 1
    tmp28 = tmp26 - tmp27
    tmp29 = -3.4028234663852886e+38
    tmp30 = tmp28 * tmp29
    tmp31 = tmp25 + tmp30
    tmp32 = tmp0 + tmp31
    tmp34 = tmp32 - tmp33
    tmp35 = tl.exp(tmp34)
    tmp37 = tmp35 / tmp36
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp37, xmask)
''')


kernel8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel9 = async_compile.triton('''
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


kernel10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp16 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = 131072 + x0
    tmp13 = tl.rand(tmp2, tmp12)
    tmp14 = tmp13 > tmp5
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp11 + tmp18
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


kernel11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tmp0 * tmp0
        _tmp2 = tl.where(xmask & rmask, _tmp2 + tmp1, _tmp2)
    tmp2 = tl.reshape(tl.sum(_tmp2, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp2, xmask)
''')


kernel12 = async_compile.triton('''
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
    tmp3 = 512
    tmp4 = tmp2 / tmp3
    tmp5 = 1e-06
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt(tmp6)
    tmp8 = 1 / tmp7
    tmp9 = tmp1 * tmp8
    tmp10 = tmp0 * tmp9
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


kernel13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 262144 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 524288 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 196608 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel16 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 655360 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 786432 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 1048576 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 393216 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 1179648 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel21 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1310720 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 1572864 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 589824 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel24 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 1703936 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1835008 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 2097152 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel27 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 786432 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel28 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 2228224 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2359296 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 2621440 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 983040 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 2752512 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2883584 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 3145728 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1179648 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 3276800 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3407872 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 3670016 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1376256 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 3801088 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel41 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3932160 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel42 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 4194304 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x2), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4325376 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp11 = 512
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sqrt(tmp14)
    tmp16 = 1 / tmp15
    tmp17 = tmp9 * tmp16
    tmp18 = tmp8 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = 1.1111111111111112
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel44 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 512],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    x0 = xindex
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr1 + (r1 + (512*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 65535
        tmp2 = tmp0 ^ tmp1
        tmp3 = 4456448 + r1 + (512*x0)
        tmp4 = tl.rand(tmp2, tmp3)
        tmp5 = 0.1
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11 * tmp11
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.reshape(tl.sum(_tmp13, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp13, xmask)
''')


kernel45 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x2), xmask)
    tmp13 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 4456448 + x2
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp14 = 512
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = tl.sqrt(tmp17)
    tmp19 = 1 / tmp18
    tmp20 = tmp12 * tmp19
    tmp21 = tmp0 * tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel46 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = r3
        tmp2 = x0
        tmp3 = tmp1 - tmp2
        tmp4 = 0
        tmp5 = tl.minimum(tmp3, tmp4)
        tmp6 = -tmp5
        tmp7 = 16
        tmp8 = tmp6 < tmp7
        tmp9 = tmp6.to(tl.float32)
        tmp10 = tmp9 / tmp7
        tmp11 = tl.log(tmp10)
        tmp12 = 2.0794415416798357
        tmp13 = tmp11 / tmp12
        tmp14 = tmp13 * tmp7
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15 + tmp7
        tmp17 = 31
        tmp18 = tl.minimum(tmp16, tmp17)
        tmp19 = tl.where(tmp8, tmp6, tmp18)
        tmp20 = tmp19 + tmp4
        tmp21 = tl.load(in_ptr1 + (x1 + (6*tmp20) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp22 = 1.0
        tmp23 = tmp1 <= tmp2
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 1
        tmp26 = tmp24 * tmp25
        tmp27 = tmp22 - tmp26
        tmp28 = -3.4028234663852886e+38
        tmp29 = tmp27 * tmp28
        tmp30 = tmp21 + tmp29
        tmp31 = tmp0 + tmp30
        _tmp32 = tl.where(xmask & rmask & (_tmp32 < tmp31), tmp31, _tmp32)
    tmp32 = tl.reshape(tl.max(_tmp32, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x4, tmp32, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 6
    tmp32 = tl.load(in_ptr2 + (x4), xmask)
    _tmp35 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = r3
        tmp2 = x0
        tmp3 = tmp1 - tmp2
        tmp4 = 0
        tmp5 = tl.minimum(tmp3, tmp4)
        tmp6 = -tmp5
        tmp7 = 16
        tmp8 = tmp6 < tmp7
        tmp9 = tmp6.to(tl.float32)
        tmp10 = tmp9 / tmp7
        tmp11 = tl.log(tmp10)
        tmp12 = 2.0794415416798357
        tmp13 = tmp11 / tmp12
        tmp14 = tmp13 * tmp7
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15 + tmp7
        tmp17 = 31
        tmp18 = tl.minimum(tmp16, tmp17)
        tmp19 = tl.where(tmp8, tmp6, tmp18)
        tmp20 = tmp19 + tmp4
        tmp21 = tl.load(in_ptr1 + (x1 + (6*tmp20) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp22 = 1.0
        tmp23 = tmp1 <= tmp2
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 1
        tmp26 = tmp24 * tmp25
        tmp27 = tmp22 - tmp26
        tmp28 = -3.4028234663852886e+38
        tmp29 = tmp27 * tmp28
        tmp30 = tmp21 + tmp29
        tmp31 = tmp0 + tmp30
        tmp33 = tmp31 - tmp32
        tmp34 = tl.exp(tmp33)
        _tmp35 = tl.where(xmask & rmask, _tmp35 + tmp34, _tmp35)
    tmp35 = tl.reshape(tl.sum(_tmp35, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x4, tmp35, xmask)
''')


kernel48 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x2 = (xindex // 16384) % 6
    x6 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp32 = tl.load(in_ptr2 + (x6), xmask)
    tmp35 = tl.load(in_ptr3 + (x6), xmask)
    tmp1 = x0
    tmp2 = x1
    tmp3 = tmp1 - tmp2
    tmp4 = 0
    tmp5 = tl.minimum(tmp3, tmp4)
    tmp6 = -tmp5
    tmp7 = 16
    tmp8 = tmp6 < tmp7
    tmp9 = tmp6.to(tl.float32)
    tmp10 = tmp9 / tmp7
    tmp11 = tl.log(tmp10)
    tmp12 = 2.0794415416798357
    tmp13 = tmp11 / tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15 + tmp7
    tmp17 = 31
    tmp18 = tl.minimum(tmp16, tmp17)
    tmp19 = tl.where(tmp8, tmp6, tmp18)
    tmp20 = tmp19 + tmp4
    tmp21 = tl.load(in_ptr1 + (x2 + (6*tmp20) + tl.zeros([XBLOCK], tl.int32)), xmask)
    tmp22 = 1.0
    tmp23 = tmp1 <= tmp2
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 1
    tmp26 = tmp24 * tmp25
    tmp27 = tmp22 - tmp26
    tmp28 = -3.4028234663852886e+38
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 + tmp29
    tmp31 = tmp0 + tmp30
    tmp33 = tmp31 - tmp32
    tmp34 = tl.exp(tmp33)
    tmp36 = tmp34 / tmp35
    tl.store(out_ptr0 + (x4 + tl.zeros([XBLOCK], tl.int32)), tmp36, xmask)
''')


kernel49 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1572864 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel50 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, seed0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp16 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4456448 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = 4587520 + x0
    tmp13 = tl.rand(tmp2, tmp12)
    tmp14 = tmp13 > tmp5
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = tmp11 + tmp18
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp10 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 0
        tmp2 = 1.0
        tmp3 = 1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 - tmp4
        tmp6 = -3.4028234663852886e+38
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 + tmp7
        tmp9 = tmp0 + tmp8
        _tmp10 = tl.where(xmask & rmask & (_tmp10 < tmp9), tmp9, _tmp10)
    tmp10 = tl.reshape(tl.max(_tmp10, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp10, xmask)
''')


kernel52 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[2048, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    tmp10 = tl.load(in_ptr1 + (x0), xmask)
    _tmp13 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = 0
        tmp2 = 1.0
        tmp3 = 1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 - tmp4
        tmp6 = -3.4028234663852886e+38
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 + tmp7
        tmp9 = tmp0 + tmp8
        tmp11 = tmp9 - tmp10
        tmp12 = tl.exp(tmp11)
        _tmp13 = tl.where(xmask & rmask, _tmp13 + tmp12, _tmp13)
    tmp13 = tl.reshape(tl.sum(_tmp13, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp13, xmask)
''')


kernel53 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp10 = tl.load(in_ptr1 + (x1), xmask)
    tmp13 = tl.load(in_ptr2 + (x1), xmask)
    tmp1 = 0
    tmp2 = 1.0
    tmp3 = 1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 - tmp4
    tmp6 = -3.4028234663852886e+38
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tmp0 + tmp8
    tmp11 = tmp9 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp14 = tmp12 / tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel54 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1769472 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel55 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 4718592 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel56 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4849664 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel57 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 5111808 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel58 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1966080 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel59 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 5242880 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel60 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2162688 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel61 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 5373952 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel62 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5505024 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel63 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 5767168 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel64 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2359296 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel65 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 5898240 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel66 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2555904 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel67 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 6029312 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel68 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6160384 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel69 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 6422528 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel70 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2752512 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel71 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 6553600 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel72 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2949120 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel73 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 6684672 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel74 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6815744 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel75 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7077888 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel76 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3145728 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel77 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7208960 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel78 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3342336 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel79 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7340032 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel80 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7471104 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel81 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7733248 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel82 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3538944 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel83 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7864320 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel84 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3735552 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel85 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 7995392 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel86 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8126464 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel87 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 8388608 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel88 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3932160 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel89 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 8519680 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel90 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 4128768 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel91 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 8650752 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel92 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8781824 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel93 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 9043968 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel94 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 4325376 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel95 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 9175040 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel96 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 4521984 + x0
    tmp2 = tl.rand(tmp0, tmp1)
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
''')


kernel97 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 9306112 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel98 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp29 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9437184 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 * tmp8
    tmp12 = tmp11 * tmp8
    tmp13 = 0.044715
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = 0.7978845608028654
    tmp17 = tmp15 * tmp16
    tmp18 = -2.0
    tmp19 = tmp17 * tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = 1 / tmp22
    tmp24 = 2.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25 - tmp21
    tmp27 = tmp26 + tmp21
    tmp28 = tmp10 * tmp27
    tmp30 = tmp28 * tmp29
    tmp31 = tmp7 * tmp30
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp33, xmask)
''')


kernel99 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, seed1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    in_ptr2 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(seed1 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp2 = 65535
    tmp3 = tmp1 ^ tmp2
    tmp4 = 9699328 + x0
    tmp5 = tl.rand(tmp3, tmp4)
    tmp6 = 0.1
    tmp7 = tmp5 > tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 + tmp12
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp13, xmask)
''')


kernel100 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp8 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x2), xmask)
    tmp10 = tl.load(in_ptr3 + (x1), xmask)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9830400 + x2
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp11 = 512
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tl.sqrt(tmp14)
    tmp16 = 1 / tmp15
    tmp17 = tmp9 * tmp16
    tmp18 = tmp8 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = 1.1111111111111112
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp21, xmask)
''')


kernel101 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[256, 262144],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 250112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (250112*x0)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    tmp1 = tl.reshape(tl.max(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


kernel102 = async_compile.triton('''
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
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (250112*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 - tmp1
        tmp3 = tl.exp(tmp2)
        _tmp4 = tl.where(xmask & rmask, _tmp4 + tmp3, _tmp4)
    tmp4 = tl.reshape(tl.sum(_tmp4, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp4, xmask)
''')


kernel103 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64028672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 250112)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.log(tmp3)
    tmp5 = tmp2 - tmp4
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel104 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 256],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 256
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
        tmp1 = tl.load(in_ptr1 + (tmp0 + (250112*r0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = -tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp3, None)
''')


kernel105 = async_compile.triton('''
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
    tmp1 = 256
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (0 + tl.zeros([XBLOCK], tl.int32)), tmp2, None)
''')


kernel106 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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


kernel107 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 512
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = tl.sqrt(tmp4)
    tmp6 = 1 / tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel108 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 131072 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 * tmp0
    tmp3 = 0.044715
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 0.7978845608028654
    tmp7 = tmp5 * tmp6
    tmp8 = -2.0
    tmp9 = tmp7 * tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = 1 / tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 - tmp11
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp16, xmask)
''')


kernel110 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 262144 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 524288 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 655360 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 786432 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1048576 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1179648 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1310720 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1572864 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1703936 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 1835008 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2097152 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2228224 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2359296 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2621440 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2752512 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 2883584 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3145728 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3276800 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3407872 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3670016 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3801088 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 3932160 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4194304 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4325376 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4456448 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4587520 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4718592 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 4849664 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5111808 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5242880 + x0
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5373952 + x0
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

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5505024 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel142 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5767168 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel143 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 5898240 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel144 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6029312 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel145 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6160384 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel146 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6422528 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel147 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6553600 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel148 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6684672 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel149 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 6815744 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel150 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7077888 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel151 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7208960 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel152 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7340032 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel153 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7471104 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel154 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7733248 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel155 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7864320 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel156 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 7995392 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel157 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8126464 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel158 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8388608 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel159 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8519680 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel160 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8650752 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel161 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 8781824 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel162 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9043968 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel163 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9175040 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel164 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9306112 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel165 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9437184 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel166 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9699328 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel167 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(seed0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(seed0 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp1 = 65535
    tmp2 = tmp0 ^ tmp1
    tmp3 = 9830400 + x0
    tmp4 = tl.rand(tmp2, tmp3)
    tmp5 = 0.1
    tmp6 = tmp4 > tmp5
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
''')


kernel168 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 - tmp1
    tmp3 = 0
    tmp4 = tl.minimum(tmp2, tmp3)
    tmp5 = -tmp4
    tmp6 = 16
    tmp7 = tmp5 < tmp6
    tmp8 = tmp5.to(tl.float32)
    tmp9 = tmp8 / tmp6
    tmp10 = tl.log(tmp9)
    tmp11 = 2.0794415416798357
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tmp14 + tmp6
    tmp16 = 31
    tmp17 = tl.minimum(tmp15, tmp16)
    tmp18 = tl.where(tmp7, tmp5, tmp17)
    tmp19 = tmp18 + tmp3
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp19, xmask)
''')


kernel169 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 - tmp1
    tmp3 = 0
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.int64)
    tmp6 = 16
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 + tmp3
    tmp9 = tl.abs(tmp2)
    tmp10 = 8
    tmp11 = tmp9 < tmp10
    tmp12 = tmp9.to(tl.float32)
    tmp13 = tmp12 / tmp10
    tmp14 = tl.log(tmp13)
    tmp15 = 2.772588722239781
    tmp16 = tmp14 / tmp15
    tmp17 = tmp16 * tmp10
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tmp18 + tmp10
    tmp20 = 15
    tmp21 = tl.minimum(tmp19, tmp20)
    tmp22 = tl.where(tmp11, tmp9, tmp21)
    tmp23 = tmp8 + tmp22
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp23, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193 = args
    args.clear()
    torch.randint(2**31, size=(), dtype=torch.int64, out=seed_cuda_0)
    buf0 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_191, primals_43, buf0, 131072, grid=grid(131072), stream=stream0)
    buf1 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel1.run(seed_cuda_0, buf0, buf1, 256, 512, grid=grid(256), stream=stream0)
    buf2 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel2.run(primals_1, seed_cuda_0, buf0, buf1, buf2, 131072, grid=grid(131072), stream=stream0)
    buf3 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf2, (256, 512), (512, 1)), as_strided(primals_44, (512, 384), (1, 512)), out=buf3)
    buf4 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf2, (256, 512), (512, 1)), as_strided(primals_45, (512, 384), (1, 512)), out=buf4)
    buf5 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf2, (256, 512), (512, 1)), as_strided(primals_46, (512, 384), (1, 512)), out=buf5)
    buf6 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf3, buf6, 98304, grid=grid(98304), stream=stream0)
    buf7 = as_strided(buf3, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf3  # reuse
    kernel4.run(buf4, buf7, 768, 128, grid=grid(768, 128), stream=stream0)
    buf8 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf6, (12, 128, 64), (8192, 64, 1)), as_strided(buf7, (12, 64, 128), (8192, 128, 1)), out=buf8)
    buf9 = empty_strided((2, 6, 128, 1), (768, 128, 1, 1536), device='cuda', dtype=torch.float32)
    kernel5.run(buf8, primals_47, buf9, 1536, 128, grid=grid(1536), stream=stream0)
    buf10 = empty_strided((2, 6, 128, 1), (768, 128, 1, 1536), device='cuda', dtype=torch.float32)
    kernel6.run(buf8, primals_47, buf9, buf10, 1536, 128, grid=grid(1536), stream=stream0)
    buf11 = as_strided(buf8, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf8  # reuse
    kernel7.run(buf11, primals_47, buf9, buf10, 196608, grid=grid(196608), stream=stream0)
    buf12 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel8.run(seed_cuda_0, buf11, buf12, 196608, grid=grid(196608), stream=stream0)
    buf13 = as_strided(buf4, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf4  # reuse
    kernel3.run(buf5, buf13, 98304, grid=grid(98304), stream=stream0)
    buf14 = as_strided(buf5, (12, 128, 64), (8192, 64, 1)); del buf5  # reuse
    aten.bmm.out(as_strided(buf12, (12, 128, 128), (16384, 128, 1)), as_strided(buf13, (12, 128, 64), (8192, 64, 1)), out=buf14)
    buf15 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf14, buf15, 98304, grid=grid(98304), stream=stream0)
    buf16 = as_strided(buf2, (256, 512), (512, 1)); del buf2  # reuse
    aten.mm.out(as_strided(buf15, (256, 384), (384, 1)), as_strided(primals_48, (384, 512), (1, 384)), out=buf16)
    buf17 = as_strided(buf16, (2, 128, 512), (65536, 512, 1)); del buf16  # reuse
    kernel10.run(buf17, seed_cuda_0, buf0, 131072, grid=grid(131072), stream=stream0)
    buf18 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf17, buf18, 256, 512, grid=grid(256), stream=stream0)
    buf19 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_2, buf17, buf18, buf19, 131072, grid=grid(131072), stream=stream0)
    buf20 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf19, (256, 512), (512, 1)), as_strided(primals_49, (512, 1024), (1, 512)), out=buf20)
    buf21 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf19, (256, 512), (512, 1)), as_strided(primals_50, (512, 1024), (1, 512)), out=buf21)
    buf22 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel13.run(seed_cuda_0, buf20, buf21, buf22, 262144, grid=grid(262144), stream=stream0)
    buf23 = as_strided(buf19, (256, 512), (512, 1)); del buf19  # reuse
    aten.mm.out(as_strided(buf22, (256, 1024), (1024, 1)), as_strided(primals_51, (1024, 512), (1, 1024)), out=buf23)
    buf24 = as_strided(buf23, (2, 128, 512), (65536, 512, 1)); del buf23  # reuse
    kernel14.run(buf24, buf17, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf25 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf24, buf25, 256, 512, grid=grid(256), stream=stream0)
    buf26 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_3, buf24, buf25, buf26, 131072, grid=grid(131072), stream=stream0)
    buf27 = as_strided(buf14, (256, 384), (384, 1)); del buf14  # reuse
    aten.mm.out(as_strided(buf26, (256, 512), (512, 1)), as_strided(primals_52, (512, 384), (1, 512)), out=buf27)
    buf28 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf26, (256, 512), (512, 1)), as_strided(primals_53, (512, 384), (1, 512)), out=buf28)
    buf29 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf26, (256, 512), (512, 1)), as_strided(primals_54, (512, 384), (1, 512)), out=buf29)
    buf30 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf27, buf30, 98304, grid=grid(98304), stream=stream0)
    buf31 = as_strided(buf27, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf27  # reuse
    kernel4.run(buf28, buf31, 768, 128, grid=grid(768, 128), stream=stream0)
    buf32 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf30, (12, 128, 64), (8192, 64, 1)), as_strided(buf31, (12, 64, 128), (8192, 128, 1)), out=buf32)
    buf33 = buf9; del buf9  # reuse
    kernel5.run(buf32, primals_47, buf33, 1536, 128, grid=grid(1536), stream=stream0)
    buf34 = buf10; del buf10  # reuse
    kernel6.run(buf32, primals_47, buf33, buf34, 1536, 128, grid=grid(1536), stream=stream0)
    buf35 = as_strided(buf32, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf32  # reuse
    kernel7.run(buf35, primals_47, buf33, buf34, 196608, grid=grid(196608), stream=stream0)
    buf36 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel15.run(seed_cuda_0, buf35, buf36, 196608, grid=grid(196608), stream=stream0)
    buf37 = as_strided(buf28, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf28  # reuse
    kernel3.run(buf29, buf37, 98304, grid=grid(98304), stream=stream0)
    buf38 = as_strided(buf29, (12, 128, 64), (8192, 64, 1)); del buf29  # reuse
    aten.bmm.out(as_strided(buf36, (12, 128, 128), (16384, 128, 1)), as_strided(buf37, (12, 128, 64), (8192, 64, 1)), out=buf38)
    buf39 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf38, buf39, 98304, grid=grid(98304), stream=stream0)
    buf40 = as_strided(buf26, (256, 512), (512, 1)); del buf26  # reuse
    aten.mm.out(as_strided(buf39, (256, 384), (384, 1)), as_strided(primals_55, (384, 512), (1, 384)), out=buf40)
    buf41 = as_strided(buf40, (2, 128, 512), (65536, 512, 1)); del buf40  # reuse
    kernel16.run(buf41, buf24, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf42 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf41, buf42, 256, 512, grid=grid(256), stream=stream0)
    buf43 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_4, buf41, buf42, buf43, 131072, grid=grid(131072), stream=stream0)
    buf44 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf43, (256, 512), (512, 1)), as_strided(primals_56, (512, 1024), (1, 512)), out=buf44)
    buf45 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf43, (256, 512), (512, 1)), as_strided(primals_57, (512, 1024), (1, 512)), out=buf45)
    buf46 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel17.run(seed_cuda_0, buf44, buf45, buf46, 262144, grid=grid(262144), stream=stream0)
    buf47 = as_strided(buf43, (256, 512), (512, 1)); del buf43  # reuse
    aten.mm.out(as_strided(buf46, (256, 1024), (1024, 1)), as_strided(primals_58, (1024, 512), (1, 1024)), out=buf47)
    buf48 = as_strided(buf47, (2, 128, 512), (65536, 512, 1)); del buf47  # reuse
    kernel18.run(buf48, buf41, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf49 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf48, buf49, 256, 512, grid=grid(256), stream=stream0)
    buf50 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_5, buf48, buf49, buf50, 131072, grid=grid(131072), stream=stream0)
    buf51 = as_strided(buf38, (256, 384), (384, 1)); del buf38  # reuse
    aten.mm.out(as_strided(buf50, (256, 512), (512, 1)), as_strided(primals_59, (512, 384), (1, 512)), out=buf51)
    buf52 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf50, (256, 512), (512, 1)), as_strided(primals_60, (512, 384), (1, 512)), out=buf52)
    buf53 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf50, (256, 512), (512, 1)), as_strided(primals_61, (512, 384), (1, 512)), out=buf53)
    buf54 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf51, buf54, 98304, grid=grid(98304), stream=stream0)
    buf55 = as_strided(buf51, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf51  # reuse
    kernel4.run(buf52, buf55, 768, 128, grid=grid(768, 128), stream=stream0)
    buf56 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf54, (12, 128, 64), (8192, 64, 1)), as_strided(buf55, (12, 64, 128), (8192, 128, 1)), out=buf56)
    buf57 = buf34; del buf34  # reuse
    kernel5.run(buf56, primals_47, buf57, 1536, 128, grid=grid(1536), stream=stream0)
    buf58 = buf33; del buf33  # reuse
    kernel6.run(buf56, primals_47, buf57, buf58, 1536, 128, grid=grid(1536), stream=stream0)
    buf59 = as_strided(buf56, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf56  # reuse
    kernel7.run(buf59, primals_47, buf57, buf58, 196608, grid=grid(196608), stream=stream0)
    buf60 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel19.run(seed_cuda_0, buf59, buf60, 196608, grid=grid(196608), stream=stream0)
    buf61 = as_strided(buf52, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf52  # reuse
    kernel3.run(buf53, buf61, 98304, grid=grid(98304), stream=stream0)
    buf62 = as_strided(buf53, (12, 128, 64), (8192, 64, 1)); del buf53  # reuse
    aten.bmm.out(as_strided(buf60, (12, 128, 128), (16384, 128, 1)), as_strided(buf61, (12, 128, 64), (8192, 64, 1)), out=buf62)
    buf63 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf62, buf63, 98304, grid=grid(98304), stream=stream0)
    buf64 = as_strided(buf50, (256, 512), (512, 1)); del buf50  # reuse
    aten.mm.out(as_strided(buf63, (256, 384), (384, 1)), as_strided(primals_62, (384, 512), (1, 384)), out=buf64)
    buf65 = as_strided(buf64, (2, 128, 512), (65536, 512, 1)); del buf64  # reuse
    kernel20.run(buf65, buf48, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf66 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf65, buf66, 256, 512, grid=grid(256), stream=stream0)
    buf67 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_6, buf65, buf66, buf67, 131072, grid=grid(131072), stream=stream0)
    buf68 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf67, (256, 512), (512, 1)), as_strided(primals_63, (512, 1024), (1, 512)), out=buf68)
    buf69 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf67, (256, 512), (512, 1)), as_strided(primals_64, (512, 1024), (1, 512)), out=buf69)
    buf70 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel21.run(seed_cuda_0, buf68, buf69, buf70, 262144, grid=grid(262144), stream=stream0)
    buf71 = as_strided(buf67, (256, 512), (512, 1)); del buf67  # reuse
    aten.mm.out(as_strided(buf70, (256, 1024), (1024, 1)), as_strided(primals_65, (1024, 512), (1, 1024)), out=buf71)
    buf72 = as_strided(buf71, (2, 128, 512), (65536, 512, 1)); del buf71  # reuse
    kernel22.run(buf72, buf65, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf73 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf72, buf73, 256, 512, grid=grid(256), stream=stream0)
    buf74 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_7, buf72, buf73, buf74, 131072, grid=grid(131072), stream=stream0)
    buf75 = as_strided(buf62, (256, 384), (384, 1)); del buf62  # reuse
    aten.mm.out(as_strided(buf74, (256, 512), (512, 1)), as_strided(primals_66, (512, 384), (1, 512)), out=buf75)
    buf76 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf74, (256, 512), (512, 1)), as_strided(primals_67, (512, 384), (1, 512)), out=buf76)
    buf77 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf74, (256, 512), (512, 1)), as_strided(primals_68, (512, 384), (1, 512)), out=buf77)
    buf78 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf75, buf78, 98304, grid=grid(98304), stream=stream0)
    buf79 = as_strided(buf75, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf75  # reuse
    kernel4.run(buf76, buf79, 768, 128, grid=grid(768, 128), stream=stream0)
    buf80 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf78, (12, 128, 64), (8192, 64, 1)), as_strided(buf79, (12, 64, 128), (8192, 128, 1)), out=buf80)
    buf81 = buf58; del buf58  # reuse
    kernel5.run(buf80, primals_47, buf81, 1536, 128, grid=grid(1536), stream=stream0)
    buf82 = buf57; del buf57  # reuse
    kernel6.run(buf80, primals_47, buf81, buf82, 1536, 128, grid=grid(1536), stream=stream0)
    buf83 = as_strided(buf80, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf80  # reuse
    kernel7.run(buf83, primals_47, buf81, buf82, 196608, grid=grid(196608), stream=stream0)
    buf84 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel23.run(seed_cuda_0, buf83, buf84, 196608, grid=grid(196608), stream=stream0)
    buf85 = as_strided(buf76, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf76  # reuse
    kernel3.run(buf77, buf85, 98304, grid=grid(98304), stream=stream0)
    buf86 = as_strided(buf77, (12, 128, 64), (8192, 64, 1)); del buf77  # reuse
    aten.bmm.out(as_strided(buf84, (12, 128, 128), (16384, 128, 1)), as_strided(buf85, (12, 128, 64), (8192, 64, 1)), out=buf86)
    buf87 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf86, buf87, 98304, grid=grid(98304), stream=stream0)
    buf88 = as_strided(buf74, (256, 512), (512, 1)); del buf74  # reuse
    aten.mm.out(as_strided(buf87, (256, 384), (384, 1)), as_strided(primals_69, (384, 512), (1, 384)), out=buf88)
    buf89 = as_strided(buf88, (2, 128, 512), (65536, 512, 1)); del buf88  # reuse
    kernel24.run(buf89, buf72, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf90 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf89, buf90, 256, 512, grid=grid(256), stream=stream0)
    buf91 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_8, buf89, buf90, buf91, 131072, grid=grid(131072), stream=stream0)
    buf92 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf91, (256, 512), (512, 1)), as_strided(primals_70, (512, 1024), (1, 512)), out=buf92)
    buf93 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf91, (256, 512), (512, 1)), as_strided(primals_71, (512, 1024), (1, 512)), out=buf93)
    buf94 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel25.run(seed_cuda_0, buf92, buf93, buf94, 262144, grid=grid(262144), stream=stream0)
    buf95 = as_strided(buf91, (256, 512), (512, 1)); del buf91  # reuse
    aten.mm.out(as_strided(buf94, (256, 1024), (1024, 1)), as_strided(primals_72, (1024, 512), (1, 1024)), out=buf95)
    buf96 = as_strided(buf95, (2, 128, 512), (65536, 512, 1)); del buf95  # reuse
    kernel26.run(buf96, buf89, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf97 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf96, buf97, 256, 512, grid=grid(256), stream=stream0)
    buf98 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_9, buf96, buf97, buf98, 131072, grid=grid(131072), stream=stream0)
    buf99 = as_strided(buf86, (256, 384), (384, 1)); del buf86  # reuse
    aten.mm.out(as_strided(buf98, (256, 512), (512, 1)), as_strided(primals_73, (512, 384), (1, 512)), out=buf99)
    buf100 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf98, (256, 512), (512, 1)), as_strided(primals_74, (512, 384), (1, 512)), out=buf100)
    buf101 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf98, (256, 512), (512, 1)), as_strided(primals_75, (512, 384), (1, 512)), out=buf101)
    buf102 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf99, buf102, 98304, grid=grid(98304), stream=stream0)
    buf103 = as_strided(buf99, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf99  # reuse
    kernel4.run(buf100, buf103, 768, 128, grid=grid(768, 128), stream=stream0)
    buf104 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf102, (12, 128, 64), (8192, 64, 1)), as_strided(buf103, (12, 64, 128), (8192, 128, 1)), out=buf104)
    buf105 = buf82; del buf82  # reuse
    kernel5.run(buf104, primals_47, buf105, 1536, 128, grid=grid(1536), stream=stream0)
    buf106 = buf81; del buf81  # reuse
    kernel6.run(buf104, primals_47, buf105, buf106, 1536, 128, grid=grid(1536), stream=stream0)
    buf107 = as_strided(buf104, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf104  # reuse
    kernel7.run(buf107, primals_47, buf105, buf106, 196608, grid=grid(196608), stream=stream0)
    buf108 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel27.run(seed_cuda_0, buf107, buf108, 196608, grid=grid(196608), stream=stream0)
    buf109 = as_strided(buf100, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf100  # reuse
    kernel3.run(buf101, buf109, 98304, grid=grid(98304), stream=stream0)
    buf110 = as_strided(buf101, (12, 128, 64), (8192, 64, 1)); del buf101  # reuse
    aten.bmm.out(as_strided(buf108, (12, 128, 128), (16384, 128, 1)), as_strided(buf109, (12, 128, 64), (8192, 64, 1)), out=buf110)
    buf111 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf110, buf111, 98304, grid=grid(98304), stream=stream0)
    buf112 = as_strided(buf98, (256, 512), (512, 1)); del buf98  # reuse
    aten.mm.out(as_strided(buf111, (256, 384), (384, 1)), as_strided(primals_76, (384, 512), (1, 384)), out=buf112)
    buf113 = as_strided(buf112, (2, 128, 512), (65536, 512, 1)); del buf112  # reuse
    kernel28.run(buf113, buf96, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf114 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf113, buf114, 256, 512, grid=grid(256), stream=stream0)
    buf115 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_10, buf113, buf114, buf115, 131072, grid=grid(131072), stream=stream0)
    buf116 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf115, (256, 512), (512, 1)), as_strided(primals_77, (512, 1024), (1, 512)), out=buf116)
    buf117 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf115, (256, 512), (512, 1)), as_strided(primals_78, (512, 1024), (1, 512)), out=buf117)
    buf118 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel29.run(seed_cuda_0, buf116, buf117, buf118, 262144, grid=grid(262144), stream=stream0)
    buf119 = as_strided(buf115, (256, 512), (512, 1)); del buf115  # reuse
    aten.mm.out(as_strided(buf118, (256, 1024), (1024, 1)), as_strided(primals_79, (1024, 512), (1, 1024)), out=buf119)
    buf120 = as_strided(buf119, (2, 128, 512), (65536, 512, 1)); del buf119  # reuse
    kernel30.run(buf120, buf113, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf121 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf120, buf121, 256, 512, grid=grid(256), stream=stream0)
    buf122 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_11, buf120, buf121, buf122, 131072, grid=grid(131072), stream=stream0)
    buf123 = as_strided(buf110, (256, 384), (384, 1)); del buf110  # reuse
    aten.mm.out(as_strided(buf122, (256, 512), (512, 1)), as_strided(primals_80, (512, 384), (1, 512)), out=buf123)
    buf124 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf122, (256, 512), (512, 1)), as_strided(primals_81, (512, 384), (1, 512)), out=buf124)
    buf125 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf122, (256, 512), (512, 1)), as_strided(primals_82, (512, 384), (1, 512)), out=buf125)
    buf126 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf123, buf126, 98304, grid=grid(98304), stream=stream0)
    buf127 = as_strided(buf123, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf123  # reuse
    kernel4.run(buf124, buf127, 768, 128, grid=grid(768, 128), stream=stream0)
    buf128 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf126, (12, 128, 64), (8192, 64, 1)), as_strided(buf127, (12, 64, 128), (8192, 128, 1)), out=buf128)
    buf129 = buf106; del buf106  # reuse
    kernel5.run(buf128, primals_47, buf129, 1536, 128, grid=grid(1536), stream=stream0)
    buf130 = buf105; del buf105  # reuse
    kernel6.run(buf128, primals_47, buf129, buf130, 1536, 128, grid=grid(1536), stream=stream0)
    buf131 = as_strided(buf128, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf128  # reuse
    kernel7.run(buf131, primals_47, buf129, buf130, 196608, grid=grid(196608), stream=stream0)
    buf132 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel31.run(seed_cuda_0, buf131, buf132, 196608, grid=grid(196608), stream=stream0)
    buf133 = as_strided(buf124, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf124  # reuse
    kernel3.run(buf125, buf133, 98304, grid=grid(98304), stream=stream0)
    buf134 = as_strided(buf125, (12, 128, 64), (8192, 64, 1)); del buf125  # reuse
    aten.bmm.out(as_strided(buf132, (12, 128, 128), (16384, 128, 1)), as_strided(buf133, (12, 128, 64), (8192, 64, 1)), out=buf134)
    buf135 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf134, buf135, 98304, grid=grid(98304), stream=stream0)
    buf136 = as_strided(buf122, (256, 512), (512, 1)); del buf122  # reuse
    aten.mm.out(as_strided(buf135, (256, 384), (384, 1)), as_strided(primals_83, (384, 512), (1, 384)), out=buf136)
    buf137 = as_strided(buf136, (2, 128, 512), (65536, 512, 1)); del buf136  # reuse
    kernel32.run(buf137, buf120, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf138 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf137, buf138, 256, 512, grid=grid(256), stream=stream0)
    buf139 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_12, buf137, buf138, buf139, 131072, grid=grid(131072), stream=stream0)
    buf140 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf139, (256, 512), (512, 1)), as_strided(primals_84, (512, 1024), (1, 512)), out=buf140)
    buf141 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf139, (256, 512), (512, 1)), as_strided(primals_85, (512, 1024), (1, 512)), out=buf141)
    buf142 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel33.run(seed_cuda_0, buf140, buf141, buf142, 262144, grid=grid(262144), stream=stream0)
    buf143 = as_strided(buf139, (256, 512), (512, 1)); del buf139  # reuse
    aten.mm.out(as_strided(buf142, (256, 1024), (1024, 1)), as_strided(primals_86, (1024, 512), (1, 1024)), out=buf143)
    buf144 = as_strided(buf143, (2, 128, 512), (65536, 512, 1)); del buf143  # reuse
    kernel34.run(buf144, buf137, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf145 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf144, buf145, 256, 512, grid=grid(256), stream=stream0)
    buf146 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_13, buf144, buf145, buf146, 131072, grid=grid(131072), stream=stream0)
    buf147 = as_strided(buf134, (256, 384), (384, 1)); del buf134  # reuse
    aten.mm.out(as_strided(buf146, (256, 512), (512, 1)), as_strided(primals_87, (512, 384), (1, 512)), out=buf147)
    buf148 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf146, (256, 512), (512, 1)), as_strided(primals_88, (512, 384), (1, 512)), out=buf148)
    buf149 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf146, (256, 512), (512, 1)), as_strided(primals_89, (512, 384), (1, 512)), out=buf149)
    buf150 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf147, buf150, 98304, grid=grid(98304), stream=stream0)
    buf151 = as_strided(buf147, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf147  # reuse
    kernel4.run(buf148, buf151, 768, 128, grid=grid(768, 128), stream=stream0)
    buf152 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf150, (12, 128, 64), (8192, 64, 1)), as_strided(buf151, (12, 64, 128), (8192, 128, 1)), out=buf152)
    buf153 = buf130; del buf130  # reuse
    kernel5.run(buf152, primals_47, buf153, 1536, 128, grid=grid(1536), stream=stream0)
    buf154 = buf129; del buf129  # reuse
    kernel6.run(buf152, primals_47, buf153, buf154, 1536, 128, grid=grid(1536), stream=stream0)
    buf155 = as_strided(buf152, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf152  # reuse
    kernel7.run(buf155, primals_47, buf153, buf154, 196608, grid=grid(196608), stream=stream0)
    buf156 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel35.run(seed_cuda_0, buf155, buf156, 196608, grid=grid(196608), stream=stream0)
    buf157 = as_strided(buf148, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf148  # reuse
    kernel3.run(buf149, buf157, 98304, grid=grid(98304), stream=stream0)
    buf158 = as_strided(buf149, (12, 128, 64), (8192, 64, 1)); del buf149  # reuse
    aten.bmm.out(as_strided(buf156, (12, 128, 128), (16384, 128, 1)), as_strided(buf157, (12, 128, 64), (8192, 64, 1)), out=buf158)
    buf159 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf158, buf159, 98304, grid=grid(98304), stream=stream0)
    buf160 = as_strided(buf146, (256, 512), (512, 1)); del buf146  # reuse
    aten.mm.out(as_strided(buf159, (256, 384), (384, 1)), as_strided(primals_90, (384, 512), (1, 384)), out=buf160)
    buf161 = as_strided(buf160, (2, 128, 512), (65536, 512, 1)); del buf160  # reuse
    kernel36.run(buf161, buf144, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf162 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf161, buf162, 256, 512, grid=grid(256), stream=stream0)
    buf163 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_14, buf161, buf162, buf163, 131072, grid=grid(131072), stream=stream0)
    buf164 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf163, (256, 512), (512, 1)), as_strided(primals_91, (512, 1024), (1, 512)), out=buf164)
    buf165 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf163, (256, 512), (512, 1)), as_strided(primals_92, (512, 1024), (1, 512)), out=buf165)
    buf166 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel37.run(seed_cuda_0, buf164, buf165, buf166, 262144, grid=grid(262144), stream=stream0)
    buf167 = as_strided(buf163, (256, 512), (512, 1)); del buf163  # reuse
    aten.mm.out(as_strided(buf166, (256, 1024), (1024, 1)), as_strided(primals_93, (1024, 512), (1, 1024)), out=buf167)
    buf168 = as_strided(buf167, (2, 128, 512), (65536, 512, 1)); del buf167  # reuse
    kernel38.run(buf168, buf161, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf169 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf168, buf169, 256, 512, grid=grid(256), stream=stream0)
    buf170 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_15, buf168, buf169, buf170, 131072, grid=grid(131072), stream=stream0)
    buf171 = as_strided(buf158, (256, 384), (384, 1)); del buf158  # reuse
    aten.mm.out(as_strided(buf170, (256, 512), (512, 1)), as_strided(primals_94, (512, 384), (1, 512)), out=buf171)
    buf172 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf170, (256, 512), (512, 1)), as_strided(primals_95, (512, 384), (1, 512)), out=buf172)
    buf173 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf170, (256, 512), (512, 1)), as_strided(primals_96, (512, 384), (1, 512)), out=buf173)
    buf174 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf171, buf174, 98304, grid=grid(98304), stream=stream0)
    buf175 = as_strided(buf171, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf171  # reuse
    kernel4.run(buf172, buf175, 768, 128, grid=grid(768, 128), stream=stream0)
    buf176 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf174, (12, 128, 64), (8192, 64, 1)), as_strided(buf175, (12, 64, 128), (8192, 128, 1)), out=buf176)
    buf177 = buf154; del buf154  # reuse
    kernel5.run(buf176, primals_47, buf177, 1536, 128, grid=grid(1536), stream=stream0)
    buf178 = buf153; del buf153  # reuse
    kernel6.run(buf176, primals_47, buf177, buf178, 1536, 128, grid=grid(1536), stream=stream0)
    buf179 = as_strided(buf176, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf176  # reuse
    kernel7.run(buf179, primals_47, buf177, buf178, 196608, grid=grid(196608), stream=stream0)
    del primals_47
    buf180 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel39.run(seed_cuda_0, buf179, buf180, 196608, grid=grid(196608), stream=stream0)
    buf181 = as_strided(buf172, (2, 6, 128, 64), (49152, 8192, 64, 1)); del buf172  # reuse
    kernel3.run(buf173, buf181, 98304, grid=grid(98304), stream=stream0)
    buf182 = as_strided(buf173, (12, 128, 64), (8192, 64, 1)); del buf173  # reuse
    aten.bmm.out(as_strided(buf180, (12, 128, 128), (16384, 128, 1)), as_strided(buf181, (12, 128, 64), (8192, 64, 1)), out=buf182)
    buf183 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf182, buf183, 98304, grid=grid(98304), stream=stream0)
    buf184 = as_strided(buf170, (256, 512), (512, 1)); del buf170  # reuse
    aten.mm.out(as_strided(buf183, (256, 384), (384, 1)), as_strided(primals_97, (384, 512), (1, 384)), out=buf184)
    buf185 = as_strided(buf184, (2, 128, 512), (65536, 512, 1)); del buf184  # reuse
    kernel40.run(buf185, buf168, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf186 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf185, buf186, 256, 512, grid=grid(256), stream=stream0)
    buf187 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_16, buf185, buf186, buf187, 131072, grid=grid(131072), stream=stream0)
    buf188 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf187, (256, 512), (512, 1)), as_strided(primals_98, (512, 1024), (1, 512)), out=buf188)
    buf189 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf187, (256, 512), (512, 1)), as_strided(primals_99, (512, 1024), (1, 512)), out=buf189)
    buf190 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel41.run(seed_cuda_0, buf188, buf189, buf190, 262144, grid=grid(262144), stream=stream0)
    buf191 = as_strided(buf187, (256, 512), (512, 1)); del buf187  # reuse
    aten.mm.out(as_strided(buf190, (256, 1024), (1024, 1)), as_strided(primals_100, (1024, 512), (1, 1024)), out=buf191)
    buf192 = as_strided(buf191, (2, 128, 512), (65536, 512, 1)); del buf191  # reuse
    kernel42.run(buf192, buf185, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf193 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf192, buf193, 256, 512, grid=grid(256), stream=stream0)
    buf194 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel43.run(seed_cuda_0, primals_17, buf192, buf193, buf194, 131072, grid=grid(131072), stream=stream0)
    buf195 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel0.run(primals_192, primals_43, buf195, 131072, grid=grid(131072), stream=stream0)
    del primals_43
    buf196 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel44.run(seed_cuda_0, buf195, buf196, 256, 512, grid=grid(256), stream=stream0)
    buf197 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel45.run(primals_18, seed_cuda_0, buf195, buf196, buf197, 131072, grid=grid(131072), stream=stream0)
    buf198 = as_strided(buf182, (256, 384), (384, 1)); del buf182  # reuse
    aten.mm.out(as_strided(buf197, (256, 512), (512, 1)), as_strided(primals_101, (512, 384), (1, 512)), out=buf198)
    buf199 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf197, (256, 512), (512, 1)), as_strided(primals_102, (512, 384), (1, 512)), out=buf199)
    buf200 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf197, (256, 512), (512, 1)), as_strided(primals_103, (512, 384), (1, 512)), out=buf200)
    buf201 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf198, buf201, 98304, grid=grid(98304), stream=stream0)
    buf202 = as_strided(buf198, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf198  # reuse
    kernel4.run(buf199, buf202, 768, 128, grid=grid(768, 128), stream=stream0)
    buf203 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf201, (12, 128, 64), (8192, 64, 1)), as_strided(buf202, (12, 64, 128), (8192, 128, 1)), out=buf203)
    buf204 = buf178; del buf178  # reuse
    kernel46.run(buf203, primals_104, buf204, 1536, 128, grid=grid(1536), stream=stream0)
    buf205 = buf177; del buf177  # reuse
    kernel47.run(buf203, primals_104, buf204, buf205, 1536, 128, grid=grid(1536), stream=stream0)
    buf206 = as_strided(buf203, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf203  # reuse
    kernel48.run(buf206, primals_104, buf204, buf205, 196608, grid=grid(196608), stream=stream0)
    buf207 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel49.run(seed_cuda_0, buf206, buf207, 196608, grid=grid(196608), stream=stream0)
    buf208 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf200, buf208, 98304, grid=grid(98304), stream=stream0)
    buf209 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf207, (12, 128, 128), (16384, 128, 1)), as_strided(buf208, (12, 128, 64), (8192, 64, 1)), out=buf209)
    buf210 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf209, buf210, 98304, grid=grid(98304), stream=stream0)
    buf211 = as_strided(buf197, (256, 512), (512, 1)); del buf197  # reuse
    aten.mm.out(as_strided(buf210, (256, 384), (384, 1)), as_strided(primals_105, (384, 512), (1, 384)), out=buf211)
    buf212 = as_strided(buf211, (2, 128, 512), (65536, 512, 1)); del buf211  # reuse
    kernel50.run(buf212, seed_cuda_0, buf195, 131072, grid=grid(131072), stream=stream0)
    buf213 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf212, buf213, 256, 512, grid=grid(256), stream=stream0)
    buf214 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_19, buf212, buf213, buf214, 131072, grid=grid(131072), stream=stream0)
    buf215 = as_strided(buf209, (256, 384), (384, 1)); del buf209  # reuse
    aten.mm.out(as_strided(buf214, (256, 512), (512, 1)), as_strided(primals_106, (512, 384), (1, 512)), out=buf215)
    buf216 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_107, (512, 384), (1, 512)), out=buf216)
    buf217 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_108, (512, 384), (1, 512)), out=buf217)
    buf218 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf215, buf218, 98304, grid=grid(98304), stream=stream0)
    buf219 = as_strided(buf215, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf215  # reuse
    kernel4.run(buf216, buf219, 768, 128, grid=grid(768, 128), stream=stream0)
    buf220 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf218, (12, 128, 64), (8192, 64, 1)), as_strided(buf219, (12, 64, 128), (8192, 128, 1)), out=buf220)
    buf221 = buf205; del buf205  # reuse
    kernel51.run(buf220, buf221, 1536, 128, grid=grid(1536), stream=stream0)
    buf222 = buf204; del buf204  # reuse
    kernel52.run(buf220, buf221, buf222, 1536, 128, grid=grid(1536), stream=stream0)
    buf223 = as_strided(buf220, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf220  # reuse
    kernel53.run(buf223, buf221, buf222, 196608, grid=grid(196608), stream=stream0)
    buf224 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel54.run(seed_cuda_0, buf223, buf224, 196608, grid=grid(196608), stream=stream0)
    buf225 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf217, buf225, 98304, grid=grid(98304), stream=stream0)
    buf226 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf224, (12, 128, 128), (16384, 128, 1)), as_strided(buf225, (12, 128, 64), (8192, 64, 1)), out=buf226)
    buf227 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf226, buf227, 98304, grid=grid(98304), stream=stream0)
    buf228 = as_strided(buf214, (256, 512), (512, 1)); del buf214  # reuse
    aten.mm.out(as_strided(buf227, (256, 384), (384, 1)), as_strided(primals_109, (384, 512), (1, 384)), out=buf228)
    buf229 = as_strided(buf228, (2, 128, 512), (65536, 512, 1)); del buf228  # reuse
    kernel55.run(buf229, buf212, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf230 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf229, buf230, 256, 512, grid=grid(256), stream=stream0)
    buf231 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_20, buf229, buf230, buf231, 131072, grid=grid(131072), stream=stream0)
    buf232 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf231, (256, 512), (512, 1)), as_strided(primals_110, (512, 1024), (1, 512)), out=buf232)
    buf233 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf231, (256, 512), (512, 1)), as_strided(primals_111, (512, 1024), (1, 512)), out=buf233)
    buf234 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel56.run(seed_cuda_0, buf232, buf233, buf234, 262144, grid=grid(262144), stream=stream0)
    buf235 = as_strided(buf231, (256, 512), (512, 1)); del buf231  # reuse
    aten.mm.out(as_strided(buf234, (256, 1024), (1024, 1)), as_strided(primals_112, (1024, 512), (1, 1024)), out=buf235)
    buf236 = as_strided(buf235, (2, 128, 512), (65536, 512, 1)); del buf235  # reuse
    kernel57.run(buf236, buf229, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf237 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf236, buf237, 256, 512, grid=grid(256), stream=stream0)
    buf238 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_21, buf236, buf237, buf238, 131072, grid=grid(131072), stream=stream0)
    buf239 = as_strided(buf226, (256, 384), (384, 1)); del buf226  # reuse
    aten.mm.out(as_strided(buf238, (256, 512), (512, 1)), as_strided(primals_113, (512, 384), (1, 512)), out=buf239)
    buf240 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf238, (256, 512), (512, 1)), as_strided(primals_114, (512, 384), (1, 512)), out=buf240)
    buf241 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf238, (256, 512), (512, 1)), as_strided(primals_115, (512, 384), (1, 512)), out=buf241)
    buf242 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf239, buf242, 98304, grid=grid(98304), stream=stream0)
    buf243 = as_strided(buf239, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf239  # reuse
    kernel4.run(buf240, buf243, 768, 128, grid=grid(768, 128), stream=stream0)
    buf244 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf242, (12, 128, 64), (8192, 64, 1)), as_strided(buf243, (12, 64, 128), (8192, 128, 1)), out=buf244)
    buf245 = buf222; del buf222  # reuse
    kernel46.run(buf244, primals_104, buf245, 1536, 128, grid=grid(1536), stream=stream0)
    buf246 = buf221; del buf221  # reuse
    kernel47.run(buf244, primals_104, buf245, buf246, 1536, 128, grid=grid(1536), stream=stream0)
    buf247 = as_strided(buf244, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf244  # reuse
    kernel48.run(buf247, primals_104, buf245, buf246, 196608, grid=grid(196608), stream=stream0)
    buf248 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel58.run(seed_cuda_0, buf247, buf248, 196608, grid=grid(196608), stream=stream0)
    buf249 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf241, buf249, 98304, grid=grid(98304), stream=stream0)
    buf250 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf248, (12, 128, 128), (16384, 128, 1)), as_strided(buf249, (12, 128, 64), (8192, 64, 1)), out=buf250)
    buf251 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf250, buf251, 98304, grid=grid(98304), stream=stream0)
    buf252 = as_strided(buf238, (256, 512), (512, 1)); del buf238  # reuse
    aten.mm.out(as_strided(buf251, (256, 384), (384, 1)), as_strided(primals_116, (384, 512), (1, 384)), out=buf252)
    buf253 = as_strided(buf252, (2, 128, 512), (65536, 512, 1)); del buf252  # reuse
    kernel59.run(buf253, buf236, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf254 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf253, buf254, 256, 512, grid=grid(256), stream=stream0)
    buf255 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_22, buf253, buf254, buf255, 131072, grid=grid(131072), stream=stream0)
    buf256 = as_strided(buf250, (256, 384), (384, 1)); del buf250  # reuse
    aten.mm.out(as_strided(buf255, (256, 512), (512, 1)), as_strided(primals_117, (512, 384), (1, 512)), out=buf256)
    buf257 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_118, (512, 384), (1, 512)), out=buf257)
    buf258 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_119, (512, 384), (1, 512)), out=buf258)
    buf259 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf256, buf259, 98304, grid=grid(98304), stream=stream0)
    buf260 = as_strided(buf256, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf256  # reuse
    kernel4.run(buf257, buf260, 768, 128, grid=grid(768, 128), stream=stream0)
    buf261 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf259, (12, 128, 64), (8192, 64, 1)), as_strided(buf260, (12, 64, 128), (8192, 128, 1)), out=buf261)
    buf262 = buf246; del buf246  # reuse
    kernel51.run(buf261, buf262, 1536, 128, grid=grid(1536), stream=stream0)
    buf263 = buf245; del buf245  # reuse
    kernel52.run(buf261, buf262, buf263, 1536, 128, grid=grid(1536), stream=stream0)
    buf264 = as_strided(buf261, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf261  # reuse
    kernel53.run(buf264, buf262, buf263, 196608, grid=grid(196608), stream=stream0)
    buf265 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel60.run(seed_cuda_0, buf264, buf265, 196608, grid=grid(196608), stream=stream0)
    buf266 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf258, buf266, 98304, grid=grid(98304), stream=stream0)
    buf267 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf265, (12, 128, 128), (16384, 128, 1)), as_strided(buf266, (12, 128, 64), (8192, 64, 1)), out=buf267)
    buf268 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf267, buf268, 98304, grid=grid(98304), stream=stream0)
    buf269 = as_strided(buf255, (256, 512), (512, 1)); del buf255  # reuse
    aten.mm.out(as_strided(buf268, (256, 384), (384, 1)), as_strided(primals_120, (384, 512), (1, 384)), out=buf269)
    buf270 = as_strided(buf269, (2, 128, 512), (65536, 512, 1)); del buf269  # reuse
    kernel61.run(buf270, buf253, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf271 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf270, buf271, 256, 512, grid=grid(256), stream=stream0)
    buf272 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_23, buf270, buf271, buf272, 131072, grid=grid(131072), stream=stream0)
    buf273 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf272, (256, 512), (512, 1)), as_strided(primals_121, (512, 1024), (1, 512)), out=buf273)
    buf274 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf272, (256, 512), (512, 1)), as_strided(primals_122, (512, 1024), (1, 512)), out=buf274)
    buf275 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel62.run(seed_cuda_0, buf273, buf274, buf275, 262144, grid=grid(262144), stream=stream0)
    buf276 = as_strided(buf272, (256, 512), (512, 1)); del buf272  # reuse
    aten.mm.out(as_strided(buf275, (256, 1024), (1024, 1)), as_strided(primals_123, (1024, 512), (1, 1024)), out=buf276)
    buf277 = as_strided(buf276, (2, 128, 512), (65536, 512, 1)); del buf276  # reuse
    kernel63.run(buf277, buf270, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf278 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf277, buf278, 256, 512, grid=grid(256), stream=stream0)
    buf279 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_24, buf277, buf278, buf279, 131072, grid=grid(131072), stream=stream0)
    buf280 = as_strided(buf267, (256, 384), (384, 1)); del buf267  # reuse
    aten.mm.out(as_strided(buf279, (256, 512), (512, 1)), as_strided(primals_124, (512, 384), (1, 512)), out=buf280)
    buf281 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf279, (256, 512), (512, 1)), as_strided(primals_125, (512, 384), (1, 512)), out=buf281)
    buf282 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf279, (256, 512), (512, 1)), as_strided(primals_126, (512, 384), (1, 512)), out=buf282)
    buf283 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf280, buf283, 98304, grid=grid(98304), stream=stream0)
    buf284 = as_strided(buf280, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf280  # reuse
    kernel4.run(buf281, buf284, 768, 128, grid=grid(768, 128), stream=stream0)
    buf285 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf283, (12, 128, 64), (8192, 64, 1)), as_strided(buf284, (12, 64, 128), (8192, 128, 1)), out=buf285)
    buf286 = buf263; del buf263  # reuse
    kernel46.run(buf285, primals_104, buf286, 1536, 128, grid=grid(1536), stream=stream0)
    buf287 = buf262; del buf262  # reuse
    kernel47.run(buf285, primals_104, buf286, buf287, 1536, 128, grid=grid(1536), stream=stream0)
    buf288 = as_strided(buf285, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf285  # reuse
    kernel48.run(buf288, primals_104, buf286, buf287, 196608, grid=grid(196608), stream=stream0)
    buf289 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel64.run(seed_cuda_0, buf288, buf289, 196608, grid=grid(196608), stream=stream0)
    buf290 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf282, buf290, 98304, grid=grid(98304), stream=stream0)
    buf291 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf289, (12, 128, 128), (16384, 128, 1)), as_strided(buf290, (12, 128, 64), (8192, 64, 1)), out=buf291)
    buf292 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf291, buf292, 98304, grid=grid(98304), stream=stream0)
    buf293 = as_strided(buf279, (256, 512), (512, 1)); del buf279  # reuse
    aten.mm.out(as_strided(buf292, (256, 384), (384, 1)), as_strided(primals_127, (384, 512), (1, 384)), out=buf293)
    buf294 = as_strided(buf293, (2, 128, 512), (65536, 512, 1)); del buf293  # reuse
    kernel65.run(buf294, buf277, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf295 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf294, buf295, 256, 512, grid=grid(256), stream=stream0)
    buf296 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_25, buf294, buf295, buf296, 131072, grid=grid(131072), stream=stream0)
    buf297 = as_strided(buf291, (256, 384), (384, 1)); del buf291  # reuse
    aten.mm.out(as_strided(buf296, (256, 512), (512, 1)), as_strided(primals_128, (512, 384), (1, 512)), out=buf297)
    buf298 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_129, (512, 384), (1, 512)), out=buf298)
    buf299 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_130, (512, 384), (1, 512)), out=buf299)
    buf300 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf297, buf300, 98304, grid=grid(98304), stream=stream0)
    buf301 = as_strided(buf297, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf297  # reuse
    kernel4.run(buf298, buf301, 768, 128, grid=grid(768, 128), stream=stream0)
    buf302 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf300, (12, 128, 64), (8192, 64, 1)), as_strided(buf301, (12, 64, 128), (8192, 128, 1)), out=buf302)
    buf303 = buf287; del buf287  # reuse
    kernel51.run(buf302, buf303, 1536, 128, grid=grid(1536), stream=stream0)
    buf304 = buf286; del buf286  # reuse
    kernel52.run(buf302, buf303, buf304, 1536, 128, grid=grid(1536), stream=stream0)
    buf305 = as_strided(buf302, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf302  # reuse
    kernel53.run(buf305, buf303, buf304, 196608, grid=grid(196608), stream=stream0)
    buf306 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel66.run(seed_cuda_0, buf305, buf306, 196608, grid=grid(196608), stream=stream0)
    buf307 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf299, buf307, 98304, grid=grid(98304), stream=stream0)
    buf308 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf306, (12, 128, 128), (16384, 128, 1)), as_strided(buf307, (12, 128, 64), (8192, 64, 1)), out=buf308)
    buf309 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf308, buf309, 98304, grid=grid(98304), stream=stream0)
    buf310 = as_strided(buf296, (256, 512), (512, 1)); del buf296  # reuse
    aten.mm.out(as_strided(buf309, (256, 384), (384, 1)), as_strided(primals_131, (384, 512), (1, 384)), out=buf310)
    buf311 = as_strided(buf310, (2, 128, 512), (65536, 512, 1)); del buf310  # reuse
    kernel67.run(buf311, buf294, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf312 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf311, buf312, 256, 512, grid=grid(256), stream=stream0)
    buf313 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_26, buf311, buf312, buf313, 131072, grid=grid(131072), stream=stream0)
    buf314 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf313, (256, 512), (512, 1)), as_strided(primals_132, (512, 1024), (1, 512)), out=buf314)
    buf315 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf313, (256, 512), (512, 1)), as_strided(primals_133, (512, 1024), (1, 512)), out=buf315)
    buf316 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel68.run(seed_cuda_0, buf314, buf315, buf316, 262144, grid=grid(262144), stream=stream0)
    buf317 = as_strided(buf313, (256, 512), (512, 1)); del buf313  # reuse
    aten.mm.out(as_strided(buf316, (256, 1024), (1024, 1)), as_strided(primals_134, (1024, 512), (1, 1024)), out=buf317)
    buf318 = as_strided(buf317, (2, 128, 512), (65536, 512, 1)); del buf317  # reuse
    kernel69.run(buf318, buf311, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf319 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf318, buf319, 256, 512, grid=grid(256), stream=stream0)
    buf320 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_27, buf318, buf319, buf320, 131072, grid=grid(131072), stream=stream0)
    buf321 = as_strided(buf308, (256, 384), (384, 1)); del buf308  # reuse
    aten.mm.out(as_strided(buf320, (256, 512), (512, 1)), as_strided(primals_135, (512, 384), (1, 512)), out=buf321)
    buf322 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf320, (256, 512), (512, 1)), as_strided(primals_136, (512, 384), (1, 512)), out=buf322)
    buf323 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf320, (256, 512), (512, 1)), as_strided(primals_137, (512, 384), (1, 512)), out=buf323)
    buf324 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf321, buf324, 98304, grid=grid(98304), stream=stream0)
    buf325 = as_strided(buf321, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf321  # reuse
    kernel4.run(buf322, buf325, 768, 128, grid=grid(768, 128), stream=stream0)
    buf326 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf324, (12, 128, 64), (8192, 64, 1)), as_strided(buf325, (12, 64, 128), (8192, 128, 1)), out=buf326)
    buf327 = buf304; del buf304  # reuse
    kernel46.run(buf326, primals_104, buf327, 1536, 128, grid=grid(1536), stream=stream0)
    buf328 = buf303; del buf303  # reuse
    kernel47.run(buf326, primals_104, buf327, buf328, 1536, 128, grid=grid(1536), stream=stream0)
    buf329 = as_strided(buf326, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf326  # reuse
    kernel48.run(buf329, primals_104, buf327, buf328, 196608, grid=grid(196608), stream=stream0)
    buf330 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel70.run(seed_cuda_0, buf329, buf330, 196608, grid=grid(196608), stream=stream0)
    buf331 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf323, buf331, 98304, grid=grid(98304), stream=stream0)
    buf332 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf330, (12, 128, 128), (16384, 128, 1)), as_strided(buf331, (12, 128, 64), (8192, 64, 1)), out=buf332)
    buf333 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf332, buf333, 98304, grid=grid(98304), stream=stream0)
    buf334 = as_strided(buf320, (256, 512), (512, 1)); del buf320  # reuse
    aten.mm.out(as_strided(buf333, (256, 384), (384, 1)), as_strided(primals_138, (384, 512), (1, 384)), out=buf334)
    buf335 = as_strided(buf334, (2, 128, 512), (65536, 512, 1)); del buf334  # reuse
    kernel71.run(buf335, buf318, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf336 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf335, buf336, 256, 512, grid=grid(256), stream=stream0)
    buf337 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_28, buf335, buf336, buf337, 131072, grid=grid(131072), stream=stream0)
    buf338 = as_strided(buf332, (256, 384), (384, 1)); del buf332  # reuse
    aten.mm.out(as_strided(buf337, (256, 512), (512, 1)), as_strided(primals_139, (512, 384), (1, 512)), out=buf338)
    buf339 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_140, (512, 384), (1, 512)), out=buf339)
    buf340 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_141, (512, 384), (1, 512)), out=buf340)
    buf341 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf338, buf341, 98304, grid=grid(98304), stream=stream0)
    buf342 = as_strided(buf338, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf338  # reuse
    kernel4.run(buf339, buf342, 768, 128, grid=grid(768, 128), stream=stream0)
    buf343 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf341, (12, 128, 64), (8192, 64, 1)), as_strided(buf342, (12, 64, 128), (8192, 128, 1)), out=buf343)
    buf344 = buf328; del buf328  # reuse
    kernel51.run(buf343, buf344, 1536, 128, grid=grid(1536), stream=stream0)
    buf345 = buf327; del buf327  # reuse
    kernel52.run(buf343, buf344, buf345, 1536, 128, grid=grid(1536), stream=stream0)
    buf346 = as_strided(buf343, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf343  # reuse
    kernel53.run(buf346, buf344, buf345, 196608, grid=grid(196608), stream=stream0)
    buf347 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel72.run(seed_cuda_0, buf346, buf347, 196608, grid=grid(196608), stream=stream0)
    buf348 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf340, buf348, 98304, grid=grid(98304), stream=stream0)
    buf349 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf347, (12, 128, 128), (16384, 128, 1)), as_strided(buf348, (12, 128, 64), (8192, 64, 1)), out=buf349)
    buf350 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf349, buf350, 98304, grid=grid(98304), stream=stream0)
    buf351 = as_strided(buf337, (256, 512), (512, 1)); del buf337  # reuse
    aten.mm.out(as_strided(buf350, (256, 384), (384, 1)), as_strided(primals_142, (384, 512), (1, 384)), out=buf351)
    buf352 = as_strided(buf351, (2, 128, 512), (65536, 512, 1)); del buf351  # reuse
    kernel73.run(buf352, buf335, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf353 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf352, buf353, 256, 512, grid=grid(256), stream=stream0)
    buf354 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_29, buf352, buf353, buf354, 131072, grid=grid(131072), stream=stream0)
    buf355 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf354, (256, 512), (512, 1)), as_strided(primals_143, (512, 1024), (1, 512)), out=buf355)
    buf356 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf354, (256, 512), (512, 1)), as_strided(primals_144, (512, 1024), (1, 512)), out=buf356)
    buf357 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel74.run(seed_cuda_0, buf355, buf356, buf357, 262144, grid=grid(262144), stream=stream0)
    buf358 = as_strided(buf354, (256, 512), (512, 1)); del buf354  # reuse
    aten.mm.out(as_strided(buf357, (256, 1024), (1024, 1)), as_strided(primals_145, (1024, 512), (1, 1024)), out=buf358)
    buf359 = as_strided(buf358, (2, 128, 512), (65536, 512, 1)); del buf358  # reuse
    kernel75.run(buf359, buf352, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf360 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf359, buf360, 256, 512, grid=grid(256), stream=stream0)
    buf361 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_30, buf359, buf360, buf361, 131072, grid=grid(131072), stream=stream0)
    buf362 = as_strided(buf349, (256, 384), (384, 1)); del buf349  # reuse
    aten.mm.out(as_strided(buf361, (256, 512), (512, 1)), as_strided(primals_146, (512, 384), (1, 512)), out=buf362)
    buf363 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf361, (256, 512), (512, 1)), as_strided(primals_147, (512, 384), (1, 512)), out=buf363)
    buf364 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf361, (256, 512), (512, 1)), as_strided(primals_148, (512, 384), (1, 512)), out=buf364)
    buf365 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf362, buf365, 98304, grid=grid(98304), stream=stream0)
    buf366 = as_strided(buf362, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf362  # reuse
    kernel4.run(buf363, buf366, 768, 128, grid=grid(768, 128), stream=stream0)
    buf367 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf365, (12, 128, 64), (8192, 64, 1)), as_strided(buf366, (12, 64, 128), (8192, 128, 1)), out=buf367)
    buf368 = buf345; del buf345  # reuse
    kernel46.run(buf367, primals_104, buf368, 1536, 128, grid=grid(1536), stream=stream0)
    buf369 = buf344; del buf344  # reuse
    kernel47.run(buf367, primals_104, buf368, buf369, 1536, 128, grid=grid(1536), stream=stream0)
    buf370 = as_strided(buf367, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf367  # reuse
    kernel48.run(buf370, primals_104, buf368, buf369, 196608, grid=grid(196608), stream=stream0)
    buf371 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel76.run(seed_cuda_0, buf370, buf371, 196608, grid=grid(196608), stream=stream0)
    buf372 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf364, buf372, 98304, grid=grid(98304), stream=stream0)
    buf373 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf371, (12, 128, 128), (16384, 128, 1)), as_strided(buf372, (12, 128, 64), (8192, 64, 1)), out=buf373)
    buf374 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf373, buf374, 98304, grid=grid(98304), stream=stream0)
    buf375 = as_strided(buf361, (256, 512), (512, 1)); del buf361  # reuse
    aten.mm.out(as_strided(buf374, (256, 384), (384, 1)), as_strided(primals_149, (384, 512), (1, 384)), out=buf375)
    buf376 = as_strided(buf375, (2, 128, 512), (65536, 512, 1)); del buf375  # reuse
    kernel77.run(buf376, buf359, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf377 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf376, buf377, 256, 512, grid=grid(256), stream=stream0)
    buf378 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_31, buf376, buf377, buf378, 131072, grid=grid(131072), stream=stream0)
    buf379 = as_strided(buf373, (256, 384), (384, 1)); del buf373  # reuse
    aten.mm.out(as_strided(buf378, (256, 512), (512, 1)), as_strided(primals_150, (512, 384), (1, 512)), out=buf379)
    buf380 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_151, (512, 384), (1, 512)), out=buf380)
    buf381 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_152, (512, 384), (1, 512)), out=buf381)
    buf382 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf379, buf382, 98304, grid=grid(98304), stream=stream0)
    buf383 = as_strided(buf379, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf379  # reuse
    kernel4.run(buf380, buf383, 768, 128, grid=grid(768, 128), stream=stream0)
    buf384 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf382, (12, 128, 64), (8192, 64, 1)), as_strided(buf383, (12, 64, 128), (8192, 128, 1)), out=buf384)
    buf385 = buf369; del buf369  # reuse
    kernel51.run(buf384, buf385, 1536, 128, grid=grid(1536), stream=stream0)
    buf386 = buf368; del buf368  # reuse
    kernel52.run(buf384, buf385, buf386, 1536, 128, grid=grid(1536), stream=stream0)
    buf387 = as_strided(buf384, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf384  # reuse
    kernel53.run(buf387, buf385, buf386, 196608, grid=grid(196608), stream=stream0)
    buf388 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel78.run(seed_cuda_0, buf387, buf388, 196608, grid=grid(196608), stream=stream0)
    buf389 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf381, buf389, 98304, grid=grid(98304), stream=stream0)
    buf390 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf388, (12, 128, 128), (16384, 128, 1)), as_strided(buf389, (12, 128, 64), (8192, 64, 1)), out=buf390)
    buf391 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf390, buf391, 98304, grid=grid(98304), stream=stream0)
    buf392 = as_strided(buf378, (256, 512), (512, 1)); del buf378  # reuse
    aten.mm.out(as_strided(buf391, (256, 384), (384, 1)), as_strided(primals_153, (384, 512), (1, 384)), out=buf392)
    buf393 = as_strided(buf392, (2, 128, 512), (65536, 512, 1)); del buf392  # reuse
    kernel79.run(buf393, buf376, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf394 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf393, buf394, 256, 512, grid=grid(256), stream=stream0)
    buf395 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_32, buf393, buf394, buf395, 131072, grid=grid(131072), stream=stream0)
    buf396 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf395, (256, 512), (512, 1)), as_strided(primals_154, (512, 1024), (1, 512)), out=buf396)
    buf397 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf395, (256, 512), (512, 1)), as_strided(primals_155, (512, 1024), (1, 512)), out=buf397)
    buf398 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel80.run(seed_cuda_0, buf396, buf397, buf398, 262144, grid=grid(262144), stream=stream0)
    buf399 = as_strided(buf395, (256, 512), (512, 1)); del buf395  # reuse
    aten.mm.out(as_strided(buf398, (256, 1024), (1024, 1)), as_strided(primals_156, (1024, 512), (1, 1024)), out=buf399)
    buf400 = as_strided(buf399, (2, 128, 512), (65536, 512, 1)); del buf399  # reuse
    kernel81.run(buf400, buf393, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf401 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf400, buf401, 256, 512, grid=grid(256), stream=stream0)
    buf402 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_33, buf400, buf401, buf402, 131072, grid=grid(131072), stream=stream0)
    buf403 = as_strided(buf390, (256, 384), (384, 1)); del buf390  # reuse
    aten.mm.out(as_strided(buf402, (256, 512), (512, 1)), as_strided(primals_157, (512, 384), (1, 512)), out=buf403)
    buf404 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf402, (256, 512), (512, 1)), as_strided(primals_158, (512, 384), (1, 512)), out=buf404)
    buf405 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf402, (256, 512), (512, 1)), as_strided(primals_159, (512, 384), (1, 512)), out=buf405)
    buf406 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf403, buf406, 98304, grid=grid(98304), stream=stream0)
    buf407 = as_strided(buf403, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf403  # reuse
    kernel4.run(buf404, buf407, 768, 128, grid=grid(768, 128), stream=stream0)
    buf408 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf406, (12, 128, 64), (8192, 64, 1)), as_strided(buf407, (12, 64, 128), (8192, 128, 1)), out=buf408)
    buf409 = buf386; del buf386  # reuse
    kernel46.run(buf408, primals_104, buf409, 1536, 128, grid=grid(1536), stream=stream0)
    buf410 = buf385; del buf385  # reuse
    kernel47.run(buf408, primals_104, buf409, buf410, 1536, 128, grid=grid(1536), stream=stream0)
    buf411 = as_strided(buf408, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf408  # reuse
    kernel48.run(buf411, primals_104, buf409, buf410, 196608, grid=grid(196608), stream=stream0)
    buf412 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel82.run(seed_cuda_0, buf411, buf412, 196608, grid=grid(196608), stream=stream0)
    buf413 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf405, buf413, 98304, grid=grid(98304), stream=stream0)
    buf414 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf412, (12, 128, 128), (16384, 128, 1)), as_strided(buf413, (12, 128, 64), (8192, 64, 1)), out=buf414)
    buf415 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf414, buf415, 98304, grid=grid(98304), stream=stream0)
    buf416 = as_strided(buf402, (256, 512), (512, 1)); del buf402  # reuse
    aten.mm.out(as_strided(buf415, (256, 384), (384, 1)), as_strided(primals_160, (384, 512), (1, 384)), out=buf416)
    buf417 = as_strided(buf416, (2, 128, 512), (65536, 512, 1)); del buf416  # reuse
    kernel83.run(buf417, buf400, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf418 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf417, buf418, 256, 512, grid=grid(256), stream=stream0)
    buf419 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_34, buf417, buf418, buf419, 131072, grid=grid(131072), stream=stream0)
    buf420 = as_strided(buf414, (256, 384), (384, 1)); del buf414  # reuse
    aten.mm.out(as_strided(buf419, (256, 512), (512, 1)), as_strided(primals_161, (512, 384), (1, 512)), out=buf420)
    buf421 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_162, (512, 384), (1, 512)), out=buf421)
    buf422 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_163, (512, 384), (1, 512)), out=buf422)
    buf423 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf420, buf423, 98304, grid=grid(98304), stream=stream0)
    buf424 = as_strided(buf420, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf420  # reuse
    kernel4.run(buf421, buf424, 768, 128, grid=grid(768, 128), stream=stream0)
    buf425 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf423, (12, 128, 64), (8192, 64, 1)), as_strided(buf424, (12, 64, 128), (8192, 128, 1)), out=buf425)
    buf426 = buf410; del buf410  # reuse
    kernel51.run(buf425, buf426, 1536, 128, grid=grid(1536), stream=stream0)
    buf427 = buf409; del buf409  # reuse
    kernel52.run(buf425, buf426, buf427, 1536, 128, grid=grid(1536), stream=stream0)
    buf428 = as_strided(buf425, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf425  # reuse
    kernel53.run(buf428, buf426, buf427, 196608, grid=grid(196608), stream=stream0)
    buf429 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel84.run(seed_cuda_0, buf428, buf429, 196608, grid=grid(196608), stream=stream0)
    buf430 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf422, buf430, 98304, grid=grid(98304), stream=stream0)
    buf431 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf429, (12, 128, 128), (16384, 128, 1)), as_strided(buf430, (12, 128, 64), (8192, 64, 1)), out=buf431)
    buf432 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf431, buf432, 98304, grid=grid(98304), stream=stream0)
    buf433 = as_strided(buf419, (256, 512), (512, 1)); del buf419  # reuse
    aten.mm.out(as_strided(buf432, (256, 384), (384, 1)), as_strided(primals_164, (384, 512), (1, 384)), out=buf433)
    buf434 = as_strided(buf433, (2, 128, 512), (65536, 512, 1)); del buf433  # reuse
    kernel85.run(buf434, buf417, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf435 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf434, buf435, 256, 512, grid=grid(256), stream=stream0)
    buf436 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_35, buf434, buf435, buf436, 131072, grid=grid(131072), stream=stream0)
    buf437 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf436, (256, 512), (512, 1)), as_strided(primals_165, (512, 1024), (1, 512)), out=buf437)
    buf438 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf436, (256, 512), (512, 1)), as_strided(primals_166, (512, 1024), (1, 512)), out=buf438)
    buf439 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel86.run(seed_cuda_0, buf437, buf438, buf439, 262144, grid=grid(262144), stream=stream0)
    buf440 = as_strided(buf436, (256, 512), (512, 1)); del buf436  # reuse
    aten.mm.out(as_strided(buf439, (256, 1024), (1024, 1)), as_strided(primals_167, (1024, 512), (1, 1024)), out=buf440)
    buf441 = as_strided(buf440, (2, 128, 512), (65536, 512, 1)); del buf440  # reuse
    kernel87.run(buf441, buf434, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf442 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf441, buf442, 256, 512, grid=grid(256), stream=stream0)
    buf443 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_36, buf441, buf442, buf443, 131072, grid=grid(131072), stream=stream0)
    buf444 = as_strided(buf431, (256, 384), (384, 1)); del buf431  # reuse
    aten.mm.out(as_strided(buf443, (256, 512), (512, 1)), as_strided(primals_168, (512, 384), (1, 512)), out=buf444)
    buf445 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf443, (256, 512), (512, 1)), as_strided(primals_169, (512, 384), (1, 512)), out=buf445)
    buf446 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf443, (256, 512), (512, 1)), as_strided(primals_170, (512, 384), (1, 512)), out=buf446)
    buf447 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf444, buf447, 98304, grid=grid(98304), stream=stream0)
    buf448 = as_strided(buf444, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf444  # reuse
    kernel4.run(buf445, buf448, 768, 128, grid=grid(768, 128), stream=stream0)
    buf449 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf447, (12, 128, 64), (8192, 64, 1)), as_strided(buf448, (12, 64, 128), (8192, 128, 1)), out=buf449)
    buf450 = buf427; del buf427  # reuse
    kernel46.run(buf449, primals_104, buf450, 1536, 128, grid=grid(1536), stream=stream0)
    buf451 = buf426; del buf426  # reuse
    kernel47.run(buf449, primals_104, buf450, buf451, 1536, 128, grid=grid(1536), stream=stream0)
    buf452 = as_strided(buf449, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf449  # reuse
    kernel48.run(buf452, primals_104, buf450, buf451, 196608, grid=grid(196608), stream=stream0)
    buf453 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel88.run(seed_cuda_0, buf452, buf453, 196608, grid=grid(196608), stream=stream0)
    buf454 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf446, buf454, 98304, grid=grid(98304), stream=stream0)
    buf455 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf453, (12, 128, 128), (16384, 128, 1)), as_strided(buf454, (12, 128, 64), (8192, 64, 1)), out=buf455)
    buf456 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf455, buf456, 98304, grid=grid(98304), stream=stream0)
    buf457 = as_strided(buf443, (256, 512), (512, 1)); del buf443  # reuse
    aten.mm.out(as_strided(buf456, (256, 384), (384, 1)), as_strided(primals_171, (384, 512), (1, 384)), out=buf457)
    buf458 = as_strided(buf457, (2, 128, 512), (65536, 512, 1)); del buf457  # reuse
    kernel89.run(buf458, buf441, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf459 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf458, buf459, 256, 512, grid=grid(256), stream=stream0)
    buf460 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_37, buf458, buf459, buf460, 131072, grid=grid(131072), stream=stream0)
    buf461 = as_strided(buf455, (256, 384), (384, 1)); del buf455  # reuse
    aten.mm.out(as_strided(buf460, (256, 512), (512, 1)), as_strided(primals_172, (512, 384), (1, 512)), out=buf461)
    buf462 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_173, (512, 384), (1, 512)), out=buf462)
    buf463 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_174, (512, 384), (1, 512)), out=buf463)
    buf464 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf461, buf464, 98304, grid=grid(98304), stream=stream0)
    buf465 = as_strided(buf461, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf461  # reuse
    kernel4.run(buf462, buf465, 768, 128, grid=grid(768, 128), stream=stream0)
    buf466 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf464, (12, 128, 64), (8192, 64, 1)), as_strided(buf465, (12, 64, 128), (8192, 128, 1)), out=buf466)
    buf467 = buf451; del buf451  # reuse
    kernel51.run(buf466, buf467, 1536, 128, grid=grid(1536), stream=stream0)
    buf468 = buf450; del buf450  # reuse
    kernel52.run(buf466, buf467, buf468, 1536, 128, grid=grid(1536), stream=stream0)
    buf469 = as_strided(buf466, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf466  # reuse
    kernel53.run(buf469, buf467, buf468, 196608, grid=grid(196608), stream=stream0)
    buf470 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel90.run(seed_cuda_0, buf469, buf470, 196608, grid=grid(196608), stream=stream0)
    buf471 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf463, buf471, 98304, grid=grid(98304), stream=stream0)
    buf472 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf470, (12, 128, 128), (16384, 128, 1)), as_strided(buf471, (12, 128, 64), (8192, 64, 1)), out=buf472)
    buf473 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf472, buf473, 98304, grid=grid(98304), stream=stream0)
    buf474 = as_strided(buf460, (256, 512), (512, 1)); del buf460  # reuse
    aten.mm.out(as_strided(buf473, (256, 384), (384, 1)), as_strided(primals_175, (384, 512), (1, 384)), out=buf474)
    buf475 = as_strided(buf474, (2, 128, 512), (65536, 512, 1)); del buf474  # reuse
    kernel91.run(buf475, buf458, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf476 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf475, buf476, 256, 512, grid=grid(256), stream=stream0)
    buf477 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_38, buf475, buf476, buf477, 131072, grid=grid(131072), stream=stream0)
    buf478 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf477, (256, 512), (512, 1)), as_strided(primals_176, (512, 1024), (1, 512)), out=buf478)
    buf479 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf477, (256, 512), (512, 1)), as_strided(primals_177, (512, 1024), (1, 512)), out=buf479)
    buf480 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel92.run(seed_cuda_0, buf478, buf479, buf480, 262144, grid=grid(262144), stream=stream0)
    buf481 = as_strided(buf477, (256, 512), (512, 1)); del buf477  # reuse
    aten.mm.out(as_strided(buf480, (256, 1024), (1024, 1)), as_strided(primals_178, (1024, 512), (1, 1024)), out=buf481)
    buf482 = as_strided(buf481, (2, 128, 512), (65536, 512, 1)); del buf481  # reuse
    kernel93.run(buf482, buf475, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf483 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf482, buf483, 256, 512, grid=grid(256), stream=stream0)
    buf484 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_39, buf482, buf483, buf484, 131072, grid=grid(131072), stream=stream0)
    buf485 = as_strided(buf472, (256, 384), (384, 1)); del buf472  # reuse
    aten.mm.out(as_strided(buf484, (256, 512), (512, 1)), as_strided(primals_179, (512, 384), (1, 512)), out=buf485)
    buf486 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf484, (256, 512), (512, 1)), as_strided(primals_180, (512, 384), (1, 512)), out=buf486)
    buf487 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf484, (256, 512), (512, 1)), as_strided(primals_181, (512, 384), (1, 512)), out=buf487)
    buf488 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf485, buf488, 98304, grid=grid(98304), stream=stream0)
    buf489 = as_strided(buf485, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf485  # reuse
    kernel4.run(buf486, buf489, 768, 128, grid=grid(768, 128), stream=stream0)
    buf490 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf488, (12, 128, 64), (8192, 64, 1)), as_strided(buf489, (12, 64, 128), (8192, 128, 1)), out=buf490)
    buf491 = buf468; del buf468  # reuse
    kernel46.run(buf490, primals_104, buf491, 1536, 128, grid=grid(1536), stream=stream0)
    buf492 = buf467; del buf467  # reuse
    kernel47.run(buf490, primals_104, buf491, buf492, 1536, 128, grid=grid(1536), stream=stream0)
    buf493 = as_strided(buf490, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf490  # reuse
    kernel48.run(buf493, primals_104, buf491, buf492, 196608, grid=grid(196608), stream=stream0)
    del primals_104
    buf494 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel94.run(seed_cuda_0, buf493, buf494, 196608, grid=grid(196608), stream=stream0)
    buf495 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf487, buf495, 98304, grid=grid(98304), stream=stream0)
    buf496 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf494, (12, 128, 128), (16384, 128, 1)), as_strided(buf495, (12, 128, 64), (8192, 64, 1)), out=buf496)
    buf497 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf496, buf497, 98304, grid=grid(98304), stream=stream0)
    buf498 = as_strided(buf484, (256, 512), (512, 1)); del buf484  # reuse
    aten.mm.out(as_strided(buf497, (256, 384), (384, 1)), as_strided(primals_182, (384, 512), (1, 384)), out=buf498)
    buf499 = as_strided(buf498, (2, 128, 512), (65536, 512, 1)); del buf498  # reuse
    kernel95.run(buf499, buf482, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf500 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf499, buf500, 256, 512, grid=grid(256), stream=stream0)
    buf501 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_40, buf499, buf500, buf501, 131072, grid=grid(131072), stream=stream0)
    buf502 = as_strided(buf496, (256, 384), (384, 1)); del buf496  # reuse
    aten.mm.out(as_strided(buf501, (256, 512), (512, 1)), as_strided(primals_183, (512, 384), (1, 512)), out=buf502)
    buf503 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_184, (512, 384), (1, 512)), out=buf503)
    buf504 = empty_strided((256, 384), (384, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf194, (256, 512), (512, 1)), as_strided(primals_185, (512, 384), (1, 512)), out=buf504)
    buf505 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf502, buf505, 98304, grid=grid(98304), stream=stream0)
    buf506 = as_strided(buf502, (2, 6, 64, 128), (49152, 8192, 128, 1)); del buf502  # reuse
    kernel4.run(buf503, buf506, 768, 128, grid=grid(768, 128), stream=stream0)
    buf507 = empty_strided((12, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf505, (12, 128, 64), (8192, 64, 1)), as_strided(buf506, (12, 64, 128), (8192, 128, 1)), out=buf507)
    buf508 = buf492; del buf492  # reuse
    kernel51.run(buf507, buf508, 1536, 128, grid=grid(1536), stream=stream0)
    buf509 = buf491; del buf491  # reuse
    kernel52.run(buf507, buf508, buf509, 1536, 128, grid=grid(1536), stream=stream0)
    buf510 = as_strided(buf507, (2, 6, 128, 128), (98304, 16384, 128, 1)); del buf507  # reuse
    kernel53.run(buf510, buf508, buf509, 196608, grid=grid(196608), stream=stream0)
    del buf508
    del buf509
    buf511 = empty_strided((2, 6, 128, 128), (98304, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel96.run(seed_cuda_0, buf510, buf511, 196608, grid=grid(196608), stream=stream0)
    buf512 = empty_strided((2, 6, 128, 64), (49152, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf504, buf512, 98304, grid=grid(98304), stream=stream0)
    buf513 = empty_strided((12, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf511, (12, 128, 128), (16384, 128, 1)), as_strided(buf512, (12, 128, 64), (8192, 64, 1)), out=buf513)
    buf514 = empty_strided((2, 128, 6, 64), (49152, 384, 64, 1), device='cuda', dtype=torch.float32)
    kernel9.run(buf513, buf514, 98304, grid=grid(98304), stream=stream0)
    del buf513
    buf515 = as_strided(buf501, (256, 512), (512, 1)); del buf501  # reuse
    aten.mm.out(as_strided(buf514, (256, 384), (384, 1)), as_strided(primals_186, (384, 512), (1, 384)), out=buf515)
    buf516 = as_strided(buf515, (2, 128, 512), (65536, 512, 1)); del buf515  # reuse
    kernel97.run(buf516, buf499, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf517 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf516, buf517, 256, 512, grid=grid(256), stream=stream0)
    buf518 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel12.run(primals_41, buf516, buf517, buf518, 131072, grid=grid(131072), stream=stream0)
    buf519 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf518, (256, 512), (512, 1)), as_strided(primals_187, (512, 1024), (1, 512)), out=buf519)
    buf520 = empty_strided((256, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf518, (256, 512), (512, 1)), as_strided(primals_188, (512, 1024), (1, 512)), out=buf520)
    buf521 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel98.run(seed_cuda_0, buf519, buf520, buf521, 262144, grid=grid(262144), stream=stream0)
    buf522 = as_strided(buf518, (256, 512), (512, 1)); del buf518  # reuse
    aten.mm.out(as_strided(buf521, (256, 1024), (1024, 1)), as_strided(primals_189, (1024, 512), (1, 1024)), out=buf522)
    buf523 = as_strided(buf522, (2, 128, 512), (65536, 512, 1)); del buf522  # reuse
    kernel99.run(buf523, buf516, seed_cuda_0, 131072, grid=grid(131072), stream=stream0)
    buf524 = empty_strided((2, 128, 1), (128, 1, 256), device='cuda', dtype=torch.float32)
    kernel11.run(buf523, buf524, 256, 512, grid=grid(256), stream=stream0)
    buf525 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.float32)
    kernel100.run(seed_cuda_0, primals_42, buf523, buf524, buf525, 131072, grid=grid(131072), stream=stream0)
    buf526 = empty_strided((256, 250112), (250112, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf525, (256, 512), (512, 1)), as_strided(primals_190, (512, 250112), (1, 512)), out=buf526)
    buf527 = empty_strided((256, 1), (1, 256), device='cuda', dtype=torch.float32)
    kernel101.run(buf526, buf527, 256, 250112, grid=grid(256), stream=stream0)
    buf528 = empty_strided((256, 1), (1, 256), device='cuda', dtype=torch.float32)
    kernel102.run(buf526, buf527, buf528, 256, 250112, grid=grid(256), stream=stream0)
    buf529 = empty_strided((256, 250112), (250112, 1), device='cuda', dtype=torch.float32)
    kernel103.run(buf526, buf527, buf528, buf529, 64028672, grid=grid(64028672), stream=stream0)
    del buf527
    del buf528
    buf530 = empty_strided((), (), device='cuda', dtype=torch.float32)
    kernel104.run(primals_193, buf529, buf530, 1, 256, grid=grid(1), stream=stream0)
    buf531 = buf530; del buf530  # reuse
    kernel105.run(buf531, 1, grid=grid(1), stream=stream0)
    buf532 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel106.run(seed_cuda_0, buf532, 131072, grid=grid(131072), stream=stream0)
    buf533 = buf1; del buf1  # reuse
    kernel107.run(buf533, 256, grid=grid(256), stream=stream0)
    buf534 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel108.run(seed_cuda_0, buf534, 131072, grid=grid(131072), stream=stream0)
    buf535 = buf18; del buf18  # reuse
    kernel107.run(buf535, 256, grid=grid(256), stream=stream0)
    buf536 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf20, buf536, 262144, grid=grid(262144), stream=stream0)
    buf537 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel110.run(seed_cuda_0, buf537, 262144, grid=grid(262144), stream=stream0)
    buf538 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel111.run(seed_cuda_0, buf538, 131072, grid=grid(131072), stream=stream0)
    buf539 = buf25; del buf25  # reuse
    kernel107.run(buf539, 256, grid=grid(256), stream=stream0)
    buf540 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel112.run(seed_cuda_0, buf540, 131072, grid=grid(131072), stream=stream0)
    buf541 = buf42; del buf42  # reuse
    kernel107.run(buf541, 256, grid=grid(256), stream=stream0)
    buf542 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf44, buf542, 262144, grid=grid(262144), stream=stream0)
    buf543 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel113.run(seed_cuda_0, buf543, 262144, grid=grid(262144), stream=stream0)
    buf544 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel114.run(seed_cuda_0, buf544, 131072, grid=grid(131072), stream=stream0)
    buf545 = buf49; del buf49  # reuse
    kernel107.run(buf545, 256, grid=grid(256), stream=stream0)
    buf546 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel115.run(seed_cuda_0, buf546, 131072, grid=grid(131072), stream=stream0)
    buf547 = buf66; del buf66  # reuse
    kernel107.run(buf547, 256, grid=grid(256), stream=stream0)
    buf548 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf68, buf548, 262144, grid=grid(262144), stream=stream0)
    buf549 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel116.run(seed_cuda_0, buf549, 262144, grid=grid(262144), stream=stream0)
    buf550 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel117.run(seed_cuda_0, buf550, 131072, grid=grid(131072), stream=stream0)
    buf551 = buf73; del buf73  # reuse
    kernel107.run(buf551, 256, grid=grid(256), stream=stream0)
    buf552 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel118.run(seed_cuda_0, buf552, 131072, grid=grid(131072), stream=stream0)
    buf553 = buf90; del buf90  # reuse
    kernel107.run(buf553, 256, grid=grid(256), stream=stream0)
    buf554 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf92, buf554, 262144, grid=grid(262144), stream=stream0)
    buf555 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel119.run(seed_cuda_0, buf555, 262144, grid=grid(262144), stream=stream0)
    buf556 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel120.run(seed_cuda_0, buf556, 131072, grid=grid(131072), stream=stream0)
    buf557 = buf97; del buf97  # reuse
    kernel107.run(buf557, 256, grid=grid(256), stream=stream0)
    buf558 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel121.run(seed_cuda_0, buf558, 131072, grid=grid(131072), stream=stream0)
    buf559 = buf114; del buf114  # reuse
    kernel107.run(buf559, 256, grid=grid(256), stream=stream0)
    buf560 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf116, buf560, 262144, grid=grid(262144), stream=stream0)
    buf561 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel122.run(seed_cuda_0, buf561, 262144, grid=grid(262144), stream=stream0)
    buf562 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel123.run(seed_cuda_0, buf562, 131072, grid=grid(131072), stream=stream0)
    buf563 = buf121; del buf121  # reuse
    kernel107.run(buf563, 256, grid=grid(256), stream=stream0)
    buf564 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel124.run(seed_cuda_0, buf564, 131072, grid=grid(131072), stream=stream0)
    buf565 = buf138; del buf138  # reuse
    kernel107.run(buf565, 256, grid=grid(256), stream=stream0)
    buf566 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf140, buf566, 262144, grid=grid(262144), stream=stream0)
    buf567 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel125.run(seed_cuda_0, buf567, 262144, grid=grid(262144), stream=stream0)
    buf568 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel126.run(seed_cuda_0, buf568, 131072, grid=grid(131072), stream=stream0)
    buf569 = buf145; del buf145  # reuse
    kernel107.run(buf569, 256, grid=grid(256), stream=stream0)
    buf570 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel127.run(seed_cuda_0, buf570, 131072, grid=grid(131072), stream=stream0)
    buf571 = buf162; del buf162  # reuse
    kernel107.run(buf571, 256, grid=grid(256), stream=stream0)
    buf572 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf164, buf572, 262144, grid=grid(262144), stream=stream0)
    buf573 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel128.run(seed_cuda_0, buf573, 262144, grid=grid(262144), stream=stream0)
    buf574 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel129.run(seed_cuda_0, buf574, 131072, grid=grid(131072), stream=stream0)
    buf575 = buf169; del buf169  # reuse
    kernel107.run(buf575, 256, grid=grid(256), stream=stream0)
    buf576 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel130.run(seed_cuda_0, buf576, 131072, grid=grid(131072), stream=stream0)
    buf577 = buf186; del buf186  # reuse
    kernel107.run(buf577, 256, grid=grid(256), stream=stream0)
    buf578 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf188, buf578, 262144, grid=grid(262144), stream=stream0)
    buf579 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel131.run(seed_cuda_0, buf579, 262144, grid=grid(262144), stream=stream0)
    buf580 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel132.run(seed_cuda_0, buf580, 131072, grid=grid(131072), stream=stream0)
    buf581 = buf193; del buf193  # reuse
    kernel107.run(buf581, 256, grid=grid(256), stream=stream0)
    buf582 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel133.run(seed_cuda_0, buf582, 131072, grid=grid(131072), stream=stream0)
    buf583 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel134.run(seed_cuda_0, buf583, 131072, grid=grid(131072), stream=stream0)
    buf584 = buf196; del buf196  # reuse
    kernel107.run(buf584, 256, grid=grid(256), stream=stream0)
    buf585 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel135.run(seed_cuda_0, buf585, 131072, grid=grid(131072), stream=stream0)
    buf586 = buf213; del buf213  # reuse
    kernel107.run(buf586, 256, grid=grid(256), stream=stream0)
    buf587 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel136.run(seed_cuda_0, buf587, 131072, grid=grid(131072), stream=stream0)
    buf588 = buf230; del buf230  # reuse
    kernel107.run(buf588, 256, grid=grid(256), stream=stream0)
    buf589 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf232, buf589, 262144, grid=grid(262144), stream=stream0)
    buf590 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel137.run(seed_cuda_0, buf590, 262144, grid=grid(262144), stream=stream0)
    buf591 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel138.run(seed_cuda_0, buf591, 131072, grid=grid(131072), stream=stream0)
    buf592 = buf237; del buf237  # reuse
    kernel107.run(buf592, 256, grid=grid(256), stream=stream0)
    buf593 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel139.run(seed_cuda_0, buf593, 131072, grid=grid(131072), stream=stream0)
    buf594 = buf254; del buf254  # reuse
    kernel107.run(buf594, 256, grid=grid(256), stream=stream0)
    buf595 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel140.run(seed_cuda_0, buf595, 131072, grid=grid(131072), stream=stream0)
    buf596 = buf271; del buf271  # reuse
    kernel107.run(buf596, 256, grid=grid(256), stream=stream0)
    buf597 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf273, buf597, 262144, grid=grid(262144), stream=stream0)
    buf598 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel141.run(seed_cuda_0, buf598, 262144, grid=grid(262144), stream=stream0)
    buf599 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel142.run(seed_cuda_0, buf599, 131072, grid=grid(131072), stream=stream0)
    buf600 = buf278; del buf278  # reuse
    kernel107.run(buf600, 256, grid=grid(256), stream=stream0)
    buf601 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel143.run(seed_cuda_0, buf601, 131072, grid=grid(131072), stream=stream0)
    buf602 = buf295; del buf295  # reuse
    kernel107.run(buf602, 256, grid=grid(256), stream=stream0)
    buf603 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel144.run(seed_cuda_0, buf603, 131072, grid=grid(131072), stream=stream0)
    buf604 = buf312; del buf312  # reuse
    kernel107.run(buf604, 256, grid=grid(256), stream=stream0)
    buf605 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf314, buf605, 262144, grid=grid(262144), stream=stream0)
    buf606 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel145.run(seed_cuda_0, buf606, 262144, grid=grid(262144), stream=stream0)
    buf607 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel146.run(seed_cuda_0, buf607, 131072, grid=grid(131072), stream=stream0)
    buf608 = buf319; del buf319  # reuse
    kernel107.run(buf608, 256, grid=grid(256), stream=stream0)
    buf609 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel147.run(seed_cuda_0, buf609, 131072, grid=grid(131072), stream=stream0)
    buf610 = buf336; del buf336  # reuse
    kernel107.run(buf610, 256, grid=grid(256), stream=stream0)
    buf611 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel148.run(seed_cuda_0, buf611, 131072, grid=grid(131072), stream=stream0)
    buf612 = buf353; del buf353  # reuse
    kernel107.run(buf612, 256, grid=grid(256), stream=stream0)
    buf613 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf355, buf613, 262144, grid=grid(262144), stream=stream0)
    buf614 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel149.run(seed_cuda_0, buf614, 262144, grid=grid(262144), stream=stream0)
    buf615 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel150.run(seed_cuda_0, buf615, 131072, grid=grid(131072), stream=stream0)
    buf616 = buf360; del buf360  # reuse
    kernel107.run(buf616, 256, grid=grid(256), stream=stream0)
    buf617 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel151.run(seed_cuda_0, buf617, 131072, grid=grid(131072), stream=stream0)
    buf618 = buf377; del buf377  # reuse
    kernel107.run(buf618, 256, grid=grid(256), stream=stream0)
    buf619 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel152.run(seed_cuda_0, buf619, 131072, grid=grid(131072), stream=stream0)
    buf620 = buf394; del buf394  # reuse
    kernel107.run(buf620, 256, grid=grid(256), stream=stream0)
    buf621 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf396, buf621, 262144, grid=grid(262144), stream=stream0)
    buf622 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel153.run(seed_cuda_0, buf622, 262144, grid=grid(262144), stream=stream0)
    buf623 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel154.run(seed_cuda_0, buf623, 131072, grid=grid(131072), stream=stream0)
    buf624 = buf401; del buf401  # reuse
    kernel107.run(buf624, 256, grid=grid(256), stream=stream0)
    buf625 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel155.run(seed_cuda_0, buf625, 131072, grid=grid(131072), stream=stream0)
    buf626 = buf418; del buf418  # reuse
    kernel107.run(buf626, 256, grid=grid(256), stream=stream0)
    buf627 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel156.run(seed_cuda_0, buf627, 131072, grid=grid(131072), stream=stream0)
    buf628 = buf435; del buf435  # reuse
    kernel107.run(buf628, 256, grid=grid(256), stream=stream0)
    buf629 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf437, buf629, 262144, grid=grid(262144), stream=stream0)
    buf630 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel157.run(seed_cuda_0, buf630, 262144, grid=grid(262144), stream=stream0)
    buf631 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel158.run(seed_cuda_0, buf631, 131072, grid=grid(131072), stream=stream0)
    buf632 = buf442; del buf442  # reuse
    kernel107.run(buf632, 256, grid=grid(256), stream=stream0)
    buf633 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel159.run(seed_cuda_0, buf633, 131072, grid=grid(131072), stream=stream0)
    buf634 = buf459; del buf459  # reuse
    kernel107.run(buf634, 256, grid=grid(256), stream=stream0)
    buf635 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel160.run(seed_cuda_0, buf635, 131072, grid=grid(131072), stream=stream0)
    buf636 = buf476; del buf476  # reuse
    kernel107.run(buf636, 256, grid=grid(256), stream=stream0)
    buf637 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf478, buf637, 262144, grid=grid(262144), stream=stream0)
    buf638 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel161.run(seed_cuda_0, buf638, 262144, grid=grid(262144), stream=stream0)
    buf639 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel162.run(seed_cuda_0, buf639, 131072, grid=grid(131072), stream=stream0)
    buf640 = buf483; del buf483  # reuse
    kernel107.run(buf640, 256, grid=grid(256), stream=stream0)
    buf641 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel163.run(seed_cuda_0, buf641, 131072, grid=grid(131072), stream=stream0)
    buf642 = buf500; del buf500  # reuse
    kernel107.run(buf642, 256, grid=grid(256), stream=stream0)
    buf643 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel164.run(seed_cuda_0, buf643, 131072, grid=grid(131072), stream=stream0)
    buf644 = buf517; del buf517  # reuse
    kernel107.run(buf644, 256, grid=grid(256), stream=stream0)
    buf645 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.float32)
    kernel109.run(buf519, buf645, 262144, grid=grid(262144), stream=stream0)
    buf646 = empty_strided((2, 128, 1024), (131072, 1024, 1), device='cuda', dtype=torch.bool)
    kernel165.run(seed_cuda_0, buf646, 262144, grid=grid(262144), stream=stream0)
    buf647 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel166.run(seed_cuda_0, buf647, 131072, grid=grid(131072), stream=stream0)
    buf648 = buf524; del buf524  # reuse
    kernel107.run(buf648, 256, grid=grid(256), stream=stream0)
    buf649 = empty_strided((2, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
    kernel167.run(seed_cuda_0, buf649, 131072, grid=grid(131072), stream=stream0)
    buf650 = empty_strided((128, 128), (128, 1), device='cuda', dtype=torch.int64)
    kernel168.run(buf650, 16384, grid=grid(16384), stream=stream0)
    buf651 = empty_strided((128, 128), (128, 1), device='cuda', dtype=torch.int64)
    kernel169.run(buf651, 16384, grid=grid(16384), stream=stream0)
    return (buf531, as_strided(buf526, (2, 128, 250112), (32014336, 250112, 1)), as_strided(buf199, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf200, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf216, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf217, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf240, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf241, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf257, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf258, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf281, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf282, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf298, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf299, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf322, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf323, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf339, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf340, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf363, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf364, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf380, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf381, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf404, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf405, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf421, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf422, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf445, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf446, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf462, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf463, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf486, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf487, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf503, (2, 6, 128, 64), (49152, 64, 384, 1)), as_strided(buf504, (2, 6, 128, 64), (49152, 64, 384, 1)), buf194, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, buf0, buf532, buf533, buf11, seed_cuda_0.clone(), as_strided(buf15, (256, 384), (384, 1)), buf534, buf17, buf535, buf20, buf536, buf21, buf537, as_strided(buf22, (256, 1024), (1024, 1)), buf538, buf24, buf539, buf35, as_strided(buf39, (256, 384), (384, 1)), buf540, buf41, buf541, buf44, buf542, buf45, buf543, as_strided(buf46, (256, 1024), (1024, 1)), buf544, buf48, buf545, buf59, as_strided(buf63, (256, 384), (384, 1)), buf546, buf65, buf547, buf68, buf548, buf69, buf549, as_strided(buf70, (256, 1024), (1024, 1)), buf550, buf72, buf551, buf83, as_strided(buf87, (256, 384), (384, 1)), buf552, buf89, buf553, buf92, buf554, buf93, buf555, as_strided(buf94, (256, 1024), (1024, 1)), buf556, buf96, buf557, buf107, as_strided(buf111, (256, 384), (384, 1)), buf558, buf113, buf559, buf116, buf560, buf117, buf561, as_strided(buf118, (256, 1024), (1024, 1)), buf562, buf120, buf563, buf131, as_strided(buf135, (256, 384), (384, 1)), buf564, buf137, buf565, buf140, buf566, buf141, buf567, as_strided(buf142, (256, 1024), (1024, 1)), buf568, buf144, buf569, buf155, as_strided(buf159, (256, 384), (384, 1)), buf570, buf161, buf571, buf164, buf572, buf165, buf573, as_strided(buf166, (256, 1024), (1024, 1)), buf574, buf168, buf575, buf179, as_strided(buf183, (256, 384), (384, 1)), buf576, buf185, buf577, buf188, buf578, buf189, buf579, as_strided(buf190, (256, 1024), (1024, 1)), buf580, buf192, buf581, buf582, buf195, buf583, buf584, buf206, as_strided(buf210, (256, 384), (384, 1)), buf585, buf212, buf586, buf223, as_strided(buf227, (256, 384), (384, 1)), buf587, buf229, buf588, buf232, buf589, buf233, buf590, as_strided(buf234, (256, 1024), (1024, 1)), buf591, buf236, buf592, buf247, as_strided(buf251, (256, 384), (384, 1)), buf593, buf253, buf594, buf264, as_strided(buf268, (256, 384), (384, 1)), buf595, buf270, buf596, buf273, buf597, buf274, buf598, as_strided(buf275, (256, 1024), (1024, 1)), buf599, buf277, buf600, buf288, as_strided(buf292, (256, 384), (384, 1)), buf601, buf294, buf602, buf305, as_strided(buf309, (256, 384), (384, 1)), buf603, buf311, buf604, buf314, buf605, buf315, buf606, as_strided(buf316, (256, 1024), (1024, 1)), buf607, buf318, buf608, buf329, as_strided(buf333, (256, 384), (384, 1)), buf609, buf335, buf610, buf346, as_strided(buf350, (256, 384), (384, 1)), buf611, buf352, buf612, buf355, buf613, buf356, buf614, as_strided(buf357, (256, 1024), (1024, 1)), buf615, buf359, buf616, buf370, as_strided(buf374, (256, 384), (384, 1)), buf617, buf376, buf618, buf387, as_strided(buf391, (256, 384), (384, 1)), buf619, buf393, buf620, buf396, buf621, buf397, buf622, as_strided(buf398, (256, 1024), (1024, 1)), buf623, buf400, buf624, buf411, as_strided(buf415, (256, 384), (384, 1)), buf625, buf417, buf626, buf428, as_strided(buf432, (256, 384), (384, 1)), buf627, buf434, buf628, buf437, buf629, buf438, buf630, as_strided(buf439, (256, 1024), (1024, 1)), buf631, buf441, buf632, buf452, as_strided(buf456, (256, 384), (384, 1)), buf633, buf458, buf634, buf469, as_strided(buf473, (256, 384), (384, 1)), buf635, buf475, buf636, buf478, buf637, buf479, buf638, as_strided(buf480, (256, 1024), (1024, 1)), buf639, buf482, buf640, buf493, as_strided(buf497, (256, 384), (384, 1)), buf641, buf499, buf642, buf510, as_strided(buf514, (256, 384), (384, 1)), buf643, buf516, buf644, buf519, buf645, buf520, buf646, as_strided(buf521, (256, 1024), (1024, 1)), buf647, buf523, buf648, buf649, as_strided(buf525, (256, 512), (512, 1)), buf529, as_strided(primals_193, (256, 1), (1, 1)), as_strided(primals_190, (250112, 512), (512, 1)), as_strided(primals_189, (512, 1024), (1024, 1)), as_strided(primals_188, (1024, 512), (512, 1)), as_strided(primals_187, (1024, 512), (512, 1)), as_strided(primals_186, (512, 384), (384, 1)), as_strided(buf511, (12, 128, 128), (16384, 1, 128)), as_strided(buf512, (12, 64, 128), (8192, 1, 64)), as_strided(buf505, (12, 64, 128), (8192, 1, 64)), as_strided(buf506, (12, 128, 64), (8192, 1, 128)), as_strided(primals_185, (384, 512), (512, 1)), as_strided(primals_184, (384, 512), (512, 1)), as_strided(primals_183, (384, 512), (512, 1)), as_strided(primals_182, (512, 384), (384, 1)), as_strided(buf494, (12, 128, 128), (16384, 1, 128)), as_strided(buf495, (12, 64, 128), (8192, 1, 64)), as_strided(buf488, (12, 64, 128), (8192, 1, 64)), as_strided(buf489, (12, 128, 64), (8192, 1, 128)), as_strided(primals_181, (384, 512), (512, 1)), as_strided(primals_180, (384, 512), (512, 1)), as_strided(primals_179, (384, 512), (512, 1)), as_strided(primals_178, (512, 1024), (1024, 1)), as_strided(primals_177, (1024, 512), (512, 1)), as_strided(primals_176, (1024, 512), (512, 1)), as_strided(primals_175, (512, 384), (384, 1)), as_strided(buf470, (12, 128, 128), (16384, 1, 128)), as_strided(buf471, (12, 64, 128), (8192, 1, 64)), as_strided(buf464, (12, 64, 128), (8192, 1, 64)), as_strided(buf465, (12, 128, 64), (8192, 1, 128)), as_strided(primals_174, (384, 512), (512, 1)), as_strided(primals_173, (384, 512), (512, 1)), as_strided(primals_172, (384, 512), (512, 1)), as_strided(primals_171, (512, 384), (384, 1)), as_strided(buf453, (12, 128, 128), (16384, 1, 128)), as_strided(buf454, (12, 64, 128), (8192, 1, 64)), as_strided(buf447, (12, 64, 128), (8192, 1, 64)), as_strided(buf448, (12, 128, 64), (8192, 1, 128)), as_strided(primals_170, (384, 512), (512, 1)), as_strided(primals_169, (384, 512), (512, 1)), as_strided(primals_168, (384, 512), (512, 1)), as_strided(primals_167, (512, 1024), (1024, 1)), as_strided(primals_166, (1024, 512), (512, 1)), as_strided(primals_165, (1024, 512), (512, 1)), as_strided(primals_164, (512, 384), (384, 1)), as_strided(buf429, (12, 128, 128), (16384, 1, 128)), as_strided(buf430, (12, 64, 128), (8192, 1, 64)), as_strided(buf423, (12, 64, 128), (8192, 1, 64)), as_strided(buf424, (12, 128, 64), (8192, 1, 128)), as_strided(primals_163, (384, 512), (512, 1)), as_strided(primals_162, (384, 512), (512, 1)), as_strided(primals_161, (384, 512), (512, 1)), as_strided(primals_160, (512, 384), (384, 1)), as_strided(buf412, (12, 128, 128), (16384, 1, 128)), as_strided(buf413, (12, 64, 128), (8192, 1, 64)), as_strided(buf406, (12, 64, 128), (8192, 1, 64)), as_strided(buf407, (12, 128, 64), (8192, 1, 128)), as_strided(primals_159, (384, 512), (512, 1)), as_strided(primals_158, (384, 512), (512, 1)), as_strided(primals_157, (384, 512), (512, 1)), as_strided(primals_156, (512, 1024), (1024, 1)), as_strided(primals_155, (1024, 512), (512, 1)), as_strided(primals_154, (1024, 512), (512, 1)), as_strided(primals_153, (512, 384), (384, 1)), as_strided(buf388, (12, 128, 128), (16384, 1, 128)), as_strided(buf389, (12, 64, 128), (8192, 1, 64)), as_strided(buf382, (12, 64, 128), (8192, 1, 64)), as_strided(buf383, (12, 128, 64), (8192, 1, 128)), as_strided(primals_152, (384, 512), (512, 1)), as_strided(primals_151, (384, 512), (512, 1)), as_strided(primals_150, (384, 512), (512, 1)), as_strided(primals_149, (512, 384), (384, 1)), as_strided(buf371, (12, 128, 128), (16384, 1, 128)), as_strided(buf372, (12, 64, 128), (8192, 1, 64)), as_strided(buf365, (12, 64, 128), (8192, 1, 64)), as_strided(buf366, (12, 128, 64), (8192, 1, 128)), as_strided(primals_148, (384, 512), (512, 1)), as_strided(primals_147, (384, 512), (512, 1)), as_strided(primals_146, (384, 512), (512, 1)), as_strided(primals_145, (512, 1024), (1024, 1)), as_strided(primals_144, (1024, 512), (512, 1)), as_strided(primals_143, (1024, 512), (512, 1)), as_strided(primals_142, (512, 384), (384, 1)), as_strided(buf347, (12, 128, 128), (16384, 1, 128)), as_strided(buf348, (12, 64, 128), (8192, 1, 64)), as_strided(buf341, (12, 64, 128), (8192, 1, 64)), as_strided(buf342, (12, 128, 64), (8192, 1, 128)), as_strided(primals_141, (384, 512), (512, 1)), as_strided(primals_140, (384, 512), (512, 1)), as_strided(primals_139, (384, 512), (512, 1)), as_strided(primals_138, (512, 384), (384, 1)), as_strided(buf330, (12, 128, 128), (16384, 1, 128)), as_strided(buf331, (12, 64, 128), (8192, 1, 64)), as_strided(buf324, (12, 64, 128), (8192, 1, 64)), as_strided(buf325, (12, 128, 64), (8192, 1, 128)), as_strided(primals_137, (384, 512), (512, 1)), as_strided(primals_136, (384, 512), (512, 1)), as_strided(primals_135, (384, 512), (512, 1)), as_strided(primals_134, (512, 1024), (1024, 1)), as_strided(primals_133, (1024, 512), (512, 1)), as_strided(primals_132, (1024, 512), (512, 1)), as_strided(primals_131, (512, 384), (384, 1)), as_strided(buf306, (12, 128, 128), (16384, 1, 128)), as_strided(buf307, (12, 64, 128), (8192, 1, 64)), as_strided(buf300, (12, 64, 128), (8192, 1, 64)), as_strided(buf301, (12, 128, 64), (8192, 1, 128)), as_strided(primals_130, (384, 512), (512, 1)), as_strided(primals_129, (384, 512), (512, 1)), as_strided(primals_128, (384, 512), (512, 1)), as_strided(primals_127, (512, 384), (384, 1)), as_strided(buf289, (12, 128, 128), (16384, 1, 128)), as_strided(buf290, (12, 64, 128), (8192, 1, 64)), as_strided(buf283, (12, 64, 128), (8192, 1, 64)), as_strided(buf284, (12, 128, 64), (8192, 1, 128)), as_strided(primals_126, (384, 512), (512, 1)), as_strided(primals_125, (384, 512), (512, 1)), as_strided(primals_124, (384, 512), (512, 1)), as_strided(primals_123, (512, 1024), (1024, 1)), as_strided(primals_122, (1024, 512), (512, 1)), as_strided(primals_121, (1024, 512), (512, 1)), as_strided(primals_120, (512, 384), (384, 1)), as_strided(buf265, (12, 128, 128), (16384, 1, 128)), as_strided(buf266, (12, 64, 128), (8192, 1, 64)), as_strided(buf259, (12, 64, 128), (8192, 1, 64)), as_strided(buf260, (12, 128, 64), (8192, 1, 128)), as_strided(primals_119, (384, 512), (512, 1)), as_strided(primals_118, (384, 512), (512, 1)), as_strided(primals_117, (384, 512), (512, 1)), as_strided(primals_116, (512, 384), (384, 1)), as_strided(buf248, (12, 128, 128), (16384, 1, 128)), as_strided(buf249, (12, 64, 128), (8192, 1, 64)), as_strided(buf242, (12, 64, 128), (8192, 1, 64)), as_strided(buf243, (12, 128, 64), (8192, 1, 128)), as_strided(primals_115, (384, 512), (512, 1)), as_strided(primals_114, (384, 512), (512, 1)), as_strided(primals_113, (384, 512), (512, 1)), as_strided(primals_112, (512, 1024), (1024, 1)), as_strided(primals_111, (1024, 512), (512, 1)), as_strided(primals_110, (1024, 512), (512, 1)), as_strided(primals_109, (512, 384), (384, 1)), as_strided(buf224, (12, 128, 128), (16384, 1, 128)), as_strided(buf225, (12, 64, 128), (8192, 1, 64)), as_strided(buf218, (12, 64, 128), (8192, 1, 64)), as_strided(buf219, (12, 128, 64), (8192, 1, 128)), as_strided(primals_108, (384, 512), (512, 1)), as_strided(primals_107, (384, 512), (512, 1)), as_strided(primals_106, (384, 512), (512, 1)), as_strided(primals_105, (512, 384), (384, 1)), as_strided(buf207, (12, 128, 128), (16384, 1, 128)), as_strided(buf208, (12, 64, 128), (8192, 1, 64)), as_strided(buf650, (16384, ), (1, )), as_strided(buf201, (12, 64, 128), (8192, 1, 64)), as_strided(buf202, (12, 128, 64), (8192, 1, 128)), as_strided(primals_103, (384, 512), (512, 1)), as_strided(primals_102, (384, 512), (512, 1)), as_strided(primals_101, (384, 512), (512, 1)), as_strided(primals_192, (256, ), (1, )), as_strided(primals_100, (512, 1024), (1024, 1)), as_strided(primals_99, (1024, 512), (512, 1)), as_strided(primals_98, (1024, 512), (512, 1)), as_strided(primals_97, (512, 384), (384, 1)), as_strided(buf180, (12, 128, 128), (16384, 1, 128)), as_strided(buf181, (12, 64, 128), (8192, 1, 64)), as_strided(buf174, (12, 64, 128), (8192, 1, 64)), as_strided(buf175, (12, 128, 64), (8192, 1, 128)), as_strided(primals_96, (384, 512), (512, 1)), as_strided(primals_95, (384, 512), (512, 1)), as_strided(primals_94, (384, 512), (512, 1)), as_strided(primals_93, (512, 1024), (1024, 1)), as_strided(primals_92, (1024, 512), (512, 1)), as_strided(primals_91, (1024, 512), (512, 1)), as_strided(primals_90, (512, 384), (384, 1)), as_strided(buf156, (12, 128, 128), (16384, 1, 128)), as_strided(buf157, (12, 64, 128), (8192, 1, 64)), as_strided(buf150, (12, 64, 128), (8192, 1, 64)), as_strided(buf151, (12, 128, 64), (8192, 1, 128)), as_strided(primals_89, (384, 512), (512, 1)), as_strided(primals_88, (384, 512), (512, 1)), as_strided(primals_87, (384, 512), (512, 1)), as_strided(primals_86, (512, 1024), (1024, 1)), as_strided(primals_85, (1024, 512), (512, 1)), as_strided(primals_84, (1024, 512), (512, 1)), as_strided(primals_83, (512, 384), (384, 1)), as_strided(buf132, (12, 128, 128), (16384, 1, 128)), as_strided(buf133, (12, 64, 128), (8192, 1, 64)), as_strided(buf126, (12, 64, 128), (8192, 1, 64)), as_strided(buf127, (12, 128, 64), (8192, 1, 128)), as_strided(primals_82, (384, 512), (512, 1)), as_strided(primals_81, (384, 512), (512, 1)), as_strided(primals_80, (384, 512), (512, 1)), as_strided(primals_79, (512, 1024), (1024, 1)), as_strided(primals_78, (1024, 512), (512, 1)), as_strided(primals_77, (1024, 512), (512, 1)), as_strided(primals_76, (512, 384), (384, 1)), as_strided(buf108, (12, 128, 128), (16384, 1, 128)), as_strided(buf109, (12, 64, 128), (8192, 1, 64)), as_strided(buf102, (12, 64, 128), (8192, 1, 64)), as_strided(buf103, (12, 128, 64), (8192, 1, 128)), as_strided(primals_75, (384, 512), (512, 1)), as_strided(primals_74, (384, 512), (512, 1)), as_strided(primals_73, (384, 512), (512, 1)), as_strided(primals_72, (512, 1024), (1024, 1)), as_strided(primals_71, (1024, 512), (512, 1)), as_strided(primals_70, (1024, 512), (512, 1)), as_strided(primals_69, (512, 384), (384, 1)), as_strided(buf84, (12, 128, 128), (16384, 1, 128)), as_strided(buf85, (12, 64, 128), (8192, 1, 64)), as_strided(buf78, (12, 64, 128), (8192, 1, 64)), as_strided(buf79, (12, 128, 64), (8192, 1, 128)), as_strided(primals_68, (384, 512), (512, 1)), as_strided(primals_67, (384, 512), (512, 1)), as_strided(primals_66, (384, 512), (512, 1)), as_strided(primals_65, (512, 1024), (1024, 1)), as_strided(primals_64, (1024, 512), (512, 1)), as_strided(primals_63, (1024, 512), (512, 1)), as_strided(primals_62, (512, 384), (384, 1)), as_strided(buf60, (12, 128, 128), (16384, 1, 128)), as_strided(buf61, (12, 64, 128), (8192, 1, 64)), as_strided(buf54, (12, 64, 128), (8192, 1, 64)), as_strided(buf55, (12, 128, 64), (8192, 1, 128)), as_strided(primals_61, (384, 512), (512, 1)), as_strided(primals_60, (384, 512), (512, 1)), as_strided(primals_59, (384, 512), (512, 1)), as_strided(primals_58, (512, 1024), (1024, 1)), as_strided(primals_57, (1024, 512), (512, 1)), as_strided(primals_56, (1024, 512), (512, 1)), as_strided(primals_55, (512, 384), (384, 1)), as_strided(buf36, (12, 128, 128), (16384, 1, 128)), as_strided(buf37, (12, 64, 128), (8192, 1, 64)), as_strided(buf30, (12, 64, 128), (8192, 1, 64)), as_strided(buf31, (12, 128, 64), (8192, 1, 128)), as_strided(primals_54, (384, 512), (512, 1)), as_strided(primals_53, (384, 512), (512, 1)), as_strided(primals_52, (384, 512), (512, 1)), as_strided(primals_51, (512, 1024), (1024, 1)), as_strided(primals_50, (1024, 512), (512, 1)), as_strided(primals_49, (1024, 512), (512, 1)), as_strided(primals_48, (512, 384), (384, 1)), as_strided(buf12, (12, 128, 128), (16384, 1, 128)), as_strided(buf13, (12, 64, 128), (8192, 1, 64)), as_strided(buf651, (16384, ), (1, )), as_strided(buf6, (12, 64, 128), (8192, 1, 64)), as_strided(buf7, (12, 128, 64), (8192, 1, 128)), as_strided(primals_46, (384, 512), (512, 1)), as_strided(primals_45, (384, 512), (512, 1)), as_strided(primals_44, (384, 512), (512, 1)), as_strided(primals_191, (256, ), (1, )), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    seed_cuda_0 = rand_strided((), (), device='cuda', dtype=torch.int64)
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
    primals_43 = rand_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_44 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_45 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_46 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_47 = rand_strided((32, 6), (6, 1), device='cuda', dtype=torch.float32)
    primals_48 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_49 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_50 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_51 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_52 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_53 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_54 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_55 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_56 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_57 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_58 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_59 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_60 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_61 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_62 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_63 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_64 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_65 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_66 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_67 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_68 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_69 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_70 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_71 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_72 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_73 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_74 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_75 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_76 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_77 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_78 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_79 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_80 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_81 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_82 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_83 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_84 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_85 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_86 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_87 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_88 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_89 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_90 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_92 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_93 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_94 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_95 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_96 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_97 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_98 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_99 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_100 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_101 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_102 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_103 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_104 = rand_strided((32, 6), (6, 1), device='cuda', dtype=torch.float32)
    primals_105 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_106 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_107 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_108 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_109 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_110 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_111 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_113 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_114 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_115 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_116 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_117 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_118 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_119 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_120 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_121 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_122 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_123 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_124 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_125 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_126 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_127 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_128 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_129 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_130 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_131 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_132 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_133 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_134 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_135 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_136 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_137 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_138 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_139 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_140 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_141 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_142 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_143 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_144 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_145 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_146 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_147 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_148 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_149 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_150 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_151 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_152 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_153 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_154 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_155 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_156 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_157 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_158 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_159 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_160 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_161 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_162 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_163 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_164 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_165 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_166 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_167 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_168 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_169 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_170 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_171 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_172 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_173 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_174 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_175 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_176 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_177 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_179 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_180 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_181 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_182 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_183 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_184 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_185 = rand_strided((384, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_186 = rand_strided((512, 384), (384, 1), device='cuda', dtype=torch.float32)
    primals_187 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_188 = rand_strided((1024, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_189 = rand_strided((512, 1024), (1024, 1), device='cuda', dtype=torch.float32)
    primals_190 = rand_strided((250112, 512), (512, 1), device='cuda', dtype=torch.float32)
    primals_191 = rand_strided((2, 128), (128, 1), device='cuda', dtype=torch.int64)
    primals_192 = rand_strided((2, 128), (128, 1), device='cuda', dtype=torch.int64)
    primals_193 = rand_strided((2, 128), (128, 1), device='cuda', dtype=torch.int64)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193]))
