
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
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, seed6, in_ptr7, in_ptr8, out_ptr0, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    x0 = xindex % 128
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp5 = tl.load(in_ptr4 + (x0), xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.load(in_ptr1 + (r2 + (768*tmp0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr3 + (r2 + (768*tmp2) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp4 = tmp1 + tmp3
        tmp6 = tl.load(in_ptr5 + (r2 + (768*tmp5) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tmp4 + tmp6
        tl.store(out_ptr0 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp7, xmask & rmask)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(out_ptr0 + (r2 + (768*x3)), xmask & rmask, eviction_policy='evict_last')
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
    tmp9 = tl.reshape(tl.sum(_tmp9, 1), [XBLOCK, 1])
    _tmp15 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp16 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(out_ptr0 + (r2 + (768*x3)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 768
        tmp12 = tmp9 / tmp11
        tmp13 = tmp10 - tmp12
        tmp14 = tmp13 * tmp13
        _tmp15 = tl.where(xmask & rmask, _tmp15 + tmp14, _tmp15)
        _tmp16 = tl.where(xmask & rmask, _tmp16 + tmp10, _tmp16)
    tmp15 = tl.reshape(tl.sum(_tmp15, 1), [XBLOCK, 1])
    tmp16 = tl.reshape(tl.sum(_tmp16, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(out_ptr0 + (r2 + (768*x3)), xmask & rmask, eviction_policy='evict_last')
        tmp18 = 768
        tmp19 = tmp16 / tmp18
        tmp20 = tmp17 - tmp19
        tmp21 = tmp15 / tmp18
        tmp22 = 1e-12
        tmp23 = tmp21 + tmp22
        tmp24 = tl.sqrt(tmp23)
        tmp25 = 1 / tmp24
        tmp26 = tmp20 * tmp25
        tl.store(out_ptr4 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp26, xmask & rmask)
    tmp27 = tl.load(seed6 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp35 = tl.load(out_ptr4 + (r2 + (768*x3)), xmask & rmask, eviction_policy='evict_last')
        tmp36 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last')
        tmp38 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last')
        tmp28 = 65535
        tmp29 = tmp27 ^ tmp28
        tmp30 = r2 + (768*x3)
        tmp31 = tl.rand(tmp29, tmp30)
        tmp32 = 0.1
        tmp33 = tmp31 > tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp37 = tmp35 * tmp36
        tmp39 = tmp37 + tmp38
        tmp40 = tmp34 * tmp39
        tmp41 = 1.1111111111111112
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr5 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp42, xmask & rmask)
    tmp43 = 768
    tmp44 = tmp15 / tmp43
    tmp45 = 1e-12
    tmp46 = tmp44 + tmp45
    tmp47 = tl.sqrt(tmp46)
    tmp48 = 1 / tmp47
    tmp49 = tmp48 / tmp43
    tl.store(out_ptr6 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp49, xmask)
''')


kernel1 = async_compile.triton('''
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


kernel2 = async_compile.triton('''
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


kernel3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 6291456 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel4 = async_compile.triton('''
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


kernel5 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 18874368 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 18874368 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


kernel7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 25165824 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 25165824 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 31457280 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel9 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 44040192 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 44040192 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 50331648 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 50331648 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 56623104 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 69206016 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 69206016 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 75497472 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 75497472 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 81788928 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 94371840 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 94371840 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 100663296 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 100663296 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel17 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 106954752 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 119537664 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 119537664 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 125829120 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 125829120 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 132120576 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 144703488 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 144703488 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 150994944 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 150994944 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 157286400 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 169869312 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 169869312 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 176160768 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 176160768 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 182452224 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 195035136 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 195035136 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 201326592 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 201326592 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 207618048 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 220200960 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 220200960 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 226492416 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 226492416 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 232783872 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 245366784 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 245366784 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 251658240 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 251658240 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel35 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 257949696 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
''')


kernel36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 270532608 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 270532608 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
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
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 276824064 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 276824064 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: '*i1', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: 'i32', 17: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, seed1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp21 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp11 = 8.0
        tmp12 = tmp10 / tmp11
        tmp13 = 1.0
        tmp14 = 1
        tmp15 = tmp13 - tmp14
        tmp16 = -3.4028234663852886e+38
        tmp17 = tmp15 * tmp16
        tmp18 = tmp12 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        _tmp21 = tl.where(xmask & rmask, _tmp21 + tmp20, _tmp21)
    tmp21 = tl.reshape(tl.sum(_tmp21, 1), [XBLOCK, 1])
    tmp22 = tl.load(seed1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp23 = 65535
        tmp24 = tmp22 ^ tmp23
        tmp25 = 283115520 + r1 + (128*x0)
        tmp26 = tl.rand(tmp24, tmp25)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = 1.0
        tmp34 = 1
        tmp35 = tmp33 - tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tmp35 * tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp21
        tmp42 = tmp29 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp45 = 6291456 + r1 + (128*x0)
        tmp46 = tl.rand(tmp24, tmp45)
        tmp47 = tmp46 > tmp27
        tmp48 = 31457280 + r1 + (128*x0)
        tmp49 = tl.rand(tmp24, tmp48)
        tmp50 = tmp49 > tmp27
        tmp51 = 56623104 + r1 + (128*x0)
        tmp52 = tl.rand(tmp24, tmp51)
        tmp53 = tmp52 > tmp27
        tmp54 = 81788928 + r1 + (128*x0)
        tmp55 = tl.rand(tmp24, tmp54)
        tmp56 = tmp55 > tmp27
        tmp57 = 106954752 + r1 + (128*x0)
        tmp58 = tl.rand(tmp24, tmp57)
        tmp59 = tmp58 > tmp27
        tmp60 = 132120576 + r1 + (128*x0)
        tmp61 = tl.rand(tmp24, tmp60)
        tmp62 = tmp61 > tmp27
        tmp63 = 157286400 + r1 + (128*x0)
        tmp64 = tl.rand(tmp24, tmp63)
        tmp65 = tmp64 > tmp27
        tmp66 = 182452224 + r1 + (128*x0)
        tmp67 = tl.rand(tmp24, tmp66)
        tmp68 = tmp67 > tmp27
        tmp69 = 207618048 + r1 + (128*x0)
        tmp70 = tl.rand(tmp24, tmp69)
        tmp71 = tmp70 > tmp27
        tmp72 = 232783872 + r1 + (128*x0)
        tmp73 = tl.rand(tmp24, tmp72)
        tmp74 = tmp73 > tmp27
        tmp75 = 257949696 + r1 + (128*x0)
        tmp76 = tl.rand(tmp24, tmp75)
        tmp77 = tmp76 > tmp27
        tl.store(out_ptr2 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp44, xmask & rmask)
        tl.store(out_ptr3 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp41, xmask & rmask)
        tl.store(out_ptr4 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp47, xmask & rmask)
        tl.store(out_ptr5 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp50, xmask & rmask)
        tl.store(out_ptr6 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp53, xmask & rmask)
        tl.store(out_ptr7 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp56, xmask & rmask)
        tl.store(out_ptr8 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp59, xmask & rmask)
        tl.store(out_ptr9 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp62, xmask & rmask)
        tl.store(out_ptr10 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp65, xmask & rmask)
        tl.store(out_ptr11 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp68, xmask & rmask)
        tl.store(out_ptr12 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp71, xmask & rmask)
        tl.store(out_ptr13 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp74, xmask & rmask)
        tl.store(out_ptr14 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp77, xmask & rmask)
        tl.store(out_ptr15 + (r1 + (128*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp28, xmask & rmask)
''')


kernel39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 295698432 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 295698432 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
    tmp61 = 768
    tmp62 = tmp32 / tmp61
    tmp63 = 1e-12
    tmp64 = tmp62 + tmp63
    tmp65 = tl.sqrt(tmp64)
    tmp66 = 1 / tmp65
    tmp67 = tmp66 / tmp61
    tl.store(out_ptr5 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp67, xmask)
''')


kernel40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: '*i1', 17: '*i1', 18: '*i1', 19: '*i1', 20: '*i1', 21: '*i1', 22: '*i1', 23: '*i1', 24: '*i1', 25: '*i1', 26: '*i1', 27: '*i1', 28: '*i1', 29: '*i1', 30: '*i1', 31: '*i1', 32: '*fp32', 33: 'i32', 34: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34), equal_to_1=())]})
@triton.jit
def kernel(seed0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp32 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp33 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp15 = 65535
        tmp16 = tmp0 ^ tmp15
        tmp17 = 301989888 + r1 + (768*x0)
        tmp18 = tl.rand(tmp16, tmp17)
        tmp19 = 0.1
        tmp20 = tmp18 > tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = 768
        tmp29 = tmp14 / tmp28
        tmp30 = tmp27 - tmp29
        tmp31 = tmp30 * tmp30
        _tmp32 = tl.where(xmask & rmask, _tmp32 + tmp31, _tmp32)
        _tmp33 = tl.where(xmask & rmask, _tmp33 + tmp27, _tmp33)
    tmp32 = tl.reshape(tl.sum(_tmp32, 1), [XBLOCK, 1])
    tmp33 = tl.reshape(tl.sum(_tmp33, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp45 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp34 = 65535
        tmp35 = tmp0 ^ tmp34
        tmp36 = 301989888 + r1 + (768*x0)
        tmp37 = tl.rand(tmp35, tmp36)
        tmp38 = 0.1
        tmp39 = tmp37 > tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp42 = tmp40 * tmp41
        tmp43 = 1.1111111111111112
        tmp44 = tmp42 * tmp43
        tmp46 = tmp44 + tmp45
        tmp47 = 768
        tmp48 = tmp33 / tmp47
        tmp49 = tmp46 - tmp48
        tmp50 = tmp32 / tmp47
        tmp51 = 1e-12
        tmp52 = tmp50 + tmp51
        tmp53 = tl.sqrt(tmp52)
        tmp54 = 1 / tmp53
        tmp55 = tmp49 * tmp54
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp55, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last')
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last')
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = 65535
        tmp62 = tmp0 ^ tmp61
        tmp63 = r1 + (768*x0)
        tmp64 = tl.rand(tmp62, tmp63)
        tmp65 = 0.1
        tmp66 = tmp64 > tmp65
        tmp67 = 18874368 + r1 + (768*x0)
        tmp68 = tl.rand(tmp62, tmp67)
        tmp69 = tmp68 > tmp65
        tmp70 = 25165824 + r1 + (768*x0)
        tmp71 = tl.rand(tmp62, tmp70)
        tmp72 = tmp71 > tmp65
        tmp73 = 44040192 + r1 + (768*x0)
        tmp74 = tl.rand(tmp62, tmp73)
        tmp75 = tmp74 > tmp65
        tmp76 = 50331648 + r1 + (768*x0)
        tmp77 = tl.rand(tmp62, tmp76)
        tmp78 = tmp77 > tmp65
        tmp79 = 69206016 + r1 + (768*x0)
        tmp80 = tl.rand(tmp62, tmp79)
        tmp81 = tmp80 > tmp65
        tmp82 = 75497472 + r1 + (768*x0)
        tmp83 = tl.rand(tmp62, tmp82)
        tmp84 = tmp83 > tmp65
        tmp85 = 94371840 + r1 + (768*x0)
        tmp86 = tl.rand(tmp62, tmp85)
        tmp87 = tmp86 > tmp65
        tmp88 = 100663296 + r1 + (768*x0)
        tmp89 = tl.rand(tmp62, tmp88)
        tmp90 = tmp89 > tmp65
        tmp91 = 119537664 + r1 + (768*x0)
        tmp92 = tl.rand(tmp62, tmp91)
        tmp93 = tmp92 > tmp65
        tmp94 = 125829120 + r1 + (768*x0)
        tmp95 = tl.rand(tmp62, tmp94)
        tmp96 = tmp95 > tmp65
        tmp97 = 144703488 + r1 + (768*x0)
        tmp98 = tl.rand(tmp62, tmp97)
        tmp99 = tmp98 > tmp65
        tmp100 = 150994944 + r1 + (768*x0)
        tmp101 = tl.rand(tmp62, tmp100)
        tmp102 = tmp101 > tmp65
        tmp103 = 169869312 + r1 + (768*x0)
        tmp104 = tl.rand(tmp62, tmp103)
        tmp105 = tmp104 > tmp65
        tmp106 = 176160768 + r1 + (768*x0)
        tmp107 = tl.rand(tmp62, tmp106)
        tmp108 = tmp107 > tmp65
        tmp109 = 195035136 + r1 + (768*x0)
        tmp110 = tl.rand(tmp62, tmp109)
        tmp111 = tmp110 > tmp65
        tmp112 = 201326592 + r1 + (768*x0)
        tmp113 = tl.rand(tmp62, tmp112)
        tmp114 = tmp113 > tmp65
        tmp115 = 220200960 + r1 + (768*x0)
        tmp116 = tl.rand(tmp62, tmp115)
        tmp117 = tmp116 > tmp65
        tmp118 = 226492416 + r1 + (768*x0)
        tmp119 = tl.rand(tmp62, tmp118)
        tmp120 = tmp119 > tmp65
        tmp121 = 245366784 + r1 + (768*x0)
        tmp122 = tl.rand(tmp62, tmp121)
        tmp123 = tmp122 > tmp65
        tmp124 = 251658240 + r1 + (768*x0)
        tmp125 = tl.rand(tmp62, tmp124)
        tmp126 = tmp125 > tmp65
        tmp127 = 270532608 + r1 + (768*x0)
        tmp128 = tl.rand(tmp62, tmp127)
        tmp129 = tmp128 > tmp65
        tmp130 = 276824064 + r1 + (768*x0)
        tmp131 = tl.rand(tmp62, tmp130)
        tmp132 = tmp131 > tmp65
        tmp133 = 295698432 + r1 + (768*x0)
        tmp134 = tl.rand(tmp62, tmp133)
        tmp135 = tmp134 > tmp65
        tmp136 = 301989888 + r1 + (768*x0)
        tmp137 = tl.rand(tmp62, tmp136)
        tmp138 = tmp137 > tmp65
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp60, xmask & rmask)
        tl.store(out_ptr5 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp66, xmask & rmask)
        tl.store(out_ptr6 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp69, xmask & rmask)
        tl.store(out_ptr7 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp72, xmask & rmask)
        tl.store(out_ptr8 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp75, xmask & rmask)
        tl.store(out_ptr9 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp78, xmask & rmask)
        tl.store(out_ptr10 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp81, xmask & rmask)
        tl.store(out_ptr11 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp84, xmask & rmask)
        tl.store(out_ptr12 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp87, xmask & rmask)
        tl.store(out_ptr13 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp90, xmask & rmask)
        tl.store(out_ptr14 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp93, xmask & rmask)
        tl.store(out_ptr15 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp96, xmask & rmask)
        tl.store(out_ptr16 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp99, xmask & rmask)
        tl.store(out_ptr17 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp102, xmask & rmask)
        tl.store(out_ptr18 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp105, xmask & rmask)
        tl.store(out_ptr19 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp108, xmask & rmask)
        tl.store(out_ptr20 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp111, xmask & rmask)
        tl.store(out_ptr21 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp114, xmask & rmask)
        tl.store(out_ptr22 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp117, xmask & rmask)
        tl.store(out_ptr23 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp120, xmask & rmask)
        tl.store(out_ptr24 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp123, xmask & rmask)
        tl.store(out_ptr25 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp126, xmask & rmask)
        tl.store(out_ptr26 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp129, xmask & rmask)
        tl.store(out_ptr27 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp132, xmask & rmask)
        tl.store(out_ptr28 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp135, xmask & rmask)
        tl.store(out_ptr29 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp138, xmask & rmask)
    tmp139 = 768
    tmp140 = tmp32 / tmp139
    tmp141 = 1e-12
    tmp142 = tmp140 + tmp141
    tmp143 = tl.sqrt(tmp142)
    tmp144 = 1 / tmp143
    tmp145 = tmp144 / tmp139
    tl.store(out_ptr30 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp145, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp79 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    _tmp80 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp38 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp39 = 0.5
        tmp40 = tmp38 * tmp39
        tmp41 = 0.7071067811865476
        tmp42 = tmp38 * tmp41
        tmp43 = tl.where(tmp42 < 0, -1, 1)
        tmp44 = tl.where(tmp42 == 0, 0, tmp43)
        tmp45 = 1.0
        tmp46 = tl.abs(tmp42)
        tmp47 = 0.3275911
        tmp48 = tmp46 * tmp47
        tmp49 = tmp48 + tmp45
        tmp50 = 1 / tmp49
        tmp51 = tmp50 * tmp45
        tmp52 = 1.061405429
        tmp53 = tmp51 * tmp52
        tmp54 = -1.453152027
        tmp55 = tmp53 + tmp54
        tmp56 = tmp55 * tmp51
        tmp57 = 1.421413741
        tmp58 = tmp56 + tmp57
        tmp59 = tmp58 * tmp51
        tmp60 = -0.284496736
        tmp61 = tmp59 + tmp60
        tmp62 = tmp61 * tmp51
        tmp63 = 0.254829592
        tmp64 = tmp62 + tmp63
        tmp65 = tmp64 * tmp51
        tmp66 = -tmp46
        tmp67 = tmp66 * tmp46
        tmp68 = tl.exp(tmp67)
        tmp69 = tmp65 * tmp68
        tmp70 = tmp45 - tmp69
        tmp71 = tmp44 * tmp70
        tmp72 = 1
        tmp73 = tmp71 + tmp72
        tmp74 = tmp40 * tmp73
        tmp75 = 768
        tmp76 = tmp37 / tmp75
        tmp77 = tmp74 - tmp76
        tmp78 = tmp77 * tmp77
        _tmp79 = tl.where(xmask & rmask, _tmp79 + tmp78, _tmp79)
        _tmp80 = tl.where(xmask & rmask, _tmp80 + tmp74, _tmp80)
    tmp79 = tl.reshape(tl.sum(_tmp79, 1), [XBLOCK, 1])
    tmp80 = tl.reshape(tl.sum(_tmp80, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp81 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp82 = 0.5
        tmp83 = tmp81 * tmp82
        tmp84 = 0.7071067811865476
        tmp85 = tmp81 * tmp84
        tmp86 = tl.where(tmp85 < 0, -1, 1)
        tmp87 = tl.where(tmp85 == 0, 0, tmp86)
        tmp88 = 1.0
        tmp89 = tl.abs(tmp85)
        tmp90 = 0.3275911
        tmp91 = tmp89 * tmp90
        tmp92 = tmp91 + tmp88
        tmp93 = 1 / tmp92
        tmp94 = tmp93 * tmp88
        tmp95 = 1.061405429
        tmp96 = tmp94 * tmp95
        tmp97 = -1.453152027
        tmp98 = tmp96 + tmp97
        tmp99 = tmp98 * tmp94
        tmp100 = 1.421413741
        tmp101 = tmp99 + tmp100
        tmp102 = tmp101 * tmp94
        tmp103 = -0.284496736
        tmp104 = tmp102 + tmp103
        tmp105 = tmp104 * tmp94
        tmp106 = 0.254829592
        tmp107 = tmp105 + tmp106
        tmp108 = tmp107 * tmp94
        tmp109 = -tmp89
        tmp110 = tmp109 * tmp89
        tmp111 = tl.exp(tmp110)
        tmp112 = tmp108 * tmp111
        tmp113 = tmp88 - tmp112
        tmp114 = tmp87 * tmp113
        tmp115 = 1
        tmp116 = tmp114 + tmp115
        tmp117 = tmp83 * tmp116
        tmp118 = 768
        tmp119 = tmp80 / tmp118
        tmp120 = tmp117 - tmp119
        tmp121 = tmp79 / tmp118
        tmp122 = 1e-12
        tmp123 = tmp121 + tmp122
        tmp124 = tl.sqrt(tmp123)
        tmp125 = 1 / tmp124
        tmp126 = tmp120 * tmp125
        tl.store(out_ptr3 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp126, xmask & rmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp127 = tl.load(out_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp128 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp130 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp132 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp129 = tmp127 * tmp128
        tmp131 = tmp129 + tmp130
        tmp133 = 0.7071067811865476
        tmp134 = tmp132 * tmp133
        tmp135 = tl.where(tmp134 < 0, -1, 1)
        tmp136 = tl.where(tmp134 == 0, 0, tmp135)
        tmp137 = 1.0
        tmp138 = tl.abs(tmp134)
        tmp139 = 0.3275911
        tmp140 = tmp138 * tmp139
        tmp141 = tmp140 + tmp137
        tmp142 = 1 / tmp141
        tmp143 = tmp142 * tmp137
        tmp144 = 1.061405429
        tmp145 = tmp143 * tmp144
        tmp146 = -1.453152027
        tmp147 = tmp145 + tmp146
        tmp148 = tmp147 * tmp143
        tmp149 = 1.421413741
        tmp150 = tmp148 + tmp149
        tmp151 = tmp150 * tmp143
        tmp152 = -0.284496736
        tmp153 = tmp151 + tmp152
        tmp154 = tmp153 * tmp143
        tmp155 = 0.254829592
        tmp156 = tmp154 + tmp155
        tmp157 = tmp156 * tmp143
        tmp158 = -tmp138
        tmp159 = tmp158 * tmp138
        tmp160 = tl.exp(tmp159)
        tmp161 = tmp157 * tmp160
        tmp162 = tmp137 - tmp161
        tmp163 = tmp136 * tmp162
        tmp164 = 1
        tmp165 = tmp163 + tmp164
        tmp166 = 0.5
        tmp167 = tmp165 * tmp166
        tmp168 = tmp132 * tmp132
        tmp169 = -0.5
        tmp170 = tmp168 * tmp169
        tmp171 = tl.exp(tmp170)
        tmp172 = 0.3989422804014327
        tmp173 = tmp171 * tmp172
        tmp174 = tmp132 * tmp173
        tmp175 = tmp167 + tmp174
        tl.store(out_ptr4 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp131, xmask & rmask)
        tl.store(out_ptr5 + (r1 + (768*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp175, xmask & rmask)
    tmp176 = 768
    tmp177 = tmp79 / tmp176
    tmp178 = 1e-12
    tmp179 = tmp177 + tmp178
    tmp180 = tl.sqrt(tmp179)
    tmp181 = 1 / tmp180
    tmp182 = tmp181 / tmp176
    tl.store(out_ptr6 + (x0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp182, xmask)
''')


kernel42 = async_compile.triton('''
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
def kernel(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp2 = tl.load(in_ptr0 + (r1 + (30522*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tmp2 - tmp1
        tmp4 = tl.exp(tmp3)
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (30522*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tmp6 - tmp1
        tmp8 = tl.log(tmp5)
        tmp9 = tmp7 - tmp8
        tl.store(out_ptr2 + (r1 + (30522*x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp9, xmask & rmask)
''')


kernel43 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 8192],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8192
    out_ptr1 = in_out_ptr0
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
    tmp4 = 8192
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr1 + (0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp5, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    torch.randint(2**31, size=(), dtype=torch.int64, out=seed_cuda_0)
    buf0 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf4 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf5 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf417 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(primals_205, primals_1, primals_203, primals_2, primals_204, primals_3, seed_cuda_0, primals_4, primals_5, buf0, buf4, buf5, buf417, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_2
    del primals_3
    del primals_5
    buf6 = as_strided(buf0, (8192, 768), (768, 1)); del buf0  # reuse
    aten.addmm.out(primals_7, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_6, (768, 768), (1, 768)), beta=1, alpha=1, out=buf6)
    del primals_7
    buf7 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_9, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_8, (768, 768), (1, 768)), beta=1, alpha=1, out=buf7)
    del primals_9
    buf8 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_11, as_strided(buf5, (8192, 768), (768, 1)), as_strided(primals_10, (768, 768), (1, 768)), beta=1, alpha=1, out=buf8)
    del primals_11
    buf9 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf6, buf9, 6291456, grid=grid(6291456), stream=stream0)
    buf10 = as_strided(buf6, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf6  # reuse
    kernel2.run(buf7, buf10, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf11 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf9, (768, 128, 64), (8192, 64, 1)), as_strided(buf10, (768, 64, 128), (8192, 128, 1)), out=buf11)
    buf14 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf416 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel3.run(buf11, seed_cuda_0, buf14, buf416, 98304, 128, grid=grid(98304), stream=stream0)
    buf15 = as_strided(buf7, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf7  # reuse
    kernel1.run(buf8, buf15, 6291456, grid=grid(6291456), stream=stream0)
    buf16 = as_strided(buf8, (768, 128, 64), (8192, 64, 1)); del buf8  # reuse
    aten.bmm.out(as_strided(buf14, (768, 128, 128), (16384, 128, 1)), as_strided(buf15, (768, 128, 64), (8192, 64, 1)), out=buf16)
    buf17 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf16, buf17, 6291456, grid=grid(6291456), stream=stream0)
    buf18 = as_strided(buf16, (8192, 768), (768, 1)); del buf16  # reuse
    aten.addmm.out(primals_13, as_strided(buf17, (8192, 768), (768, 1)), as_strided(primals_12, (768, 768), (1, 768)), beta=1, alpha=1, out=buf18)
    del primals_13
    buf22 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf23 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf415 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel5.run(seed_cuda_0, buf18, buf5, primals_14, primals_15, buf22, buf23, buf415, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_15
    buf24 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_17, as_strided(buf23, (8192, 768), (768, 1)), as_strided(primals_16, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf24)
    del primals_17
    buf25 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf414 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf24, buf25, buf414, 25165824, grid=grid(25165824), stream=stream0)
    buf26 = buf18; del buf18  # reuse
    aten.addmm.out(primals_19, as_strided(buf25, (8192, 3072), (3072, 1)), as_strided(primals_18, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf26)
    del primals_19
    buf30 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf31 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf413 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel7.run(seed_cuda_0, buf26, buf23, primals_20, primals_21, buf30, buf31, buf413, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_21
    buf32 = buf26; del buf26  # reuse
    aten.addmm.out(primals_23, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_22, (768, 768), (1, 768)), beta=1, alpha=1, out=buf32)
    del primals_23
    buf33 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_25, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_24, (768, 768), (1, 768)), beta=1, alpha=1, out=buf33)
    del primals_25
    buf34 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_27, as_strided(buf31, (8192, 768), (768, 1)), as_strided(primals_26, (768, 768), (1, 768)), beta=1, alpha=1, out=buf34)
    del primals_27
    buf35 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf32, buf35, 6291456, grid=grid(6291456), stream=stream0)
    buf36 = as_strided(buf32, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf32  # reuse
    kernel2.run(buf33, buf36, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf37 = buf11; del buf11  # reuse
    aten.bmm.out(as_strided(buf35, (768, 128, 64), (8192, 64, 1)), as_strided(buf36, (768, 64, 128), (8192, 128, 1)), out=buf37)
    buf40 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf412 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf37, seed_cuda_0, buf40, buf412, 98304, 128, grid=grid(98304), stream=stream0)
    buf41 = as_strided(buf33, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf33  # reuse
    kernel1.run(buf34, buf41, 6291456, grid=grid(6291456), stream=stream0)
    buf42 = as_strided(buf34, (768, 128, 64), (8192, 64, 1)); del buf34  # reuse
    aten.bmm.out(as_strided(buf40, (768, 128, 128), (16384, 128, 1)), as_strided(buf41, (768, 128, 64), (8192, 64, 1)), out=buf42)
    buf43 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf42, buf43, 6291456, grid=grid(6291456), stream=stream0)
    buf44 = as_strided(buf42, (8192, 768), (768, 1)); del buf42  # reuse
    aten.addmm.out(primals_29, as_strided(buf43, (8192, 768), (768, 1)), as_strided(primals_28, (768, 768), (1, 768)), beta=1, alpha=1, out=buf44)
    del primals_29
    buf48 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf49 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf411 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel9.run(seed_cuda_0, buf44, buf31, primals_30, primals_31, buf48, buf49, buf411, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_31
    buf50 = buf24; del buf24  # reuse
    aten.addmm.out(primals_33, as_strided(buf49, (8192, 768), (768, 1)), as_strided(primals_32, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf50)
    del primals_33
    buf51 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf410 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf50, buf51, buf410, 25165824, grid=grid(25165824), stream=stream0)
    buf52 = buf44; del buf44  # reuse
    aten.addmm.out(primals_35, as_strided(buf51, (8192, 3072), (3072, 1)), as_strided(primals_34, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf52)
    del primals_35
    buf56 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf57 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf409 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel10.run(seed_cuda_0, buf52, buf49, primals_36, primals_37, buf56, buf57, buf409, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_37
    buf58 = buf52; del buf52  # reuse
    aten.addmm.out(primals_39, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_38, (768, 768), (1, 768)), beta=1, alpha=1, out=buf58)
    del primals_39
    buf59 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_41, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_40, (768, 768), (1, 768)), beta=1, alpha=1, out=buf59)
    del primals_41
    buf60 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_43, as_strided(buf57, (8192, 768), (768, 1)), as_strided(primals_42, (768, 768), (1, 768)), beta=1, alpha=1, out=buf60)
    del primals_43
    buf61 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf58, buf61, 6291456, grid=grid(6291456), stream=stream0)
    buf62 = as_strided(buf58, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf58  # reuse
    kernel2.run(buf59, buf62, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf63 = buf37; del buf37  # reuse
    aten.bmm.out(as_strided(buf61, (768, 128, 64), (8192, 64, 1)), as_strided(buf62, (768, 64, 128), (8192, 128, 1)), out=buf63)
    buf66 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf408 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel11.run(buf63, seed_cuda_0, buf66, buf408, 98304, 128, grid=grid(98304), stream=stream0)
    buf67 = as_strided(buf59, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf59  # reuse
    kernel1.run(buf60, buf67, 6291456, grid=grid(6291456), stream=stream0)
    buf68 = as_strided(buf60, (768, 128, 64), (8192, 64, 1)); del buf60  # reuse
    aten.bmm.out(as_strided(buf66, (768, 128, 128), (16384, 128, 1)), as_strided(buf67, (768, 128, 64), (8192, 64, 1)), out=buf68)
    buf69 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf68, buf69, 6291456, grid=grid(6291456), stream=stream0)
    buf70 = as_strided(buf68, (8192, 768), (768, 1)); del buf68  # reuse
    aten.addmm.out(primals_45, as_strided(buf69, (8192, 768), (768, 1)), as_strided(primals_44, (768, 768), (1, 768)), beta=1, alpha=1, out=buf70)
    del primals_45
    buf74 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf75 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf407 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel12.run(seed_cuda_0, buf70, buf57, primals_46, primals_47, buf74, buf75, buf407, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_47
    buf76 = buf50; del buf50  # reuse
    aten.addmm.out(primals_49, as_strided(buf75, (8192, 768), (768, 1)), as_strided(primals_48, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf76)
    del primals_49
    buf77 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf406 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf76, buf77, buf406, 25165824, grid=grid(25165824), stream=stream0)
    buf78 = buf70; del buf70  # reuse
    aten.addmm.out(primals_51, as_strided(buf77, (8192, 3072), (3072, 1)), as_strided(primals_50, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf78)
    del primals_51
    buf82 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf83 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf405 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel13.run(seed_cuda_0, buf78, buf75, primals_52, primals_53, buf82, buf83, buf405, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_53
    buf84 = buf78; del buf78  # reuse
    aten.addmm.out(primals_55, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_54, (768, 768), (1, 768)), beta=1, alpha=1, out=buf84)
    del primals_55
    buf85 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_57, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_56, (768, 768), (1, 768)), beta=1, alpha=1, out=buf85)
    del primals_57
    buf86 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_59, as_strided(buf83, (8192, 768), (768, 1)), as_strided(primals_58, (768, 768), (1, 768)), beta=1, alpha=1, out=buf86)
    del primals_59
    buf87 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf84, buf87, 6291456, grid=grid(6291456), stream=stream0)
    buf88 = as_strided(buf84, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf84  # reuse
    kernel2.run(buf85, buf88, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf89 = buf63; del buf63  # reuse
    aten.bmm.out(as_strided(buf87, (768, 128, 64), (8192, 64, 1)), as_strided(buf88, (768, 64, 128), (8192, 128, 1)), out=buf89)
    buf92 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf404 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel14.run(buf89, seed_cuda_0, buf92, buf404, 98304, 128, grid=grid(98304), stream=stream0)
    buf93 = as_strided(buf85, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf85  # reuse
    kernel1.run(buf86, buf93, 6291456, grid=grid(6291456), stream=stream0)
    buf94 = as_strided(buf86, (768, 128, 64), (8192, 64, 1)); del buf86  # reuse
    aten.bmm.out(as_strided(buf92, (768, 128, 128), (16384, 128, 1)), as_strided(buf93, (768, 128, 64), (8192, 64, 1)), out=buf94)
    buf95 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf94, buf95, 6291456, grid=grid(6291456), stream=stream0)
    buf96 = as_strided(buf94, (8192, 768), (768, 1)); del buf94  # reuse
    aten.addmm.out(primals_61, as_strided(buf95, (8192, 768), (768, 1)), as_strided(primals_60, (768, 768), (1, 768)), beta=1, alpha=1, out=buf96)
    del primals_61
    buf100 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf101 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf403 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel15.run(seed_cuda_0, buf96, buf83, primals_62, primals_63, buf100, buf101, buf403, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_63
    buf102 = buf76; del buf76  # reuse
    aten.addmm.out(primals_65, as_strided(buf101, (8192, 768), (768, 1)), as_strided(primals_64, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf102)
    del primals_65
    buf103 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf402 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf102, buf103, buf402, 25165824, grid=grid(25165824), stream=stream0)
    buf104 = buf96; del buf96  # reuse
    aten.addmm.out(primals_67, as_strided(buf103, (8192, 3072), (3072, 1)), as_strided(primals_66, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf104)
    del primals_67
    buf108 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf109 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf401 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel16.run(seed_cuda_0, buf104, buf101, primals_68, primals_69, buf108, buf109, buf401, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_69
    buf110 = buf104; del buf104  # reuse
    aten.addmm.out(primals_71, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_70, (768, 768), (1, 768)), beta=1, alpha=1, out=buf110)
    del primals_71
    buf111 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_73, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_72, (768, 768), (1, 768)), beta=1, alpha=1, out=buf111)
    del primals_73
    buf112 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_75, as_strided(buf109, (8192, 768), (768, 1)), as_strided(primals_74, (768, 768), (1, 768)), beta=1, alpha=1, out=buf112)
    del primals_75
    buf113 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf110, buf113, 6291456, grid=grid(6291456), stream=stream0)
    buf114 = as_strided(buf110, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf110  # reuse
    kernel2.run(buf111, buf114, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf115 = buf89; del buf89  # reuse
    aten.bmm.out(as_strided(buf113, (768, 128, 64), (8192, 64, 1)), as_strided(buf114, (768, 64, 128), (8192, 128, 1)), out=buf115)
    buf118 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf400 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel17.run(buf115, seed_cuda_0, buf118, buf400, 98304, 128, grid=grid(98304), stream=stream0)
    buf119 = as_strided(buf111, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf111  # reuse
    kernel1.run(buf112, buf119, 6291456, grid=grid(6291456), stream=stream0)
    buf120 = as_strided(buf112, (768, 128, 64), (8192, 64, 1)); del buf112  # reuse
    aten.bmm.out(as_strided(buf118, (768, 128, 128), (16384, 128, 1)), as_strided(buf119, (768, 128, 64), (8192, 64, 1)), out=buf120)
    buf121 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf120, buf121, 6291456, grid=grid(6291456), stream=stream0)
    buf122 = as_strided(buf120, (8192, 768), (768, 1)); del buf120  # reuse
    aten.addmm.out(primals_77, as_strided(buf121, (8192, 768), (768, 1)), as_strided(primals_76, (768, 768), (1, 768)), beta=1, alpha=1, out=buf122)
    del primals_77
    buf126 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf127 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf399 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel18.run(seed_cuda_0, buf122, buf109, primals_78, primals_79, buf126, buf127, buf399, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_79
    buf128 = buf102; del buf102  # reuse
    aten.addmm.out(primals_81, as_strided(buf127, (8192, 768), (768, 1)), as_strided(primals_80, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf128)
    del primals_81
    buf129 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf398 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf128, buf129, buf398, 25165824, grid=grid(25165824), stream=stream0)
    buf130 = buf122; del buf122  # reuse
    aten.addmm.out(primals_83, as_strided(buf129, (8192, 3072), (3072, 1)), as_strided(primals_82, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf130)
    del primals_83
    buf134 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf135 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf397 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel19.run(seed_cuda_0, buf130, buf127, primals_84, primals_85, buf134, buf135, buf397, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_85
    buf136 = buf130; del buf130  # reuse
    aten.addmm.out(primals_87, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_86, (768, 768), (1, 768)), beta=1, alpha=1, out=buf136)
    del primals_87
    buf137 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_89, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_88, (768, 768), (1, 768)), beta=1, alpha=1, out=buf137)
    del primals_89
    buf138 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_91, as_strided(buf135, (8192, 768), (768, 1)), as_strided(primals_90, (768, 768), (1, 768)), beta=1, alpha=1, out=buf138)
    del primals_91
    buf139 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf136, buf139, 6291456, grid=grid(6291456), stream=stream0)
    buf140 = as_strided(buf136, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf136  # reuse
    kernel2.run(buf137, buf140, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf141 = buf115; del buf115  # reuse
    aten.bmm.out(as_strided(buf139, (768, 128, 64), (8192, 64, 1)), as_strided(buf140, (768, 64, 128), (8192, 128, 1)), out=buf141)
    buf144 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf396 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel20.run(buf141, seed_cuda_0, buf144, buf396, 98304, 128, grid=grid(98304), stream=stream0)
    buf145 = as_strided(buf137, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf137  # reuse
    kernel1.run(buf138, buf145, 6291456, grid=grid(6291456), stream=stream0)
    buf146 = as_strided(buf138, (768, 128, 64), (8192, 64, 1)); del buf138  # reuse
    aten.bmm.out(as_strided(buf144, (768, 128, 128), (16384, 128, 1)), as_strided(buf145, (768, 128, 64), (8192, 64, 1)), out=buf146)
    buf147 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf146, buf147, 6291456, grid=grid(6291456), stream=stream0)
    buf148 = as_strided(buf146, (8192, 768), (768, 1)); del buf146  # reuse
    aten.addmm.out(primals_93, as_strided(buf147, (8192, 768), (768, 1)), as_strided(primals_92, (768, 768), (1, 768)), beta=1, alpha=1, out=buf148)
    del primals_93
    buf152 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf153 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf395 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel21.run(seed_cuda_0, buf148, buf135, primals_94, primals_95, buf152, buf153, buf395, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_95
    buf154 = buf128; del buf128  # reuse
    aten.addmm.out(primals_97, as_strided(buf153, (8192, 768), (768, 1)), as_strided(primals_96, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf154)
    del primals_97
    buf155 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf394 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf154, buf155, buf394, 25165824, grid=grid(25165824), stream=stream0)
    buf156 = buf148; del buf148  # reuse
    aten.addmm.out(primals_99, as_strided(buf155, (8192, 3072), (3072, 1)), as_strided(primals_98, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf156)
    del primals_99
    buf160 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf161 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf393 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel22.run(seed_cuda_0, buf156, buf153, primals_100, primals_101, buf160, buf161, buf393, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_101
    buf162 = buf156; del buf156  # reuse
    aten.addmm.out(primals_103, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_102, (768, 768), (1, 768)), beta=1, alpha=1, out=buf162)
    del primals_103
    buf163 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_105, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_104, (768, 768), (1, 768)), beta=1, alpha=1, out=buf163)
    del primals_105
    buf164 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_107, as_strided(buf161, (8192, 768), (768, 1)), as_strided(primals_106, (768, 768), (1, 768)), beta=1, alpha=1, out=buf164)
    del primals_107
    buf165 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf162, buf165, 6291456, grid=grid(6291456), stream=stream0)
    buf166 = as_strided(buf162, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf162  # reuse
    kernel2.run(buf163, buf166, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf167 = buf141; del buf141  # reuse
    aten.bmm.out(as_strided(buf165, (768, 128, 64), (8192, 64, 1)), as_strided(buf166, (768, 64, 128), (8192, 128, 1)), out=buf167)
    buf170 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf392 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel23.run(buf167, seed_cuda_0, buf170, buf392, 98304, 128, grid=grid(98304), stream=stream0)
    buf171 = as_strided(buf163, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf163  # reuse
    kernel1.run(buf164, buf171, 6291456, grid=grid(6291456), stream=stream0)
    buf172 = as_strided(buf164, (768, 128, 64), (8192, 64, 1)); del buf164  # reuse
    aten.bmm.out(as_strided(buf170, (768, 128, 128), (16384, 128, 1)), as_strided(buf171, (768, 128, 64), (8192, 64, 1)), out=buf172)
    buf173 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf172, buf173, 6291456, grid=grid(6291456), stream=stream0)
    buf174 = as_strided(buf172, (8192, 768), (768, 1)); del buf172  # reuse
    aten.addmm.out(primals_109, as_strided(buf173, (8192, 768), (768, 1)), as_strided(primals_108, (768, 768), (1, 768)), beta=1, alpha=1, out=buf174)
    del primals_109
    buf178 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf179 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf391 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel24.run(seed_cuda_0, buf174, buf161, primals_110, primals_111, buf178, buf179, buf391, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_111
    buf180 = buf154; del buf154  # reuse
    aten.addmm.out(primals_113, as_strided(buf179, (8192, 768), (768, 1)), as_strided(primals_112, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf180)
    del primals_113
    buf181 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf390 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf180, buf181, buf390, 25165824, grid=grid(25165824), stream=stream0)
    buf182 = buf174; del buf174  # reuse
    aten.addmm.out(primals_115, as_strided(buf181, (8192, 3072), (3072, 1)), as_strided(primals_114, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf182)
    del primals_115
    buf186 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf187 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf389 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel25.run(seed_cuda_0, buf182, buf179, primals_116, primals_117, buf186, buf187, buf389, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_117
    buf188 = buf182; del buf182  # reuse
    aten.addmm.out(primals_119, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_118, (768, 768), (1, 768)), beta=1, alpha=1, out=buf188)
    del primals_119
    buf189 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_121, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_120, (768, 768), (1, 768)), beta=1, alpha=1, out=buf189)
    del primals_121
    buf190 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_123, as_strided(buf187, (8192, 768), (768, 1)), as_strided(primals_122, (768, 768), (1, 768)), beta=1, alpha=1, out=buf190)
    del primals_123
    buf191 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf188, buf191, 6291456, grid=grid(6291456), stream=stream0)
    buf192 = as_strided(buf188, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf188  # reuse
    kernel2.run(buf189, buf192, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf193 = buf167; del buf167  # reuse
    aten.bmm.out(as_strided(buf191, (768, 128, 64), (8192, 64, 1)), as_strided(buf192, (768, 64, 128), (8192, 128, 1)), out=buf193)
    buf196 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf388 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel26.run(buf193, seed_cuda_0, buf196, buf388, 98304, 128, grid=grid(98304), stream=stream0)
    buf197 = as_strided(buf189, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf189  # reuse
    kernel1.run(buf190, buf197, 6291456, grid=grid(6291456), stream=stream0)
    buf198 = as_strided(buf190, (768, 128, 64), (8192, 64, 1)); del buf190  # reuse
    aten.bmm.out(as_strided(buf196, (768, 128, 128), (16384, 128, 1)), as_strided(buf197, (768, 128, 64), (8192, 64, 1)), out=buf198)
    buf199 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf198, buf199, 6291456, grid=grid(6291456), stream=stream0)
    buf200 = as_strided(buf198, (8192, 768), (768, 1)); del buf198  # reuse
    aten.addmm.out(primals_125, as_strided(buf199, (8192, 768), (768, 1)), as_strided(primals_124, (768, 768), (1, 768)), beta=1, alpha=1, out=buf200)
    del primals_125
    buf204 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf205 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf387 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel27.run(seed_cuda_0, buf200, buf187, primals_126, primals_127, buf204, buf205, buf387, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_127
    buf206 = buf180; del buf180  # reuse
    aten.addmm.out(primals_129, as_strided(buf205, (8192, 768), (768, 1)), as_strided(primals_128, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf206)
    del primals_129
    buf207 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf386 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf206, buf207, buf386, 25165824, grid=grid(25165824), stream=stream0)
    buf208 = buf200; del buf200  # reuse
    aten.addmm.out(primals_131, as_strided(buf207, (8192, 3072), (3072, 1)), as_strided(primals_130, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf208)
    del primals_131
    buf212 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf213 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf385 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel28.run(seed_cuda_0, buf208, buf205, primals_132, primals_133, buf212, buf213, buf385, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_133
    buf214 = buf208; del buf208  # reuse
    aten.addmm.out(primals_135, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_134, (768, 768), (1, 768)), beta=1, alpha=1, out=buf214)
    del primals_135
    buf215 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_137, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_136, (768, 768), (1, 768)), beta=1, alpha=1, out=buf215)
    del primals_137
    buf216 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_139, as_strided(buf213, (8192, 768), (768, 1)), as_strided(primals_138, (768, 768), (1, 768)), beta=1, alpha=1, out=buf216)
    del primals_139
    buf217 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf214, buf217, 6291456, grid=grid(6291456), stream=stream0)
    buf218 = as_strided(buf214, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf214  # reuse
    kernel2.run(buf215, buf218, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf219 = buf193; del buf193  # reuse
    aten.bmm.out(as_strided(buf217, (768, 128, 64), (8192, 64, 1)), as_strided(buf218, (768, 64, 128), (8192, 128, 1)), out=buf219)
    buf222 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf384 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel29.run(buf219, seed_cuda_0, buf222, buf384, 98304, 128, grid=grid(98304), stream=stream0)
    buf223 = as_strided(buf215, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf215  # reuse
    kernel1.run(buf216, buf223, 6291456, grid=grid(6291456), stream=stream0)
    buf224 = as_strided(buf216, (768, 128, 64), (8192, 64, 1)); del buf216  # reuse
    aten.bmm.out(as_strided(buf222, (768, 128, 128), (16384, 128, 1)), as_strided(buf223, (768, 128, 64), (8192, 64, 1)), out=buf224)
    buf225 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf224, buf225, 6291456, grid=grid(6291456), stream=stream0)
    buf226 = as_strided(buf224, (8192, 768), (768, 1)); del buf224  # reuse
    aten.addmm.out(primals_141, as_strided(buf225, (8192, 768), (768, 1)), as_strided(primals_140, (768, 768), (1, 768)), beta=1, alpha=1, out=buf226)
    del primals_141
    buf230 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf231 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf383 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel30.run(seed_cuda_0, buf226, buf213, primals_142, primals_143, buf230, buf231, buf383, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_143
    buf232 = buf206; del buf206  # reuse
    aten.addmm.out(primals_145, as_strided(buf231, (8192, 768), (768, 1)), as_strided(primals_144, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf232)
    del primals_145
    buf233 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf382 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf232, buf233, buf382, 25165824, grid=grid(25165824), stream=stream0)
    buf234 = buf226; del buf226  # reuse
    aten.addmm.out(primals_147, as_strided(buf233, (8192, 3072), (3072, 1)), as_strided(primals_146, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf234)
    del primals_147
    buf238 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf239 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf381 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel31.run(seed_cuda_0, buf234, buf231, primals_148, primals_149, buf238, buf239, buf381, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_149
    buf240 = buf234; del buf234  # reuse
    aten.addmm.out(primals_151, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_150, (768, 768), (1, 768)), beta=1, alpha=1, out=buf240)
    del primals_151
    buf241 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_153, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_152, (768, 768), (1, 768)), beta=1, alpha=1, out=buf241)
    del primals_153
    buf242 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_155, as_strided(buf239, (8192, 768), (768, 1)), as_strided(primals_154, (768, 768), (1, 768)), beta=1, alpha=1, out=buf242)
    del primals_155
    buf243 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf240, buf243, 6291456, grid=grid(6291456), stream=stream0)
    buf244 = as_strided(buf240, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf240  # reuse
    kernel2.run(buf241, buf244, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf245 = buf219; del buf219  # reuse
    aten.bmm.out(as_strided(buf243, (768, 128, 64), (8192, 64, 1)), as_strided(buf244, (768, 64, 128), (8192, 128, 1)), out=buf245)
    buf248 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf380 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel32.run(buf245, seed_cuda_0, buf248, buf380, 98304, 128, grid=grid(98304), stream=stream0)
    buf249 = as_strided(buf241, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf241  # reuse
    kernel1.run(buf242, buf249, 6291456, grid=grid(6291456), stream=stream0)
    buf250 = as_strided(buf242, (768, 128, 64), (8192, 64, 1)); del buf242  # reuse
    aten.bmm.out(as_strided(buf248, (768, 128, 128), (16384, 128, 1)), as_strided(buf249, (768, 128, 64), (8192, 64, 1)), out=buf250)
    buf251 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf250, buf251, 6291456, grid=grid(6291456), stream=stream0)
    buf252 = as_strided(buf250, (8192, 768), (768, 1)); del buf250  # reuse
    aten.addmm.out(primals_157, as_strided(buf251, (8192, 768), (768, 1)), as_strided(primals_156, (768, 768), (1, 768)), beta=1, alpha=1, out=buf252)
    del primals_157
    buf256 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf257 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf379 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel33.run(seed_cuda_0, buf252, buf239, primals_158, primals_159, buf256, buf257, buf379, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_159
    buf258 = buf232; del buf232  # reuse
    aten.addmm.out(primals_161, as_strided(buf257, (8192, 768), (768, 1)), as_strided(primals_160, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf258)
    del primals_161
    buf259 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf378 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf258, buf259, buf378, 25165824, grid=grid(25165824), stream=stream0)
    buf260 = buf252; del buf252  # reuse
    aten.addmm.out(primals_163, as_strided(buf259, (8192, 3072), (3072, 1)), as_strided(primals_162, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf260)
    del primals_163
    buf264 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf265 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf377 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel34.run(seed_cuda_0, buf260, buf257, primals_164, primals_165, buf264, buf265, buf377, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_165
    buf266 = buf260; del buf260  # reuse
    aten.addmm.out(primals_167, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_166, (768, 768), (1, 768)), beta=1, alpha=1, out=buf266)
    del primals_167
    buf267 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_169, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_168, (768, 768), (1, 768)), beta=1, alpha=1, out=buf267)
    del primals_169
    buf268 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_171, as_strided(buf265, (8192, 768), (768, 1)), as_strided(primals_170, (768, 768), (1, 768)), beta=1, alpha=1, out=buf268)
    del primals_171
    buf269 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf266, buf269, 6291456, grid=grid(6291456), stream=stream0)
    buf270 = as_strided(buf266, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf266  # reuse
    kernel2.run(buf267, buf270, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf271 = buf245; del buf245  # reuse
    aten.bmm.out(as_strided(buf269, (768, 128, 64), (8192, 64, 1)), as_strided(buf270, (768, 64, 128), (8192, 128, 1)), out=buf271)
    buf274 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf376 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    kernel35.run(buf271, seed_cuda_0, buf274, buf376, 98304, 128, grid=grid(98304), stream=stream0)
    buf275 = as_strided(buf267, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf267  # reuse
    kernel1.run(buf268, buf275, 6291456, grid=grid(6291456), stream=stream0)
    buf276 = as_strided(buf268, (768, 128, 64), (8192, 64, 1)); del buf268  # reuse
    aten.bmm.out(as_strided(buf274, (768, 128, 128), (16384, 128, 1)), as_strided(buf275, (768, 128, 64), (8192, 64, 1)), out=buf276)
    buf277 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf276, buf277, 6291456, grid=grid(6291456), stream=stream0)
    buf278 = as_strided(buf276, (8192, 768), (768, 1)); del buf276  # reuse
    aten.addmm.out(primals_173, as_strided(buf277, (8192, 768), (768, 1)), as_strided(primals_172, (768, 768), (1, 768)), beta=1, alpha=1, out=buf278)
    del primals_173
    buf282 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf283 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf375 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel36.run(seed_cuda_0, buf278, buf265, primals_174, primals_175, buf282, buf283, buf375, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_175
    buf284 = buf258; del buf258  # reuse
    aten.addmm.out(primals_177, as_strided(buf283, (8192, 768), (768, 1)), as_strided(primals_176, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf284)
    del primals_177
    buf285 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf374 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf284, buf285, buf374, 25165824, grid=grid(25165824), stream=stream0)
    buf286 = buf278; del buf278  # reuse
    aten.addmm.out(primals_179, as_strided(buf285, (8192, 3072), (3072, 1)), as_strided(primals_178, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf286)
    del primals_179
    buf290 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf291 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf373 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel37.run(seed_cuda_0, buf286, buf283, primals_180, primals_181, buf290, buf291, buf373, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_181
    buf292 = buf286; del buf286  # reuse
    aten.addmm.out(primals_183, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_182, (768, 768), (1, 768)), beta=1, alpha=1, out=buf292)
    del primals_183
    buf293 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_185, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_184, (768, 768), (1, 768)), beta=1, alpha=1, out=buf293)
    del primals_185
    buf294 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_187, as_strided(buf291, (8192, 768), (768, 1)), as_strided(primals_186, (768, 768), (1, 768)), beta=1, alpha=1, out=buf294)
    del primals_187
    buf295 = empty_strided((64, 12, 128, 64), (98304, 8192, 64, 1), device='cuda', dtype=torch.float32)
    kernel1.run(buf292, buf295, 6291456, grid=grid(6291456), stream=stream0)
    buf296 = as_strided(buf292, (64, 12, 64, 128), (98304, 8192, 128, 1)); del buf292  # reuse
    kernel2.run(buf293, buf296, 49152, 128, grid=grid(49152, 128), stream=stream0)
    buf297 = buf271; del buf271  # reuse
    aten.bmm.out(as_strided(buf295, (768, 128, 64), (8192, 64, 1)), as_strided(buf296, (768, 64, 128), (8192, 128, 1)), out=buf297)
    buf300 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf372 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    buf331 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf334 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf337 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf340 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf343 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf346 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf349 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf352 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf355 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf358 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf361 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    buf364 = empty_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    kernel38.run(buf297, seed_cuda_0, buf300, buf372, buf331, buf334, buf337, buf340, buf343, buf346, buf349, buf352, buf355, buf358, buf361, buf364, 98304, 128, grid=grid(98304), stream=stream0)
    del buf297
    buf301 = as_strided(buf293, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf293  # reuse
    kernel1.run(buf294, buf301, 6291456, grid=grid(6291456), stream=stream0)
    buf302 = as_strided(buf294, (768, 128, 64), (8192, 64, 1)); del buf294  # reuse
    aten.bmm.out(as_strided(buf300, (768, 128, 128), (16384, 128, 1)), as_strided(buf301, (768, 128, 64), (8192, 64, 1)), out=buf302)
    buf303 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf302, buf303, 6291456, grid=grid(6291456), stream=stream0)
    buf304 = as_strided(buf302, (8192, 768), (768, 1)); del buf302  # reuse
    aten.addmm.out(primals_189, as_strided(buf303, (8192, 768), (768, 1)), as_strided(primals_188, (768, 768), (1, 768)), beta=1, alpha=1, out=buf304)
    del primals_189
    buf308 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf309 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf371 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel39.run(seed_cuda_0, buf304, buf291, primals_190, primals_191, buf308, buf309, buf371, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_191
    buf310 = buf284; del buf284  # reuse
    aten.addmm.out(primals_193, as_strided(buf309, (8192, 768), (768, 1)), as_strided(primals_192, (768, 3072), (1, 768)), beta=1, alpha=1, out=buf310)
    del primals_193
    buf311 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    buf370 = empty_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    kernel6.run(buf310, buf311, buf370, 25165824, grid=grid(25165824), stream=stream0)
    del buf310
    buf312 = buf304; del buf304  # reuse
    aten.addmm.out(primals_195, as_strided(buf311, (8192, 3072), (3072, 1)), as_strided(primals_194, (3072, 768), (1, 3072)), beta=1, alpha=1, out=buf312)
    del primals_195
    buf316 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf317 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf330 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf332 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf333 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf335 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf336 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf338 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf339 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf341 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf342 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf344 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf345 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf347 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf348 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf350 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf351 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf353 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf354 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf356 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf357 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf359 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf360 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf362 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf363 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf365 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf366 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    buf369 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel40.run(seed_cuda_0, buf312, buf309, primals_196, primals_197, buf316, buf317, buf330, buf332, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf344, buf345, buf347, buf348, buf350, buf351, buf353, buf354, buf356, buf357, buf359, buf360, buf362, buf363, buf365, buf366, buf369, 8192, 768, grid=grid(8192), stream=stream0)
    del primals_197
    buf318 = buf312; del buf312  # reuse
    aten.addmm.out(primals_199, as_strided(buf317, (8192, 768), (768, 1)), as_strided(primals_198, (768, 768), (1, 768)), beta=1, alpha=1, out=buf318)
    del primals_199
    buf322 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf323 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf368 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    buf367 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel41.run(buf318, primals_200, primals_201, buf322, buf323, buf368, buf367, 8192, 768, grid=grid(8192), stream=stream0)
    del buf318
    del primals_201
    buf324 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    aten.addmm.out(primals_202, as_strided(buf323, (8192, 768), (768, 1)), as_strided(primals_1, (768, 30522), (1, 768)), beta=1, alpha=1, out=buf324)
    del primals_202
    buf327 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    kernel42.run(buf324, buf327, 8192, 30522, grid=grid(8192), stream=stream0)
    buf328 = empty_strided((), (), device='cuda', dtype=torch.float32)
    buf329 = buf328; del buf328  # reuse
    kernel43.run(buf329, primals_206, buf327, 1, 8192, grid=grid(1), stream=stream0)
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
