
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

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250036224
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

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = -1.0
    tl.store(out_ptr0 + (tmp0 + (30522*x0) + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


kernel2 = async_compile.triton('''
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
    tmp1 = tl.load(in_ptr1 + (0 + tl.zeros([XBLOCK, RBLOCK], tl.int32)), None)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30522*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = 8192.0
        tmp3 = tmp1 / tmp2
        tmp4 = tmp0 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp5, xmask)
''')


kernel3 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250036224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 30522)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (0 + tl.zeros([XBLOCK], tl.int32)), None)
    tmp6 = tl.load(in_ptr3 + (x2), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp3 = 8192.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp1 * tmp4
    tmp7 = tl.exp(tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp5 - tmp9
    tmp11 = tmp0 + tmp10
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


kernel4 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[32768, 8192],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
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
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp3, xmask)
''')


kernel6 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[8192, 1024],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp5, xmask)
''')


kernel7 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp3, xmask)
''')


kernel8 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1024, 64],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), xmask & rmask, eviction_policy='evict_last')
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

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


kernel10 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x2), xmask)
    tmp9 = tl.load(in_ptr5 + (x1), xmask)
    tmp13 = tl.load(in_ptr6 + (x2), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = 768
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel11 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x1), xmask)
    tmp8 = tl.load(in_ptr4 + (x2), xmask)
    tmp9 = tl.load(in_ptr5 + (x1), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = 768
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 - tmp10
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp12, xmask)
''')


kernel12 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


kernel13 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


kernel14 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 256],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (786432*x1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp1, xmask)
''')


kernel15 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[4096, 32],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp5, xmask)
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
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 * tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.reshape(tl.sum(_tmp7, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp7, xmask)
''')


kernel18 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp4 = tl.load(in_ptr3 + (x0), xmask)
    tmp8 = tl.load(in_ptr4 + (x1), xmask)
    tmp10 = tl.load(in_ptr5 + (x2), xmask)
    tmp11 = tl.load(in_ptr6 + (x1), xmask)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 768
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 - tmp8
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 - tmp12
    tmp14 = tmp0 * tmp13
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp14, xmask)
''')


kernel19 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        _tmp5 = tl.where(xmask & rmask, _tmp5 + tmp4, _tmp5)
    tmp5 = tl.reshape(tl.sum(_tmp5, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp5, xmask)
''')


kernel20 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp3, xmask)
''')


kernel21 = async_compile.triton('''
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


kernel22 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 128],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex
    _tmp8 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp7 = tmp5 * tmp6
        _tmp8 = tl.where(xmask & rmask, _tmp8 + tmp7, _tmp8)
    tmp8 = tl.reshape(tl.sum(_tmp8, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp8, xmask)
''')


kernel23 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x2), xmask)
    tmp8 = tl.load(in_ptr3 + (x1), xmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp7 - tmp9
    tmp11 = 8.0
    tmp12 = tmp10 / tmp11
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp12, xmask)
''')


kernel24 = async_compile.triton('''
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


kernel25 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8192, 1024], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 8192
    ynumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (128*y2) + (98304*x1)), xmask & ymask)
    tl.store(out_ptr0 + (y2 + (768*x3) + tl.zeros([XBLOCK, YBLOCK], tl.int32)), tmp0, xmask & ymask)
''')


kernel26 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x2), xmask)
    tmp5 = tl.load(in_ptr3 + (x2), xmask)
    tmp7 = tl.load(in_ptr4 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp8, xmask)
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


kernel28 = async_compile.triton('''
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
    _tmp3 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        _tmp3 = tl.where(xmask & rmask, _tmp3 + tmp2, _tmp3)
    tmp3 = tl.reshape(tl.sum(_tmp3, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp3, xmask)
''')


kernel29 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr0, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr1 = in_out_ptr0
    out_ptr0 = in_out_ptr0
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask)
    tmp6 = tl.load(in_ptr3 + (x2), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask)
    tmp2 = 768
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 - tmp8
    tmp10 = tmp0 * tmp9
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp10, xmask)
''')


kernel30 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp9 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        _tmp9 = tl.where(xmask & rmask, _tmp9 + tmp8, _tmp9)
    tmp9 = tl.reshape(tl.sum(_tmp9, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp9, xmask)
''')


kernel31 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[65536, 128],
              reduction_hint=ReductionHint.OUTER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp7 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp5 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), xmask & rmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        _tmp7 = tl.where(xmask & rmask, _tmp7 + tmp6, _tmp7)
    tmp7 = tl.reshape(tl.sum(_tmp7, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x3, tmp7, xmask)
''')


kernel32 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    in_ptr0 = in_out_ptr0
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
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp11, xmask)
''')


kernel33 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[131072, 64],
              reduction_hint=ReductionHint.DEFAULT,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (98304*r1)), xmask & rmask, eviction_policy='evict_last')
        _tmp1 = tl.where(xmask & rmask, _tmp1 + tmp0, _tmp1)
    tmp1 = tl.reshape(tl.sum(_tmp1, 1), [XBLOCK, 1])
    tl.store(out_ptr0 + x0, tmp1, xmask)
''')


kernel34 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel35 = async_compile.triton('''
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
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = -1
    tmp2 = tmp0 != tmp1
    tmp4 = 0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel36 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel37 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1 % 128), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = -1
    tmp2 = tmp0 != tmp1
    tmp4 = 0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


kernel38 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def kernel(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


kernel39 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = 0
    tmp2 = tmp0 != tmp1
    tmp4 = tl.where(tmp2, tmp3, tmp1)
    tl.atomic_add(out_ptr0 + (x0 + (768*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


kernel40 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def kernel(in_out_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    in_ptr0 = in_out_ptr0
    out_ptr0 = in_out_ptr0
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, slice_2, mul_1, gt, view, gt_1, view_11, gt_2, mul_9, view_13, view_15, gt_3, mul_26, view_17, gt_4, view_28, gt_5, mul_32, view_30, view_32, gt_6, mul_49, view_34, gt_7, view_45, gt_8, mul_55, view_47, view_49, gt_9, mul_72, view_51, gt_10, view_62, gt_11, mul_78, view_64, view_66, gt_12, mul_95, view_68, gt_13, view_79, gt_14, mul_101, view_81, view_83, gt_15, mul_118, view_85, gt_16, view_96, gt_17, mul_124, view_98, view_100, gt_18, mul_141, view_102, gt_19, view_113, gt_20, mul_147, view_115, view_117, gt_21, mul_164, view_119, gt_22, view_130, gt_23, mul_170, view_132, view_134, gt_24, mul_187, view_136, gt_25, view_147, gt_26, mul_193, view_149, view_151, gt_27, mul_210, view_153, gt_28, view_164, gt_29, mul_216, view_166, view_168, gt_30, mul_233, view_170, gt_31, view_181, gt_32, mul_239, view_183, view_185, gt_33, mul_256, view_187, gt_34, view_198, gt_35, mul_262, view_200, view_202, gt_36, mul_279, view_204, mul_294, view_206, sub_53, unsqueeze_2, permute_134, div_25, add_175, permute_138, div_26, permute_142, add_182, permute_146, div_27, permute_150, permute_155, permute_156, alias_83, permute_157, permute_158, permute_162, permute_167, permute_171, div_29, permute_175, add_193, permute_179, div_30, permute_183, permute_188, permute_189, alias_85, permute_190, permute_191, permute_195, permute_200, permute_204, div_32, permute_208, add_204, permute_212, div_33, permute_216, permute_221, permute_222, alias_87, permute_223, permute_224, permute_228, permute_233, permute_237, div_35, permute_241, add_215, permute_245, div_36, permute_249, permute_254, permute_255, alias_89, permute_256, permute_257, permute_261, permute_266, permute_270, div_38, permute_274, add_226, permute_278, div_39, permute_282, permute_287, permute_288, alias_91, permute_289, permute_290, permute_294, permute_299, permute_303, div_41, permute_307, add_237, permute_311, div_42, permute_315, permute_320, permute_321, alias_93, permute_322, permute_323, permute_327, permute_332, permute_336, div_44, permute_340, add_248, permute_344, div_45, permute_348, permute_353, permute_354, alias_95, permute_355, permute_356, permute_360, permute_365, permute_369, div_47, permute_373, add_259, permute_377, div_48, permute_381, permute_386, permute_387, alias_97, permute_388, permute_389, permute_393, permute_398, permute_402, div_50, permute_406, add_270, permute_410, div_51, permute_414, permute_419, permute_420, alias_99, permute_421, permute_422, permute_426, permute_431, permute_435, div_53, permute_439, add_281, permute_443, div_54, permute_447, permute_452, permute_453, alias_101, permute_454, permute_455, permute_459, permute_464, permute_468, div_56, permute_472, add_292, permute_476, div_57, permute_480, permute_485, permute_486, alias_103, permute_487, permute_488, permute_492, permute_497, permute_501, div_59, permute_505, add_303, permute_509, div_60, permute_513, permute_518, permute_519, alias_105, permute_520, permute_521, permute_525, permute_530, permute_534, div_62, view_506, view_509, tangents_1, tangents_2 = args
    args.clear()
    buf0 = empty_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    stream0 = get_cuda_stream(0)
    kernel0.run(buf0, 250036224, grid=grid(250036224), stream=stream0)
    kernel1.run(unsqueeze_2, buf0, 8192, grid=grid(8192), stream=stream0)
    del unsqueeze_2
    buf2 = empty_strided((8192, 1), (1, 8192), device='cuda', dtype=torch.float32)
    kernel2.run(buf0, tangents_1, buf2, 8192, 30522, grid=grid(8192), stream=stream0)
    buf3 = empty_strided((64, 128, 30522), (3906816, 30522, 1), device='cuda', dtype=torch.float32)
    kernel3.run(tangents_2, buf0, tangents_1, sub_53, buf2, buf3, 250036224, grid=grid(250036224), stream=stream0)
    del buf0
    del sub_53
    del tangents_1
    del tangents_2
    buf4 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf3, (8192, 30522), (30522, 1)), permute_134, out=buf4)
    del permute_134
    buf5 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf3, (30522, 8192), (1, 30522)), view_206, out=buf5)
    del view_206
    buf6 = empty_strided((1, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    kernel4.run(buf3, buf6, 30522, 8192, grid=grid(30522), stream=stream0)
    del buf3
    buf7 = as_strided(buf2, (64, 128, 1), (128, 1, 8192)); del buf2  # reuse
    kernel5.run(buf4, primals_200, buf7, 8192, 768, grid=grid(8192), stream=stream0)
    buf8 = empty_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    kernel6.run(buf4, primals_200, mul_294, buf8, 8192, 768, grid=grid(8192), stream=stream0)
    buf9 = empty_strided((768, 64), (1, 768), device='cuda', dtype=torch.float32)
    kernel7.run(buf4, mul_294, buf9, 49152, 128, grid=grid(49152), stream=stream0)
    buf10 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf9, buf10, 768, 64, grid=grid(768), stream=stream0)
    buf11 = buf9; del buf9  # reuse
    kernel9.run(buf4, buf11, 49152, 128, grid=grid(49152), stream=stream0)
    buf12 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf11, buf12, 768, 64, grid=grid(768), stream=stream0)
    buf13 = as_strided(buf4, (64, 128, 768), (98304, 768, 1)); del buf4  # reuse
    kernel10.run(buf13, div_25, primals_200, buf7, mul_294, buf8, add_175, 6291456, grid=grid(6291456), stream=stream0)
    del add_175
    del div_25
    del mul_294
    del primals_200
    buf14 = empty_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf13, (8192, 768), (768, 1)), permute_138, out=buf14)
    del permute_138
    buf15 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf13, (768, 8192), (1, 768)), view_204, out=buf15)
    del view_204
    buf16 = as_strided(buf11, (1, 768, 64), (49152, 1, 768)); del buf11  # reuse
    kernel9.run(buf13, buf16, 49152, 128, grid=grid(49152), stream=stream0)
    buf17 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf16, buf17, 768, 64, grid=grid(768), stream=stream0)
    buf18 = buf8; del buf8  # reuse
    kernel5.run(buf14, primals_196, buf18, 8192, 768, grid=grid(8192), stream=stream0)
    buf19 = buf7; del buf7  # reuse
    kernel6.run(buf14, primals_196, mul_279, buf19, 8192, 768, grid=grid(8192), stream=stream0)
    buf20 = buf13; del buf13  # reuse
    kernel11.run(div_26, buf14, primals_196, buf18, mul_279, buf19, buf20, 6291456, grid=grid(6291456), stream=stream0)
    del div_26
    del primals_196
    buf21 = as_strided(buf16, (768, 64), (1, 768)); del buf16  # reuse
    kernel7.run(buf14, mul_279, buf21, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_279
    buf22 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf21, buf22, 768, 64, grid=grid(768), stream=stream0)
    buf23 = buf21; del buf21  # reuse
    kernel9.run(buf14, buf23, 49152, 128, grid=grid(49152), stream=stream0)
    buf24 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf23, buf24, 768, 64, grid=grid(768), stream=stream0)
    buf25 = as_strided(buf14, (64, 128, 768), (98304, 768, 1)); del buf14  # reuse
    kernel12.run(buf20, gt_36, buf25, 6291456, grid=grid(6291456), stream=stream0)
    del gt_36
    buf26 = empty_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf25, (8192, 768), (768, 1)), permute_142, out=buf26)
    del permute_142
    buf27 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf25, (768, 8192), (1, 768)), view_202, out=buf27)
    del view_202
    buf28 = as_strided(buf23, (1, 768, 64), (49152, 1, 768)); del buf23  # reuse
    kernel9.run(buf25, buf28, 49152, 128, grid=grid(49152), stream=stream0)
    buf29 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf28, buf29, 768, 64, grid=grid(768), stream=stream0)
    buf30 = as_strided(buf26, (64, 128, 3072), (393216, 3072, 1)); del buf26  # reuse
    kernel13.run(buf30, add_182, 25165824, grid=grid(25165824), stream=stream0)
    del add_182
    buf31 = as_strided(buf25, (8192, 768), (768, 1)); del buf25  # reuse
    aten.mm.out(as_strided(buf30, (8192, 3072), (3072, 1)), permute_146, out=buf31)
    del permute_146
    buf32 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf30, (3072, 8192), (1, 3072)), view_200, out=buf32)
    del view_200
    buf33 = empty_strided((1, 3072, 32), (98304, 1, 3072), device='cuda', dtype=torch.float32)
    kernel14.run(buf30, buf33, 98304, 256, grid=grid(98304), stream=stream0)
    buf34 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf33, buf34, 3072, 32, grid=grid(3072), stream=stream0)
    buf35 = buf19; del buf19  # reuse
    kernel16.run(buf20, buf31, primals_190, buf35, 8192, 768, grid=grid(8192), stream=stream0)
    buf36 = buf18; del buf18  # reuse
    kernel17.run(buf20, buf31, primals_190, mul_262, buf36, 8192, 768, grid=grid(8192), stream=stream0)
    buf37 = empty_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    kernel18.run(div_27, buf20, buf31, primals_190, buf35, mul_262, buf36, buf37, 6291456, grid=grid(6291456), stream=stream0)
    del div_27
    del primals_190
    buf38 = as_strided(buf28, (768, 64), (1, 768)); del buf28  # reuse
    kernel19.run(buf20, buf31, mul_262, buf38, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_262
    buf39 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf38, buf39, 768, 64, grid=grid(768), stream=stream0)
    buf40 = buf38; del buf38  # reuse
    kernel20.run(buf20, buf31, buf40, 49152, 128, grid=grid(49152), stream=stream0)
    buf41 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf40, buf41, 768, 64, grid=grid(768), stream=stream0)
    buf42 = as_strided(buf31, (64, 128, 768), (98304, 768, 1)); del buf31  # reuse
    kernel12.run(buf37, gt_35, buf42, 6291456, grid=grid(6291456), stream=stream0)
    del gt_35
    buf43 = as_strided(buf20, (8192, 768), (768, 1)); del buf20  # reuse
    aten.mm.out(as_strided(buf42, (8192, 768), (768, 1)), permute_150, out=buf43)
    del permute_150
    buf44 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf42, (768, 8192), (1, 768)), view_198, out=buf44)
    del view_198
    buf45 = as_strided(buf40, (1, 768, 64), (49152, 1, 768)); del buf40  # reuse
    kernel9.run(buf42, buf45, 49152, 128, grid=grid(49152), stream=stream0)
    buf46 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf45, buf46, 768, 64, grid=grid(768), stream=stream0)
    buf47 = as_strided(buf42, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf42  # reuse
    kernel21.run(buf43, buf47, 6291456, grid=grid(6291456), stream=stream0)
    buf48 = as_strided(buf43, (768, 128, 64), (8192, 64, 1)); del buf43  # reuse
    aten.bmm.out(permute_155, as_strided(buf47, (768, 128, 64), (8192, 64, 1)), out=buf48)
    del permute_155
    buf49 = empty_strided((768, 128, 128), (16384, 128, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf47, (768, 128, 64), (8192, 64, 1)), permute_156, out=buf49)
    del permute_156
    buf50 = as_strided(buf33, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf33  # reuse
    kernel22.run(buf49, gt_34, alias_83, buf50, 98304, 128, grid=grid(98304), stream=stream0)
    buf51 = as_strided(buf49, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf49  # reuse
    kernel23.run(buf51, gt_34, alias_83, buf50, 12582912, grid=grid(12582912), stream=stream0)
    del alias_83
    del gt_34
    buf52 = as_strided(buf47, (768, 64, 128), (8192, 128, 1)); del buf47  # reuse
    aten.bmm.out(permute_157, as_strided(buf51, (768, 128, 128), (16384, 128, 1)), out=buf52)
    del permute_157
    buf53 = empty_strided((768, 128, 64), (8192, 64, 1), device='cuda', dtype=torch.float32)
    aten.bmm.out(as_strided(buf51, (768, 128, 128), (16384, 128, 1)), permute_158, out=buf53)
    del permute_158
    buf54 = empty_strided((64, 128, 12, 64), (98304, 768, 64, 1), device='cuda', dtype=torch.float32)
    kernel24.run(buf48, buf54, 6291456, grid=grid(6291456), stream=stream0)
    buf55 = as_strided(buf48, (8192, 768), (768, 1)); del buf48  # reuse
    aten.mm.out(as_strided(buf54, (8192, 768), (768, 1)), permute_162, out=buf55)
    del permute_162
    buf56 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf54, (768, 8192), (1, 768)), view_187, out=buf56)
    buf57 = buf45; del buf45  # reuse
    kernel9.run(buf54, buf57, 49152, 128, grid=grid(49152), stream=stream0)
    buf58 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf57, buf58, 768, 64, grid=grid(768), stream=stream0)
    buf59 = as_strided(buf54, (64, 128, 768), (98304, 768, 1)); del buf54  # reuse
    kernel25.run(buf52, buf59, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf60 = as_strided(buf52, (8192, 768), (768, 1)); del buf52  # reuse
    aten.mm.out(as_strided(buf59, (8192, 768), (768, 1)), permute_167, out=buf60)
    del permute_167
    buf61 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf59, (768, 8192), (1, 768)), view_187, out=buf61)
    buf62 = buf57; del buf57  # reuse
    kernel9.run(buf59, buf62, 49152, 128, grid=grid(49152), stream=stream0)
    buf63 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf62, buf63, 768, 64, grid=grid(768), stream=stream0)
    buf64 = as_strided(buf59, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf59  # reuse
    kernel24.run(buf53, buf64, 6291456, grid=grid(6291456), stream=stream0)
    buf65 = as_strided(buf53, (8192, 768), (768, 1)); del buf53  # reuse
    aten.mm.out(as_strided(buf64, (8192, 768), (768, 1)), permute_171, out=buf65)
    del permute_171
    buf66 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf64, (768, 8192), (1, 768)), view_187, out=buf66)
    del view_187
    buf67 = buf62; del buf62  # reuse
    kernel9.run(buf64, buf67, 49152, 128, grid=grid(49152), stream=stream0)
    buf68 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf67, buf68, 768, 64, grid=grid(768), stream=stream0)
    buf69 = as_strided(buf64, (64, 128, 768), (98304, 768, 1)); del buf64  # reuse
    kernel26.run(buf37, buf55, buf60, buf65, primals_180, buf69, 6291456, grid=grid(6291456), stream=stream0)
    del primals_180
    buf70 = buf36; del buf36  # reuse
    kernel27.run(buf69, buf70, 8192, 768, grid=grid(8192), stream=stream0)
    buf71 = buf35; del buf35  # reuse
    kernel28.run(buf69, mul_256, buf71, 8192, 768, grid=grid(8192), stream=stream0)
    buf72 = buf69; del buf69  # reuse
    kernel29.run(buf72, div_29, buf70, mul_256, buf71, 6291456, grid=grid(6291456), stream=stream0)
    del div_29
    buf73 = as_strided(buf67, (768, 64), (1, 768)); del buf67  # reuse
    kernel30.run(buf37, buf55, buf60, buf65, mul_256, buf73, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_256
    buf74 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf73, buf74, 768, 64, grid=grid(768), stream=stream0)
    buf75 = buf73; del buf73  # reuse
    kernel31.run(buf37, buf55, buf60, buf65, buf75, 49152, 128, grid=grid(49152), stream=stream0)
    buf76 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf75, buf76, 768, 64, grid=grid(768), stream=stream0)
    buf77 = as_strided(buf65, (64, 128, 768), (98304, 768, 1)); del buf65  # reuse
    kernel12.run(buf72, gt_33, buf77, 6291456, grid=grid(6291456), stream=stream0)
    del gt_33
    buf78 = as_strided(buf30, (8192, 3072), (3072, 1)); del buf30  # reuse
    aten.mm.out(as_strided(buf77, (8192, 768), (768, 1)), permute_175, out=buf78)
    del permute_175
    buf79 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf77, (768, 8192), (1, 768)), view_185, out=buf79)
    del view_185
    buf80 = as_strided(buf75, (1, 768, 64), (49152, 1, 768)); del buf75  # reuse
    kernel9.run(buf77, buf80, 49152, 128, grid=grid(49152), stream=stream0)
    buf81 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf80, buf81, 768, 64, grid=grid(768), stream=stream0)
    buf82 = as_strided(buf78, (64, 128, 3072), (393216, 3072, 1)); del buf78  # reuse
    kernel13.run(buf82, add_193, 25165824, grid=grid(25165824), stream=stream0)
    del add_193
    buf83 = as_strided(buf77, (8192, 768), (768, 1)); del buf77  # reuse
    aten.mm.out(as_strided(buf82, (8192, 3072), (3072, 1)), permute_179, out=buf83)
    del permute_179
    buf84 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf82, (3072, 8192), (1, 3072)), view_183, out=buf84)
    del view_183
    buf85 = as_strided(buf50, (1, 3072, 32), (98304, 1, 3072)); del buf50  # reuse
    kernel14.run(buf82, buf85, 98304, 256, grid=grid(98304), stream=stream0)
    buf86 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf85, buf86, 3072, 32, grid=grid(3072), stream=stream0)
    buf87 = buf71; del buf71  # reuse
    kernel16.run(buf72, buf83, primals_174, buf87, 8192, 768, grid=grid(8192), stream=stream0)
    buf88 = buf70; del buf70  # reuse
    kernel17.run(buf72, buf83, primals_174, mul_239, buf88, 8192, 768, grid=grid(8192), stream=stream0)
    buf89 = as_strided(buf60, (64, 128, 768), (98304, 768, 1)); del buf60  # reuse
    kernel18.run(div_30, buf72, buf83, primals_174, buf87, mul_239, buf88, buf89, 6291456, grid=grid(6291456), stream=stream0)
    del div_30
    del primals_174
    buf90 = as_strided(buf80, (768, 64), (1, 768)); del buf80  # reuse
    kernel19.run(buf72, buf83, mul_239, buf90, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_239
    buf91 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf90, buf91, 768, 64, grid=grid(768), stream=stream0)
    buf92 = buf90; del buf90  # reuse
    kernel20.run(buf72, buf83, buf92, 49152, 128, grid=grid(49152), stream=stream0)
    buf93 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf92, buf93, 768, 64, grid=grid(768), stream=stream0)
    buf94 = as_strided(buf83, (64, 128, 768), (98304, 768, 1)); del buf83  # reuse
    kernel12.run(buf89, gt_32, buf94, 6291456, grid=grid(6291456), stream=stream0)
    del gt_32
    buf95 = as_strided(buf72, (8192, 768), (768, 1)); del buf72  # reuse
    aten.mm.out(as_strided(buf94, (8192, 768), (768, 1)), permute_183, out=buf95)
    del permute_183
    buf96 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf94, (768, 8192), (1, 768)), view_181, out=buf96)
    del view_181
    buf97 = as_strided(buf92, (1, 768, 64), (49152, 1, 768)); del buf92  # reuse
    kernel9.run(buf94, buf97, 49152, 128, grid=grid(49152), stream=stream0)
    buf98 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf97, buf98, 768, 64, grid=grid(768), stream=stream0)
    buf99 = as_strided(buf94, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf94  # reuse
    kernel21.run(buf95, buf99, 6291456, grid=grid(6291456), stream=stream0)
    buf100 = as_strided(buf95, (768, 128, 64), (8192, 64, 1)); del buf95  # reuse
    aten.bmm.out(permute_188, as_strided(buf99, (768, 128, 64), (8192, 64, 1)), out=buf100)
    del permute_188
    buf101 = as_strided(buf51, (768, 128, 128), (16384, 128, 1)); del buf51  # reuse
    aten.bmm.out(as_strided(buf99, (768, 128, 64), (8192, 64, 1)), permute_189, out=buf101)
    del permute_189
    buf102 = as_strided(buf85, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf85  # reuse
    kernel22.run(buf101, gt_31, alias_85, buf102, 98304, 128, grid=grid(98304), stream=stream0)
    buf103 = as_strided(buf101, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf101  # reuse
    kernel23.run(buf103, gt_31, alias_85, buf102, 12582912, grid=grid(12582912), stream=stream0)
    del alias_85
    del gt_31
    buf104 = as_strided(buf99, (768, 64, 128), (8192, 128, 1)); del buf99  # reuse
    aten.bmm.out(permute_190, as_strided(buf103, (768, 128, 128), (16384, 128, 1)), out=buf104)
    del permute_190
    buf105 = as_strided(buf55, (768, 128, 64), (8192, 64, 1)); del buf55  # reuse
    aten.bmm.out(as_strided(buf103, (768, 128, 128), (16384, 128, 1)), permute_191, out=buf105)
    del permute_191
    buf106 = as_strided(buf37, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf37  # reuse
    kernel24.run(buf100, buf106, 6291456, grid=grid(6291456), stream=stream0)
    buf107 = as_strided(buf100, (8192, 768), (768, 1)); del buf100  # reuse
    aten.mm.out(as_strided(buf106, (8192, 768), (768, 1)), permute_195, out=buf107)
    del permute_195
    buf108 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf106, (768, 8192), (1, 768)), view_170, out=buf108)
    buf109 = buf97; del buf97  # reuse
    kernel9.run(buf106, buf109, 49152, 128, grid=grid(49152), stream=stream0)
    buf110 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf109, buf110, 768, 64, grid=grid(768), stream=stream0)
    buf111 = as_strided(buf106, (64, 128, 768), (98304, 768, 1)); del buf106  # reuse
    kernel25.run(buf104, buf111, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf112 = as_strided(buf104, (8192, 768), (768, 1)); del buf104  # reuse
    aten.mm.out(as_strided(buf111, (8192, 768), (768, 1)), permute_200, out=buf112)
    del permute_200
    buf113 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf111, (768, 8192), (1, 768)), view_170, out=buf113)
    buf114 = buf109; del buf109  # reuse
    kernel9.run(buf111, buf114, 49152, 128, grid=grid(49152), stream=stream0)
    buf115 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf114, buf115, 768, 64, grid=grid(768), stream=stream0)
    buf116 = as_strided(buf111, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf111  # reuse
    kernel24.run(buf105, buf116, 6291456, grid=grid(6291456), stream=stream0)
    buf117 = as_strided(buf105, (8192, 768), (768, 1)); del buf105  # reuse
    aten.mm.out(as_strided(buf116, (8192, 768), (768, 1)), permute_204, out=buf117)
    del permute_204
    buf118 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf116, (768, 8192), (1, 768)), view_170, out=buf118)
    del view_170
    buf119 = buf114; del buf114  # reuse
    kernel9.run(buf116, buf119, 49152, 128, grid=grid(49152), stream=stream0)
    buf120 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf119, buf120, 768, 64, grid=grid(768), stream=stream0)
    buf121 = as_strided(buf116, (64, 128, 768), (98304, 768, 1)); del buf116  # reuse
    kernel26.run(buf89, buf107, buf112, buf117, primals_164, buf121, 6291456, grid=grid(6291456), stream=stream0)
    del primals_164
    buf122 = buf88; del buf88  # reuse
    kernel27.run(buf121, buf122, 8192, 768, grid=grid(8192), stream=stream0)
    buf123 = buf87; del buf87  # reuse
    kernel28.run(buf121, mul_233, buf123, 8192, 768, grid=grid(8192), stream=stream0)
    buf124 = buf121; del buf121  # reuse
    kernel29.run(buf124, div_32, buf122, mul_233, buf123, 6291456, grid=grid(6291456), stream=stream0)
    del div_32
    buf125 = as_strided(buf119, (768, 64), (1, 768)); del buf119  # reuse
    kernel30.run(buf89, buf107, buf112, buf117, mul_233, buf125, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_233
    buf126 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf125, buf126, 768, 64, grid=grid(768), stream=stream0)
    buf127 = buf125; del buf125  # reuse
    kernel31.run(buf89, buf107, buf112, buf117, buf127, 49152, 128, grid=grid(49152), stream=stream0)
    buf128 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf127, buf128, 768, 64, grid=grid(768), stream=stream0)
    buf129 = buf89; del buf89  # reuse
    kernel12.run(buf124, gt_30, buf129, 6291456, grid=grid(6291456), stream=stream0)
    del gt_30
    buf130 = as_strided(buf82, (8192, 3072), (3072, 1)); del buf82  # reuse
    aten.mm.out(as_strided(buf129, (8192, 768), (768, 1)), permute_208, out=buf130)
    del permute_208
    buf131 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf129, (768, 8192), (1, 768)), view_168, out=buf131)
    del view_168
    buf132 = as_strided(buf127, (1, 768, 64), (49152, 1, 768)); del buf127  # reuse
    kernel9.run(buf129, buf132, 49152, 128, grid=grid(49152), stream=stream0)
    buf133 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf132, buf133, 768, 64, grid=grid(768), stream=stream0)
    buf134 = as_strided(buf130, (64, 128, 3072), (393216, 3072, 1)); del buf130  # reuse
    kernel13.run(buf134, add_204, 25165824, grid=grid(25165824), stream=stream0)
    del add_204
    buf135 = as_strided(buf129, (8192, 768), (768, 1)); del buf129  # reuse
    aten.mm.out(as_strided(buf134, (8192, 3072), (3072, 1)), permute_212, out=buf135)
    del permute_212
    buf136 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf134, (3072, 8192), (1, 3072)), view_166, out=buf136)
    del view_166
    buf137 = as_strided(buf102, (1, 3072, 32), (98304, 1, 3072)); del buf102  # reuse
    kernel14.run(buf134, buf137, 98304, 256, grid=grid(98304), stream=stream0)
    buf138 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf137, buf138, 3072, 32, grid=grid(3072), stream=stream0)
    buf139 = buf123; del buf123  # reuse
    kernel16.run(buf124, buf135, primals_158, buf139, 8192, 768, grid=grid(8192), stream=stream0)
    buf140 = buf122; del buf122  # reuse
    kernel17.run(buf124, buf135, primals_158, mul_216, buf140, 8192, 768, grid=grid(8192), stream=stream0)
    buf141 = as_strided(buf117, (64, 128, 768), (98304, 768, 1)); del buf117  # reuse
    kernel18.run(div_33, buf124, buf135, primals_158, buf139, mul_216, buf140, buf141, 6291456, grid=grid(6291456), stream=stream0)
    del div_33
    del primals_158
    buf142 = as_strided(buf132, (768, 64), (1, 768)); del buf132  # reuse
    kernel19.run(buf124, buf135, mul_216, buf142, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_216
    buf143 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf142, buf143, 768, 64, grid=grid(768), stream=stream0)
    buf144 = buf142; del buf142  # reuse
    kernel20.run(buf124, buf135, buf144, 49152, 128, grid=grid(49152), stream=stream0)
    buf145 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf144, buf145, 768, 64, grid=grid(768), stream=stream0)
    buf146 = as_strided(buf135, (64, 128, 768), (98304, 768, 1)); del buf135  # reuse
    kernel12.run(buf141, gt_29, buf146, 6291456, grid=grid(6291456), stream=stream0)
    del gt_29
    buf147 = as_strided(buf124, (8192, 768), (768, 1)); del buf124  # reuse
    aten.mm.out(as_strided(buf146, (8192, 768), (768, 1)), permute_216, out=buf147)
    del permute_216
    buf148 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf146, (768, 8192), (1, 768)), view_164, out=buf148)
    del view_164
    buf149 = as_strided(buf144, (1, 768, 64), (49152, 1, 768)); del buf144  # reuse
    kernel9.run(buf146, buf149, 49152, 128, grid=grid(49152), stream=stream0)
    buf150 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf149, buf150, 768, 64, grid=grid(768), stream=stream0)
    buf151 = as_strided(buf146, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf146  # reuse
    kernel21.run(buf147, buf151, 6291456, grid=grid(6291456), stream=stream0)
    buf152 = as_strided(buf147, (768, 128, 64), (8192, 64, 1)); del buf147  # reuse
    aten.bmm.out(permute_221, as_strided(buf151, (768, 128, 64), (8192, 64, 1)), out=buf152)
    del permute_221
    buf153 = as_strided(buf103, (768, 128, 128), (16384, 128, 1)); del buf103  # reuse
    aten.bmm.out(as_strided(buf151, (768, 128, 64), (8192, 64, 1)), permute_222, out=buf153)
    del permute_222
    buf154 = as_strided(buf137, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf137  # reuse
    kernel22.run(buf153, gt_28, alias_87, buf154, 98304, 128, grid=grid(98304), stream=stream0)
    buf155 = as_strided(buf153, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf153  # reuse
    kernel23.run(buf155, gt_28, alias_87, buf154, 12582912, grid=grid(12582912), stream=stream0)
    del alias_87
    del gt_28
    buf156 = as_strided(buf151, (768, 64, 128), (8192, 128, 1)); del buf151  # reuse
    aten.bmm.out(permute_223, as_strided(buf155, (768, 128, 128), (16384, 128, 1)), out=buf156)
    del permute_223
    buf157 = as_strided(buf112, (768, 128, 64), (8192, 64, 1)); del buf112  # reuse
    aten.bmm.out(as_strided(buf155, (768, 128, 128), (16384, 128, 1)), permute_224, out=buf157)
    del permute_224
    buf158 = as_strided(buf107, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf107  # reuse
    kernel24.run(buf152, buf158, 6291456, grid=grid(6291456), stream=stream0)
    buf159 = as_strided(buf152, (8192, 768), (768, 1)); del buf152  # reuse
    aten.mm.out(as_strided(buf158, (8192, 768), (768, 1)), permute_228, out=buf159)
    del permute_228
    buf160 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf158, (768, 8192), (1, 768)), view_153, out=buf160)
    buf161 = buf149; del buf149  # reuse
    kernel9.run(buf158, buf161, 49152, 128, grid=grid(49152), stream=stream0)
    buf162 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf161, buf162, 768, 64, grid=grid(768), stream=stream0)
    buf163 = as_strided(buf158, (64, 128, 768), (98304, 768, 1)); del buf158  # reuse
    kernel25.run(buf156, buf163, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf164 = as_strided(buf156, (8192, 768), (768, 1)); del buf156  # reuse
    aten.mm.out(as_strided(buf163, (8192, 768), (768, 1)), permute_233, out=buf164)
    del permute_233
    buf165 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf163, (768, 8192), (1, 768)), view_153, out=buf165)
    buf166 = buf161; del buf161  # reuse
    kernel9.run(buf163, buf166, 49152, 128, grid=grid(49152), stream=stream0)
    buf167 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf166, buf167, 768, 64, grid=grid(768), stream=stream0)
    buf168 = as_strided(buf163, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf163  # reuse
    kernel24.run(buf157, buf168, 6291456, grid=grid(6291456), stream=stream0)
    buf169 = as_strided(buf157, (8192, 768), (768, 1)); del buf157  # reuse
    aten.mm.out(as_strided(buf168, (8192, 768), (768, 1)), permute_237, out=buf169)
    del permute_237
    buf170 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf168, (768, 8192), (1, 768)), view_153, out=buf170)
    del view_153
    buf171 = buf166; del buf166  # reuse
    kernel9.run(buf168, buf171, 49152, 128, grid=grid(49152), stream=stream0)
    buf172 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf171, buf172, 768, 64, grid=grid(768), stream=stream0)
    buf173 = as_strided(buf168, (64, 128, 768), (98304, 768, 1)); del buf168  # reuse
    kernel26.run(buf141, buf159, buf164, buf169, primals_148, buf173, 6291456, grid=grid(6291456), stream=stream0)
    del primals_148
    buf174 = buf140; del buf140  # reuse
    kernel27.run(buf173, buf174, 8192, 768, grid=grid(8192), stream=stream0)
    buf175 = buf139; del buf139  # reuse
    kernel28.run(buf173, mul_210, buf175, 8192, 768, grid=grid(8192), stream=stream0)
    buf176 = buf173; del buf173  # reuse
    kernel29.run(buf176, div_35, buf174, mul_210, buf175, 6291456, grid=grid(6291456), stream=stream0)
    del div_35
    buf177 = as_strided(buf171, (768, 64), (1, 768)); del buf171  # reuse
    kernel30.run(buf141, buf159, buf164, buf169, mul_210, buf177, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_210
    buf178 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf177, buf178, 768, 64, grid=grid(768), stream=stream0)
    buf179 = buf177; del buf177  # reuse
    kernel31.run(buf141, buf159, buf164, buf169, buf179, 49152, 128, grid=grid(49152), stream=stream0)
    buf180 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf179, buf180, 768, 64, grid=grid(768), stream=stream0)
    buf181 = as_strided(buf169, (64, 128, 768), (98304, 768, 1)); del buf169  # reuse
    kernel12.run(buf176, gt_27, buf181, 6291456, grid=grid(6291456), stream=stream0)
    del gt_27
    buf182 = as_strided(buf134, (8192, 3072), (3072, 1)); del buf134  # reuse
    aten.mm.out(as_strided(buf181, (8192, 768), (768, 1)), permute_241, out=buf182)
    del permute_241
    buf183 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf181, (768, 8192), (1, 768)), view_151, out=buf183)
    del view_151
    buf184 = as_strided(buf179, (1, 768, 64), (49152, 1, 768)); del buf179  # reuse
    kernel9.run(buf181, buf184, 49152, 128, grid=grid(49152), stream=stream0)
    buf185 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf184, buf185, 768, 64, grid=grid(768), stream=stream0)
    buf186 = as_strided(buf182, (64, 128, 3072), (393216, 3072, 1)); del buf182  # reuse
    kernel13.run(buf186, add_215, 25165824, grid=grid(25165824), stream=stream0)
    del add_215
    buf187 = as_strided(buf181, (8192, 768), (768, 1)); del buf181  # reuse
    aten.mm.out(as_strided(buf186, (8192, 3072), (3072, 1)), permute_245, out=buf187)
    del permute_245
    buf188 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf186, (3072, 8192), (1, 3072)), view_149, out=buf188)
    del view_149
    buf189 = as_strided(buf154, (1, 3072, 32), (98304, 1, 3072)); del buf154  # reuse
    kernel14.run(buf186, buf189, 98304, 256, grid=grid(98304), stream=stream0)
    buf190 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf189, buf190, 3072, 32, grid=grid(3072), stream=stream0)
    buf191 = buf175; del buf175  # reuse
    kernel16.run(buf176, buf187, primals_142, buf191, 8192, 768, grid=grid(8192), stream=stream0)
    buf192 = buf174; del buf174  # reuse
    kernel17.run(buf176, buf187, primals_142, mul_193, buf192, 8192, 768, grid=grid(8192), stream=stream0)
    buf193 = as_strided(buf164, (64, 128, 768), (98304, 768, 1)); del buf164  # reuse
    kernel18.run(div_36, buf176, buf187, primals_142, buf191, mul_193, buf192, buf193, 6291456, grid=grid(6291456), stream=stream0)
    del div_36
    del primals_142
    buf194 = as_strided(buf184, (768, 64), (1, 768)); del buf184  # reuse
    kernel19.run(buf176, buf187, mul_193, buf194, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_193
    buf195 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf194, buf195, 768, 64, grid=grid(768), stream=stream0)
    buf196 = buf194; del buf194  # reuse
    kernel20.run(buf176, buf187, buf196, 49152, 128, grid=grid(49152), stream=stream0)
    buf197 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf196, buf197, 768, 64, grid=grid(768), stream=stream0)
    buf198 = as_strided(buf187, (64, 128, 768), (98304, 768, 1)); del buf187  # reuse
    kernel12.run(buf193, gt_26, buf198, 6291456, grid=grid(6291456), stream=stream0)
    del gt_26
    buf199 = as_strided(buf176, (8192, 768), (768, 1)); del buf176  # reuse
    aten.mm.out(as_strided(buf198, (8192, 768), (768, 1)), permute_249, out=buf199)
    del permute_249
    buf200 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf198, (768, 8192), (1, 768)), view_147, out=buf200)
    del view_147
    buf201 = as_strided(buf196, (1, 768, 64), (49152, 1, 768)); del buf196  # reuse
    kernel9.run(buf198, buf201, 49152, 128, grid=grid(49152), stream=stream0)
    buf202 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf201, buf202, 768, 64, grid=grid(768), stream=stream0)
    buf203 = as_strided(buf198, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf198  # reuse
    kernel21.run(buf199, buf203, 6291456, grid=grid(6291456), stream=stream0)
    buf204 = as_strided(buf199, (768, 128, 64), (8192, 64, 1)); del buf199  # reuse
    aten.bmm.out(permute_254, as_strided(buf203, (768, 128, 64), (8192, 64, 1)), out=buf204)
    del permute_254
    buf205 = as_strided(buf155, (768, 128, 128), (16384, 128, 1)); del buf155  # reuse
    aten.bmm.out(as_strided(buf203, (768, 128, 64), (8192, 64, 1)), permute_255, out=buf205)
    del permute_255
    buf206 = as_strided(buf189, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf189  # reuse
    kernel22.run(buf205, gt_25, alias_89, buf206, 98304, 128, grid=grid(98304), stream=stream0)
    buf207 = as_strided(buf205, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf205  # reuse
    kernel23.run(buf207, gt_25, alias_89, buf206, 12582912, grid=grid(12582912), stream=stream0)
    del alias_89
    del gt_25
    buf208 = as_strided(buf203, (768, 64, 128), (8192, 128, 1)); del buf203  # reuse
    aten.bmm.out(permute_256, as_strided(buf207, (768, 128, 128), (16384, 128, 1)), out=buf208)
    del permute_256
    buf209 = as_strided(buf159, (768, 128, 64), (8192, 64, 1)); del buf159  # reuse
    aten.bmm.out(as_strided(buf207, (768, 128, 128), (16384, 128, 1)), permute_257, out=buf209)
    del permute_257
    buf210 = as_strided(buf141, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf141  # reuse
    kernel24.run(buf204, buf210, 6291456, grid=grid(6291456), stream=stream0)
    buf211 = as_strided(buf204, (8192, 768), (768, 1)); del buf204  # reuse
    aten.mm.out(as_strided(buf210, (8192, 768), (768, 1)), permute_261, out=buf211)
    del permute_261
    buf212 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf210, (768, 8192), (1, 768)), view_136, out=buf212)
    buf213 = buf201; del buf201  # reuse
    kernel9.run(buf210, buf213, 49152, 128, grid=grid(49152), stream=stream0)
    buf214 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf213, buf214, 768, 64, grid=grid(768), stream=stream0)
    buf215 = as_strided(buf210, (64, 128, 768), (98304, 768, 1)); del buf210  # reuse
    kernel25.run(buf208, buf215, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf216 = as_strided(buf208, (8192, 768), (768, 1)); del buf208  # reuse
    aten.mm.out(as_strided(buf215, (8192, 768), (768, 1)), permute_266, out=buf216)
    del permute_266
    buf217 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf215, (768, 8192), (1, 768)), view_136, out=buf217)
    buf218 = buf213; del buf213  # reuse
    kernel9.run(buf215, buf218, 49152, 128, grid=grid(49152), stream=stream0)
    buf219 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf218, buf219, 768, 64, grid=grid(768), stream=stream0)
    buf220 = as_strided(buf215, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf215  # reuse
    kernel24.run(buf209, buf220, 6291456, grid=grid(6291456), stream=stream0)
    buf221 = as_strided(buf209, (8192, 768), (768, 1)); del buf209  # reuse
    aten.mm.out(as_strided(buf220, (8192, 768), (768, 1)), permute_270, out=buf221)
    del permute_270
    buf222 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf220, (768, 8192), (1, 768)), view_136, out=buf222)
    del view_136
    buf223 = buf218; del buf218  # reuse
    kernel9.run(buf220, buf223, 49152, 128, grid=grid(49152), stream=stream0)
    buf224 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf223, buf224, 768, 64, grid=grid(768), stream=stream0)
    buf225 = as_strided(buf220, (64, 128, 768), (98304, 768, 1)); del buf220  # reuse
    kernel26.run(buf193, buf211, buf216, buf221, primals_132, buf225, 6291456, grid=grid(6291456), stream=stream0)
    del primals_132
    buf226 = buf192; del buf192  # reuse
    kernel27.run(buf225, buf226, 8192, 768, grid=grid(8192), stream=stream0)
    buf227 = buf191; del buf191  # reuse
    kernel28.run(buf225, mul_187, buf227, 8192, 768, grid=grid(8192), stream=stream0)
    buf228 = buf225; del buf225  # reuse
    kernel29.run(buf228, div_38, buf226, mul_187, buf227, 6291456, grid=grid(6291456), stream=stream0)
    del div_38
    buf229 = as_strided(buf223, (768, 64), (1, 768)); del buf223  # reuse
    kernel30.run(buf193, buf211, buf216, buf221, mul_187, buf229, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_187
    buf230 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf229, buf230, 768, 64, grid=grid(768), stream=stream0)
    buf231 = buf229; del buf229  # reuse
    kernel31.run(buf193, buf211, buf216, buf221, buf231, 49152, 128, grid=grid(49152), stream=stream0)
    buf232 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf231, buf232, 768, 64, grid=grid(768), stream=stream0)
    buf233 = as_strided(buf221, (64, 128, 768), (98304, 768, 1)); del buf221  # reuse
    kernel12.run(buf228, gt_24, buf233, 6291456, grid=grid(6291456), stream=stream0)
    del gt_24
    buf234 = as_strided(buf186, (8192, 3072), (3072, 1)); del buf186  # reuse
    aten.mm.out(as_strided(buf233, (8192, 768), (768, 1)), permute_274, out=buf234)
    del permute_274
    buf235 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf233, (768, 8192), (1, 768)), view_134, out=buf235)
    del view_134
    buf236 = as_strided(buf231, (1, 768, 64), (49152, 1, 768)); del buf231  # reuse
    kernel9.run(buf233, buf236, 49152, 128, grid=grid(49152), stream=stream0)
    buf237 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf236, buf237, 768, 64, grid=grid(768), stream=stream0)
    buf238 = as_strided(buf234, (64, 128, 3072), (393216, 3072, 1)); del buf234  # reuse
    kernel13.run(buf238, add_226, 25165824, grid=grid(25165824), stream=stream0)
    del add_226
    buf239 = as_strided(buf233, (8192, 768), (768, 1)); del buf233  # reuse
    aten.mm.out(as_strided(buf238, (8192, 3072), (3072, 1)), permute_278, out=buf239)
    del permute_278
    buf240 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf238, (3072, 8192), (1, 3072)), view_132, out=buf240)
    del view_132
    buf241 = as_strided(buf206, (1, 3072, 32), (98304, 1, 3072)); del buf206  # reuse
    kernel14.run(buf238, buf241, 98304, 256, grid=grid(98304), stream=stream0)
    buf242 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf241, buf242, 3072, 32, grid=grid(3072), stream=stream0)
    buf243 = buf227; del buf227  # reuse
    kernel16.run(buf228, buf239, primals_126, buf243, 8192, 768, grid=grid(8192), stream=stream0)
    buf244 = buf226; del buf226  # reuse
    kernel17.run(buf228, buf239, primals_126, mul_170, buf244, 8192, 768, grid=grid(8192), stream=stream0)
    buf245 = as_strided(buf216, (64, 128, 768), (98304, 768, 1)); del buf216  # reuse
    kernel18.run(div_39, buf228, buf239, primals_126, buf243, mul_170, buf244, buf245, 6291456, grid=grid(6291456), stream=stream0)
    del div_39
    del primals_126
    buf246 = as_strided(buf236, (768, 64), (1, 768)); del buf236  # reuse
    kernel19.run(buf228, buf239, mul_170, buf246, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_170
    buf247 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf246, buf247, 768, 64, grid=grid(768), stream=stream0)
    buf248 = buf246; del buf246  # reuse
    kernel20.run(buf228, buf239, buf248, 49152, 128, grid=grid(49152), stream=stream0)
    buf249 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf248, buf249, 768, 64, grid=grid(768), stream=stream0)
    buf250 = as_strided(buf239, (64, 128, 768), (98304, 768, 1)); del buf239  # reuse
    kernel12.run(buf245, gt_23, buf250, 6291456, grid=grid(6291456), stream=stream0)
    del gt_23
    buf251 = as_strided(buf228, (8192, 768), (768, 1)); del buf228  # reuse
    aten.mm.out(as_strided(buf250, (8192, 768), (768, 1)), permute_282, out=buf251)
    del permute_282
    buf252 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf250, (768, 8192), (1, 768)), view_130, out=buf252)
    del view_130
    buf253 = as_strided(buf248, (1, 768, 64), (49152, 1, 768)); del buf248  # reuse
    kernel9.run(buf250, buf253, 49152, 128, grid=grid(49152), stream=stream0)
    buf254 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf253, buf254, 768, 64, grid=grid(768), stream=stream0)
    buf255 = as_strided(buf250, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf250  # reuse
    kernel21.run(buf251, buf255, 6291456, grid=grid(6291456), stream=stream0)
    buf256 = as_strided(buf251, (768, 128, 64), (8192, 64, 1)); del buf251  # reuse
    aten.bmm.out(permute_287, as_strided(buf255, (768, 128, 64), (8192, 64, 1)), out=buf256)
    del permute_287
    buf257 = as_strided(buf207, (768, 128, 128), (16384, 128, 1)); del buf207  # reuse
    aten.bmm.out(as_strided(buf255, (768, 128, 64), (8192, 64, 1)), permute_288, out=buf257)
    del permute_288
    buf258 = as_strided(buf241, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf241  # reuse
    kernel22.run(buf257, gt_22, alias_91, buf258, 98304, 128, grid=grid(98304), stream=stream0)
    buf259 = as_strided(buf257, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf257  # reuse
    kernel23.run(buf259, gt_22, alias_91, buf258, 12582912, grid=grid(12582912), stream=stream0)
    del alias_91
    del gt_22
    buf260 = as_strided(buf255, (768, 64, 128), (8192, 128, 1)); del buf255  # reuse
    aten.bmm.out(permute_289, as_strided(buf259, (768, 128, 128), (16384, 128, 1)), out=buf260)
    del permute_289
    buf261 = as_strided(buf211, (768, 128, 64), (8192, 64, 1)); del buf211  # reuse
    aten.bmm.out(as_strided(buf259, (768, 128, 128), (16384, 128, 1)), permute_290, out=buf261)
    del permute_290
    buf262 = as_strided(buf193, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf193  # reuse
    kernel24.run(buf256, buf262, 6291456, grid=grid(6291456), stream=stream0)
    buf263 = as_strided(buf256, (8192, 768), (768, 1)); del buf256  # reuse
    aten.mm.out(as_strided(buf262, (8192, 768), (768, 1)), permute_294, out=buf263)
    del permute_294
    buf264 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf262, (768, 8192), (1, 768)), view_119, out=buf264)
    buf265 = buf253; del buf253  # reuse
    kernel9.run(buf262, buf265, 49152, 128, grid=grid(49152), stream=stream0)
    buf266 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf265, buf266, 768, 64, grid=grid(768), stream=stream0)
    buf267 = as_strided(buf262, (64, 128, 768), (98304, 768, 1)); del buf262  # reuse
    kernel25.run(buf260, buf267, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf268 = as_strided(buf260, (8192, 768), (768, 1)); del buf260  # reuse
    aten.mm.out(as_strided(buf267, (8192, 768), (768, 1)), permute_299, out=buf268)
    del permute_299
    buf269 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf267, (768, 8192), (1, 768)), view_119, out=buf269)
    buf270 = buf265; del buf265  # reuse
    kernel9.run(buf267, buf270, 49152, 128, grid=grid(49152), stream=stream0)
    buf271 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf270, buf271, 768, 64, grid=grid(768), stream=stream0)
    buf272 = as_strided(buf267, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf267  # reuse
    kernel24.run(buf261, buf272, 6291456, grid=grid(6291456), stream=stream0)
    buf273 = as_strided(buf261, (8192, 768), (768, 1)); del buf261  # reuse
    aten.mm.out(as_strided(buf272, (8192, 768), (768, 1)), permute_303, out=buf273)
    del permute_303
    buf274 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf272, (768, 8192), (1, 768)), view_119, out=buf274)
    del view_119
    buf275 = buf270; del buf270  # reuse
    kernel9.run(buf272, buf275, 49152, 128, grid=grid(49152), stream=stream0)
    buf276 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf275, buf276, 768, 64, grid=grid(768), stream=stream0)
    buf277 = as_strided(buf272, (64, 128, 768), (98304, 768, 1)); del buf272  # reuse
    kernel26.run(buf245, buf263, buf268, buf273, primals_116, buf277, 6291456, grid=grid(6291456), stream=stream0)
    del primals_116
    buf278 = buf244; del buf244  # reuse
    kernel27.run(buf277, buf278, 8192, 768, grid=grid(8192), stream=stream0)
    buf279 = buf243; del buf243  # reuse
    kernel28.run(buf277, mul_164, buf279, 8192, 768, grid=grid(8192), stream=stream0)
    buf280 = buf277; del buf277  # reuse
    kernel29.run(buf280, div_41, buf278, mul_164, buf279, 6291456, grid=grid(6291456), stream=stream0)
    del div_41
    buf281 = as_strided(buf275, (768, 64), (1, 768)); del buf275  # reuse
    kernel30.run(buf245, buf263, buf268, buf273, mul_164, buf281, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_164
    buf282 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf281, buf282, 768, 64, grid=grid(768), stream=stream0)
    buf283 = buf281; del buf281  # reuse
    kernel31.run(buf245, buf263, buf268, buf273, buf283, 49152, 128, grid=grid(49152), stream=stream0)
    buf284 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf283, buf284, 768, 64, grid=grid(768), stream=stream0)
    buf285 = as_strided(buf273, (64, 128, 768), (98304, 768, 1)); del buf273  # reuse
    kernel12.run(buf280, gt_21, buf285, 6291456, grid=grid(6291456), stream=stream0)
    del gt_21
    buf286 = as_strided(buf238, (8192, 3072), (3072, 1)); del buf238  # reuse
    aten.mm.out(as_strided(buf285, (8192, 768), (768, 1)), permute_307, out=buf286)
    del permute_307
    buf287 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf285, (768, 8192), (1, 768)), view_117, out=buf287)
    del view_117
    buf288 = as_strided(buf283, (1, 768, 64), (49152, 1, 768)); del buf283  # reuse
    kernel9.run(buf285, buf288, 49152, 128, grid=grid(49152), stream=stream0)
    buf289 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf288, buf289, 768, 64, grid=grid(768), stream=stream0)
    buf290 = as_strided(buf286, (64, 128, 3072), (393216, 3072, 1)); del buf286  # reuse
    kernel13.run(buf290, add_237, 25165824, grid=grid(25165824), stream=stream0)
    del add_237
    buf291 = as_strided(buf285, (8192, 768), (768, 1)); del buf285  # reuse
    aten.mm.out(as_strided(buf290, (8192, 3072), (3072, 1)), permute_311, out=buf291)
    del permute_311
    buf292 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf290, (3072, 8192), (1, 3072)), view_115, out=buf292)
    del view_115
    buf293 = as_strided(buf258, (1, 3072, 32), (98304, 1, 3072)); del buf258  # reuse
    kernel14.run(buf290, buf293, 98304, 256, grid=grid(98304), stream=stream0)
    buf294 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf293, buf294, 3072, 32, grid=grid(3072), stream=stream0)
    buf295 = buf279; del buf279  # reuse
    kernel16.run(buf280, buf291, primals_110, buf295, 8192, 768, grid=grid(8192), stream=stream0)
    buf296 = buf278; del buf278  # reuse
    kernel17.run(buf280, buf291, primals_110, mul_147, buf296, 8192, 768, grid=grid(8192), stream=stream0)
    buf297 = as_strided(buf268, (64, 128, 768), (98304, 768, 1)); del buf268  # reuse
    kernel18.run(div_42, buf280, buf291, primals_110, buf295, mul_147, buf296, buf297, 6291456, grid=grid(6291456), stream=stream0)
    del div_42
    del primals_110
    buf298 = as_strided(buf288, (768, 64), (1, 768)); del buf288  # reuse
    kernel19.run(buf280, buf291, mul_147, buf298, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_147
    buf299 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf298, buf299, 768, 64, grid=grid(768), stream=stream0)
    buf300 = buf298; del buf298  # reuse
    kernel20.run(buf280, buf291, buf300, 49152, 128, grid=grid(49152), stream=stream0)
    buf301 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf300, buf301, 768, 64, grid=grid(768), stream=stream0)
    buf302 = as_strided(buf291, (64, 128, 768), (98304, 768, 1)); del buf291  # reuse
    kernel12.run(buf297, gt_20, buf302, 6291456, grid=grid(6291456), stream=stream0)
    del gt_20
    buf303 = as_strided(buf280, (8192, 768), (768, 1)); del buf280  # reuse
    aten.mm.out(as_strided(buf302, (8192, 768), (768, 1)), permute_315, out=buf303)
    del permute_315
    buf304 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf302, (768, 8192), (1, 768)), view_113, out=buf304)
    del view_113
    buf305 = as_strided(buf300, (1, 768, 64), (49152, 1, 768)); del buf300  # reuse
    kernel9.run(buf302, buf305, 49152, 128, grid=grid(49152), stream=stream0)
    buf306 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf305, buf306, 768, 64, grid=grid(768), stream=stream0)
    buf307 = as_strided(buf302, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf302  # reuse
    kernel21.run(buf303, buf307, 6291456, grid=grid(6291456), stream=stream0)
    buf308 = as_strided(buf303, (768, 128, 64), (8192, 64, 1)); del buf303  # reuse
    aten.bmm.out(permute_320, as_strided(buf307, (768, 128, 64), (8192, 64, 1)), out=buf308)
    del permute_320
    buf309 = as_strided(buf259, (768, 128, 128), (16384, 128, 1)); del buf259  # reuse
    aten.bmm.out(as_strided(buf307, (768, 128, 64), (8192, 64, 1)), permute_321, out=buf309)
    del permute_321
    buf310 = as_strided(buf293, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf293  # reuse
    kernel22.run(buf309, gt_19, alias_93, buf310, 98304, 128, grid=grid(98304), stream=stream0)
    buf311 = as_strided(buf309, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf309  # reuse
    kernel23.run(buf311, gt_19, alias_93, buf310, 12582912, grid=grid(12582912), stream=stream0)
    del alias_93
    del gt_19
    buf312 = as_strided(buf307, (768, 64, 128), (8192, 128, 1)); del buf307  # reuse
    aten.bmm.out(permute_322, as_strided(buf311, (768, 128, 128), (16384, 128, 1)), out=buf312)
    del permute_322
    buf313 = as_strided(buf263, (768, 128, 64), (8192, 64, 1)); del buf263  # reuse
    aten.bmm.out(as_strided(buf311, (768, 128, 128), (16384, 128, 1)), permute_323, out=buf313)
    del permute_323
    buf314 = as_strided(buf245, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf245  # reuse
    kernel24.run(buf308, buf314, 6291456, grid=grid(6291456), stream=stream0)
    buf315 = as_strided(buf308, (8192, 768), (768, 1)); del buf308  # reuse
    aten.mm.out(as_strided(buf314, (8192, 768), (768, 1)), permute_327, out=buf315)
    del permute_327
    buf316 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf314, (768, 8192), (1, 768)), view_102, out=buf316)
    buf317 = buf305; del buf305  # reuse
    kernel9.run(buf314, buf317, 49152, 128, grid=grid(49152), stream=stream0)
    buf318 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf317, buf318, 768, 64, grid=grid(768), stream=stream0)
    buf319 = as_strided(buf314, (64, 128, 768), (98304, 768, 1)); del buf314  # reuse
    kernel25.run(buf312, buf319, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf320 = as_strided(buf312, (8192, 768), (768, 1)); del buf312  # reuse
    aten.mm.out(as_strided(buf319, (8192, 768), (768, 1)), permute_332, out=buf320)
    del permute_332
    buf321 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf319, (768, 8192), (1, 768)), view_102, out=buf321)
    buf322 = buf317; del buf317  # reuse
    kernel9.run(buf319, buf322, 49152, 128, grid=grid(49152), stream=stream0)
    buf323 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf322, buf323, 768, 64, grid=grid(768), stream=stream0)
    buf324 = as_strided(buf319, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf319  # reuse
    kernel24.run(buf313, buf324, 6291456, grid=grid(6291456), stream=stream0)
    buf325 = as_strided(buf313, (8192, 768), (768, 1)); del buf313  # reuse
    aten.mm.out(as_strided(buf324, (8192, 768), (768, 1)), permute_336, out=buf325)
    del permute_336
    buf326 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf324, (768, 8192), (1, 768)), view_102, out=buf326)
    del view_102
    buf327 = buf322; del buf322  # reuse
    kernel9.run(buf324, buf327, 49152, 128, grid=grid(49152), stream=stream0)
    buf328 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf327, buf328, 768, 64, grid=grid(768), stream=stream0)
    buf329 = as_strided(buf324, (64, 128, 768), (98304, 768, 1)); del buf324  # reuse
    kernel26.run(buf297, buf315, buf320, buf325, primals_100, buf329, 6291456, grid=grid(6291456), stream=stream0)
    del primals_100
    buf330 = buf296; del buf296  # reuse
    kernel27.run(buf329, buf330, 8192, 768, grid=grid(8192), stream=stream0)
    buf331 = buf295; del buf295  # reuse
    kernel28.run(buf329, mul_141, buf331, 8192, 768, grid=grid(8192), stream=stream0)
    buf332 = buf329; del buf329  # reuse
    kernel29.run(buf332, div_44, buf330, mul_141, buf331, 6291456, grid=grid(6291456), stream=stream0)
    del div_44
    buf333 = as_strided(buf327, (768, 64), (1, 768)); del buf327  # reuse
    kernel30.run(buf297, buf315, buf320, buf325, mul_141, buf333, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_141
    buf334 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf333, buf334, 768, 64, grid=grid(768), stream=stream0)
    buf335 = buf333; del buf333  # reuse
    kernel31.run(buf297, buf315, buf320, buf325, buf335, 49152, 128, grid=grid(49152), stream=stream0)
    buf336 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf335, buf336, 768, 64, grid=grid(768), stream=stream0)
    buf337 = as_strided(buf325, (64, 128, 768), (98304, 768, 1)); del buf325  # reuse
    kernel12.run(buf332, gt_18, buf337, 6291456, grid=grid(6291456), stream=stream0)
    del gt_18
    buf338 = as_strided(buf290, (8192, 3072), (3072, 1)); del buf290  # reuse
    aten.mm.out(as_strided(buf337, (8192, 768), (768, 1)), permute_340, out=buf338)
    del permute_340
    buf339 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf337, (768, 8192), (1, 768)), view_100, out=buf339)
    del view_100
    buf340 = as_strided(buf335, (1, 768, 64), (49152, 1, 768)); del buf335  # reuse
    kernel9.run(buf337, buf340, 49152, 128, grid=grid(49152), stream=stream0)
    buf341 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf340, buf341, 768, 64, grid=grid(768), stream=stream0)
    buf342 = as_strided(buf338, (64, 128, 3072), (393216, 3072, 1)); del buf338  # reuse
    kernel13.run(buf342, add_248, 25165824, grid=grid(25165824), stream=stream0)
    del add_248
    buf343 = as_strided(buf337, (8192, 768), (768, 1)); del buf337  # reuse
    aten.mm.out(as_strided(buf342, (8192, 3072), (3072, 1)), permute_344, out=buf343)
    del permute_344
    buf344 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf342, (3072, 8192), (1, 3072)), view_98, out=buf344)
    del view_98
    buf345 = as_strided(buf310, (1, 3072, 32), (98304, 1, 3072)); del buf310  # reuse
    kernel14.run(buf342, buf345, 98304, 256, grid=grid(98304), stream=stream0)
    buf346 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf345, buf346, 3072, 32, grid=grid(3072), stream=stream0)
    buf347 = buf331; del buf331  # reuse
    kernel16.run(buf332, buf343, primals_94, buf347, 8192, 768, grid=grid(8192), stream=stream0)
    buf348 = buf330; del buf330  # reuse
    kernel17.run(buf332, buf343, primals_94, mul_124, buf348, 8192, 768, grid=grid(8192), stream=stream0)
    buf349 = as_strided(buf320, (64, 128, 768), (98304, 768, 1)); del buf320  # reuse
    kernel18.run(div_45, buf332, buf343, primals_94, buf347, mul_124, buf348, buf349, 6291456, grid=grid(6291456), stream=stream0)
    del div_45
    del primals_94
    buf350 = as_strided(buf340, (768, 64), (1, 768)); del buf340  # reuse
    kernel19.run(buf332, buf343, mul_124, buf350, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_124
    buf351 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf350, buf351, 768, 64, grid=grid(768), stream=stream0)
    buf352 = buf350; del buf350  # reuse
    kernel20.run(buf332, buf343, buf352, 49152, 128, grid=grid(49152), stream=stream0)
    buf353 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf352, buf353, 768, 64, grid=grid(768), stream=stream0)
    buf354 = as_strided(buf343, (64, 128, 768), (98304, 768, 1)); del buf343  # reuse
    kernel12.run(buf349, gt_17, buf354, 6291456, grid=grid(6291456), stream=stream0)
    del gt_17
    buf355 = as_strided(buf332, (8192, 768), (768, 1)); del buf332  # reuse
    aten.mm.out(as_strided(buf354, (8192, 768), (768, 1)), permute_348, out=buf355)
    del permute_348
    buf356 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf354, (768, 8192), (1, 768)), view_96, out=buf356)
    del view_96
    buf357 = as_strided(buf352, (1, 768, 64), (49152, 1, 768)); del buf352  # reuse
    kernel9.run(buf354, buf357, 49152, 128, grid=grid(49152), stream=stream0)
    buf358 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf357, buf358, 768, 64, grid=grid(768), stream=stream0)
    buf359 = as_strided(buf354, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf354  # reuse
    kernel21.run(buf355, buf359, 6291456, grid=grid(6291456), stream=stream0)
    buf360 = as_strided(buf355, (768, 128, 64), (8192, 64, 1)); del buf355  # reuse
    aten.bmm.out(permute_353, as_strided(buf359, (768, 128, 64), (8192, 64, 1)), out=buf360)
    del permute_353
    buf361 = as_strided(buf311, (768, 128, 128), (16384, 128, 1)); del buf311  # reuse
    aten.bmm.out(as_strided(buf359, (768, 128, 64), (8192, 64, 1)), permute_354, out=buf361)
    del permute_354
    buf362 = as_strided(buf345, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf345  # reuse
    kernel22.run(buf361, gt_16, alias_95, buf362, 98304, 128, grid=grid(98304), stream=stream0)
    buf363 = as_strided(buf361, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf361  # reuse
    kernel23.run(buf363, gt_16, alias_95, buf362, 12582912, grid=grid(12582912), stream=stream0)
    del alias_95
    del gt_16
    buf364 = as_strided(buf359, (768, 64, 128), (8192, 128, 1)); del buf359  # reuse
    aten.bmm.out(permute_355, as_strided(buf363, (768, 128, 128), (16384, 128, 1)), out=buf364)
    del permute_355
    buf365 = as_strided(buf315, (768, 128, 64), (8192, 64, 1)); del buf315  # reuse
    aten.bmm.out(as_strided(buf363, (768, 128, 128), (16384, 128, 1)), permute_356, out=buf365)
    del permute_356
    buf366 = as_strided(buf297, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf297  # reuse
    kernel24.run(buf360, buf366, 6291456, grid=grid(6291456), stream=stream0)
    buf367 = as_strided(buf360, (8192, 768), (768, 1)); del buf360  # reuse
    aten.mm.out(as_strided(buf366, (8192, 768), (768, 1)), permute_360, out=buf367)
    del permute_360
    buf368 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf366, (768, 8192), (1, 768)), view_85, out=buf368)
    buf369 = buf357; del buf357  # reuse
    kernel9.run(buf366, buf369, 49152, 128, grid=grid(49152), stream=stream0)
    buf370 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf369, buf370, 768, 64, grid=grid(768), stream=stream0)
    buf371 = as_strided(buf366, (64, 128, 768), (98304, 768, 1)); del buf366  # reuse
    kernel25.run(buf364, buf371, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf372 = as_strided(buf364, (8192, 768), (768, 1)); del buf364  # reuse
    aten.mm.out(as_strided(buf371, (8192, 768), (768, 1)), permute_365, out=buf372)
    del permute_365
    buf373 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf371, (768, 8192), (1, 768)), view_85, out=buf373)
    buf374 = buf369; del buf369  # reuse
    kernel9.run(buf371, buf374, 49152, 128, grid=grid(49152), stream=stream0)
    buf375 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf374, buf375, 768, 64, grid=grid(768), stream=stream0)
    buf376 = as_strided(buf371, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf371  # reuse
    kernel24.run(buf365, buf376, 6291456, grid=grid(6291456), stream=stream0)
    buf377 = as_strided(buf365, (8192, 768), (768, 1)); del buf365  # reuse
    aten.mm.out(as_strided(buf376, (8192, 768), (768, 1)), permute_369, out=buf377)
    del permute_369
    buf378 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf376, (768, 8192), (1, 768)), view_85, out=buf378)
    del view_85
    buf379 = buf374; del buf374  # reuse
    kernel9.run(buf376, buf379, 49152, 128, grid=grid(49152), stream=stream0)
    buf380 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf379, buf380, 768, 64, grid=grid(768), stream=stream0)
    buf381 = as_strided(buf376, (64, 128, 768), (98304, 768, 1)); del buf376  # reuse
    kernel26.run(buf349, buf367, buf372, buf377, primals_84, buf381, 6291456, grid=grid(6291456), stream=stream0)
    del primals_84
    buf382 = buf348; del buf348  # reuse
    kernel27.run(buf381, buf382, 8192, 768, grid=grid(8192), stream=stream0)
    buf383 = buf347; del buf347  # reuse
    kernel28.run(buf381, mul_118, buf383, 8192, 768, grid=grid(8192), stream=stream0)
    buf384 = buf381; del buf381  # reuse
    kernel29.run(buf384, div_47, buf382, mul_118, buf383, 6291456, grid=grid(6291456), stream=stream0)
    del div_47
    buf385 = as_strided(buf379, (768, 64), (1, 768)); del buf379  # reuse
    kernel30.run(buf349, buf367, buf372, buf377, mul_118, buf385, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_118
    buf386 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf385, buf386, 768, 64, grid=grid(768), stream=stream0)
    buf387 = buf385; del buf385  # reuse
    kernel31.run(buf349, buf367, buf372, buf377, buf387, 49152, 128, grid=grid(49152), stream=stream0)
    buf388 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf387, buf388, 768, 64, grid=grid(768), stream=stream0)
    buf389 = as_strided(buf377, (64, 128, 768), (98304, 768, 1)); del buf377  # reuse
    kernel12.run(buf384, gt_15, buf389, 6291456, grid=grid(6291456), stream=stream0)
    del gt_15
    buf390 = as_strided(buf342, (8192, 3072), (3072, 1)); del buf342  # reuse
    aten.mm.out(as_strided(buf389, (8192, 768), (768, 1)), permute_373, out=buf390)
    del permute_373
    buf391 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf389, (768, 8192), (1, 768)), view_83, out=buf391)
    del view_83
    buf392 = as_strided(buf387, (1, 768, 64), (49152, 1, 768)); del buf387  # reuse
    kernel9.run(buf389, buf392, 49152, 128, grid=grid(49152), stream=stream0)
    buf393 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf392, buf393, 768, 64, grid=grid(768), stream=stream0)
    buf394 = as_strided(buf390, (64, 128, 3072), (393216, 3072, 1)); del buf390  # reuse
    kernel13.run(buf394, add_259, 25165824, grid=grid(25165824), stream=stream0)
    del add_259
    buf395 = as_strided(buf389, (8192, 768), (768, 1)); del buf389  # reuse
    aten.mm.out(as_strided(buf394, (8192, 3072), (3072, 1)), permute_377, out=buf395)
    del permute_377
    buf396 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf394, (3072, 8192), (1, 3072)), view_81, out=buf396)
    del view_81
    buf397 = as_strided(buf362, (1, 3072, 32), (98304, 1, 3072)); del buf362  # reuse
    kernel14.run(buf394, buf397, 98304, 256, grid=grid(98304), stream=stream0)
    buf398 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf397, buf398, 3072, 32, grid=grid(3072), stream=stream0)
    buf399 = buf383; del buf383  # reuse
    kernel16.run(buf384, buf395, primals_78, buf399, 8192, 768, grid=grid(8192), stream=stream0)
    buf400 = buf382; del buf382  # reuse
    kernel17.run(buf384, buf395, primals_78, mul_101, buf400, 8192, 768, grid=grid(8192), stream=stream0)
    buf401 = as_strided(buf372, (64, 128, 768), (98304, 768, 1)); del buf372  # reuse
    kernel18.run(div_48, buf384, buf395, primals_78, buf399, mul_101, buf400, buf401, 6291456, grid=grid(6291456), stream=stream0)
    del div_48
    del primals_78
    buf402 = as_strided(buf392, (768, 64), (1, 768)); del buf392  # reuse
    kernel19.run(buf384, buf395, mul_101, buf402, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_101
    buf403 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf402, buf403, 768, 64, grid=grid(768), stream=stream0)
    buf404 = buf402; del buf402  # reuse
    kernel20.run(buf384, buf395, buf404, 49152, 128, grid=grid(49152), stream=stream0)
    buf405 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf404, buf405, 768, 64, grid=grid(768), stream=stream0)
    buf406 = as_strided(buf395, (64, 128, 768), (98304, 768, 1)); del buf395  # reuse
    kernel12.run(buf401, gt_14, buf406, 6291456, grid=grid(6291456), stream=stream0)
    del gt_14
    buf407 = as_strided(buf384, (8192, 768), (768, 1)); del buf384  # reuse
    aten.mm.out(as_strided(buf406, (8192, 768), (768, 1)), permute_381, out=buf407)
    del permute_381
    buf408 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf406, (768, 8192), (1, 768)), view_79, out=buf408)
    del view_79
    buf409 = as_strided(buf404, (1, 768, 64), (49152, 1, 768)); del buf404  # reuse
    kernel9.run(buf406, buf409, 49152, 128, grid=grid(49152), stream=stream0)
    buf410 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf409, buf410, 768, 64, grid=grid(768), stream=stream0)
    buf411 = as_strided(buf406, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf406  # reuse
    kernel21.run(buf407, buf411, 6291456, grid=grid(6291456), stream=stream0)
    buf412 = as_strided(buf407, (768, 128, 64), (8192, 64, 1)); del buf407  # reuse
    aten.bmm.out(permute_386, as_strided(buf411, (768, 128, 64), (8192, 64, 1)), out=buf412)
    del permute_386
    buf413 = as_strided(buf363, (768, 128, 128), (16384, 128, 1)); del buf363  # reuse
    aten.bmm.out(as_strided(buf411, (768, 128, 64), (8192, 64, 1)), permute_387, out=buf413)
    del permute_387
    buf414 = as_strided(buf397, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf397  # reuse
    kernel22.run(buf413, gt_13, alias_97, buf414, 98304, 128, grid=grid(98304), stream=stream0)
    buf415 = as_strided(buf413, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf413  # reuse
    kernel23.run(buf415, gt_13, alias_97, buf414, 12582912, grid=grid(12582912), stream=stream0)
    del alias_97
    del gt_13
    buf416 = as_strided(buf411, (768, 64, 128), (8192, 128, 1)); del buf411  # reuse
    aten.bmm.out(permute_388, as_strided(buf415, (768, 128, 128), (16384, 128, 1)), out=buf416)
    del permute_388
    buf417 = as_strided(buf367, (768, 128, 64), (8192, 64, 1)); del buf367  # reuse
    aten.bmm.out(as_strided(buf415, (768, 128, 128), (16384, 128, 1)), permute_389, out=buf417)
    del permute_389
    buf418 = as_strided(buf349, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf349  # reuse
    kernel24.run(buf412, buf418, 6291456, grid=grid(6291456), stream=stream0)
    buf419 = as_strided(buf412, (8192, 768), (768, 1)); del buf412  # reuse
    aten.mm.out(as_strided(buf418, (8192, 768), (768, 1)), permute_393, out=buf419)
    del permute_393
    buf420 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf418, (768, 8192), (1, 768)), view_68, out=buf420)
    buf421 = buf409; del buf409  # reuse
    kernel9.run(buf418, buf421, 49152, 128, grid=grid(49152), stream=stream0)
    buf422 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf421, buf422, 768, 64, grid=grid(768), stream=stream0)
    buf423 = as_strided(buf418, (64, 128, 768), (98304, 768, 1)); del buf418  # reuse
    kernel25.run(buf416, buf423, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf424 = as_strided(buf416, (8192, 768), (768, 1)); del buf416  # reuse
    aten.mm.out(as_strided(buf423, (8192, 768), (768, 1)), permute_398, out=buf424)
    del permute_398
    buf425 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf423, (768, 8192), (1, 768)), view_68, out=buf425)
    buf426 = buf421; del buf421  # reuse
    kernel9.run(buf423, buf426, 49152, 128, grid=grid(49152), stream=stream0)
    buf427 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf426, buf427, 768, 64, grid=grid(768), stream=stream0)
    buf428 = as_strided(buf423, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf423  # reuse
    kernel24.run(buf417, buf428, 6291456, grid=grid(6291456), stream=stream0)
    buf429 = as_strided(buf417, (8192, 768), (768, 1)); del buf417  # reuse
    aten.mm.out(as_strided(buf428, (8192, 768), (768, 1)), permute_402, out=buf429)
    del permute_402
    buf430 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf428, (768, 8192), (1, 768)), view_68, out=buf430)
    del view_68
    buf431 = buf426; del buf426  # reuse
    kernel9.run(buf428, buf431, 49152, 128, grid=grid(49152), stream=stream0)
    buf432 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf431, buf432, 768, 64, grid=grid(768), stream=stream0)
    buf433 = as_strided(buf428, (64, 128, 768), (98304, 768, 1)); del buf428  # reuse
    kernel26.run(buf401, buf419, buf424, buf429, primals_68, buf433, 6291456, grid=grid(6291456), stream=stream0)
    del primals_68
    buf434 = buf400; del buf400  # reuse
    kernel27.run(buf433, buf434, 8192, 768, grid=grid(8192), stream=stream0)
    buf435 = buf399; del buf399  # reuse
    kernel28.run(buf433, mul_95, buf435, 8192, 768, grid=grid(8192), stream=stream0)
    buf436 = buf433; del buf433  # reuse
    kernel29.run(buf436, div_50, buf434, mul_95, buf435, 6291456, grid=grid(6291456), stream=stream0)
    del div_50
    buf437 = as_strided(buf431, (768, 64), (1, 768)); del buf431  # reuse
    kernel30.run(buf401, buf419, buf424, buf429, mul_95, buf437, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_95
    buf438 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf437, buf438, 768, 64, grid=grid(768), stream=stream0)
    buf439 = buf437; del buf437  # reuse
    kernel31.run(buf401, buf419, buf424, buf429, buf439, 49152, 128, grid=grid(49152), stream=stream0)
    buf440 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf439, buf440, 768, 64, grid=grid(768), stream=stream0)
    buf441 = as_strided(buf429, (64, 128, 768), (98304, 768, 1)); del buf429  # reuse
    kernel12.run(buf436, gt_12, buf441, 6291456, grid=grid(6291456), stream=stream0)
    del gt_12
    buf442 = as_strided(buf394, (8192, 3072), (3072, 1)); del buf394  # reuse
    aten.mm.out(as_strided(buf441, (8192, 768), (768, 1)), permute_406, out=buf442)
    del permute_406
    buf443 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf441, (768, 8192), (1, 768)), view_66, out=buf443)
    del view_66
    buf444 = as_strided(buf439, (1, 768, 64), (49152, 1, 768)); del buf439  # reuse
    kernel9.run(buf441, buf444, 49152, 128, grid=grid(49152), stream=stream0)
    buf445 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf444, buf445, 768, 64, grid=grid(768), stream=stream0)
    buf446 = as_strided(buf442, (64, 128, 3072), (393216, 3072, 1)); del buf442  # reuse
    kernel13.run(buf446, add_270, 25165824, grid=grid(25165824), stream=stream0)
    del add_270
    buf447 = as_strided(buf441, (8192, 768), (768, 1)); del buf441  # reuse
    aten.mm.out(as_strided(buf446, (8192, 3072), (3072, 1)), permute_410, out=buf447)
    del permute_410
    buf448 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf446, (3072, 8192), (1, 3072)), view_64, out=buf448)
    del view_64
    buf449 = as_strided(buf414, (1, 3072, 32), (98304, 1, 3072)); del buf414  # reuse
    kernel14.run(buf446, buf449, 98304, 256, grid=grid(98304), stream=stream0)
    buf450 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf449, buf450, 3072, 32, grid=grid(3072), stream=stream0)
    buf451 = buf435; del buf435  # reuse
    kernel16.run(buf436, buf447, primals_62, buf451, 8192, 768, grid=grid(8192), stream=stream0)
    buf452 = buf434; del buf434  # reuse
    kernel17.run(buf436, buf447, primals_62, mul_78, buf452, 8192, 768, grid=grid(8192), stream=stream0)
    buf453 = as_strided(buf424, (64, 128, 768), (98304, 768, 1)); del buf424  # reuse
    kernel18.run(div_51, buf436, buf447, primals_62, buf451, mul_78, buf452, buf453, 6291456, grid=grid(6291456), stream=stream0)
    del div_51
    del primals_62
    buf454 = as_strided(buf444, (768, 64), (1, 768)); del buf444  # reuse
    kernel19.run(buf436, buf447, mul_78, buf454, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_78
    buf455 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf454, buf455, 768, 64, grid=grid(768), stream=stream0)
    buf456 = buf454; del buf454  # reuse
    kernel20.run(buf436, buf447, buf456, 49152, 128, grid=grid(49152), stream=stream0)
    buf457 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf456, buf457, 768, 64, grid=grid(768), stream=stream0)
    buf458 = as_strided(buf447, (64, 128, 768), (98304, 768, 1)); del buf447  # reuse
    kernel12.run(buf453, gt_11, buf458, 6291456, grid=grid(6291456), stream=stream0)
    del gt_11
    buf459 = as_strided(buf436, (8192, 768), (768, 1)); del buf436  # reuse
    aten.mm.out(as_strided(buf458, (8192, 768), (768, 1)), permute_414, out=buf459)
    del permute_414
    buf460 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf458, (768, 8192), (1, 768)), view_62, out=buf460)
    del view_62
    buf461 = as_strided(buf456, (1, 768, 64), (49152, 1, 768)); del buf456  # reuse
    kernel9.run(buf458, buf461, 49152, 128, grid=grid(49152), stream=stream0)
    buf462 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf461, buf462, 768, 64, grid=grid(768), stream=stream0)
    buf463 = as_strided(buf458, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf458  # reuse
    kernel21.run(buf459, buf463, 6291456, grid=grid(6291456), stream=stream0)
    buf464 = as_strided(buf459, (768, 128, 64), (8192, 64, 1)); del buf459  # reuse
    aten.bmm.out(permute_419, as_strided(buf463, (768, 128, 64), (8192, 64, 1)), out=buf464)
    del permute_419
    buf465 = as_strided(buf415, (768, 128, 128), (16384, 128, 1)); del buf415  # reuse
    aten.bmm.out(as_strided(buf463, (768, 128, 64), (8192, 64, 1)), permute_420, out=buf465)
    del permute_420
    buf466 = as_strided(buf449, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf449  # reuse
    kernel22.run(buf465, gt_10, alias_99, buf466, 98304, 128, grid=grid(98304), stream=stream0)
    buf467 = as_strided(buf465, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf465  # reuse
    kernel23.run(buf467, gt_10, alias_99, buf466, 12582912, grid=grid(12582912), stream=stream0)
    del alias_99
    del gt_10
    buf468 = as_strided(buf463, (768, 64, 128), (8192, 128, 1)); del buf463  # reuse
    aten.bmm.out(permute_421, as_strided(buf467, (768, 128, 128), (16384, 128, 1)), out=buf468)
    del permute_421
    buf469 = as_strided(buf419, (768, 128, 64), (8192, 64, 1)); del buf419  # reuse
    aten.bmm.out(as_strided(buf467, (768, 128, 128), (16384, 128, 1)), permute_422, out=buf469)
    del permute_422
    buf470 = as_strided(buf401, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf401  # reuse
    kernel24.run(buf464, buf470, 6291456, grid=grid(6291456), stream=stream0)
    buf471 = as_strided(buf464, (8192, 768), (768, 1)); del buf464  # reuse
    aten.mm.out(as_strided(buf470, (8192, 768), (768, 1)), permute_426, out=buf471)
    del permute_426
    buf472 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf470, (768, 8192), (1, 768)), view_51, out=buf472)
    buf473 = buf461; del buf461  # reuse
    kernel9.run(buf470, buf473, 49152, 128, grid=grid(49152), stream=stream0)
    buf474 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf473, buf474, 768, 64, grid=grid(768), stream=stream0)
    buf475 = as_strided(buf470, (64, 128, 768), (98304, 768, 1)); del buf470  # reuse
    kernel25.run(buf468, buf475, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf476 = as_strided(buf468, (8192, 768), (768, 1)); del buf468  # reuse
    aten.mm.out(as_strided(buf475, (8192, 768), (768, 1)), permute_431, out=buf476)
    del permute_431
    buf477 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf475, (768, 8192), (1, 768)), view_51, out=buf477)
    buf478 = buf473; del buf473  # reuse
    kernel9.run(buf475, buf478, 49152, 128, grid=grid(49152), stream=stream0)
    buf479 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf478, buf479, 768, 64, grid=grid(768), stream=stream0)
    buf480 = as_strided(buf475, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf475  # reuse
    kernel24.run(buf469, buf480, 6291456, grid=grid(6291456), stream=stream0)
    buf481 = as_strided(buf469, (8192, 768), (768, 1)); del buf469  # reuse
    aten.mm.out(as_strided(buf480, (8192, 768), (768, 1)), permute_435, out=buf481)
    del permute_435
    buf482 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf480, (768, 8192), (1, 768)), view_51, out=buf482)
    del view_51
    buf483 = buf478; del buf478  # reuse
    kernel9.run(buf480, buf483, 49152, 128, grid=grid(49152), stream=stream0)
    buf484 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf483, buf484, 768, 64, grid=grid(768), stream=stream0)
    buf485 = as_strided(buf480, (64, 128, 768), (98304, 768, 1)); del buf480  # reuse
    kernel26.run(buf453, buf471, buf476, buf481, primals_52, buf485, 6291456, grid=grid(6291456), stream=stream0)
    del primals_52
    buf486 = buf452; del buf452  # reuse
    kernel27.run(buf485, buf486, 8192, 768, grid=grid(8192), stream=stream0)
    buf487 = buf451; del buf451  # reuse
    kernel28.run(buf485, mul_72, buf487, 8192, 768, grid=grid(8192), stream=stream0)
    buf488 = buf485; del buf485  # reuse
    kernel29.run(buf488, div_53, buf486, mul_72, buf487, 6291456, grid=grid(6291456), stream=stream0)
    del div_53
    buf489 = as_strided(buf483, (768, 64), (1, 768)); del buf483  # reuse
    kernel30.run(buf453, buf471, buf476, buf481, mul_72, buf489, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_72
    buf490 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf489, buf490, 768, 64, grid=grid(768), stream=stream0)
    buf491 = buf489; del buf489  # reuse
    kernel31.run(buf453, buf471, buf476, buf481, buf491, 49152, 128, grid=grid(49152), stream=stream0)
    buf492 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf491, buf492, 768, 64, grid=grid(768), stream=stream0)
    buf493 = as_strided(buf481, (64, 128, 768), (98304, 768, 1)); del buf481  # reuse
    kernel12.run(buf488, gt_9, buf493, 6291456, grid=grid(6291456), stream=stream0)
    del gt_9
    buf494 = as_strided(buf446, (8192, 3072), (3072, 1)); del buf446  # reuse
    aten.mm.out(as_strided(buf493, (8192, 768), (768, 1)), permute_439, out=buf494)
    del permute_439
    buf495 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf493, (768, 8192), (1, 768)), view_49, out=buf495)
    del view_49
    buf496 = as_strided(buf491, (1, 768, 64), (49152, 1, 768)); del buf491  # reuse
    kernel9.run(buf493, buf496, 49152, 128, grid=grid(49152), stream=stream0)
    buf497 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf496, buf497, 768, 64, grid=grid(768), stream=stream0)
    buf498 = as_strided(buf494, (64, 128, 3072), (393216, 3072, 1)); del buf494  # reuse
    kernel13.run(buf498, add_281, 25165824, grid=grid(25165824), stream=stream0)
    del add_281
    buf499 = as_strided(buf493, (8192, 768), (768, 1)); del buf493  # reuse
    aten.mm.out(as_strided(buf498, (8192, 3072), (3072, 1)), permute_443, out=buf499)
    del permute_443
    buf500 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf498, (3072, 8192), (1, 3072)), view_47, out=buf500)
    del view_47
    buf501 = as_strided(buf466, (1, 3072, 32), (98304, 1, 3072)); del buf466  # reuse
    kernel14.run(buf498, buf501, 98304, 256, grid=grid(98304), stream=stream0)
    buf502 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf501, buf502, 3072, 32, grid=grid(3072), stream=stream0)
    buf503 = buf487; del buf487  # reuse
    kernel16.run(buf488, buf499, primals_46, buf503, 8192, 768, grid=grid(8192), stream=stream0)
    buf504 = buf486; del buf486  # reuse
    kernel17.run(buf488, buf499, primals_46, mul_55, buf504, 8192, 768, grid=grid(8192), stream=stream0)
    buf505 = as_strided(buf476, (64, 128, 768), (98304, 768, 1)); del buf476  # reuse
    kernel18.run(div_54, buf488, buf499, primals_46, buf503, mul_55, buf504, buf505, 6291456, grid=grid(6291456), stream=stream0)
    del div_54
    del primals_46
    buf506 = as_strided(buf496, (768, 64), (1, 768)); del buf496  # reuse
    kernel19.run(buf488, buf499, mul_55, buf506, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_55
    buf507 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf506, buf507, 768, 64, grid=grid(768), stream=stream0)
    buf508 = buf506; del buf506  # reuse
    kernel20.run(buf488, buf499, buf508, 49152, 128, grid=grid(49152), stream=stream0)
    buf509 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf508, buf509, 768, 64, grid=grid(768), stream=stream0)
    buf510 = as_strided(buf499, (64, 128, 768), (98304, 768, 1)); del buf499  # reuse
    kernel12.run(buf505, gt_8, buf510, 6291456, grid=grid(6291456), stream=stream0)
    del gt_8
    buf511 = as_strided(buf488, (8192, 768), (768, 1)); del buf488  # reuse
    aten.mm.out(as_strided(buf510, (8192, 768), (768, 1)), permute_447, out=buf511)
    del permute_447
    buf512 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf510, (768, 8192), (1, 768)), view_45, out=buf512)
    del view_45
    buf513 = as_strided(buf508, (1, 768, 64), (49152, 1, 768)); del buf508  # reuse
    kernel9.run(buf510, buf513, 49152, 128, grid=grid(49152), stream=stream0)
    buf514 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf513, buf514, 768, 64, grid=grid(768), stream=stream0)
    buf515 = as_strided(buf510, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf510  # reuse
    kernel21.run(buf511, buf515, 6291456, grid=grid(6291456), stream=stream0)
    buf516 = as_strided(buf511, (768, 128, 64), (8192, 64, 1)); del buf511  # reuse
    aten.bmm.out(permute_452, as_strided(buf515, (768, 128, 64), (8192, 64, 1)), out=buf516)
    del permute_452
    buf517 = as_strided(buf467, (768, 128, 128), (16384, 128, 1)); del buf467  # reuse
    aten.bmm.out(as_strided(buf515, (768, 128, 64), (8192, 64, 1)), permute_453, out=buf517)
    del permute_453
    buf518 = as_strided(buf501, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf501  # reuse
    kernel22.run(buf517, gt_7, alias_101, buf518, 98304, 128, grid=grid(98304), stream=stream0)
    buf519 = as_strided(buf517, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf517  # reuse
    kernel23.run(buf519, gt_7, alias_101, buf518, 12582912, grid=grid(12582912), stream=stream0)
    del alias_101
    del gt_7
    buf520 = as_strided(buf515, (768, 64, 128), (8192, 128, 1)); del buf515  # reuse
    aten.bmm.out(permute_454, as_strided(buf519, (768, 128, 128), (16384, 128, 1)), out=buf520)
    del permute_454
    buf521 = as_strided(buf471, (768, 128, 64), (8192, 64, 1)); del buf471  # reuse
    aten.bmm.out(as_strided(buf519, (768, 128, 128), (16384, 128, 1)), permute_455, out=buf521)
    del permute_455
    buf522 = as_strided(buf453, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf453  # reuse
    kernel24.run(buf516, buf522, 6291456, grid=grid(6291456), stream=stream0)
    buf523 = as_strided(buf516, (8192, 768), (768, 1)); del buf516  # reuse
    aten.mm.out(as_strided(buf522, (8192, 768), (768, 1)), permute_459, out=buf523)
    del permute_459
    buf524 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf522, (768, 8192), (1, 768)), view_34, out=buf524)
    buf525 = buf513; del buf513  # reuse
    kernel9.run(buf522, buf525, 49152, 128, grid=grid(49152), stream=stream0)
    buf526 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf525, buf526, 768, 64, grid=grid(768), stream=stream0)
    buf527 = as_strided(buf522, (64, 128, 768), (98304, 768, 1)); del buf522  # reuse
    kernel25.run(buf520, buf527, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf528 = as_strided(buf520, (8192, 768), (768, 1)); del buf520  # reuse
    aten.mm.out(as_strided(buf527, (8192, 768), (768, 1)), permute_464, out=buf528)
    del permute_464
    buf529 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf527, (768, 8192), (1, 768)), view_34, out=buf529)
    buf530 = buf525; del buf525  # reuse
    kernel9.run(buf527, buf530, 49152, 128, grid=grid(49152), stream=stream0)
    buf531 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf530, buf531, 768, 64, grid=grid(768), stream=stream0)
    buf532 = as_strided(buf527, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf527  # reuse
    kernel24.run(buf521, buf532, 6291456, grid=grid(6291456), stream=stream0)
    buf533 = as_strided(buf521, (8192, 768), (768, 1)); del buf521  # reuse
    aten.mm.out(as_strided(buf532, (8192, 768), (768, 1)), permute_468, out=buf533)
    del permute_468
    buf534 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf532, (768, 8192), (1, 768)), view_34, out=buf534)
    del view_34
    buf535 = buf530; del buf530  # reuse
    kernel9.run(buf532, buf535, 49152, 128, grid=grid(49152), stream=stream0)
    buf536 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf535, buf536, 768, 64, grid=grid(768), stream=stream0)
    buf537 = as_strided(buf532, (64, 128, 768), (98304, 768, 1)); del buf532  # reuse
    kernel26.run(buf505, buf523, buf528, buf533, primals_36, buf537, 6291456, grid=grid(6291456), stream=stream0)
    del primals_36
    buf538 = buf504; del buf504  # reuse
    kernel27.run(buf537, buf538, 8192, 768, grid=grid(8192), stream=stream0)
    buf539 = buf503; del buf503  # reuse
    kernel28.run(buf537, mul_49, buf539, 8192, 768, grid=grid(8192), stream=stream0)
    buf540 = buf537; del buf537  # reuse
    kernel29.run(buf540, div_56, buf538, mul_49, buf539, 6291456, grid=grid(6291456), stream=stream0)
    del div_56
    buf541 = as_strided(buf535, (768, 64), (1, 768)); del buf535  # reuse
    kernel30.run(buf505, buf523, buf528, buf533, mul_49, buf541, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_49
    buf542 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf541, buf542, 768, 64, grid=grid(768), stream=stream0)
    buf543 = buf541; del buf541  # reuse
    kernel31.run(buf505, buf523, buf528, buf533, buf543, 49152, 128, grid=grid(49152), stream=stream0)
    buf544 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf543, buf544, 768, 64, grid=grid(768), stream=stream0)
    buf545 = as_strided(buf533, (64, 128, 768), (98304, 768, 1)); del buf533  # reuse
    kernel12.run(buf540, gt_6, buf545, 6291456, grid=grid(6291456), stream=stream0)
    del gt_6
    buf546 = as_strided(buf498, (8192, 3072), (3072, 1)); del buf498  # reuse
    aten.mm.out(as_strided(buf545, (8192, 768), (768, 1)), permute_472, out=buf546)
    del permute_472
    buf547 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf545, (768, 8192), (1, 768)), view_32, out=buf547)
    del view_32
    buf548 = as_strided(buf543, (1, 768, 64), (49152, 1, 768)); del buf543  # reuse
    kernel9.run(buf545, buf548, 49152, 128, grid=grid(49152), stream=stream0)
    buf549 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf548, buf549, 768, 64, grid=grid(768), stream=stream0)
    buf550 = as_strided(buf546, (64, 128, 3072), (393216, 3072, 1)); del buf546  # reuse
    kernel13.run(buf550, add_292, 25165824, grid=grid(25165824), stream=stream0)
    del add_292
    buf551 = as_strided(buf545, (8192, 768), (768, 1)); del buf545  # reuse
    aten.mm.out(as_strided(buf550, (8192, 3072), (3072, 1)), permute_476, out=buf551)
    del permute_476
    buf552 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf550, (3072, 8192), (1, 3072)), view_30, out=buf552)
    del view_30
    buf553 = as_strided(buf518, (1, 3072, 32), (98304, 1, 3072)); del buf518  # reuse
    kernel14.run(buf550, buf553, 98304, 256, grid=grid(98304), stream=stream0)
    buf554 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf553, buf554, 3072, 32, grid=grid(3072), stream=stream0)
    buf555 = buf539; del buf539  # reuse
    kernel16.run(buf540, buf551, primals_30, buf555, 8192, 768, grid=grid(8192), stream=stream0)
    buf556 = buf538; del buf538  # reuse
    kernel17.run(buf540, buf551, primals_30, mul_32, buf556, 8192, 768, grid=grid(8192), stream=stream0)
    buf557 = as_strided(buf528, (64, 128, 768), (98304, 768, 1)); del buf528  # reuse
    kernel18.run(div_57, buf540, buf551, primals_30, buf555, mul_32, buf556, buf557, 6291456, grid=grid(6291456), stream=stream0)
    del div_57
    del primals_30
    buf558 = as_strided(buf548, (768, 64), (1, 768)); del buf548  # reuse
    kernel19.run(buf540, buf551, mul_32, buf558, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_32
    buf559 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf558, buf559, 768, 64, grid=grid(768), stream=stream0)
    buf560 = buf558; del buf558  # reuse
    kernel20.run(buf540, buf551, buf560, 49152, 128, grid=grid(49152), stream=stream0)
    buf561 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf560, buf561, 768, 64, grid=grid(768), stream=stream0)
    buf562 = as_strided(buf551, (64, 128, 768), (98304, 768, 1)); del buf551  # reuse
    kernel12.run(buf557, gt_5, buf562, 6291456, grid=grid(6291456), stream=stream0)
    del gt_5
    buf563 = as_strided(buf540, (8192, 768), (768, 1)); del buf540  # reuse
    aten.mm.out(as_strided(buf562, (8192, 768), (768, 1)), permute_480, out=buf563)
    del permute_480
    buf564 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf562, (768, 8192), (1, 768)), view_28, out=buf564)
    del view_28
    buf565 = as_strided(buf560, (1, 768, 64), (49152, 1, 768)); del buf560  # reuse
    kernel9.run(buf562, buf565, 49152, 128, grid=grid(49152), stream=stream0)
    buf566 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf565, buf566, 768, 64, grid=grid(768), stream=stream0)
    buf567 = as_strided(buf562, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf562  # reuse
    kernel21.run(buf563, buf567, 6291456, grid=grid(6291456), stream=stream0)
    buf568 = as_strided(buf563, (768, 128, 64), (8192, 64, 1)); del buf563  # reuse
    aten.bmm.out(permute_485, as_strided(buf567, (768, 128, 64), (8192, 64, 1)), out=buf568)
    del permute_485
    buf569 = as_strided(buf519, (768, 128, 128), (16384, 128, 1)); del buf519  # reuse
    aten.bmm.out(as_strided(buf567, (768, 128, 64), (8192, 64, 1)), permute_486, out=buf569)
    del permute_486
    buf570 = as_strided(buf553, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf553  # reuse
    kernel22.run(buf569, gt_4, alias_103, buf570, 98304, 128, grid=grid(98304), stream=stream0)
    buf571 = as_strided(buf569, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf569  # reuse
    kernel23.run(buf571, gt_4, alias_103, buf570, 12582912, grid=grid(12582912), stream=stream0)
    del alias_103
    del gt_4
    buf572 = as_strided(buf567, (768, 64, 128), (8192, 128, 1)); del buf567  # reuse
    aten.bmm.out(permute_487, as_strided(buf571, (768, 128, 128), (16384, 128, 1)), out=buf572)
    del permute_487
    buf573 = as_strided(buf523, (768, 128, 64), (8192, 64, 1)); del buf523  # reuse
    aten.bmm.out(as_strided(buf571, (768, 128, 128), (16384, 128, 1)), permute_488, out=buf573)
    del permute_488
    buf574 = as_strided(buf505, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf505  # reuse
    kernel24.run(buf568, buf574, 6291456, grid=grid(6291456), stream=stream0)
    buf575 = as_strided(buf568, (8192, 768), (768, 1)); del buf568  # reuse
    aten.mm.out(as_strided(buf574, (8192, 768), (768, 1)), permute_492, out=buf575)
    del permute_492
    buf576 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf574, (768, 8192), (1, 768)), view_17, out=buf576)
    buf577 = buf565; del buf565  # reuse
    kernel9.run(buf574, buf577, 49152, 128, grid=grid(49152), stream=stream0)
    buf578 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf577, buf578, 768, 64, grid=grid(768), stream=stream0)
    buf579 = as_strided(buf574, (64, 128, 768), (98304, 768, 1)); del buf574  # reuse
    kernel25.run(buf572, buf579, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf580 = as_strided(buf572, (8192, 768), (768, 1)); del buf572  # reuse
    aten.mm.out(as_strided(buf579, (8192, 768), (768, 1)), permute_497, out=buf580)
    del permute_497
    buf581 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf579, (768, 8192), (1, 768)), view_17, out=buf581)
    buf582 = buf577; del buf577  # reuse
    kernel9.run(buf579, buf582, 49152, 128, grid=grid(49152), stream=stream0)
    buf583 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf582, buf583, 768, 64, grid=grid(768), stream=stream0)
    buf584 = as_strided(buf579, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf579  # reuse
    kernel24.run(buf573, buf584, 6291456, grid=grid(6291456), stream=stream0)
    buf585 = as_strided(buf573, (8192, 768), (768, 1)); del buf573  # reuse
    aten.mm.out(as_strided(buf584, (8192, 768), (768, 1)), permute_501, out=buf585)
    del permute_501
    buf586 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf584, (768, 8192), (1, 768)), view_17, out=buf586)
    del view_17
    buf587 = buf582; del buf582  # reuse
    kernel9.run(buf584, buf587, 49152, 128, grid=grid(49152), stream=stream0)
    buf588 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf587, buf588, 768, 64, grid=grid(768), stream=stream0)
    buf589 = as_strided(buf584, (64, 128, 768), (98304, 768, 1)); del buf584  # reuse
    kernel26.run(buf557, buf575, buf580, buf585, primals_20, buf589, 6291456, grid=grid(6291456), stream=stream0)
    del primals_20
    buf590 = buf556; del buf556  # reuse
    kernel27.run(buf589, buf590, 8192, 768, grid=grid(8192), stream=stream0)
    buf591 = buf555; del buf555  # reuse
    kernel28.run(buf589, mul_26, buf591, 8192, 768, grid=grid(8192), stream=stream0)
    buf592 = buf589; del buf589  # reuse
    kernel29.run(buf592, div_59, buf590, mul_26, buf591, 6291456, grid=grid(6291456), stream=stream0)
    del div_59
    buf593 = as_strided(buf587, (768, 64), (1, 768)); del buf587  # reuse
    kernel30.run(buf557, buf575, buf580, buf585, mul_26, buf593, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_26
    buf594 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf593, buf594, 768, 64, grid=grid(768), stream=stream0)
    buf595 = buf593; del buf593  # reuse
    kernel31.run(buf557, buf575, buf580, buf585, buf595, 49152, 128, grid=grid(49152), stream=stream0)
    buf596 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf595, buf596, 768, 64, grid=grid(768), stream=stream0)
    buf597 = as_strided(buf585, (64, 128, 768), (98304, 768, 1)); del buf585  # reuse
    kernel12.run(buf592, gt_3, buf597, 6291456, grid=grid(6291456), stream=stream0)
    del gt_3
    buf598 = as_strided(buf550, (8192, 3072), (3072, 1)); del buf550  # reuse
    aten.mm.out(as_strided(buf597, (8192, 768), (768, 1)), permute_505, out=buf598)
    del permute_505
    buf599 = empty_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf597, (768, 8192), (1, 768)), view_15, out=buf599)
    del view_15
    buf600 = as_strided(buf595, (1, 768, 64), (49152, 1, 768)); del buf595  # reuse
    kernel9.run(buf597, buf600, 49152, 128, grid=grid(49152), stream=stream0)
    buf601 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf600, buf601, 768, 64, grid=grid(768), stream=stream0)
    buf602 = as_strided(buf598, (64, 128, 3072), (393216, 3072, 1)); del buf598  # reuse
    kernel13.run(buf602, add_303, 25165824, grid=grid(25165824), stream=stream0)
    del add_303
    buf603 = as_strided(buf597, (8192, 768), (768, 1)); del buf597  # reuse
    aten.mm.out(as_strided(buf602, (8192, 3072), (3072, 1)), permute_509, out=buf603)
    del permute_509
    buf604 = empty_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf602, (3072, 8192), (1, 3072)), view_13, out=buf604)
    del view_13
    buf605 = as_strided(buf570, (1, 3072, 32), (98304, 1, 3072)); del buf570  # reuse
    kernel14.run(buf602, buf605, 98304, 256, grid=grid(98304), stream=stream0)
    del buf602
    buf606 = empty_strided((1, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    kernel15.run(buf605, buf606, 3072, 32, grid=grid(3072), stream=stream0)
    buf607 = buf591; del buf591  # reuse
    kernel16.run(buf592, buf603, primals_14, buf607, 8192, 768, grid=grid(8192), stream=stream0)
    buf608 = buf590; del buf590  # reuse
    kernel17.run(buf592, buf603, primals_14, mul_9, buf608, 8192, 768, grid=grid(8192), stream=stream0)
    buf609 = as_strided(buf580, (64, 128, 768), (98304, 768, 1)); del buf580  # reuse
    kernel18.run(div_60, buf592, buf603, primals_14, buf607, mul_9, buf608, buf609, 6291456, grid=grid(6291456), stream=stream0)
    del div_60
    del primals_14
    buf610 = as_strided(buf600, (768, 64), (1, 768)); del buf600  # reuse
    kernel19.run(buf592, buf603, mul_9, buf610, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_9
    buf611 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf610, buf611, 768, 64, grid=grid(768), stream=stream0)
    buf612 = buf610; del buf610  # reuse
    kernel20.run(buf592, buf603, buf612, 49152, 128, grid=grid(49152), stream=stream0)
    buf613 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf612, buf613, 768, 64, grid=grid(768), stream=stream0)
    buf614 = as_strided(buf603, (64, 128, 768), (98304, 768, 1)); del buf603  # reuse
    kernel12.run(buf609, gt_2, buf614, 6291456, grid=grid(6291456), stream=stream0)
    del gt_2
    buf615 = as_strided(buf592, (8192, 768), (768, 1)); del buf592  # reuse
    aten.mm.out(as_strided(buf614, (8192, 768), (768, 1)), permute_513, out=buf615)
    del permute_513
    buf616 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf614, (768, 8192), (1, 768)), view_11, out=buf616)
    del view_11
    buf617 = as_strided(buf612, (1, 768, 64), (49152, 1, 768)); del buf612  # reuse
    kernel9.run(buf614, buf617, 49152, 128, grid=grid(49152), stream=stream0)
    buf618 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf617, buf618, 768, 64, grid=grid(768), stream=stream0)
    buf619 = as_strided(buf614, (64, 12, 128, 64), (98304, 8192, 64, 1)); del buf614  # reuse
    kernel21.run(buf615, buf619, 6291456, grid=grid(6291456), stream=stream0)
    buf620 = as_strided(buf615, (768, 128, 64), (8192, 64, 1)); del buf615  # reuse
    aten.bmm.out(permute_518, as_strided(buf619, (768, 128, 64), (8192, 64, 1)), out=buf620)
    del permute_518
    buf621 = as_strided(buf571, (768, 128, 128), (16384, 128, 1)); del buf571  # reuse
    aten.bmm.out(as_strided(buf619, (768, 128, 64), (8192, 64, 1)), permute_519, out=buf621)
    del permute_519
    buf622 = as_strided(buf605, (64, 12, 128, 1), (1536, 128, 1, 98304)); del buf605  # reuse
    kernel22.run(buf621, gt_1, alias_105, buf622, 98304, 128, grid=grid(98304), stream=stream0)
    buf623 = as_strided(buf621, (64, 12, 128, 128), (196608, 16384, 128, 1)); del buf621  # reuse
    kernel23.run(buf623, gt_1, alias_105, buf622, 12582912, grid=grid(12582912), stream=stream0)
    del alias_105
    del gt_1
    buf624 = as_strided(buf619, (768, 64, 128), (8192, 128, 1)); del buf619  # reuse
    aten.bmm.out(permute_520, as_strided(buf623, (768, 128, 128), (16384, 128, 1)), out=buf624)
    del permute_520
    buf625 = as_strided(buf575, (768, 128, 64), (8192, 64, 1)); del buf575  # reuse
    aten.bmm.out(as_strided(buf623, (768, 128, 128), (16384, 128, 1)), permute_521, out=buf625)
    del buf623
    del permute_521
    buf626 = as_strided(buf557, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf557  # reuse
    kernel24.run(buf620, buf626, 6291456, grid=grid(6291456), stream=stream0)
    buf627 = as_strided(buf620, (8192, 768), (768, 1)); del buf620  # reuse
    aten.mm.out(as_strided(buf626, (8192, 768), (768, 1)), permute_525, out=buf627)
    del permute_525
    buf628 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf626, (768, 8192), (1, 768)), view, out=buf628)
    buf629 = buf617; del buf617  # reuse
    kernel9.run(buf626, buf629, 49152, 128, grid=grid(49152), stream=stream0)
    buf630 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf629, buf630, 768, 64, grid=grid(768), stream=stream0)
    buf631 = as_strided(buf626, (64, 128, 768), (98304, 768, 1)); del buf626  # reuse
    kernel25.run(buf624, buf631, 8192, 768, grid=grid(8192, 768), stream=stream0)
    buf632 = as_strided(buf624, (8192, 768), (768, 1)); del buf624  # reuse
    aten.mm.out(as_strided(buf631, (8192, 768), (768, 1)), permute_530, out=buf632)
    del permute_530
    buf633 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf631, (768, 8192), (1, 768)), view, out=buf633)
    buf634 = buf629; del buf629  # reuse
    kernel9.run(buf631, buf634, 49152, 128, grid=grid(49152), stream=stream0)
    buf635 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf634, buf635, 768, 64, grid=grid(768), stream=stream0)
    buf636 = as_strided(buf631, (64, 128, 12, 64), (98304, 768, 64, 1)); del buf631  # reuse
    kernel24.run(buf625, buf636, 6291456, grid=grid(6291456), stream=stream0)
    buf637 = as_strided(buf625, (8192, 768), (768, 1)); del buf625  # reuse
    aten.mm.out(as_strided(buf636, (8192, 768), (768, 1)), permute_534, out=buf637)
    del permute_534
    buf638 = empty_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    aten.mm.out(as_strided(buf636, (768, 8192), (1, 768)), view, out=buf638)
    del view
    buf639 = buf634; del buf634  # reuse
    kernel9.run(buf636, buf639, 49152, 128, grid=grid(49152), stream=stream0)
    del buf636
    buf640 = empty_strided((1, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel8.run(buf639, buf640, 768, 64, grid=grid(768), stream=stream0)
    buf641 = buf609; del buf609  # reuse
    kernel32.run(buf641, buf627, buf632, buf637, gt, 6291456, grid=grid(6291456), stream=stream0)
    del buf627
    del buf632
    del gt
    buf642 = buf608; del buf608  # reuse
    kernel5.run(buf641, primals_4, buf642, 8192, 768, grid=grid(8192), stream=stream0)
    buf643 = buf607; del buf607  # reuse
    kernel6.run(buf641, primals_4, mul_1, buf643, 8192, 768, grid=grid(8192), stream=stream0)
    buf644 = as_strided(buf637, (64, 128, 768), (98304, 768, 1)); del buf637  # reuse
    kernel11.run(div_62, buf641, primals_4, buf642, mul_1, buf643, buf644, 6291456, grid=grid(6291456), stream=stream0)
    del buf642
    del buf643
    del div_62
    del primals_4
    buf645 = as_strided(buf639, (768, 64), (1, 768)); del buf639  # reuse
    kernel7.run(buf641, mul_1, buf645, 49152, 128, grid=grid(49152), stream=stream0)
    del mul_1
    buf646 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf645, buf646, 768, 64, grid=grid(768), stream=stream0)
    buf647 = buf645; del buf645  # reuse
    kernel9.run(buf641, buf647, 49152, 128, grid=grid(49152), stream=stream0)
    del buf641
    buf648 = empty_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    kernel8.run(buf647, buf648, 768, 64, grid=grid(768), stream=stream0)
    del buf647
    buf649 = as_strided(buf622, (1, 128, 768), (98304, 768, 1)); del buf622  # reuse
    kernel33.run(buf644, buf649, 98304, 64, grid=grid(98304), stream=stream0)
    buf650 = empty_strided((512, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel34.run(buf650, 393216, grid=grid(393216), stream=stream0)
    kernel35.run(view_506, buf649, buf650, 98304, grid=grid(98304), stream=stream0)
    del buf649
    del view_506
    buf652 = empty_strided((2, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel36.run(buf652, 1536, grid=grid(1536), stream=stream0)
    kernel37.run(slice_2, buf644, buf652, 6291456, grid=grid(6291456), stream=stream0)
    del slice_2
    buf654 = empty_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
    kernel38.run(buf654, 23440896, grid=grid(23440896), stream=stream0)
    kernel39.run(view_509, buf644, buf654, 6291456, grid=grid(6291456), stream=stream0)
    del buf644
    del view_509
    buf656 = buf5; del buf5  # reuse
    kernel40.run(buf656, buf654, 23440896, grid=grid(23440896), stream=stream0)
    return (buf656, buf652, buf650, buf646, buf648, as_strided(buf638, (768, 768), (768, 1)), as_strided(buf640, (768, ), (1, )), as_strided(buf633, (768, 768), (768, 1)), as_strided(buf635, (768, ), (1, )), as_strided(buf628, (768, 768), (768, 1)), as_strided(buf630, (768, ), (1, )), as_strided(buf616, (768, 768), (768, 1)), as_strided(buf618, (768, ), (1, )), buf611, buf613, as_strided(buf604, (3072, 768), (768, 1)), as_strided(buf606, (3072, ), (1, )), as_strided(buf599, (768, 3072), (3072, 1)), as_strided(buf601, (768, ), (1, )), buf594, buf596, as_strided(buf586, (768, 768), (768, 1)), as_strided(buf588, (768, ), (1, )), as_strided(buf581, (768, 768), (768, 1)), as_strided(buf583, (768, ), (1, )), as_strided(buf576, (768, 768), (768, 1)), as_strided(buf578, (768, ), (1, )), as_strided(buf564, (768, 768), (768, 1)), as_strided(buf566, (768, ), (1, )), buf559, buf561, as_strided(buf552, (3072, 768), (768, 1)), as_strided(buf554, (3072, ), (1, )), as_strided(buf547, (768, 3072), (3072, 1)), as_strided(buf549, (768, ), (1, )), buf542, buf544, as_strided(buf534, (768, 768), (768, 1)), as_strided(buf536, (768, ), (1, )), as_strided(buf529, (768, 768), (768, 1)), as_strided(buf531, (768, ), (1, )), as_strided(buf524, (768, 768), (768, 1)), as_strided(buf526, (768, ), (1, )), as_strided(buf512, (768, 768), (768, 1)), as_strided(buf514, (768, ), (1, )), buf507, buf509, as_strided(buf500, (3072, 768), (768, 1)), as_strided(buf502, (3072, ), (1, )), as_strided(buf495, (768, 3072), (3072, 1)), as_strided(buf497, (768, ), (1, )), buf490, buf492, as_strided(buf482, (768, 768), (768, 1)), as_strided(buf484, (768, ), (1, )), as_strided(buf477, (768, 768), (768, 1)), as_strided(buf479, (768, ), (1, )), as_strided(buf472, (768, 768), (768, 1)), as_strided(buf474, (768, ), (1, )), as_strided(buf460, (768, 768), (768, 1)), as_strided(buf462, (768, ), (1, )), buf455, buf457, as_strided(buf448, (3072, 768), (768, 1)), as_strided(buf450, (3072, ), (1, )), as_strided(buf443, (768, 3072), (3072, 1)), as_strided(buf445, (768, ), (1, )), buf438, buf440, as_strided(buf430, (768, 768), (768, 1)), as_strided(buf432, (768, ), (1, )), as_strided(buf425, (768, 768), (768, 1)), as_strided(buf427, (768, ), (1, )), as_strided(buf420, (768, 768), (768, 1)), as_strided(buf422, (768, ), (1, )), as_strided(buf408, (768, 768), (768, 1)), as_strided(buf410, (768, ), (1, )), buf403, buf405, as_strided(buf396, (3072, 768), (768, 1)), as_strided(buf398, (3072, ), (1, )), as_strided(buf391, (768, 3072), (3072, 1)), as_strided(buf393, (768, ), (1, )), buf386, buf388, as_strided(buf378, (768, 768), (768, 1)), as_strided(buf380, (768, ), (1, )), as_strided(buf373, (768, 768), (768, 1)), as_strided(buf375, (768, ), (1, )), as_strided(buf368, (768, 768), (768, 1)), as_strided(buf370, (768, ), (1, )), as_strided(buf356, (768, 768), (768, 1)), as_strided(buf358, (768, ), (1, )), buf351, buf353, as_strided(buf344, (3072, 768), (768, 1)), as_strided(buf346, (3072, ), (1, )), as_strided(buf339, (768, 3072), (3072, 1)), as_strided(buf341, (768, ), (1, )), buf334, buf336, as_strided(buf326, (768, 768), (768, 1)), as_strided(buf328, (768, ), (1, )), as_strided(buf321, (768, 768), (768, 1)), as_strided(buf323, (768, ), (1, )), as_strided(buf316, (768, 768), (768, 1)), as_strided(buf318, (768, ), (1, )), as_strided(buf304, (768, 768), (768, 1)), as_strided(buf306, (768, ), (1, )), buf299, buf301, as_strided(buf292, (3072, 768), (768, 1)), as_strided(buf294, (3072, ), (1, )), as_strided(buf287, (768, 3072), (3072, 1)), as_strided(buf289, (768, ), (1, )), buf282, buf284, as_strided(buf274, (768, 768), (768, 1)), as_strided(buf276, (768, ), (1, )), as_strided(buf269, (768, 768), (768, 1)), as_strided(buf271, (768, ), (1, )), as_strided(buf264, (768, 768), (768, 1)), as_strided(buf266, (768, ), (1, )), as_strided(buf252, (768, 768), (768, 1)), as_strided(buf254, (768, ), (1, )), buf247, buf249, as_strided(buf240, (3072, 768), (768, 1)), as_strided(buf242, (3072, ), (1, )), as_strided(buf235, (768, 3072), (3072, 1)), as_strided(buf237, (768, ), (1, )), buf230, buf232, as_strided(buf222, (768, 768), (768, 1)), as_strided(buf224, (768, ), (1, )), as_strided(buf217, (768, 768), (768, 1)), as_strided(buf219, (768, ), (1, )), as_strided(buf212, (768, 768), (768, 1)), as_strided(buf214, (768, ), (1, )), as_strided(buf200, (768, 768), (768, 1)), as_strided(buf202, (768, ), (1, )), buf195, buf197, as_strided(buf188, (3072, 768), (768, 1)), as_strided(buf190, (3072, ), (1, )), as_strided(buf183, (768, 3072), (3072, 1)), as_strided(buf185, (768, ), (1, )), buf178, buf180, as_strided(buf170, (768, 768), (768, 1)), as_strided(buf172, (768, ), (1, )), as_strided(buf165, (768, 768), (768, 1)), as_strided(buf167, (768, ), (1, )), as_strided(buf160, (768, 768), (768, 1)), as_strided(buf162, (768, ), (1, )), as_strided(buf148, (768, 768), (768, 1)), as_strided(buf150, (768, ), (1, )), buf143, buf145, as_strided(buf136, (3072, 768), (768, 1)), as_strided(buf138, (3072, ), (1, )), as_strided(buf131, (768, 3072), (3072, 1)), as_strided(buf133, (768, ), (1, )), buf126, buf128, as_strided(buf118, (768, 768), (768, 1)), as_strided(buf120, (768, ), (1, )), as_strided(buf113, (768, 768), (768, 1)), as_strided(buf115, (768, ), (1, )), as_strided(buf108, (768, 768), (768, 1)), as_strided(buf110, (768, ), (1, )), as_strided(buf96, (768, 768), (768, 1)), as_strided(buf98, (768, ), (1, )), buf91, buf93, as_strided(buf84, (3072, 768), (768, 1)), as_strided(buf86, (3072, ), (1, )), as_strided(buf79, (768, 3072), (3072, 1)), as_strided(buf81, (768, ), (1, )), buf74, buf76, as_strided(buf66, (768, 768), (768, 1)), as_strided(buf68, (768, ), (1, )), as_strided(buf61, (768, 768), (768, 1)), as_strided(buf63, (768, ), (1, )), as_strided(buf56, (768, 768), (768, 1)), as_strided(buf58, (768, ), (1, )), as_strided(buf44, (768, 768), (768, 1)), as_strided(buf46, (768, ), (1, )), buf39, buf41, as_strided(buf32, (3072, 768), (768, 1)), as_strided(buf34, (3072, ), (1, )), as_strided(buf27, (768, 3072), (3072, 1)), as_strided(buf29, (768, ), (1, )), buf22, buf24, as_strided(buf15, (768, 768), (768, 1)), as_strided(buf17, (768, ), (1, )), buf10, buf12, as_strided(buf6, (30522, ), (1, )), None, None, None, None, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda', dtype=torch.float32)
    slice_2 = rand_strided((1, 128), (512, 1), device='cuda', dtype=torch.int64)
    mul_1 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    gt = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    view = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_1 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_11 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_2 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_9 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_13 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_15 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_3 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_26 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_17 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_4 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_28 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_5 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_32 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_30 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_32 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_6 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_49 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_34 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_7 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_45 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_8 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_55 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_47 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_49 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_9 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_72 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_51 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_10 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_62 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_11 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_78 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_64 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_66 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_12 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_95 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_68 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_13 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_79 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_14 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_101 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_81 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_83 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_15 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_118 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_85 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_16 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_96 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_17 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_124 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_98 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_100 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_18 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_141 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_102 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_19 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_113 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_20 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_147 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_115 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_117 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_21 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_164 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_119 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_22 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_130 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_23 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_170 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_132 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_134 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_24 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_187 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_136 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_25 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_147 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_26 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_193 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_149 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_151 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_27 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_210 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_153 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_28 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_164 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_29 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_216 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_166 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_168 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_30 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_233 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_170 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_31 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_181 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_32 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_239 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_183 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_185 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_33 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_256 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_187 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_34 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.bool)
    view_198 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    gt_35 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_262 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_200 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    view_202 = rand_strided((8192, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    gt_36 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.bool)
    mul_279 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_204 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    mul_294 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    view_206 = rand_strided((8192, 768), (768, 1), device='cuda', dtype=torch.float32)
    sub_53 = rand_strided((8192, 30522), (30522, 1), device='cuda', dtype=torch.float32)
    unsqueeze_2 = rand_strided((8192, 1), (1, 1), device='cuda', dtype=torch.int64)
    permute_134 = rand_strided((30522, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_25 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    add_175 = rand_strided((64, 128, 768), (98304, 768, 1), device='cuda', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_26 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_182 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_27 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_155 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_156 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_83 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_157 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_158 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_29 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_193 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_30 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_188 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_189 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_85 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_190 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_191 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_32 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_204 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_33 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_221 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_222 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_87 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_223 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_224 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_35 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_215 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_36 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_254 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_255 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_89 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_256 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_257 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_38 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_226 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_39 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_287 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_288 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_91 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_289 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_290 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_41 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_237 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_42 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_320 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_321 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_93 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_322 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_323 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_44 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_248 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_45 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_353 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_354 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_95 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_355 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_356 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_47 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_259 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_48 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_386 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_387 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_97 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_388 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_389 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_50 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_270 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_51 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_419 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_420 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_99 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_421 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_422 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_53 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_281 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_54 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_452 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_453 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_101 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_454 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_455 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_56 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_292 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_57 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_485 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_486 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_103 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_487 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_488 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_59 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda', dtype=torch.float32)
    add_303 = rand_strided((64, 128, 3072), (393216, 3072, 1), device='cuda', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_60 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_518 = rand_strided((768, 128, 128), (16384, 1, 128), device='cuda', dtype=torch.float32)
    permute_519 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    alias_105 = rand_strided((64, 12, 128, 128), (196608, 16384, 128, 1), device='cuda', dtype=torch.float32)
    permute_520 = rand_strided((768, 64, 128), (8192, 1, 64), device='cuda', dtype=torch.float32)
    permute_521 = rand_strided((768, 128, 64), (8192, 1, 128), device='cuda', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda', dtype=torch.float32)
    div_62 = rand_strided((64, 128, 1), (128, 1, 8192), device='cuda', dtype=torch.float32)
    view_506 = rand_strided((128, ), (1, ), device='cuda', dtype=torch.int64)
    view_509 = rand_strided((8192, ), (1, ), device='cuda', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda', dtype=torch.float32)
    tangents_2 = rand_strided((64, 128, 30522), (3906816, 30522, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, slice_2, mul_1, gt, view, gt_1, view_11, gt_2, mul_9, view_13, view_15, gt_3, mul_26, view_17, gt_4, view_28, gt_5, mul_32, view_30, view_32, gt_6, mul_49, view_34, gt_7, view_45, gt_8, mul_55, view_47, view_49, gt_9, mul_72, view_51, gt_10, view_62, gt_11, mul_78, view_64, view_66, gt_12, mul_95, view_68, gt_13, view_79, gt_14, mul_101, view_81, view_83, gt_15, mul_118, view_85, gt_16, view_96, gt_17, mul_124, view_98, view_100, gt_18, mul_141, view_102, gt_19, view_113, gt_20, mul_147, view_115, view_117, gt_21, mul_164, view_119, gt_22, view_130, gt_23, mul_170, view_132, view_134, gt_24, mul_187, view_136, gt_25, view_147, gt_26, mul_193, view_149, view_151, gt_27, mul_210, view_153, gt_28, view_164, gt_29, mul_216, view_166, view_168, gt_30, mul_233, view_170, gt_31, view_181, gt_32, mul_239, view_183, view_185, gt_33, mul_256, view_187, gt_34, view_198, gt_35, mul_262, view_200, view_202, gt_36, mul_279, view_204, mul_294, view_206, sub_53, unsqueeze_2, permute_134, div_25, add_175, permute_138, div_26, permute_142, add_182, permute_146, div_27, permute_150, permute_155, permute_156, alias_83, permute_157, permute_158, permute_162, permute_167, permute_171, div_29, permute_175, add_193, permute_179, div_30, permute_183, permute_188, permute_189, alias_85, permute_190, permute_191, permute_195, permute_200, permute_204, div_32, permute_208, add_204, permute_212, div_33, permute_216, permute_221, permute_222, alias_87, permute_223, permute_224, permute_228, permute_233, permute_237, div_35, permute_241, add_215, permute_245, div_36, permute_249, permute_254, permute_255, alias_89, permute_256, permute_257, permute_261, permute_266, permute_270, div_38, permute_274, add_226, permute_278, div_39, permute_282, permute_287, permute_288, alias_91, permute_289, permute_290, permute_294, permute_299, permute_303, div_41, permute_307, add_237, permute_311, div_42, permute_315, permute_320, permute_321, alias_93, permute_322, permute_323, permute_327, permute_332, permute_336, div_44, permute_340, add_248, permute_344, div_45, permute_348, permute_353, permute_354, alias_95, permute_355, permute_356, permute_360, permute_365, permute_369, div_47, permute_373, add_259, permute_377, div_48, permute_381, permute_386, permute_387, alias_97, permute_388, permute_389, permute_393, permute_398, permute_402, div_50, permute_406, add_270, permute_410, div_51, permute_414, permute_419, permute_420, alias_99, permute_421, permute_422, permute_426, permute_431, permute_435, div_53, permute_439, add_281, permute_443, div_54, permute_447, permute_452, permute_453, alias_101, permute_454, permute_455, permute_459, permute_464, permute_468, div_56, permute_472, add_292, permute_476, div_57, permute_480, permute_485, permute_486, alias_103, permute_487, permute_488, permute_492, permute_497, permute_501, div_59, permute_505, add_303, permute_509, div_60, permute_513, permute_518, permute_519, alias_105, permute_520, permute_521, permute_525, permute_530, permute_534, div_62, view_506, view_509, tangents_1, tangents_2]))
