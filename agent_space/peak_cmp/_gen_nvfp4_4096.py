import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 16
    nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // nested_R0_LOCAL_REDUCTION_SIZE
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    reduced_r0_index = r0_offset // nested_R0_LOCAL_REDUCTION_SIZE + tl.arange(0, nested_R0_REDUCED_BLOCK)[None, :]
    reduced_r0_index_mask = reduced_r0_index < 256
    r0_4 = reduced_r0_index
    lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
    lane2_r0_index_mask = lane2_r0_index < 2048
    r0_7 = lane2_r0_index
    tmp12 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.full([1, 1], 4096.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-06, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp1 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tl.reshape(tmp16, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
    tmp18 = triton_helpers.max2(tmp17, 2)
    tmp19 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full([1, 1], 1e-12, tl.float32)
    tmp23 = tl.maximum(tmp21, tmp22, tl.PropagateNan.ALL)
    tmp24 = tl.full([1, 1], 448.0, tl.float32)
    tmp25 = tl.minimum(tmp23, tmp24, tl.PropagateNan.ALL)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26.to(tl.float8e4nv)
    tmp35, tmp36 = tl.split(tl.reshape(tmp15, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp37 = tmp35.to(tl.float32)
    tmp38 = tmp27.to(tl.float32)
    tmp39 = tl.full([1, 1], 1.0, tl.float32)
    tmp40 = (tmp39 / tmp38)
    tmp41 = tl.reshape(tl.broadcast_to(tmp40[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE]), [XBLOCK, R0_BLOCK])
    tmp42, tmp43 = tl.split(tl.reshape(tmp41, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp44 = tmp37 * tmp42
    tmp52 = tmp36.to(tl.float32)
    tmp53 = tmp52 * tmp42
    tmp54 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp44, tmp53], dtype=tl.int32, is_pure=True, pack=1)
    tmp55 = tmp54.to(tl.uint8)
    tl.store(out_ptr2 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 32768*(x0 // 128) + ((r0_4 % 4))), tmp27, reduced_r0_index_mask)
    tl.store(out_ptr4 + (r0_7 + 2048*x0), tmp55, lane2_r0_index_mask)
