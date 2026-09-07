import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 32
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
    reduced_r0_index_mask = reduced_r0_index < 128
    r0_4 = reduced_r0_index
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
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.reshape(tmp17, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
    tmp19 = triton_helpers.max2(tmp18, 2)
    tmp20 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tl.full([1, 1], 1.1754943508222875e-38, tl.float32)
    tmp23 = tl.maximum(tmp21, tmp22, tl.PropagateNan.ALL)
    tmp24 = tl.inline_asm_elementwise('cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;', '=h,r', [tmp23], dtype=tl.uint16, is_pure=True, pack=1)
    tmp25 = tmp24.to(tl.int16).to(tl.uint8)
    tmp26 = tmp25.to(tl.int32)
    tmp27 = tl.inline_asm_elementwise('{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}', '=f,r', [tmp26], dtype=tl.float32, is_pure=True, pack=1)
    tmp28 = tmp15.to(tl.float32)
    tmp29 = tl.reshape(tl.broadcast_to(tmp27[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE]), [XBLOCK, R0_BLOCK])
    tmp30 = tmp28 * tmp29
    tmp31 = tl.full([1, 1], -448.0, tl.float32)
    tmp32 = tl.maximum(tmp30, tmp31, tl.PropagateNan.ALL)
    tmp33 = tl.full([1, 1], 448.0, tl.float32)
    tmp34 = tl.minimum(tmp32, tmp33, tl.PropagateNan.ALL)
    tmp35 = tmp34.to(tl.float8e4nv)
    tl.store(out_ptr4 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 16384*(x0 // 128) + ((r0_4 % 4))), tmp25, reduced_r0_index_mask)
    tl.store(out_ptr5 + (r0_1 + 4096*x0), tmp35, None)
