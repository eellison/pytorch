import statistics, torch, triton, triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
CVT = tl.constexpr("cvt.f32.bf16 $0, $1;")  # non-pure so it is not CSE'd with the first upcast
RCP_E8M0 = tl.constexpr("{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}")
E8M0 = tl.constexpr("cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;")

@triton.jit
def scale_and_store(y, x0, r0, out_sf, K: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 32
    amax = tl.max(tl.reshape(tl_math.abs(y), [XBLOCK, G, 32]), 2)
    raw = tl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
    sf16 = tl.inline_asm_elementwise(E8M0, '=h,r', [raw], dtype=tl.uint16, is_pure=True, pack=1)
    sf = sf16.to(tl.uint8)
    inv = tl.inline_asm_elementwise(RCP_E8M0, '=f,r', [sf.to(tl.int32)], dtype=tl.float32, is_pure=True, pack=1)
    r0_4 = tl.arange(0, G)[None, :]
    SW: tl.constexpr = (K // 128) * 512
    tl.store(out_sf + (4 * ((x0 // 32) % 4) + 16 * (x0 % 32) + 512 * (r0_4 // 4) + SW * (x0 // 128) + (r0_4 % 4)), sf, None)
    return inv

@triton.jit
def k_fp32_resident(in_ptr0, in_ptr1, out_sf, out_q, K: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 32
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]; r0 = tl.arange(0, R0_BLOCK)[None, :]
    x = tl.load(in_ptr0 + (r0 + K * x0), None, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last').to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x * x, 1)[:, None] / K + 1e-06)
    y = x * rstd * w
    inv = scale_and_store(y, x0, r0, out_sf, K, XBLOCK, R0_BLOCK)
    q = tl.reshape(tl.reshape(y, [XBLOCK, G, 32]) * inv[:, :, None], [XBLOCK, R0_BLOCK])
    tl.store(out_q + (r0 + K * x0), tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv), None)

@triton.jit
def k_bf16_resident_y_shared(in_ptr0, in_ptr1, out_sf, out_q, K: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 32
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]; r0 = tl.arange(0, R0_BLOCK)[None, :]
    xb = tl.load(in_ptr0 + (r0 + K * x0), None, eviction_policy='evict_first')
    x1 = xb.to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x1 * x1, 1)[:, None] / K + 1e-06)
    x2 = tl.inline_asm_elementwise(CVT, '=r,h', [xb], dtype=tl.float32, is_pure=False, pack=1)
    w = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last').to(tl.float32)
    y = x2 * rstd * w
    inv = scale_and_store(y, x0, r0, out_sf, K, XBLOCK, R0_BLOCK)
    q = tl.reshape(tl.reshape(y, [XBLOCK, G, 32]) * inv[:, :, None], [XBLOCK, R0_BLOCK])
    tl.store(out_q + (r0 + K * x0), tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv), None)

@triton.jit
def k_bf16_resident_y_recompute(in_ptr0, in_ptr1, out_sf, out_q, K: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 32
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]; r0 = tl.arange(0, R0_BLOCK)[None, :]
    xb = tl.load(in_ptr0 + (r0 + K * x0), None, eviction_policy='evict_first')
    x1 = xb.to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x1 * x1, 1)[:, None] / K + 1e-06)
    wb = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last')
    x2 = tl.inline_asm_elementwise(CVT, '=r,h', [xb], dtype=tl.float32, is_pure=False, pack=1)
    y1 = x2 * rstd * wb.to(tl.float32)
    inv = scale_and_store(y1, x0, r0, out_sf, K, XBLOCK, R0_BLOCK)
    x3 = tl.inline_asm_elementwise(CVT, '=r,h', [xb], dtype=tl.float32, is_pure=False, pack=1)
    w3 = tl.inline_asm_elementwise(CVT, '=r,h', [wb], dtype=tl.float32, is_pure=False, pack=1)
    y2 = x3 * rstd * w3
    q = tl.reshape(tl.reshape(y2, [XBLOCK, G, 32]) * inv[:, :, None], [XBLOCK, R0_BLOCK])
    tl.store(out_q + (r0 + K * x0), tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv), None)


@triton.jit
def k_noclamp(in_ptr0, in_ptr1, out_sf, out_q, K: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 32
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]; r0 = tl.arange(0, R0_BLOCK)[None, :]
    x = tl.load(in_ptr0 + (r0 + K * x0), None, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(in_ptr1 + r0, None, eviction_policy='evict_last').to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x * x, 1)[:, None] / K + 1e-06)
    y = x * rstd * w
    inv = scale_and_store(y, x0, r0, out_sf, K, XBLOCK, R0_BLOCK)
    q = tl.reshape(tl.reshape(y, [XBLOCK, G, 32]) * inv[:, :, None], [XBLOCK, R0_BLOCK])
    tl.store(out_q + (r0 + K * x0), q.to(tl.float8e4nv), None)

