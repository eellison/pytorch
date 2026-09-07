import sys, statistics, torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
_ = ("gl has:", {n: hasattr(gl, n) for n in ("abs", "sqrt", "rsqrt", "minimum", "maximum", "float8e4nv", "uint16", "exp2")})

RECIP = gl.constexpr("{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}")
E8M0 = gl.constexpr("cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;")

@gluon.jit
def rms_mxfp8_kernel(x_ptr, w_ptr, q_ptr, sf_ptr, K: gl.constexpr, GPT: gl.constexpr, NUM_WARPS: gl.constexpr):
    # Tile [1, G, 32]; each thread owns GPT whole 32-element groups.
    G: gl.constexpr = K // 32
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, GPT, 32], threads_per_warp=[1, 32, 1], warps_per_cta=[1, NUM_WARPS, 1], order=[2, 1, 0])
    row = gl.program_id(0)
    L2: gl.constexpr = gl.SliceLayout(0, L)  # [G, 32]
    g = gl.arange(0, G, layout=gl.SliceLayout(1, L2))
    i = gl.arange(0, 32, layout=gl.SliceLayout(0, L2))
    offs = gl.expand_dims(gl.expand_dims(g, 1) * 32 + gl.expand_dims(i, 0), 0)  # [1, G, 32], layout L
    x = gl.load(x_ptr + row * K + offs).to(gl.float32)
    w = gl.load(w_ptr + offs).to(gl.float32)
    ss = gl.sum(gl.sum(x * x, axis=2), axis=1)  # [1]
    rstd = gl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [ss * (1.0 / K) + 1e-6], dtype=gl.float32, is_pure=True, pack=1)
    y = x * gl.expand_dims(gl.expand_dims(rstd, 1), 2) * w
    amax = gl.max(gl.maximum(y, -y), axis=2)  # [1, G], in-thread
    raw = gl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
    sf = gl.inline_asm_elementwise(E8M0, "=h,r", [raw], dtype=gl.uint16, is_pure=True, pack=1).to(gl.uint8)
    inv = gl.inline_asm_elementwise(RECIP, "=f,r", [sf.to(gl.int32)], dtype=gl.float32, is_pure=True, pack=1)
    q = y * gl.expand_dims(inv, 2)
    q = gl.minimum(gl.maximum(q, -448.0), 448.0)
    gl.store(q_ptr + row * K + offs, q.to(gl.float8e4nv))
    gs = gl.arange(0, G, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    g2 = gl.expand_dims(gs, 0)  # [1, G] matching sf layout
    sw = 4 * ((row // 32) % 4) + 16 * (row % 32) + 512 * (g2 // 4) + (K // 128) * 512 * (row // 128) + (g2 % 4)
    gl.store(sf_ptr + sw, sf)

@gluon.jit
def rms_mxfp8_convert(x_ptr, w_ptr, q_ptr, sf_ptr, K: gl.constexpr, GPT: gl.constexpr, NUM_WARPS: gl.constexpr):
    # Tile [1, G, 32]; each thread owns GPT whole 32-element groups.
    G: gl.constexpr = K // 32
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, GPT, 32], threads_per_warp=[1, 32, 1], warps_per_cta=[1, NUM_WARPS, 1], order=[2, 1, 0])
    row = gl.program_id(0)
    L2: gl.constexpr = gl.SliceLayout(0, L)  # [G, 32]
    g = gl.arange(0, G, layout=gl.SliceLayout(1, L2))
    i = gl.arange(0, 32, layout=gl.SliceLayout(0, L2))
    offs = gl.expand_dims(gl.expand_dims(g, 1) * 32 + gl.expand_dims(i, 0), 0)  # [1, G, 32], layout L
    LC: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 1, 8], threads_per_warp=[1, 8, 4], warps_per_cta=[1, NUM_WARPS, 1], order=[2, 1, 0])
    LC2: gl.constexpr = gl.SliceLayout(0, LC)
    gc = gl.arange(0, G, layout=gl.SliceLayout(1, LC2))
    ic = gl.arange(0, 32, layout=gl.SliceLayout(0, LC2))
    offc = gl.expand_dims(gl.expand_dims(gc, 1) * 32 + gl.expand_dims(ic, 0), 0)
    x = gl.convert_layout(gl.load(x_ptr + row * K + offc), L).to(gl.float32)
    w = gl.convert_layout(gl.load(w_ptr + offc), L).to(gl.float32)
    ss = gl.sum(gl.sum(x * x, axis=2), axis=1)  # [1]
    rstd = gl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [ss * (1.0 / K) + 1e-6], dtype=gl.float32, is_pure=True, pack=1)
    y = x * gl.expand_dims(gl.expand_dims(rstd, 1), 2) * w
    amax = gl.max(gl.maximum(y, -y), axis=2)  # [1, G], in-thread
    raw = gl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
    sf = gl.inline_asm_elementwise(E8M0, "=h,r", [raw], dtype=gl.uint16, is_pure=True, pack=1).to(gl.uint8)
    inv = gl.inline_asm_elementwise(RECIP, "=f,r", [sf.to(gl.int32)], dtype=gl.float32, is_pure=True, pack=1)
    q = y * gl.expand_dims(inv, 2)
    q = gl.minimum(gl.maximum(q, -448.0), 448.0)
    gl.store(q_ptr + row * K + offs, q.to(gl.float8e4nv))
    gs = gl.arange(0, G, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    g2 = gl.expand_dims(gs, 0)  # [1, G] matching sf layout
    sw = 4 * ((row // 32) % 4) + 16 * (row % 32) + 512 * (g2 // 4) + (K // 128) * 512 * (row // 128) + (g2 % 4)
    gl.store(sf_ptr + sw, sf)


@gluon.jit
def rms_mxfp8_half(x_ptr, w_ptr, q_ptr, sf_ptr, K: gl.constexpr, GPT: gl.constexpr, NUM_WARPS: gl.constexpr):
    # Tile [1, G, 32]; each thread owns GPT whole 32-element groups.
    G: gl.constexpr = K // 32
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, GPT, 16], threads_per_warp=[1, 16, 2], warps_per_cta=[1, NUM_WARPS, 1], order=[2, 1, 0])
    row = gl.program_id(0)
    L2: gl.constexpr = gl.SliceLayout(0, L)  # [G, 32]
    g = gl.arange(0, G, layout=gl.SliceLayout(1, L2))
    i = gl.arange(0, 32, layout=gl.SliceLayout(0, L2))
    offs = gl.expand_dims(gl.expand_dims(g, 1) * 32 + gl.expand_dims(i, 0), 0)  # [1, G, 32], layout L
    x = gl.load(x_ptr + row * K + offs).to(gl.float32)
    w = gl.load(w_ptr + offs).to(gl.float32)
    ss = gl.sum(gl.sum(x * x, axis=2), axis=1)  # [1]
    rstd = gl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [ss * (1.0 / K) + 1e-6], dtype=gl.float32, is_pure=True, pack=1)
    y = x * gl.expand_dims(gl.expand_dims(rstd, 1), 2) * w
    amax = gl.max(gl.maximum(y, -y), axis=2)  # [1, G], in-thread
    raw = gl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
    sf = gl.inline_asm_elementwise(E8M0, "=h,r", [raw], dtype=gl.uint16, is_pure=True, pack=1).to(gl.uint8)
    inv = gl.inline_asm_elementwise(RECIP, "=f,r", [sf.to(gl.int32)], dtype=gl.float32, is_pure=True, pack=1)
    q = y * gl.expand_dims(inv, 2)
    q = gl.minimum(gl.maximum(q, -448.0), 448.0)
    gl.store(q_ptr + row * K + offs, q.to(gl.float8e4nv))
    gs = gl.arange(0, G, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    g2 = gl.expand_dims(gs, 0)  # [1, G] matching sf layout
    sw = 4 * ((row // 32) % 4) + 16 * (row % 32) + 512 * (g2 // 4) + (K // 128) * 512 * (row // 128) + (g2 % 4)
    gl.store(sf_ptr + sw, sf)


