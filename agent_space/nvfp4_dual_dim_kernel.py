"""Combined dim0+dim1 NVFP4 quantization: ONE kernel, ONE load of x.

Semantics match agent_space/bench_dim1_quant_casts.py exactly:
  dim0 (rowwise):  groups of 16 along K. scale = e4m3((amax/6).clamp(1e-12,448)),
                   value = x / scale.float(), adjacent COLUMN pairs packed per
                   byte via cvt.rn.satfinite.e2m1x2.f32 (even col -> low nibble).
                   payload [M, K//2] u8, scale [M, K//16] e4m3.
  dim1 (colwise):  groups of 16 along M, adjacent ROW pairs packed per byte.
                   row-major layout: payload [M//2, K] u8, scale [M//16, K].
                   transposed layout (DIM1_TRANSPOSED): payload [K, M//2],
                   scale [K, M//16], both contiguous.

No per-tensor scale (the bench reference does not use one). Tile is
BLOCK_M x BLOCK_K with 16 | BLOCK_M and 16 | BLOCK_K, so every 16-group in
either direction is tile-local. M and K must be multiples of 16 (reference
requirement); partial tiles are handled with masks and always cover whole
16-groups, so masked-out lanes never contaminate a stored group.
"""

import torch
import triton
import triton.language as tl


E2M1X2_PACK_ASM = (
    "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
)


@triton.jit
def _pack_e2m1x2(even, odd):
    packed = tl.inline_asm_elementwise(
        "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}",
        "=r,f,f",
        [even, odd],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8)


@triton.jit
def nvfp4_dual_dim_kernel(
    x_ptr,
    q0_ptr,
    s0_ptr,
    q1_ptr,
    s1_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIM1_TRANSPOSED: tl.constexpr,
    EVEN: tl.constexpr,
    PRECISE: tl.constexpr,
):
    tiles_k: tl.constexpr = tl.cdiv(K, BLOCK_K)
    tile = tl.program_id(0)
    tile_m = tile // tiles_k
    tile_k = tile % tiles_k
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)

    if EVEN:
        x = tl.load(x_ptr + rows[:, None] * K + cols[None, :]).to(tl.float32)
    else:
        ld_mask = (rows[:, None] < M) & (cols[None, :] < K)
        x = tl.load(
            x_ptr + rows[:, None] * K + cols[None, :], mask=ld_mask, other=0.0
        ).to(tl.float32)

    g0: tl.constexpr = BLOCK_K // 16
    g1: tl.constexpr = BLOCK_M // 16

    # dim0: 16-groups along K
    x_g0 = tl.reshape(x, (BLOCK_M, g0, 16))
    amax0 = tl.max(tl.abs(x_g0), axis=2)
    s0 = tl.clamp(amax0 / 6.0, 1e-12, 448.0).to(tl.float8e4nv)
    if PRECISE:
        y0 = x_g0 / s0.to(tl.float32)[:, :, None]
    else:
        y0 = x_g0 * (1.0 / s0.to(tl.float32))[:, :, None]
    even0, odd0 = tl.split(tl.reshape(y0, (BLOCK_M, BLOCK_K // 2, 2)))
    q0 = _pack_e2m1x2(even0, odd0)

    cols2 = tile_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    gcols = tile_k * g0 + tl.arange(0, g0)
    if EVEN:
        tl.store(q0_ptr + rows[:, None] * (K // 2) + cols2[None, :], q0)
        tl.store(s0_ptr + rows[:, None] * (K // 16) + gcols[None, :], s0)
    else:
        q0_mask = (rows[:, None] < M) & (cols2[None, :] < K // 2)
        s0_mask = (rows[:, None] < M) & (gcols[None, :] < K // 16)
        tl.store(q0_ptr + rows[:, None] * (K // 2) + cols2[None, :], q0, mask=q0_mask)
        tl.store(s0_ptr + rows[:, None] * (K // 16) + gcols[None, :], s0, mask=s0_mask)

    # dim1: 16-groups along M, computed on an explicitly transposed register
    # tile so the group axis is trailing (structurally identical to dim0).
    # This measured faster than permute/split on the row-major tile because
    # Triton then needs a single layout conversion for the whole dim1 side.
    xt = tl.trans(x)
    xt_g = tl.reshape(xt, (BLOCK_K, g1, 16))
    amax1 = tl.max(tl.abs(xt_g), axis=2)
    s1 = tl.clamp(amax1 / 6.0, 1e-12, 448.0).to(tl.float8e4nv)
    if PRECISE:
        y1 = xt_g / s1.to(tl.float32)[:, :, None]
    else:
        y1 = xt_g * (1.0 / s1.to(tl.float32))[:, :, None]
    even1, odd1 = tl.split(tl.reshape(y1, (BLOCK_K, BLOCK_M // 2, 2)))
    q1t = _pack_e2m1x2(even1, odd1)

    rows2 = tile_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
    grows = tile_m * g1 + tl.arange(0, g1)
    if DIM1_TRANSPOSED:
        if EVEN:
            tl.store(q1_ptr + cols[:, None] * (M // 2) + rows2[None, :], q1t)
            tl.store(s1_ptr + cols[:, None] * (M // 16) + grows[None, :], s1)
        else:
            q1_mask = (cols[:, None] < K) & (rows2[None, :] < M // 2)
            s1_mask = (cols[:, None] < K) & (grows[None, :] < M // 16)
            tl.store(q1_ptr + cols[:, None] * (M // 2) + rows2[None, :], q1t, mask=q1_mask)
            tl.store(s1_ptr + cols[:, None] * (M // 16) + grows[None, :], s1, mask=s1_mask)
    else:
        q1 = tl.trans(q1t)
        s1_rm = tl.trans(s1)
        if EVEN:
            tl.store(q1_ptr + rows2[:, None] * K + cols[None, :], q1)
            tl.store(s1_ptr + grows[:, None] * K + cols[None, :], s1_rm)
        else:
            q1_mask = (rows2[:, None] < M // 2) & (cols[None, :] < K)
            s1_mask = (grows[:, None] < M // 16) & (cols[None, :] < K)
            tl.store(q1_ptr + rows2[:, None] * K + cols[None, :], q1, mask=q1_mask)
            tl.store(s1_ptr + grows[:, None] * K + cols[None, :], s1_rm, mask=s1_mask)


def nvfp4_dual_dim(x, block_m=32, block_k=128, num_warps=4, dim1_transposed=False, precise=True):
    M, K = x.shape
    if M % 16 or K % 16:
        raise ValueError(f"M and K must be multiples of 16, got {M}x{K}")
    dev = x.device
    fp8 = torch.float8_e4m3fn
    q0 = torch.empty(M, K // 2, device=dev, dtype=torch.uint8)
    s0 = torch.empty(M, K // 16, device=dev, dtype=fp8)
    if dim1_transposed:
        q1 = torch.empty(K, M // 2, device=dev, dtype=torch.uint8)
        s1 = torch.empty(K, M // 16, device=dev, dtype=fp8)
    else:
        q1 = torch.empty(M // 2, K, device=dev, dtype=torch.uint8)
        s1 = torch.empty(M // 16, K, device=dev, dtype=fp8)
    even = M % block_m == 0 and K % block_k == 0
    grid = (triton.cdiv(M, block_m) * triton.cdiv(K, block_k),)
    nvfp4_dual_dim_kernel[grid](
        x, q0, s0, q1, s1, M, K, block_m, block_k,
        DIM1_TRANSPOSED=dim1_transposed, EVEN=even, PRECISE=precise, num_warps=num_warps,
    )
    return q0, s0, q1, s1


if __name__ == "__main__":
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    for t in (False, True):
        outs = nvfp4_dual_dim(x, dim1_transposed=t)
        print(t, [(o.shape, o.dtype) for o in outs])
    # partial-tile smoke test
    x = torch.randn(528, 1040, device="cuda", dtype=torch.bfloat16)
    outs = nvfp4_dual_dim(x, block_m=32, block_k=128)
    print("edge", [(o.shape, o.dtype) for o in outs])
