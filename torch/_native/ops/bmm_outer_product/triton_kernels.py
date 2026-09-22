import functools

import triton
import triton.language as tl

import torch
from torch._native.instrumentation import instrumented_triton_cache

from ...triton import ConstTensorWrapper


# Pin Triton's current default so the launch and its safety check use the
# same number of warps.
_TRITON_DEFAULT_NUM_WARPS = 4


def _bmm_log_key(a, b, out, B, M, N, *strides, BLOCK_M, BLOCK_N, num_warps) -> str:
    # Receives the kernel's launch args; BLOCK_M/BLOCK_N are the constexprs
    # that (with shapes/dtype) form the Triton compile key.
    return (
        f"bmm_outer B={B} M={M} N={N} {a.dtype} "
        f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} num_warps={num_warps}"
    )


@instrumented_triton_cache("aten::bmm", key_fn=_bmm_log_key)
def _bmm_outer_product_kernel(
    A_ptr,
    B_ptr,
    OUT_ptr,
    B_dim,
    M,
    N,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The program id is promoted to int64 once, so every index derived from it
    # (batch, tile, row and column offsets) is 64-bit. Program ids and the
    # int32-range strides are i32, and both pid_b * stride_ob (once
    # (batch - 1) * M * N > INT32_MAX, e.g. (512, 8209, 512)) and
    # pid_m * BLOCK_M (once M > INT32_MAX) used to wrap and write the tail
    # of the output gigabytes before its buffer.
    pid = tl.program_id(0).to(tl.int64)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N

    a = tl.load(A_ptr + pid_b * stride_ab + rm * stride_am, mask=mask_m, other=0.0)
    b = tl.load(B_ptr + pid_b * stride_bb + rn * stride_bn, mask=mask_n, other=0.0)

    out = a[:, None] * b[None, :]

    mask = mask_m[:, None] & mask_n[None, :]  # pyrefly: ignore[bad-index]
    tl.store(
        OUT_ptr + pid_b * stride_ob + rm[:, None] * stride_om + rn[None, :] * stride_on,
        out,
        mask=mask,
    )


def _next_power_of_2(n, cap: int | None = None):
    # triton.next_power_of_2 for an int. A symbolic size (a host trace,
    # torch/cuda/_host_trace.py, hands the override SymInt sizes) has no
    # bitwise form: the comparison ladder up to `cap`, each step a guard of
    # the trace; the caller's min() bounds an int result the same way.
    if type(n) is int:
        return triton.next_power_of_2(n)
    p = 1
    while (cap is None or p < cap) and p < n:
        p *= 2
    return p


def _pick_block_sizes(m: int, n: int) -> tuple[int, int]:
    """I swept over some shapes and in the future we should figure out @autotune story"""
    if m <= 32:
        block_m = _next_power_of_2(m)
    elif m <= 96:
        block_m = 32
    elif m <= 192:
        block_m = 64
    else:
        block_m = 128
    return block_m, min(_next_power_of_2(n, 128), 128)


@functools.lru_cache(maxsize=1024)
def _bmm_outer_product_launch_config(
    batch: int, m: int, n: int
) -> tuple[int, int, int]:
    """Return the 1D grid size and block sizes used to launch the kernel.

    The grid has one entry for every (batch, M tile, N tile). Sharing this
    calculation with the safety guard keeps the checked and launched grids identical.
    """
    if type(m) is int and type(n) is int:
        block_m, block_n = _pick_block_sizes(m, n)
    else:
        # symbolic sizes (a host trace): the block-size ladder's comparisons
        # pick the launch configuration, a kernel choice of the trace
        with torch._C._HostTraceKernelChoice():
            block_m, block_n = _pick_block_sizes(m, n)
    grid_size = batch * triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    return grid_size, block_m, block_n


def bmm_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, _ = a.shape
    N = b.shape[2]

    out = torch.empty(B, M, N, dtype=a.dtype, device=a.device)

    try:
        grid_size, BLOCK_M, BLOCK_N = _bmm_outer_product_launch_config(B, M, N)
    except TypeError:
        # symbolic sizes (a host trace) are not hashable: the same selection, uncached
        config = _bmm_outer_product_launch_config.__wrapped__
        grid_size, BLOCK_M, BLOCK_N = config(B, M, N)

    # a and b are read-only inputs; wrap them so a copy-on-write tensor is read
    # through const_data_ptr() and not materialized. out is written directly.
    _bmm_outer_product_kernel[(grid_size,)](
        ConstTensorWrapper(a),
        ConstTensorWrapper(b),
        out,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_TRITON_DEFAULT_NUM_WARPS,
    )
    return out
