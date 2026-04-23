"""
Push-pattern allreduce + bias + rmsnorm kernels in Triton.

Variant A: push + signal_pad sync (2 barriers, local reads)
Variant B: Lamport triple-buffer (0 barriers, local reads, neg-zero polling)
"""

import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.language.math import rsqrt as tl_rsqrt

from kraken._ptx_utils import symm_mem_sync

NEG_ZERO_I16 = tl.constexpr(-32768)  # 0x8000 = bf16 negative zero


# =========================================================================
# Inline PTX helpers for Lamport protocol
# =========================================================================

@triton.jit
def _cta_register(counter_ptr):
    """Thread 0 of each CTA atomically increments the CTA counter."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t_id, %tmp;
            .reg .pred %p;
            mov.u32 %t_id, %tid.x;
            setp.eq.u32 %p, %t_id, 0;
            mov.u32 $0, 0;
            @%p atom.global.gpu.add.u32 %tmp, [$1], 1;
        }
        """,
        "=r, l",
        [counter_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _lamport_update(counter_ptr, phase_ptr, num_ctas, new_phase):
    """CTA 0, thread 0: spin until all CTAs done, advance phase, reset counter."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %b_id, %t_id, %cnt, %nctas, %nph;
            .reg .pred %p_b, %p_t, %p_bt, %p_done;

            mov.u32 %b_id, %ctaid.x;
            mov.u32 %t_id, %tid.x;
            setp.eq.u32 %p_b, %b_id, 0;
            setp.eq.u32 %p_t, %t_id, 0;
            and.pred %p_bt, %p_b, %p_t;
            @!%p_bt bra lamport_done;

            mov.u32 %nctas, $1;
            mov.u32 %nph, $3;

        lamport_spin:
            ld.volatile.global.u32 %cnt, [$2];
            setp.ne.u32 %p_done, %cnt, %nctas;
            @%p_done bra lamport_spin;

            st.volatile.global.u32 [$4], %nph;
            mov.u32 %cnt, 0;
            st.volatile.global.u32 [$2], %cnt;

        lamport_done:
            mov.u32 $0, 0;
        }
        """,
        "=r, r, l, r, l",
        [num_ctas, counter_ptr, new_phase, phase_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# =========================================================================
# Variant A: Push + signal pad sync (same barriers as Kraken, but local reads)
# =========================================================================

@triton.jit
def push_sync_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D

    # --- Push: write our data to ALL remote buffers ---
    inp = tl.load(input_ptr + row_idx * D + cols, mask=mask)
    for r in tl.static_range(world_size):
        buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
        buf = tl.multiple_of(buf, 16)
        tl.store(buf + rank * stride + row_idx * D + cols, inp, mask=mask)

    # --- Barrier: all ranks have pushed ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # --- Read LOCAL buffer + reduce + bias ---
    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)
    acc = tl.load(bias_ptr + row_idx * D + cols, mask=mask).to(tl.float32)
    for r in tl.static_range(world_size):
        val = tl.load(my_buf + r * stride + row_idx * D + cols, mask=mask)
        acc += val.to(tl.float32)

    # --- RMS Norm ---
    variance = tl.sum(acc * acc, axis=0) / D
    rstd = tl_rsqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row_idx * D + cols, (acc * rstd * w).to(tl.bfloat16), mask=mask)

    # --- Final barrier: safe to reuse buffer ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=False)


def push_sync_ar_bias_rmsnorm(
    symm_mem_buffer: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)

    push_sync_ar_bias_rmsnorm_kernel[(N,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x, bias, w, y, N,
        eps=eps,
        D=D,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=32,
    )


# =========================================================================
# Variant A0: Input already in symm_mem (no copy/push, read-remote pattern)
# =========================================================================

@triton.jit
def inplace_sync_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Allreduce + bias + rmsnorm when input is already in symm_mem.

    Each rank's data is in buffer_ptrs[rank]. No copy/push needed.
    Barrier, read all W peers (remote reads), reduce + bias + rmsnorm, barrier.
    """
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # --- Barrier: ensure all ranks' GEMM outputs are visible ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # --- Read from all W peers + reduce + bias ---
    acc = tl.load(bias_ptr + row_idx * D + cols, mask=mask).to(tl.float32)
    for r in tl.static_range(world_size):
        buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
        buf = tl.multiple_of(buf, 16)
        val = tl.load(buf + row_idx * D + cols, mask=mask)
        acc += val.to(tl.float32)

    # --- RMS Norm ---
    variance = tl.sum(acc * acc, axis=0) / D
    rstd = tl_rsqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row_idx * D + cols, (acc * rstd * w).to(tl.bfloat16), mask=mask)

    # --- Barrier: safe to reuse buffer ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=False)


def inplace_sync_ar_bias_rmsnorm(
    symm_mem_buffer: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Fused allreduce+bias+rmsnorm. Input must already be in symm_mem_buffer."""
    D = symm_mem_buffer.shape[-1]
    N = math.prod(symm_mem_buffer.shape[:-1])
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)

    inplace_sync_ar_bias_rmsnorm_kernel[(N,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        bias, w, y, N,
        eps=eps,
        D=D,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=32,
    )


# =========================================================================
# Variant A2: Push + signal pad sync, two-shot (reduce-scatter + all-gather)
# =========================================================================

@triton.jit
def push_sync_twoshot_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    chunk_start,
    chunk_end,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-shot allreduce + bias + rmsnorm with barrier sync.

    Buffer layout: region1 (W*N*D) for initial push + region2 (N*D) for reduced.
    Only owning CTAs (row in [chunk_start, chunk_end)) do reduce + push.
    Non-owning CTAs skip phase 2 entirely, read from local region2 after barrier.
    Per-CTA barriers work because row i is always handled by CTA i on all ranks.
    """
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D
    r1_size = world_size * stride

    # --- Phase 1: Push input to all remote buffers (region 1) ---
    inp = tl.load(input_ptr + row_idx * D + cols, mask=mask)
    for r in tl.static_range(world_size):
        buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
        buf = tl.multiple_of(buf, 16)
        tl.store(buf + rank * stride + row_idx * D + cols, inp, mask=mask)

    # --- Barrier 1: all ranks have pushed ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # --- Phase 2+3: Reduce/push/barrier/rmsnorm (branched to keep vars scoped) ---
    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)
    is_my_chunk = (row_idx >= chunk_start) & (row_idx < chunk_end)

    if is_my_chunk:
        # Owning CTA: reduce all W copies from local region1
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for r in tl.static_range(world_size):
            val = tl.load(my_buf + r * stride + row_idx * D + cols, mask=mask)
            acc += val.to(tl.float32)
        # Push reduced to all peers' region2
        reduced = acc.to(tl.bfloat16)
        for r in tl.static_range(world_size):
            buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
            buf = tl.multiple_of(buf, 16)
            tl.store(buf + r1_size + row_idx * D + cols, reduced, mask=mask)
        # Barrier 2
        symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                      hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
        # Already have acc in f32
        _rmsnorm_epilogue(acc, bias_ptr, w_ptr, y_ptr, row_idx, cols, mask,
                          eps=eps, D=D, BLOCK_D=BLOCK_D)
    else:
        # Non-owning CTA: skip reduce, just wait at barrier
        # Barrier 2
        symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                      hasPreviousMemAccess=False, hasSubsequentMemAccess=True)
        # Read reduced from local region2
        val = tl.load(my_buf + r1_size + row_idx * D + cols, mask=mask)
        _rmsnorm_epilogue(val.to(tl.float32), bias_ptr, w_ptr, y_ptr, row_idx, cols, mask,
                          eps=eps, D=D, BLOCK_D=BLOCK_D)

    # --- Barrier 3: safe to reuse buffer ---
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=False)


def push_sync_twoshot_ar_bias_rmsnorm(
    symm_mem_buffer: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Two-shot barrier-based fused allreduce+bias+rmsnorm. Requires N >= world_size."""
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)
    W = symm_mem_hdl.world_size
    R = symm_mem_hdl.rank
    assert N >= W, f"Two-shot requires N ({N}) >= world_size ({W})"

    chunk_size = N // W
    chunk_start = R * chunk_size
    chunk_end = chunk_start + chunk_size if R < W - 1 else N

    push_sync_twoshot_ar_bias_rmsnorm_kernel[(N,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x, bias, w, y, N,
        chunk_start, chunk_end,
        eps=eps,
        D=D,
        rank=R,
        world_size=W,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=32,
    )


# =========================================================================
# Variant B: Lamport triple-buffer (0 barriers, neg-zero polling with .cv)
# =========================================================================

@triton.jit
def lamport_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    counter_ptr,
    phase_ptr,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    num_ctas = N

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D                    # elements per rank-section within a slice
    slice_size = world_size * stride  # elements per slice (one of the 3 Lamport buffers)

    # --- Read Lamport phase ---
    phase = tl.load(phase_ptr)
    data_offset = phase % 3
    clear_offset = (phase + 2) % 3

    # --- Register this CTA ---
    _cta_register(counter_ptr)
    tl.debug_barrier()

    # --- PUSH: write to data_offset slice of all remote buffers ---
    inp = tl.load(input_ptr + row_idx * D + cols, mask=mask, other=0.0)
    # Replace neg zeros in input (sentinel collision avoidance)
    inp_bits = inp.to(tl.int16, bitcast=True)
    inp = tl.where(inp_bits == NEG_ZERO_I16,
                   tl.zeros([BLOCK_D], dtype=tl.bfloat16), inp)

    for r in tl.static_range(world_size):
        buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
        buf = tl.multiple_of(buf, 16)
        tl.store(buf + data_offset * slice_size + rank * stride + row_idx * D + cols,
                 inp, mask=mask)

    # --- CLEAR: write sentinels to clear_offset slice of OWN buffer ---
    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)
    sentinel = tl.full([BLOCK_D], NEG_ZERO_I16, dtype=tl.int16).to(tl.bfloat16, bitcast=True)
    for r in tl.static_range(world_size):
        tl.store(my_buf + clear_offset * slice_size + r * stride + row_idx * D + cols,
                 sentinel, mask=mask)

    # --- POLL + REDUCE: read data_offset slice of OWN buffer with .cv ---
    acc = tl.load(bias_ptr + row_idx * D + cols, mask=mask, other=0.0).to(tl.float32)

    for r in tl.static_range(world_size):
        base = data_offset * slice_size + r * stride + row_idx * D
        val = tl.load(my_buf + base + cols, mask=mask, other=0.0,
                      volatile=True)
        val_bits = val.to(tl.int16, bitcast=True)
        n_sentinel = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
        while n_sentinel > 0:
            val = tl.load(my_buf + base + cols, mask=mask, other=0.0,
                          volatile=True)
            val_bits = val.to(tl.int16, bitcast=True)
            n_sentinel = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
        acc += val.to(tl.float32)

    # --- RMS Norm ---
    variance = tl.sum(acc * acc, axis=0) / D
    rstd = tl_rsqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row_idx * D + cols, (acc * rstd * w).to(tl.bfloat16), mask=mask)

    # --- LAMPORT UPDATE: CTA 0 thread 0 advances phase ---
    new_phase = (phase + 1) % 3
    _lamport_update(counter_ptr, phase_ptr, num_ctas, new_phase)


@triton.jit
def _rmsnorm_epilogue(acc, bias_ptr, w_ptr, y_ptr, row_idx, cols, mask,
                      eps: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """Shared bias + rmsnorm + store logic."""
    acc += tl.load(bias_ptr + row_idx * D + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(acc * acc, axis=0) / D
    rstd = tl_rsqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row_idx * D + cols, (acc * rstd * w).to(tl.bfloat16), mask=mask)


# =========================================================================
# Two-shot Lamport (0 barriers)
# =========================================================================

@triton.jit
def lamport_twoshot_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    counter_ptr,
    phase_ptr,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    chunk_start,
    chunk_end,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-shot allreduce + bias + rmsnorm using Lamport protocol (0 barriers).

    Buffer per Lamport slice: region1 (W*N*D) + region2 (N*D).
    Region 1: initial data push (one section per rank).
    Region 2: reduced results (one entry per token, written by owning rank).
    """
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    num_ctas = N

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D
    r1_size = world_size * stride   # region 1 elements
    slice_size = r1_size + stride   # region 1 + region 2

    # --- Read Lamport phase ---
    phase = tl.load(phase_ptr)
    data_off = phase % 3
    clear_off = (phase + 2) % 3

    # --- Register CTA ---
    _cta_register(counter_ptr)
    tl.debug_barrier()

    # --- Phase 1: Push input to all peers' region 1 ---
    inp = tl.load(input_ptr + row_idx * D + cols, mask=mask, other=0.0)
    inp_bits = inp.to(tl.int16, bitcast=True)
    inp = tl.where(inp_bits == NEG_ZERO_I16,
                   tl.zeros([BLOCK_D], dtype=tl.bfloat16), inp)

    for r in tl.static_range(world_size):
        buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
        buf = tl.multiple_of(buf, 16)
        tl.store(buf + data_off * slice_size + rank * stride + row_idx * D + cols,
                 inp, mask=mask)

    # --- Clear stale sentinels (both regions) ---
    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)
    sentinel = tl.full([BLOCK_D], NEG_ZERO_I16, dtype=tl.int16).to(tl.bfloat16, bitcast=True)
    for r in tl.static_range(world_size):
        tl.store(my_buf + clear_off * slice_size + r * stride + row_idx * D + cols,
                 sentinel, mask=mask)
    tl.store(my_buf + clear_off * slice_size + r1_size + row_idx * D + cols,
             sentinel, mask=mask)

    # --- Phase 2: Reduce my chunk + push to region 2 ---
    is_my_chunk = (row_idx >= chunk_start) & (row_idx < chunk_end)
    if is_my_chunk:
        # Poll region 1 for all W copies of this row
        acc_reduce = tl.zeros([BLOCK_D], dtype=tl.float32)
        for r in tl.static_range(world_size):
            base = data_off * slice_size + r * stride + row_idx * D
            val = tl.load(my_buf + base + cols, mask=mask, other=0.0, volatile=True)
            val_bits = val.to(tl.int16, bitcast=True)
            n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
            while n_sent > 0:
                val = tl.load(my_buf + base + cols, mask=mask, other=0.0, volatile=True)
                val_bits = val.to(tl.int16, bitcast=True)
                n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
            acc_reduce += val.to(tl.float32)

        # Push reduced row to all peers' region 2
        reduced = acc_reduce.to(tl.bfloat16)
        reduced_bits = reduced.to(tl.int16, bitcast=True)
        reduced = tl.where(reduced_bits == NEG_ZERO_I16,
                           tl.zeros([BLOCK_D], dtype=tl.bfloat16), reduced)
        for r in tl.static_range(world_size):
            buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
            buf = tl.multiple_of(buf, 16)
            tl.store(buf + data_off * slice_size + r1_size + row_idx * D + cols,
                     reduced, mask=mask)

        # Fused op: we already have the reduced value, use it directly
        # (avoids volatile-read-after-local-write issue)
        _rmsnorm_epilogue(acc_reduce, bias_ptr, w_ptr, y_ptr, row_idx, cols, mask,
                          eps=eps, D=D, BLOCK_D=BLOCK_D)
    else:
        # --- Phase 3: Poll region 2 for reduced value ---
        r2_base = data_off * slice_size + r1_size + row_idx * D
        val = tl.load(my_buf + r2_base + cols, mask=mask, other=0.0, volatile=True)
        val_bits = val.to(tl.int16, bitcast=True)
        n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
        while n_sent > 0:
            val = tl.load(my_buf + r2_base + cols, mask=mask, other=0.0, volatile=True)
            val_bits = val.to(tl.int16, bitcast=True)
            n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
        _rmsnorm_epilogue(val.to(tl.float32), bias_ptr, w_ptr, y_ptr, row_idx, cols, mask,
                          eps=eps, D=D, BLOCK_D=BLOCK_D)

    # --- Lamport update ---
    new_phase = (phase + 1) % 3
    _lamport_update(counter_ptr, phase_ptr, num_ctas, new_phase)


def init_sentinel_buffer(buf: torch.Tensor) -> None:
    """Fill symmetric memory buffer with bf16 negative zeros (sentinel)."""
    buf.view(torch.int16).fill_(-32768)


class LamportWorkspace:
    """Manages triple-buffer + counter + phase for Lamport allreduce."""

    def __init__(self, max_N: int, D: int, dtype: torch.dtype, device: torch.device,
                 group: dist.ProcessGroup | None = None, twoshot: bool = False):
        group = group or dist.group.WORLD
        self.world_size = dist.get_world_size(group)
        self.D = D
        self.dtype = dtype
        self.device = device
        self.group = group
        self.twoshot = twoshot

        # One-shot: 3 slices × W × N × D
        # Two-shot: 3 slices × (W × N × D + N × D) = 3 × (W+1) × N × D
        rows_per_slice = (self.world_size + (1 if twoshot else 0)) * max_N
        self.triple_buf = symm_mem.empty(
            3 * rows_per_slice, D, dtype=dtype, device=device
        )
        self.symm_hdl = symm_mem.rendezvous(self.triple_buf, group.group_name)
        init_sentinel_buffer(self.triple_buf)

        # Device-local state (not shared across ranks)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.phase = torch.zeros(1, dtype=torch.int32, device=device)

    @property
    def rank(self):
        return self.symm_hdl.rank


def lamport_ar_bias_rmsnorm(
    workspace: LamportWorkspace,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """One-shot Lamport fused allreduce+bias+rmsnorm."""
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])

    lamport_ar_bias_rmsnorm_kernel[(N,)](
        workspace.symm_hdl.buffer_ptrs_dev,
        workspace.counter,
        workspace.phase,
        x, bias, w, y, N,
        eps=eps,
        D=D,
        rank=workspace.rank,
        world_size=workspace.world_size,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=32,
    )


def lamport_twoshot_ar_bias_rmsnorm(
    workspace: LamportWorkspace,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """Two-shot Lamport fused allreduce+bias+rmsnorm. Requires N >= world_size."""
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])
    W = workspace.world_size
    R = workspace.rank
    assert N >= W, f"Two-shot requires N ({N}) >= world_size ({W})"

    chunk_size = N // W
    chunk_start = R * chunk_size
    chunk_end = chunk_start + chunk_size if R < W - 1 else N  # last rank gets remainder

    lamport_twoshot_ar_bias_rmsnorm_kernel[(N,)](
        workspace.symm_hdl.buffer_ptrs_dev,
        workspace.counter,
        workspace.phase,
        x, bias, w, y, N,
        chunk_start, chunk_end,
        eps=eps,
        D=D,
        rank=R,
        world_size=W,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=32,
    )


# =========================================================================
# Variant C: Butterfly (recursive-doubling) allreduce + bias + rmsnorm
#
# log2(W) rounds of pairwise exchange. Each round: 1 remote read, 1 local
# read/write, 1 barrier. No remote writes (peers read from our buffer).
# Total remote traffic = log2(W) * N * D vs W * N * D for one-shot push.
# Requires world_size to be a power of 2.
# =========================================================================

@triton.jit
def butterfly_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2_W: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D

    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)

    # Write input to own buffer (slice A = offset 0), keep f32 accumulator
    inp = tl.load(input_ptr + row_idx * D + cols, mask=mask)
    tl.store(my_buf + row_idx * D + cols, inp, mask=mask)
    acc = inp.to(tl.float32)

    # Barrier: all ranks have written their input
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # Butterfly rounds: ping-pong between slice A (offset 0) and slice B (offset stride)
    for rnd in tl.static_range(LOG2_W):
        peer = rank ^ (1 << rnd)
        read_off = (rnd % 2) * stride
        write_off = ((rnd + 1) % 2) * stride

        # Read peer's current slice (remote read)
        peer_buf = tl.load(buf_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        peer_buf = tl.multiple_of(peer_buf, 16)
        peer_val = tl.load(peer_buf + read_off + row_idx * D + cols, mask=mask)

        # Accumulate in f32, store bf16 for next round's peers to read
        acc += peer_val.to(tl.float32)
        tl.store(my_buf + write_off + row_idx * D + cols,
                 acc.to(tl.bfloat16), mask=mask)

        # Barrier: all ranks have read before anyone overwrites
        symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                      hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

    # acc holds the full allreduce result in f32
    acc += tl.load(bias_ptr + row_idx * D + cols, mask=mask).to(tl.float32)
    variance = tl.sum(acc * acc, axis=0) / D
    rstd = tl_rsqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row_idx * D + cols, (acc * rstd * w).to(tl.bfloat16), mask=mask)

    # Barrier: safe to reuse buffer
    symm_mem_sync(signal_pad_ptrs, None, rank, world_size,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=False)


def butterfly_ar_bias_rmsnorm(
    symm_mem_buffer: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Butterfly (recursive-doubling) fused allreduce+bias+rmsnorm."""
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)
    W = symm_mem_hdl.world_size
    assert W & (W - 1) == 0, f"Butterfly requires power-of-2 world_size, got {W}"
    LOG2_W = int(math.log2(W))

    butterfly_ar_bias_rmsnorm_kernel[(N,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x, bias, w, y, N,
        eps=eps,
        D=D,
        rank=symm_mem_hdl.rank,
        world_size=W,
        BLOCK_D=triton.next_power_of_2(D),
        LOG2_W=LOG2_W,
        num_warps=32,
    )


# =========================================================================
# Variant D: Tiled Lamport — polls in TILE_D chunks to avoid large reductions
#
# Same Lamport protocol as Variant B but tiles the D dimension. Each tile
# is pushed, cleared, polled, and accumulated independently. The rmsnorm
# variance is accumulated across tiles, then a second pass normalizes.
# =========================================================================

@triton.jit
def lamport_tiled_ar_bias_rmsnorm_kernel(
    buffer_ptrs,
    counter_ptr,
    phase_ptr,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    N,
    eps: tl.constexpr,
    D: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    TILE_D: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    num_ctas = N

    buf_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    stride = N * D
    slice_size = world_size * stride

    phase = tl.load(phase_ptr)
    data_offset = phase % 3
    clear_offset = (phase + 2) % 3

    _cta_register(counter_ptr)
    tl.debug_barrier()

    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    my_buf = tl.multiple_of(my_buf, 16)

    # --- Pass 1: Push + Clear + Poll + Reduce (per tile), accumulate variance ---
    sum_sq = tl.zeros([1], dtype=tl.float32)

    for tile in tl.static_range(NUM_TILES):
        tile_off = tile * TILE_D
        cols = tl.arange(0, TILE_D) + tile_off
        mask = cols < D

        # Push this tile to all peers
        inp = tl.load(input_ptr + row_idx * D + cols, mask=mask, other=0.0)
        inp_bits = inp.to(tl.int16, bitcast=True)
        inp = tl.where(inp_bits == NEG_ZERO_I16,
                       tl.zeros([TILE_D], dtype=tl.bfloat16), inp)
        for r in tl.static_range(world_size):
            buf = tl.load(buf_ptrs + r).to(tl.pointer_type(tl.bfloat16))
            buf = tl.multiple_of(buf, 16)
            tl.store(buf + data_offset * slice_size + rank * stride + row_idx * D + cols,
                     inp, mask=mask)

        # Clear stale sentinels for this tile
        sentinel = tl.full([TILE_D], NEG_ZERO_I16, dtype=tl.int16).to(tl.bfloat16, bitcast=True)
        for r in tl.static_range(world_size):
            tl.store(my_buf + clear_offset * slice_size + r * stride + row_idx * D + cols,
                     sentinel, mask=mask)

        # Poll + reduce this tile from all peers
        bias_val = tl.load(bias_ptr + row_idx * D + cols, mask=mask, other=0.0).to(tl.float32)
        acc = bias_val
        for r in tl.static_range(world_size):
            base = data_offset * slice_size + r * stride + row_idx * D
            val = tl.load(my_buf + base + cols, mask=mask, other=0.0, volatile=True)
            val_bits = val.to(tl.int16, bitcast=True)
            n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
            while n_sent > 0:
                val = tl.load(my_buf + base + cols, mask=mask, other=0.0, volatile=True)
                val_bits = val.to(tl.int16, bitcast=True)
                n_sent = tl.sum(tl.where(mask, (val_bits == NEG_ZERO_I16).to(tl.int32), 0))
            acc += val.to(tl.float32)

        # Accumulate variance and store reduced+biased values to output (will normalize in pass 2)
        sum_sq += tl.sum(acc * acc, axis=0)
        tl.store(y_ptr + row_idx * D + cols, acc.to(tl.bfloat16), mask=mask)

    # --- Pass 2: Normalize (read back from y, apply rsqrt * w, overwrite) ---
    rstd = tl_rsqrt(sum_sq / D + eps)
    for tile in tl.static_range(NUM_TILES):
        tile_off = tile * TILE_D
        cols = tl.arange(0, TILE_D) + tile_off
        mask = cols < D
        val = tl.load(y_ptr + row_idx * D + cols, mask=mask).to(tl.float32)
        wt = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        tl.store(y_ptr + row_idx * D + cols, (val * rstd * wt).to(tl.bfloat16), mask=mask)

    new_phase = (phase + 1) % 3
    _lamport_update(counter_ptr, phase_ptr, num_ctas, new_phase)


def lamport_tiled_ar_bias_rmsnorm(
    workspace: LamportWorkspace,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
    tile_d: int = 4096,
) -> None:
    """Tiled Lamport fused allreduce+bias+rmsnorm. Tiles D dimension for large hidden dims."""
    D = x.shape[-1]
    N = math.prod(x.shape[:-1])
    TILE_D = min(tile_d, triton.next_power_of_2(D))
    NUM_TILES = triton.cdiv(D, TILE_D)

    lamport_tiled_ar_bias_rmsnorm_kernel[(N,)](
        workspace.symm_hdl.buffer_ptrs_dev,
        workspace.counter,
        workspace.phase,
        x, bias, w, y, N,
        eps=eps,
        D=D,
        rank=workspace.rank,
        world_size=workspace.world_size,
        TILE_D=TILE_D,
        NUM_TILES=NUM_TILES,
        num_warps=32,
    )
