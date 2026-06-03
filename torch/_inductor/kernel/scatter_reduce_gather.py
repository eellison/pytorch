"""
Fused scatter-reduce-gather kernel for the bilinear upsample backward pattern.

This kernel eliminates the scatter intermediate by iterating over the OUTPUT
positions and gathering contributions from source positions, applying the mask
inline, and accumulating into per-channel results.

Target pattern (from UNet backward):
    zeros[B, C, H_out, W_out] = 0
    for each of 4 bilinear corners:
        index_put_(zeros, [None, None, row_idx, col_idx], weighted_src, accumulate=True)
    masked = where(bn_relu_gate <= 0, 0, zeros)
    result[C] = sum(masked, dim=[0, 2, 3])

Rewritten as:
    for c in range(C):
        for b in range(B):
            for h in range(H_out):
                for w in range(W_out):
                    if mask[b, c, h, w]:  # BN/ReLU gate
                        # Gather: what did the scatter put at (h, w)?
                        # For bilinear 2x: it's the sum of contributions from
                        # source positions that map to this output position.
                        val = gather_bilinear_contributions(sources, h, w)
                        result[c] += val

This avoids:
- Allocating the [B, C, H_out, W_out] intermediate
- Writing 4 * B*C*H_src*W_src atomic operations
- Reading B*C*H_out*W_out for the reduction

Instead does:
- Reading B*C*H_out*W_out of mask (needed regardless)
- For each unmasked position, gathers from source (~4 reads per position)
- Per-channel tree reduction
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


def scatter_reduce_gather_reference(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    output_shape: tuple[int, ...],
    reduction_dims: list[int],
) -> torch.Tensor:
    """Reference implementation of the fused scatter-reduce-gather.

    This computes the same result as:
        zeros = torch.zeros(output_shape)
        for src, row_idx, col_idx in zip(sources, row_indices, col_indices):
            zeros = index_put(zeros, [None, None, row_idx, col_idx], src, True)
        masked = where(mask, 0, zeros)
        result = sum(masked, dim=reduction_dims)

    But without materializing the intermediate 'zeros' buffer.

    Args:
        sources: List of source tensors being scattered [B, C, H_src, W_src]
        row_indices: List of row index tensors [H_src, 1] or [H_out, 1]
        col_indices: List of col index tensors [W_src] or [W_out]
        mask: Boolean mask in output space [B, C, H_out, W_out] (True = masked out)
        output_shape: Shape of the scatter output [B, C, H_out, W_out]
        reduction_dims: Dimensions to reduce [0, 2, 3]
    """
    B, C, H_out, W_out = output_shape

    # The key insight: we iterate over OUTPUT positions.
    # For each output position (b, c, h, w), we need to find which source
    # positions mapped to it (reverse mapping).
    #
    # For bilinear 2x upsample backward:
    # Source[b, c, r, s] maps to output[b, c, row_idx[r], col_idx[s]]
    # So output[b, c, h, w] receives from all (r, s) where row_idx[r]=h and col_idx[s]=w.
    #
    # Instead of building this reverse mapping explicitly, we can iterate
    # over SOURCE positions and accumulate into the reduction, gating by
    # whether the OUTPUT mask allows it.
    #
    # output[h, w] is masked => skip ALL source contributions to (h, w)
    # output[h, w] is NOT masked => accumulate all source contributions to (h, w)

    # Channel dimension is preserved (not reduced), so result shape is [C]
    # assuming reduction_dims = [0, 2, 3]
    result = torch.zeros(C, device=sources[0].device, dtype=sources[0].dtype)

    for src, row_idx, col_idx in zip(sources, row_indices, col_indices):
        # src shape: [B, C, H_src, W_src]
        # row_idx shape: [H_src, 1] - each source row maps to this output row
        # col_idx shape: [W_src] - each source col maps to this output col
        H_src = row_idx.shape[0]
        W_src = col_idx.shape[0]

        for b in range(B):
            for r in range(H_src):
                h = row_idx[r, 0].item()  # output row
                for s in range(W_src):
                    w = col_idx[s].item()  # output col
                    # Check mask at the output position
                    if not mask[b, :, h, w].any():
                        # Not masked - accumulate this contribution
                        # Note: mask is True where we ZERO out, so ~mask means "keep"
                        pass
                    # Per-channel: mask[b, c, h, w] determines per-channel
                    for c in range(C):
                        if not mask[b, c, h, w].item():
                            result[c] += src[b, c, r, s].item()

    return result


def scatter_reduce_gather_fast(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    output_shape: tuple[int, ...],
) -> torch.Tensor:
    """Vectorized PyTorch implementation of the fused scatter-reduce-gather.

    Strategy: iterate over source positions, look up the output mask for
    each source position's target, and accumulate into per-channel sums.

    This is O(B * C * H_src * W_src * N_corners) which is the same work as
    the original scatter, but avoids atomic contention and intermediate memory.

    Memory-efficient: gathers the mask per batch to avoid materializing
    huge [B, C, H_src, W_src] index tensors.
    """
    B, C, H_out, W_out = output_shape
    device = sources[0].device
    dtype = sources[0].dtype

    result = torch.zeros(C, device=device, dtype=dtype)

    for src, row_idx, col_idx in zip(sources, row_indices, col_indices):
        # src: [B, C, H_src, W_src]
        # row_idx: [H_src, 1]  -> target output row for each source row
        # col_idx: [W_src]     -> target output col for each source col

        H_src = row_idx.shape[0]
        W_src = col_idx.shape[0]

        # Gather mask into source space efficiently by processing per-batch.
        # mask shape: [B, C, H_out, W_out]
        # We want: gathered_mask[b, c, r, s] = mask[b, c, row_idx[r, 0], col_idx[s]]
        #
        # Since row_idx and col_idx are spatial-only (no batch/channel dependence),
        # we can use indexing on the spatial dims directly per batch element.
        # mask[b, :, row_idx[:, 0], col_idx[:]] gives [C, H_src, W_src] per batch.

        # Flatten row_idx to [H_src]
        row_flat = row_idx.view(-1)  # [H_src]

        # Gather: mask[b, :, row_flat, :] first, then index cols
        # mask[:, :, row_flat, :] -> [B, C, H_src, W_out]
        # Then index cols: [..., col_idx] -> [B, C, H_src, W_src]
        # This avoids creating [B, C, H_src, W_src] index tensors for batch/channel.
        gathered_mask = mask[:, :, row_flat, :][:, :, :, col_idx]
        # gathered_mask: [B, C, H_src, W_src] - True where output is masked

        # Zero out masked source contributions and reduce
        # ~gathered_mask means "this source value's target is NOT masked"
        unmasked_src = src * (~gathered_mask).to(dtype)

        # Sum over batch and spatial dims [0, 2, 3] -> [C]
        result += unmasked_src.sum(dim=[0, 2, 3])

    return result


def try_scatter_reduce_gather_triton(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    output_shape: tuple[int, ...],
) -> Optional[torch.Tensor]:
    """Try to use Triton for the fused kernel. Falls back to PyTorch if unavailable."""
    try:
        import triton
        import triton.language as tl
        return _triton_scatter_reduce_gather(sources, row_indices, col_indices, mask, output_shape)
    except ImportError:
        return None


def _triton_scatter_reduce_gather(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    output_shape: tuple[int, ...],
) -> torch.Tensor:
    """Triton implementation of the fused scatter-reduce-gather kernel.

    Launch strategy: one program per channel, each program iterates over
    (batch, H_src, W_src) and accumulates unmasked contributions.
    """
    import triton
    import triton.language as tl

    B, C, H_out, W_out = output_shape

    @triton.jit
    def _kernel_single_source(
        src_ptr, row_idx_ptr, col_idx_ptr, mask_ptr, result_ptr,
        B: tl.constexpr, C: tl.constexpr,
        H_src: tl.constexpr, W_src: tl.constexpr,
        H_out: tl.constexpr, W_out: tl.constexpr,
        src_stride_b, src_stride_c, src_stride_h, src_stride_w,
        mask_stride_b, mask_stride_c, mask_stride_h, mask_stride_w,
        BLOCK_HW: tl.constexpr,
    ):
        """Process one channel, accumulating across all source positions."""
        c_idx = tl.program_id(0)
        if c_idx >= C:
            return

        acc = 0.0

        # Iterate over batch
        for b in range(B):
            # Iterate over source spatial positions in blocks
            for hw_start in range(0, H_src * W_src, BLOCK_HW):
                hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
                hw_mask_valid = hw_offsets < (H_src * W_src)

                # Decompose linear index to (r, s)
                r = hw_offsets // W_src
                s = hw_offsets % W_src

                # Look up target output position
                # row_idx[r, 0] and col_idx[s]
                target_h = tl.load(row_idx_ptr + r, mask=hw_mask_valid, other=0)
                target_w = tl.load(col_idx_ptr + s, mask=hw_mask_valid, other=0)

                # Check mask at target position: mask[b, c, target_h, target_w]
                mask_offset = (
                    b * mask_stride_b +
                    c_idx * mask_stride_c +
                    target_h * mask_stride_h +
                    target_w * mask_stride_w
                )
                is_masked = tl.load(mask_ptr + mask_offset, mask=hw_mask_valid, other=True)

                # Load source value: src[b, c, r, s]
                src_offset = (
                    b * src_stride_b +
                    c_idx * src_stride_c +
                    r * src_stride_h +
                    s * src_stride_w
                )
                src_val = tl.load(src_ptr + src_offset, mask=hw_mask_valid, other=0.0)

                # Accumulate unmasked values
                # is_masked=True means zero it out, so we want ~is_masked
                contribution = tl.where(is_masked, 0.0, src_val)
                acc += tl.sum(contribution * hw_mask_valid.to(tl.float32), axis=0)

        # Store result (accumulate into existing value for multi-source)
        old_val = tl.load(result_ptr + c_idx)
        tl.store(result_ptr + c_idx, old_val + acc)

    # Allocate output
    device = sources[0].device
    dtype = sources[0].dtype
    result = torch.zeros(C, device=device, dtype=dtype)

    # Process each source tensor
    for src, row_idx, col_idx in zip(sources, row_indices, col_indices):
        H_src = row_idx.shape[0]
        W_src = col_idx.shape[0]

        # Flatten row_idx from [H_src, 1] to [H_src]
        row_idx_flat = row_idx.view(-1).contiguous()
        col_idx_flat = col_idx.view(-1).contiguous()

        # Determine block size
        BLOCK_HW = min(1024, triton.next_power_of_2(H_src * W_src))

        grid = (C,)
        _kernel_single_source[grid](
            src, row_idx_flat, col_idx_flat, mask, result,
            B, C, H_src, W_src, H_out, W_out,
            src.stride(0), src.stride(1), src.stride(2), src.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            BLOCK_HW=BLOCK_HW,
        )

    return result.to(dtype)
