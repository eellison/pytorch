"""Padding strategies for the 128x4 blocked scale swizzle, all layout-identical."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from torch._inductor import inductor_prims


def _padded_shape(rows: int, cols: int) -> tuple[int, int]:
    return (rows + 127) // 128 * 128, (cols + 3) // 4 * 4


def to_blocked_fpad(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows, padded_cols = _padded_shape(rows, cols)
    scale = F.pad(scale, (0, padded_cols - cols, 0, padded_rows - rows))
    blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
    return blocks.permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def _scatter_valid(scale: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    rows, cols = scale.shape
    padded_rows, padded_cols = _padded_shape(rows, cols)
    row = torch.arange(rows, device=scale.device)[:, None]
    col = torch.arange(cols, device=scale.device)[None, :]
    offset = row // 128 * (padded_cols // 4) * 512
    offset = offset + col // 4 * 512
    offset = offset + row % 32 * 16
    offset = offset + row // 32 % 4 * 4
    offset = offset + col % 4
    out = torch.empty(padded_rows * padded_cols, dtype=scale.dtype, device=scale.device)
    out = torch.ops.aten._unsafe_index_put.default(out, [offset], scale, False)
    return out, padded_rows, padded_cols


def _padding_mask(out: torch.Tensor, rows: int, cols: int, padded_cols: int) -> torch.Tensor:
    linear = torch.arange(out.numel(), device=out.device)
    col_inner = linear % 4
    linear = linear // 4
    row_outer = linear % 4
    linear = linear // 4
    row_lane = linear % 32
    linear = linear // 32
    col_outer = linear % (padded_cols // 4)
    row_chunk = linear // (padded_cols // 4)
    logical_row = row_chunk * 128 + row_outer * 32 + row_lane
    logical_col = col_outer * 4 + col_inner
    return (logical_row >= rows) | (logical_col >= cols)


def to_blocked_predicated(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    out, _, padded_cols = _scatter_valid(scale)
    mask = _padding_mask(out, rows, cols, padded_cols)
    zero = torch.zeros((), dtype=scale.dtype, device=scale.device)
    return inductor_prims.predicated_masked_fill(out, mask, zero)


def to_blocked_index_put(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    out, _, padded_cols = _scatter_valid(scale)
    mask = _padding_mask(out, rows, cols, padded_cols)
    zero = torch.zeros((), dtype=scale.dtype, device=scale.device)
    return torch.ops.aten.index_put.default(out, [mask], zero, False)


def to_blocked_zeros_scatter(scale: torch.Tensor) -> torch.Tensor:
    """No masked fill: zero-init the destination, then scatter the valid lanes."""
    rows, cols = scale.shape
    padded_rows, padded_cols = _padded_shape(rows, cols)
    row = torch.arange(rows, device=scale.device)[:, None]
    col = torch.arange(cols, device=scale.device)[None, :]
    offset = row // 128 * (padded_cols // 4) * 512
    offset = offset + col // 4 * 512
    offset = offset + row % 32 * 16
    offset = offset + row // 32 % 4 * 4
    offset = offset + col % 4
    out = torch.zeros(padded_rows * padded_cols, dtype=scale.dtype, device=scale.device)
    return torch.ops.aten._unsafe_index_put.default(out, [offset], scale, False)


def to_blocked_zeros_tail(scale: torch.Tensor) -> torch.Tensor:
    """Zero only the region that can contain pad lanes, then scatter.

    With column padding every 512B block holds pad lanes, but for row-only
    padding they all live in the final 128-row chunk.
    """
    rows, cols = scale.shape
    padded_rows, padded_cols = _padded_shape(rows, cols)
    total = padded_rows * padded_cols
    if padded_cols == cols:
        tail = (padded_rows - 128) // 128 * (padded_cols // 4) * 512
    else:
        tail = 0
    out = torch.empty(total, dtype=scale.dtype, device=scale.device)
    out[tail:] = 0
    row = torch.arange(rows, device=scale.device)[:, None]
    col = torch.arange(cols, device=scale.device)[None, :]
    offset = row // 128 * (padded_cols // 4) * 512
    offset = offset + col // 4 * 512
    offset = offset + row % 32 * 16
    offset = offset + row // 32 % 4 * 4
    offset = offset + col % 4
    return torch.ops.aten._unsafe_index_put.default(out, [offset], scale, False)


def _offsets(rows_iota, cols_iota, padded_cols):
    r = rows_iota[:, None]
    c = cols_iota[None, :]
    return (
        r // 128 * (padded_cols // 4) * 512
        + c // 4 * 512
        + r % 32 * 16
        + r // 32 % 4 * 4
        + c % 4
    )


def to_blocked_pad_scatter(scale: torch.Tensor) -> torch.Tensor:
    """Scatter the valid lanes, then scatter zeros over only the pad lanes.

    The pad set is two disjoint rectangles, so it is an affine index space; the
    zeroing grid is sized to the padding instead of to the whole output.
    """
    rows, cols = scale.shape
    padded_rows, padded_cols = _padded_shape(rows, cols)
    device = scale.device
    out = torch.empty(padded_rows * padded_cols, dtype=scale.dtype, device=device)
    offset = _offsets(torch.arange(rows, device=device), torch.arange(cols, device=device), padded_cols)
    out = torch.ops.aten._unsafe_index_put.default(out, [offset], scale, False)
    for row_lo, row_hi, col_lo, col_hi in (
        (rows, padded_rows, 0, padded_cols),
        (0, rows, cols, padded_cols),
    ):
        if row_lo >= row_hi or col_lo >= col_hi:
            continue
        pad_offset = _offsets(
            torch.arange(row_lo, row_hi, device=device),
            torch.arange(col_lo, col_hi, device=device),
            padded_cols,
        )
        zeros = torch.zeros((row_hi - row_lo, col_hi - col_lo), dtype=scale.dtype, device=device)
        out = torch.ops.aten._unsafe_index_put.default(out, [pad_offset], zeros, False)
    return out


VARIANTS = {
    "fpad": to_blocked_fpad,
    "fpad_cat": to_blocked_fpad,
    "index_put": to_blocked_index_put,
    "predicated": to_blocked_predicated,
    "zeros_scatter": to_blocked_zeros_scatter,
    "zeros_tail": to_blocked_zeros_tail,
    "pad_scatter": to_blocked_pad_scatter,
}
