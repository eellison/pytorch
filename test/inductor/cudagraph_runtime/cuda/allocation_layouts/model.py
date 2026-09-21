"""Materialize contiguous and permuted rank-three allocation roots."""

from kernels import dynamic_user_add_one as USER_KERNEL
import torch
import triton


class AllocationLayouts(torch.nn.Module):
    def forward(self, value):
        batch = value.shape[1]
        source = value.sin()
        dense = torch.empty_like(source)
        torch.library.wrap_triton(USER_KERNEL)[(triton.cdiv(source.numel(), 128),)](
            source, dense, source.numel(), BLOCK=128)
        strided = torch.empty_strided(
            (batch, 3, 128), (128, batch * 128, 1), dtype=value.dtype, device=value.device)
        torch.library.wrap_triton(USER_KERNEL)[(triton.cdiv(dense.numel(), 128),)](
            dense, strided, dense.numel(), BLOCK=128)
        result = strided.cos() + dense.permute(1, 0, 2)
        return result, dense, strided


def eager_reference(value):
    dense = value.sin() + 1
    strided = (dense + 1).permute(1, 0, 2)
    return strided.cos() + dense.permute(1, 0, 2), dense, strided
