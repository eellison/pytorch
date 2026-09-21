"""Tensor-list views feed the existing direct CuTe and Triton composition."""

import torch
import triton

OPERATION = None


def make_operation(add, cute):
    def operation(n, m, value, block):
        count = n * m * 128
        root = torch.empty_strided((3 * count,), (1,), dtype=value.dtype, device=value.device)
        add[lambda meta: (triton.cdiv(3 * count, meta["BLOCK"]),)](value, root, 3 * count, BLOCK=block)
        tail = root[count:]
        source_flat, residual_flat = tail.chunk(2, dim=0)
        source = source_flat.view(n, m, 128).permute(1, 0, 2)
        residual = residual_flat.view(n, m, 128).permute(1, 0, 2)
        destination = torch.empty_strided((m, n, 128), (128, m * 128, 1), dtype=value.dtype, device=value.device)
        cute(source, residual, destination, destination, n + m)
        add[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](destination, destination, count, BLOCK=block)
        return destination, destination.permute(1, 0, 2)

    return operation


def host(box):
    n, m, value = box
    box.clear()
    block = 256 if n > m else 128
    return OPERATION(n, m, value, block)
