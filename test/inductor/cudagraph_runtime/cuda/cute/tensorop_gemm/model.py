"""Ordinary Triton producers followed by the unchanged upstream CuTe GEMM."""

import torch
import triton


ADD = None
GEMM = None


def host(box):
    rows, a, b = box
    box.clear()
    m = 8 * rows
    count_a, count_b = 2 * m * 128, 2 * 128 * 128
    root_a = torch.empty_strided((count_a,), (1,), dtype=a.dtype, device=a.device)
    root_b = torch.empty_strided((count_b,), (1,), dtype=b.dtype, device=b.device)
    ADD[lambda meta: (triton.cdiv(count_a, meta["BLOCK"]),)](a, root_a, count_a, BLOCK=128)
    ADD[lambda meta: (triton.cdiv(count_b, meta["BLOCK"]),)](b, root_b, count_b, BLOCK=128)
    matrix_a = root_a[m * 128:].view(1, m, 128)
    matrix_b = root_b[128 * 128:].view(1, 128, 128)
    result = torch.empty_strided((1, m, 128), (m * 128, 128, 1), dtype=a.dtype, device=a.device)
    GEMM(matrix_a, matrix_b, result)
    return result, result.transpose(1, 2)


def reference(rows, a, b):
    matrix_a = (a[1024 * rows:] + 1).view(1, 8 * rows, 128)
    matrix_b = (b[128 * 128:] + 1).view(1, 128, 128)
    result = torch.bmm(matrix_a.float(), matrix_b.float()).to(a.dtype)
    return result, result.transpose(1, 2)
