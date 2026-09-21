"""An ordinary autotuned GEMM followed by CuTe centering and a Triton epilogue."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["COLS"],
    cache_results=False,
)
@triton.jit(do_not_specialize_on_alignment=["a", "b", "out"])
def matmul(a, b, out, ROWS, COLS, K: tl.constexpr,
           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        offsets = start + reduction
        left = tl.load(a + rows[:, None] * stride_am + offsets[None, :] * stride_ak,
                       (rows[:, None] < ROWS) & (offsets[None, :] < K), other=0)
        right = tl.load(b + offsets[:, None] * stride_bk + columns[None, :] * stride_bn,
                        (offsets[:, None] < K) & (columns[None, :] < COLS), other=0)
        accumulator += tl.dot(left, right)
    tl.store(out + rows[:, None] * stride_cm + columns[None, :] * stride_cn,
             accumulator, (rows[:, None] < ROWS) & (columns[None, :] < COLS))


OPERATION = None


def make_operation(gemm, cute, add):
    def operation(n, m, left, right):
        rows, columns = 2 * n, 128 * m
        projection = torch.empty_strided((rows, columns), (columns, 1), dtype=torch.float32, device=left.device)
        gemm[lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(columns, meta["BLOCK_N"]))](
            left, right, projection, rows, columns, 128,
            left.stride(0), left.stride(1), right.stride(0), right.stride(1),
            projection.stride(0), projection.stride(1),
        )
        count = n * m * 128
        flat = projection.view(-1)
        source = flat[:count].view(n, m, 128).permute(1, 0, 2)
        residual = flat[count:].view(n, m, 128).permute(1, 0, 2)
        destination = torch.empty_strided((m, n, 128), (128, m * 128, 1), dtype=torch.float32, device=left.device)
        cute(source, residual, destination, destination, n + m)
        add[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](destination, destination, count, BLOCK=128)
        return destination, destination.permute(1, 0, 2)

    return operation


def host(box):
    n, m, left, right = box
    box.clear()
    return OPERATION(n, m, left, right)


def torch_reference(n, m, left, right):
    projection = left.float() @ right.float()
    count = n * m * 128
    flat = projection.flatten()
    source = flat[:count].view(n, m, 128).permute(1, 0, 2)
    residual = flat[count:].view(n, m, 128).permute(1, 0, 2)
    destination = source - source.mean(dim=-1, keepdim=True) + residual * 2 + n + m + 1
    return destination, destination.permute(1, 0, 2)
