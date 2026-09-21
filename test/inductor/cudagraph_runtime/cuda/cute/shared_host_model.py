"""One ordinary Python composition of CuTe and Triton kernels."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["x", "out"])
def dynamic_user_add_one(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < n, other=0)
    tl.store(out + offsets, value + 1, offsets < n)


ADD = None
CUTE = None


def host(box):
    n, m, value = box
    box.clear()
    count = n * m * 128
    root = torch.empty_strided((3 * count,), (1,), dtype=value.dtype, device=value.device)
    ADD[lambda meta: (triton.cdiv(3 * count, meta["BLOCK"]),)](value, root, 3 * count, BLOCK=128)
    tail = root[count:]
    source = tail[:count].view(n, m, 128).permute(1, 0, 2)
    residual = tail[count:].view(n, m, 128).permute(1, 0, 2)
    destination = torch.empty_strided((m, n, 128), (128, m * 128, 1), dtype=value.dtype, device=value.device)
    CUTE(source, residual, destination, destination, n + m)
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](destination, destination, count, BLOCK=128)
    return destination, destination.permute(1, 0, 2)


def eager_reference(n, m, value):
    count = n * m * 128
    source = (value[count:2 * count] + 1).view(n, m, 128).permute(1, 0, 2)
    residual = (value[2 * count:] + 1).view(n, m, 128).permute(1, 0, 2)
    destination = source - source.mean(dim=-1, keepdim=True) + residual * 2 + n + m + 1
    return destination, destination.permute(1, 0, 2)
