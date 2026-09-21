"""A one-warp-per-row CuTe RMSNorm with no affine parameters."""

import cutlass
import cutlass.cute as cute
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry


WIDTH = 256
EPSILON = 1e-5


@cute.kernel
def rmsnorm(source: cute.Tensor, destination: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    square_sum = cutlass.Float32(0.0)
    for part in cutlass.range_constexpr(WIDTH // 32):
        value = source[row, lane + part * 32]
        square_sum += value * value
    mean_square = cute.arch.warp_reduction_sum(square_sum) / WIDTH
    scale = cute.math.rsqrt(mean_square + EPSILON)
    for part in cutlass.range_constexpr(WIDTH // 32):
        column = lane + part * 32
        destination[row, column] = source[row, column] * scale


@cute.jit
def invocation(source: cute.Tensor, destination: cute.Tensor, stream):
    rmsnorm(source, destination).launch(
        grid=(source.shape[0], 1, 1), block=(32, 1, 1), smem=0, stream=stream,
    )


ENTRY = PythonEntry(invocation)
