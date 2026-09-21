"""Two static CuTe launch configurations for the ordinary cache workload."""

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["source", "destination"])
def add_one(source, destination, count, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < count, other=0)
    tl.store(destination + index, value + 1, index < count)


@cute.kernel
def affine(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32,
           rows_per_cta: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    column = thread % 128
    row = block * rows_per_cta + thread // 128
    if row < source.shape[0]:
        destination[row, column] = source[row, column] * 2.0 + bias


@cute.jit
def launch_one(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32, stream):
    affine(source, destination, bias, 1).launch(
        grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=0, stream=stream,
    )


@cute.jit
def launch_two(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32, stream):
    affine(source, destination, bias, 2).launch(
        grid=(cute.ceil_div(source.shape[0], 2), 1, 1), block=(256, 1, 1), smem=0, stream=stream,
    )


def convert_arguments(source, destination, bias):
    values = tuple(from_dlpack(tensor, assumed_align=16, use_32bit_stride=False)
                   for tensor in (source, destination))
    for value in values:
        value.mark_compact_shape_dynamic(0, stride_order=(0, 1), divisibility=1)
    return (*values, cutlass.Int32(bias))

