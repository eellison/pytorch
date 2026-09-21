"""Two ordinary compiler selections from one constexpr-tiled CuTe kernel source."""

import importlib.util
from itertools import count
from pathlib import Path
import sys

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry


ROWS_PER_CTA = 1


def convert_arguments(source, residual, destination, repeated_destination, bias):
    values = tuple(from_dlpack(tensor, assumed_align=256, use_32bit_stride=False)
                   for tensor in (source, residual, destination, repeated_destination))
    for value in values:
        for axis in range(source.ndim - 1):
            value.mark_compact_shape_dynamic(axis, stride_order=(1, 0, 2), divisibility=1)
    return (*values, cutlass.Int32(bias))


@cute.kernel
def residual_center(source: cute.Tensor, residual: cute.Tensor, destination: cute.Tensor,
                    repeated_destination: cute.Tensor, bias: cutlass.Int32, rows_per_cta: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, column, _ = cute.arch.block_idx()
    lane = thread % 32
    row = block * rows_per_cta + thread // 32
    if row < source.shape[0]:
        x0 = source[row, column, lane]
        x1 = source[row, column, lane + 32]
        x2 = source[row, column, lane + 64]
        x3 = source[row, column, lane + 96]
        mean = cute.arch.warp_reduction_sum(x0 + x1 + x2 + x3) * 0.0078125
        destination[row, column, lane] = x0 - mean + residual[row, column, lane] + bias
        destination[row, column, lane] = repeated_destination[row, column, lane] + residual[row, column, lane]
        destination[row, column, lane + 32] = x1 - mean + residual[row, column, lane + 32] + bias
        destination[row, column, lane + 32] = repeated_destination[row, column, lane + 32] + residual[row, column, lane + 32]
        destination[row, column, lane + 64] = x2 - mean + residual[row, column, lane + 64] + bias
        destination[row, column, lane + 64] = repeated_destination[row, column, lane + 64] + residual[row, column, lane + 64]
        destination[row, column, lane + 96] = x3 - mean + residual[row, column, lane + 96] + bias
        destination[row, column, lane + 96] = repeated_destination[row, column, lane + 96] + residual[row, column, lane + 96]


@cute.jit
def invocation(source: cute.Tensor, residual: cute.Tensor, destination: cute.Tensor,
               repeated_destination: cute.Tensor, bias: cutlass.Int32, stream):
    residual_center(source, residual, destination, repeated_destination, bias, ROWS_PER_CTA).launch(
        grid=(cute.ceil_div(source.shape[0], ROWS_PER_CTA), source.shape[1], 1), block=(32 * ROWS_PER_CTA, 1, 1), smem=0, stream=stream,
    )


ENTRY = PythonEntry(invocation)
_generation = count()


def make_fixture(rows_per_cta):
    name = f"{__name__}_ordinary_{next(_generation)}"
    spec = importlib.util.spec_from_file_location(name, Path(__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("The CuTe argument fixture has no source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ROWS_PER_CTA = rows_per_cta
    return module.ENTRY, module.residual_center


def make_selection(one, two):
    def selected(source, residual, destination, repeated_destination, bias):
        if source.shape[0] > source.shape[1]:
            two(source, residual, destination, repeated_destination, bias)
        else:
            one(source, residual, destination, repeated_destination, bias)
    return selected
