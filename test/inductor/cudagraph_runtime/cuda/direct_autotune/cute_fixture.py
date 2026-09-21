"""A real CuTe reduction with three tensors, a repeated destination and an integer."""

import importlib.util
from itertools import count
from pathlib import Path
import sys

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry


def convert_arguments(source, residual, destination, repeated_destination, bias):
    values = tuple(from_dlpack(tensor, assumed_align=16, use_32bit_stride=False)
                   for tensor in (source, residual, destination, repeated_destination))
    for value in values:
        for axis in range(source.ndim - 1):
            value.mark_compact_shape_dynamic(axis, stride_order=(1, 0, 2), divisibility=1)
    return (*values, cutlass.Int32(bias))


@cute.kernel
def residual_center(source: cute.Tensor, residual: cute.Tensor, destination: cute.Tensor,
                    repeated_destination: cute.Tensor, bias: cutlass.Int32):
    lane, _, _ = cute.arch.thread_idx()
    row, column, _ = cute.arch.block_idx()
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
    residual_center(source, residual, destination, repeated_destination, bias).launch(
        grid=(source.shape[0], source.shape[1], 1), block=(32, 1, 1), smem=0, stream=stream,
    )


ENTRY = PythonEntry(invocation)
_generation = count()


def make_fixture():
    name = f"{__name__}_ordinary_{next(_generation)}"
    spec = importlib.util.spec_from_file_location(name, Path(__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("The CuTe argument fixture has no source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ENTRY, module.residual_center
