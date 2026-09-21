"""Stable ordinary CuTe invocation and conversion for the TMA replay test."""

from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import get_leading_dim


GEMM = None


@cute.jit
def invocation(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    GEMM(a, b, c, stream)


def convert_arguments(a, b, c):
    return tuple(from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=get_leading_dim(value))
                 for value in (a, b, c))
