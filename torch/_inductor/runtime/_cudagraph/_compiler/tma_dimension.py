"""Encoded TMA dimensions from the compiler's ordered recast basis."""

from math import gcd

import sympy


def tma_dimension(*values):
    if (not values or (len(values) != 1 and len(values) % 2)
            or any(type(value) is not int for value in values)):
        raise ValueError("TMA dimension requires a shape or ordered integer shape/stride pairs")
    values = tuple(value % (1 << 64) for value in values)
    if len(values) == 1:
        return values[0]
    # CUTLASS fill_tma_gmem_shape_stride uses uint64 intermediates and preserves order.
    shape, stride = 1, 0
    for next_shape, next_stride in zip(values[::2], values[1::2], strict=True):
        divisor = gcd(stride, next_stride)
        shape = (((shape - 1) * (stride // divisor)
                  + (next_shape - 1) * (next_stride // divisor) + 1) % (1 << 64)
                 if divisor else next_shape)
        stride = divisor
    return shape


class TmaDimension(sympy.Function):
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, *args):
        if (not args or (len(args) != 1 and len(args) % 2)
                or any(arg.is_integer is not True for arg in args)):
            raise ValueError("TMA dimension requires a shape or ordered integer shape/stride pairs")
        if all(isinstance(arg, sympy.Integer) for arg in args):
            return sympy.Integer(tma_dimension(*(int(arg) for arg in args)))
