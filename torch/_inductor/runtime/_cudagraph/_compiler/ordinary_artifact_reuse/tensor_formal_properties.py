"""Read tensor specialization directly from a CuTe host formal type."""

from dataclasses import dataclass

from cutlass._mlir import ir
from cutlass._mlir.dialects import cute, func
from cutlass.cute.core import _unpack_x_tuple


@dataclass(frozen=True)
class IntegerProperty:
    constant: int | None
    bits: int | None
    divisibility: int | None


def tensor_properties(tensor_type):
    if not isinstance(tensor_type, cute.MemRefType):
        raise ValueError("Expected an actual CuTe tensor formal type")
    properties = []
    with tensor_type.context, ir.Location.unknown(), ir.raw_values():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("__tensor_properties", ([tensor_type], []))
        block = function.add_entry_block()
        with ir.InsertionPoint(block):
            layout = cute.GetLayoutOp(block.arguments[0]).result
            for value in (cute.GetShapeOp(layout).result, cute.GetStrideOp(layout).result):
                leaves = tuple(cute.GetLeavesOp(value).results)
                if value.type.rank != len(leaves):
                    raise ValueError("Nested tensor properties need explicit metadata correspondence")
                row = []
                for axis, leaf in enumerate(leaves):
                    if cute.is_static(leaf.type):
                        constant = _unpack_x_tuple(leaf)
                        if type(constant) is not int:
                            raise ValueError("Expected an integer tensor property")
                        row.append(IntegerProperty(constant, None, None))
                    else:
                        scalars = tuple(cute.GetScalarsOp(leaf).results)
                        if len(scalars) != 1 or not isinstance(scalars[0].type, ir.IntegerType):
                            raise ValueError("Expected one dynamic integer tensor property")
                        bits = scalars[0].type.width
                        divisor = value.type.get_divisibility([axis])
                        if bits not in (32, 64) or type(divisor) is not int or divisor <= 0:
                            raise ValueError("Unsupported tensor property width or divisibility")
                        row.append(IntegerProperty(None, bits, divisor))
                properties.append(tuple(row))
            func.ReturnOp([])
        if not module.operation.verify():
            raise ValueError("Tensor property extraction failed verification")
    return tuple(properties)
