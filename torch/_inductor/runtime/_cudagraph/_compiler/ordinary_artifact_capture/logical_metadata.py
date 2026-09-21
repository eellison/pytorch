"""Project compiler-elided singleton strides without changing Python tensors."""

import cutlass.compiler as compiler
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute, func
from cutlass.cute.core import _unpack_x_tuple


def project_metadata(metadata, tensor_types):
    """Update fresh metadata using the actual compiler tensor formal types."""
    tensors = tuple(parameter for parameter in metadata.params if type(parameter) is compiler.Tensor)
    if set(tensor_types) != {parameter.name for parameter in tensors}:
        raise ValueError("Logical tensor types must cover the exact metadata tensor formals")
    with ir.Location.unknown(), ir.raw_values():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("__logical_tensor_metadata",
                                   ([tensor_types[parameter.name] for parameter in tensors], []))
        block = function.add_entry_block()
        with ir.InsertionPoint(block):
            for parameter, argument in zip(tensors, block.arguments, strict=True):
                if not isinstance(argument.type, cute.MemRefType):
                    raise ValueError("Logical tensor metadata requires a compiler MemRef formal")
                layout = cute.GetLayoutOp(argument).result
                shape = cute.GetShapeOp(layout).result
                stride = cute.GetStrideOp(layout).result
                shapes = tuple(cute.GetLeavesOp(shape).results)
                strides = tuple(cute.GetLeavesOp(stride).results)
                if (shape.type.rank != len(parameter.shape) or stride.type.rank != len(parameter.strides)
                        or len(shapes) != len(parameter.shape) or len(strides) != len(parameter.strides)):
                    raise ValueError("Logical tensor property rank differs from its metadata")
                for axis, (size, step) in enumerate(zip(shapes, strides, strict=True)):
                    raw_size = parameter.shape[axis]
                    if (raw_size.is_const and raw_size.value == 1 and cute.is_static(size.type)
                            and _unpack_x_tuple(size) == 1 and cute.is_static(step.type)
                            and _unpack_x_tuple(step) == 0):
                        parameter.strides[axis] = compiler.Dim(0)
            func.ReturnOp([])
        if not module.operation.verify():
            raise ValueError("Logical tensor metadata projection failed verification")
    return metadata
