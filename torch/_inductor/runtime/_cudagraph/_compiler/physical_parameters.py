"""Project compiled parameter values using the compiler's physical type layout."""

from dataclasses import dataclass

from torch._inductor.runtime._cudagraph._compiler.argument_flow import ParameterFlow, ValueFlow, _Values, parameter_leaves
from torch._inductor.runtime._cudagraph._compiler.compiler_type_layout import CompiledTypeLayouts


@dataclass(frozen=True)
class ParameterLeaf:
    path: tuple[int, ...]
    llvm_type: str
    kind: str
    byte_offset: int
    byte_size: int
    source: ValueFlow


@dataclass(frozen=True)
class ParameterLiteral:
    field: ParameterLeaf
    data: bytes


@dataclass(frozen=True)
class ParameterLayout:
    parameter: ParameterFlow
    size: int
    alignment: int
    fields: tuple[ParameterLeaf, ...]
    undefined: tuple[ParameterLeaf, ...]
    padding: tuple[tuple[int, int], ...]
    constants: tuple[ParameterLiteral, ...] = ()


def project_parameters(parameters, compiled_layouts):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    if (type(parameters) is not tuple or type(compiled_layouts) is not CompiledTypeLayouts
            or len(parameters) != len(compiled_layouts.types)
            or len(parameters) != len(compiled_layouts.layouts)):
        raise TypeError("Expected ordered parameters and their compiler type layouts")
    result = []
    for parameter, typ, layout in zip(parameters, compiled_layouts.types, compiled_layouts.layouts):
        if (type(parameter) is not ParameterFlow or parameter.index != len(result)
                or parameter.llvm_type != str(typ) or parameter.llvm_type != layout.llvm_type
                or parameter.source.llvm_type != parameter.llvm_type):
            raise ValueError("Physical layout differs from its exact compiled parameter")
        with typ.context:
            leaves = parameter_leaves(parameter.source, typ)
        if tuple((path, str(leaf)) for path, leaf, _ in leaves) != tuple(
                (field.path, field.llvm_type) for field in layout.fields):
            raise ValueError("Compiler layout and parameter projection disagree")
        fields, undefined, padding, constants = [], [], [], []
        end = 0
        for (path, leaf, source), field in zip(leaves, layout.fields):
            if field.offset < end or field.offset + field.size > layout.size:
                raise ValueError("Parameter fields overlap or exceed the compiler size")
            if field.offset > end:
                padding.append((end, field.offset - end))
            end = field.offset + field.size
            kind = "pointer" if isinstance(leaf, llvm.PointerType) else str(leaf)
            row = ParameterLeaf(path, str(leaf), kind, field.offset, field.size, source)
            if source.kind == "undef":
                if (not isinstance(leaf, ir.IntegerType) or not leaf.is_signless
                        or leaf.width not in (1, 8, 16, 32, 64) or source.operands or source.attributes
                        or source.argument is not None or source.path or source.value is not None):
                    raise ValueError("Only pure integer undef leaves can define unspecified bytes")
                undefined.append(row)
            elif kind in ("i1", "i8", "i16") and source.kind in ("constant", "zero"):
                from torch._inductor.runtime._cudagraph._compiler.values import evaluate, ScalarValue

                if source.operands or source.attributes or source.argument is not None or source.path:
                    raise ValueError("Narrow fields require an exact compiler integer literal")
                literal = evaluate(source, (), typ.context)
                if type(literal) is not ScalarValue or literal.size != field.size:
                    raise ValueError("Compiler literal size differs from its physical field")
                constants.append(ParameterLiteral(row, literal.data()))
            elif kind in ("i32", "i64", "pointer", "f32"):
                with typ.context:
                    _Values(())._defined(source)
                if field.size != (4 if kind in ("i32", "f32") else 8):
                    raise ValueError("Unsupported native scalar or pointer width")
                fields.append(row)
            else:
                raise ValueError("Unsupported physical parameter leaf: " + str(leaf))
        if end < layout.size:
            padding.append((end, layout.size - end))
        result.append(ParameterLayout(parameter, layout.size, layout.alignment,
                                      tuple(fields), tuple(undefined), tuple(padding), tuple(constants)))
    return tuple(result)
