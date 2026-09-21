from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SOURCE_ARG_ATTR = "cudagraph.source_arg"


@dataclass(frozen=True)
class MetadataSlot:
    path: tuple[int, ...]
    name: str
    kind: str
    ir_arg_index: int | None
    abi_arg_index: int | None


@dataclass(frozen=True)
class FormalLowering:
    metadata: MetadataSlot
    ir_arg_index: int
    llvm_arg_index: int
    source_type: str
    llvm_type: str


def _metadata_slots(metadata: Any) -> tuple[MetadataSlot, ...]:
    import cutlass.compiler as compiler

    if type(metadata) is not compiler.FunctionMetadata:
        raise TypeError("Expected the original compiler FunctionMetadata")
    forwarded = (compiler.Var, compiler.Tensor, compiler.Stream, compiler.EnvStream, compiler.Pointer)
    structural = (compiler.Const, compiler.Shape, compiler.Unit, compiler.Tuple)
    slots = []

    def visit(binding: Any, path: tuple[int, ...]) -> None:
        index, abi_index = binding.ir_arg_index, binding.abi_arg_index
        if abi_index is not None and (type(abi_index) is not int or abi_index < 0):
            raise ValueError("Invalid metadata ABI argument index")
        if isinstance(binding, forwarded):
            if type(index) is not int or index < 0:
                raise ValueError("Forwarded metadata leaf lacks its compiler IR index")
        elif isinstance(binding, structural):
            if index is not None:
                raise ValueError("Nonforwarded metadata binding has an IR index")
        else:
            raise ValueError("Unsupported compiler metadata binding")
        slots.append(MetadataSlot(path, binding.name, type(binding).__name__, index, abi_index))
        if isinstance(binding, compiler.Tuple):
            for child_index, child in enumerate(binding.values):
                visit(child, (*path, child_index))

    for index, binding in enumerate(metadata.params):
        visit(binding, (index,))
    return tuple(slots)


def _formals(module: Any, function_name: str, operation_name: str) -> tuple[tuple[int, str], ...]:
    from cutlass._mlir import ir

    with module.context, ir.raw_values():
        found = [view.operation for view in module.body.operations
                 if view.operation.name == operation_name
                 and view.operation.attributes["sym_name"].value == function_name]
        if len(found) != 1 or len(found[0].regions) != 1 or not found[0].regions[0].blocks:
            raise ValueError("Expected the exact original host function definition")
        function = found[0]
        arguments = function.regions[0].blocks[0].arguments
        attributes = function.attributes.get("arg_attrs")
        if not isinstance(attributes, ir.ArrayAttr) or len(attributes) != len(arguments):
            raise ValueError("Host formals lack complete compiler-preserved source markers")
        result = []
        for argument, dictionary in zip(arguments, attributes):
            if not isinstance(dictionary, ir.DictAttr):
                raise ValueError("Expected a dictionary of host argument attributes")
            marker = dictionary[SOURCE_ARG_ATTR] if SOURCE_ARG_ATTR in dictionary else None
            if not isinstance(marker, ir.IntegerAttr) or str(marker.type) != "i64" or marker.value < 0:
                raise ValueError("Host formal lacks an exact source argument marker")
            result.append((marker.value, str(argument.type)))
        return tuple(result)


def _bind(program: Any, metadata: Any, expected_source_types: tuple[str, ...]):
    program.check()
    owned = program.source_metadata
    owned = owned if isinstance(owned, (tuple, list)) else (owned,)
    if not any(metadata is item for item in owned) or metadata.symbol_name != program.function_name:
        raise RuntimeError("Metadata does not belong to the original compiler artifact")
    slots = _metadata_slots(metadata)
    forwarded = [slot for slot in slots if slot.ir_arg_index is not None]
    source = _formals(program.source_module, program.function_name, "func.func")
    lowered = _formals(program.module, program.function_name, "llvm.func")
    if tuple(typ for _, typ in source) != expected_source_types:
        raise ValueError("Actual compiler argument types differ from the original host formals")
    source_indices = tuple(index for index, _ in source)
    if source_indices != tuple(range(len(source))):
        raise ValueError("Original source marker does not identify its compiler IR argument")
    indices = [slot.ir_arg_index for slot in forwarded]
    if len(set(indices)) != len(indices) or set(indices) != set(source_indices):
        raise ValueError("Metadata IR indices do not cover the original formals exactly")
    lowered_indices = [index for index, _ in lowered]
    if len(set(lowered_indices)) != len(lowered_indices) or set(lowered_indices) != set(source_indices):
        raise ValueError("Lowering dropped, duplicated, split, or invented a source formal")
    lowered_by_source = {source_index: (index, typ) for index, (source_index, typ) in enumerate(lowered)}
    records = []
    for slot in forwarded:
        source_index = slot.ir_arg_index
        llvm_index, llvm_type = lowered_by_source[source_index]
        source_type = source[source_index][1]
        if slot.kind == "Var" and source_type != llvm_type:
            raise ValueError("This subset requires unchanged scalar widths and types at the host boundary")
        if slot.kind in ("Stream", "EnvStream") and (source_type != "!cuda.stream" or llvm_type != "!llvm.ptr"):
            raise ValueError("Unsupported host stream lowering")
        records.append(FormalLowering(slot, source_index, llvm_index, source_type, llvm_type))
    program.check()
    return tuple(records), tuple(slot for slot in slots if slot.ir_arg_index is None)


@dataclass(frozen=True)
class FormalLoweringSet:
    program: Any = field(repr=False, compare=False)
    metadata: Any = field(repr=False, compare=False)
    expected_source_types: tuple[str, ...]
    formals: tuple[FormalLowering, ...]
    nonforwarded: tuple[MetadataSlot, ...]
    _owners: tuple[Any, Any] = field(repr=False, compare=False)

    def check(self) -> None:
        if self.program is not self._owners[0] or self.metadata is not self._owners[1]:
            raise RuntimeError("Original formal-lowering ownership changed")
        if _bind(self.program, self.metadata, self.expected_source_types) != (self.formals, self.nonforwarded):
            raise RuntimeError("Original formal-lowering correspondence changed")


def bind_host_formals(program: Any, metadata: Any, expected_source_types: tuple[str, ...]) -> FormalLoweringSet:
    if type(expected_source_types) is not tuple or any(type(typ) is not str for typ in expected_source_types):
        raise TypeError("Expected exact MLIR type strings from the actual compiler arguments")
    formals, nonforwarded = _bind(program, metadata, expected_source_types)
    return FormalLoweringSet(program, metadata, expected_source_types, formals, nonforwarded, (program, metadata))
