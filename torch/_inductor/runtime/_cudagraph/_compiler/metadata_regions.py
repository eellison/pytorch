from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from torch._inductor.runtime._cudagraph._compiler.cfg_values import CFGProgram, ValueType
from torch._inductor.runtime._cudagraph._compiler.values import ScalarValue


_NUMERIC = frozenset({"i1", "i8", "i16", "i32", "i64", "f32", "f64"})


@dataclass(frozen=True)
class UnavailablePointer:
    llvm_type: str
    size: int

    def data(self) -> bytes:
        raise TypeError("An unavailable pointer has no bytes")

    def __bytes__(self) -> bytes:
        return self.data()

    def __reduce_ex__(self, protocol):
        raise TypeError("An unavailable pointer cannot be serialized")


@dataclass(frozen=True)
class MetadataAggregate:
    llvm_type: str
    fields: tuple[tuple[tuple[int, ...], ScalarValue | UnavailablePointer], ...]

    def data(self) -> bytes:
        raise TypeError("A metadata aggregate has no descriptor bytes")

    def __bytes__(self) -> bytes:
        return self.data()

    def __reduce_ex__(self, protocol):
        raise TypeError("A metadata aggregate cannot be serialized")


def _validate(value: Any, typ: ValueType, *, aggregate_leaf: bool = False) -> None:
    if type(value) not in (MetadataAggregate, UnavailablePointer, ScalarValue) or value.llvm_type != typ.llvm_type:
        raise ValueError("Metadata value does not match its exact compiler type")
    if typ.leaves is not None:
        if (type(value) is not MetadataAggregate or type(value.fields) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[1]) not in (ScalarValue, UnavailablePointer) for item in value.fields)
                or tuple((path, item.llvm_type) for path, item in value.fields) != typ.leaves):
            raise ValueError("Metadata aggregate lacks complete typed leaf coverage")
        for path, item in value.fields:
            if type(path) is not tuple or any(type(index) is not int or index < 0 for index in path):
                raise ValueError("Metadata aggregate paths must be exact component indices")
            _validate(item, ValueType(item.llvm_type), aggregate_leaf=True)
    elif typ.llvm_type in _NUMERIC:
        if type(value) is not ScalarValue:
            raise ValueError("Numeric metadata must contain a concrete scalar")
        value.data()
    elif typ.llvm_type.startswith("!llvm.ptr"):
        if type(value) not in (UnavailablePointer, ScalarValue) or type(value.size) is not int or value.size <= 0:
            raise ValueError("Pointer metadata requires a positive supplied storage width")
        if type(value) is UnavailablePointer:
            return
        if (aggregate_leaf or type(value) is not ScalarValue or type(value.value) is not int
                or not 0 <= value.value < 2**(8 * value.size)):
            raise ValueError("Tensor pointer leaves must be unavailable; only a scalar environment handle may be known")
    else:
        raise ValueError("Unsupported metadata value type")


def _check_computation(cfg: CFGProgram) -> None:
    if type(cfg) is not CFGProgram:
        raise TypeError("Expected an owned compiler-derived CFGProgram")
    cfg.check()

    def numeric_slot(slot: int) -> None:
        typ = cfg.types[slot]
        if typ.leaves is not None or typ.llvm_type not in _NUMERIC:
            raise ValueError("Metadata computation reads a pointer or whole aggregate")

    def expression(flow: Any) -> None:
        if flow.llvm_type not in _NUMERIC:
            raise ValueError("Metadata computation reads a pointer or whole aggregate")
        if flow.kind == "argument":
            typ = cfg.types[flow.argument]
            if flow.path:
                if typ.leaves is None or (flow.path, flow.llvm_type) not in typ.leaves:
                    raise ValueError("Metadata projection lacks its exact scalar-leaf source")
            else:
                numeric_slot(flow.argument)
        for operand in flow.operands:
            expression(operand)

    for index, block in enumerate(cfg.blocks):
        if index:
            for slot in block.arguments:
                numeric_slot(slot)
        for instruction in block.instructions:
            numeric_slot(instruction.result)
            expression(instruction.expression)
        for slot in block.terminator.values:
            numeric_slot(slot)
        for edge in block.terminator.edges:
            for slot in edge.arguments:
                numeric_slot(slot)
