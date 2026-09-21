from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.argument_flow import ValueFlow


@dataclass(frozen=True)
class FieldSource:
    kind: str
    ir_arg_index: int | None
    formal_name: str | None
    metadata_path: tuple[int, ...]
    property: str
    property_path: tuple[int, ...]
    value: ValueFlow


@dataclass(frozen=True)
class PointerField:
    parameter: int
    byte_offset: int
    source: FieldSource


@dataclass(frozen=True)
class IntegerField:
    parameter: int
    byte_offset: int
    dtype: str
    source: FieldSource


@dataclass(frozen=True)
class FixedField:
    """A required capture/source guard, not proof that this value is fixed."""

    parameter: int
    byte_offset: int
    byte_size: int
    dtype: str
    source: FieldSource


@dataclass(frozen=True)
class Padding:
    """Bytes to preserve from checked capture; never an inferred scalar field."""

    parameter: int
    byte_offset: int
    byte_size: int


@dataclass(frozen=True)
class UndefinedField:
    parameter: int
    byte_offset: int
    byte_size: int
    source: ValueFlow


@dataclass(frozen=True)
class ConstantField:
    parameter: int
    byte_offset: int
    data: bytes
    source: ValueFlow


@dataclass(frozen=True)
class StaticProperty:
    ir_arg_index: int
    formal_name: str
    property: str
    property_path: tuple[int, ...]
    value: ValueFlow


@dataclass(frozen=True)
class NodeFields:
    launch: int
    kernel_symbol: str
    parameter_sizes: tuple[int, ...]
    pointers: tuple[PointerField, ...]
    integers: tuple[IntegerField, ...]
    fixed: tuple[FixedField, ...]
    padding: tuple[Padding, ...]
    undefined: tuple[UndefinedField, ...] = ()
    constants: tuple[ConstantField, ...] = ()

    @property
    def pointer_descriptors(self):
        return tuple((item.parameter, item.byte_offset) for item in self.pointers)

    @property
    def scalar_descriptors(self):
        return tuple(
            (item.parameter, item.byte_offset, item.dtype) for item in self.integers
        )
