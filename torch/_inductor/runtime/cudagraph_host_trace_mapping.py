"""Translate live host-trace storage symbols to shared replay address roots."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import sympy
from sympy.functions.elementary.piecewise import ExprCondPair

import torch
from torch.fx.experimental.sym_node import SymNode

from .cudagraph_arg_mapping import BufferSource, InputSource, IntExpr, PointerSource
from .cudagraph_launch_association import UnsupportedCapture


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.cuda._host_trace import _Root, Tape


@dataclass(frozen=True)
class HostTraceInputMetadata:
    index: int
    property: str
    dimension: int | None = None


class HostTraceSymbolMapping:
    r"""Map recorded tensor metadata and storage symbols to replay sources.

    Input storage bases are rebased onto runtime data pointers, retaining the
    recorded storage offsets and allocation alignment facts.
    """

    def __init__(
        self, tape: Tape, *, input_indices: dict[int, int] | None = None
    ) -> None:
        self.tape = tape
        self.metadata_symbols: dict[sympy.Symbol, HostTraceInputMetadata] = {}
        self.opaque_symbols: dict[sympy.Symbol, dict] = {}
        self.host_address_symbols: dict[sympy.Symbol, int] = {}
        self.host_original_symbols: dict[sympy.Symbol, int] = {}
        self.address_symbols: dict[sympy.Symbol, InputSource | BufferSource] = {}
        self.root_alignments: dict[InputSource | BufferSource, int] = {}
        self.substitutions: dict[sympy.Symbol, sympy.Expr] = {}
        self._roots: dict[int, tuple[_Root, InputSource | BufferSource]] = {}
        if input_indices is None:
            input_indices = {record.position: record.position for record in tape.inputs}
        if (
            set(input_indices) != {record.position for record in tape.inputs}
            or any(
                type(index) is not int or index < 0 for index in input_indices.values()
            )
            or len(set(input_indices.values())) != len(input_indices)
        ):
            raise UnsupportedCapture(
                "Host-trace inputs require distinct exact runtime argument indices"
            )
        self.input_indices = dict(input_indices)
        positions = set()
        for record in tape.inputs:
            if record.position in positions or not 0 <= record.position < tape.nargs:
                raise UnsupportedCapture(
                    "Host-trace input root has an ambiguous argument position"
                )
            positions.add(record.position)
            if record.root.itemsize != record.dtype.itemsize:
                raise UnsupportedCapture(
                    "Host-trace input root lost its input element width"
                )
            if len(record.sizes) != len(record.strides):
                raise UnsupportedCapture("Host-trace input sizes and strides disagree")
            index = self.input_indices[record.position]
            for property, values in (
                ("size", record.sizes),
                ("stride", record.strides),
            ):
                for dimension, value in enumerate(values):
                    self._metadata(
                        value, HostTraceInputMetadata(index, property, dimension)
                    )
            offset = self._metadata(
                record.offset, HostTraceInputMetadata(index, "storage_offset")
            )
            source = InputSource(index)
            pointer = self._root(record.root, source, 1)
            base = self._original(record.root.sym)
            self._substitute(base, pointer - offset * record.dtype.itemsize)

        names = set()
        for record in tape.allocs:
            if (
                not isinstance(record.name, str)
                or not record.name
                or record.name in names
            ):
                raise UnsupportedCapture(
                    "Host-trace allocation root has an ambiguous buffer name"
                )
            names.add(record.name)
            if record.root.itemsize != record.dtype.itemsize:
                raise UnsupportedCapture(
                    "Host-trace allocation root lost its element width"
                )
            quotient = self._original(record.q)
            base = self._original(record.root.sym)
            if not isinstance(quotient, sympy.Symbol) or base != 256 * quotient:
                raise UnsupportedCapture(
                    "Host-trace allocation lost its allocator alignment fact"
                )
            pointer = self._root(record.root, BufferSource(record.name), 256)
            # The recorder and replay allocator both guarantee this exact quotient.
            self._substitute(quotient, pointer / 256)

        for index, record in enumerate(getattr(tape, "host_buffers", ())):
            original = self._original(record["root"])
            pointer = sympy.Dummy(f"host_table_{index}", integer=True)
            self._substitute(original, pointer)
            self.host_original_symbols[original] = index
            self.host_address_symbols[pointer] = index

        previous = -1
        for record in tape.opaque:
            symbol = self._original(record["sym"])
            if (
                not isinstance(symbol, sympy.Symbol)
                or symbol.is_integer is not True
                or symbol in self.metadata_symbols
                or symbol in self.substitutions
                or symbol in self.opaque_symbols
                or record["kind"] not in ("guard", "rebind")
                or type(record["seq"]) is not int
                or record["seq"] <= previous
                or type(record.get("impl")) is not int
                or record["impl"] <= 0
                or not callable(record["call"])
                or type(record["expected"]) is not int
                or not -(1 << 63) <= record["expected"] < 1 << 63
            ):
                raise UnsupportedCapture(
                    "Host-trace opaque call lost its ordered native contract"
                )
            for argument in record["args"]:
                value = self.translate(argument)
                if value.is_integer is not True:
                    raise UnsupportedCapture(
                        "Host-trace opaque arguments require integer values"
                    )
            self.opaque_symbols[symbol] = record
            previous = record["seq"]

    def _expression(self, value: object, *, original: bool = False) -> sympy.Basic:
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value.node
        if isinstance(value, SymNode):
            if value.shape_env is not self.tape.shape_env:
                raise UnsupportedCapture("Host-trace value belongs to another ShapeEnv")
            value = value._expr if original else value.expr
        if isinstance(value, sympy.Basic):
            return value
        if type(value) in (int, float, bool):
            return sympy.sympify(value)
        raise UnsupportedCapture("Host-trace value has no live symbolic expression")

    def _original(self, value: object) -> sympy.Basic:
        return self._expression(value, original=True)

    def _metadata(self, value: object, source: HostTraceInputMetadata) -> sympy.Expr:
        symbol = self._original(value)
        if not isinstance(symbol, sympy.Symbol) or symbol.is_integer is not True:
            raise UnsupportedCapture(
                "Host-trace input metadata lost its original integer symbol"
            )
        if symbol in self.metadata_symbols or symbol in self.substitutions:
            raise UnsupportedCapture(
                "Host-trace input metadata has an ambiguous symbolic source"
            )
        self.metadata_symbols[symbol] = source
        return symbol

    def _substitute(self, symbol: sympy.Basic, value: sympy.Expr) -> None:
        if not isinstance(symbol, sympy.Symbol) or symbol.is_integer is not True:
            raise UnsupportedCapture(
                "Host-trace storage root lost its original integer symbol"
            )
        if symbol in self.substitutions or symbol in self.metadata_symbols:
            raise UnsupportedCapture(
                "Host-trace storage root has an ambiguous symbolic source"
            )
        self.substitutions[symbol] = value

    def _root(
        self, root: _Root, source: InputSource | BufferSource, alignment: int
    ) -> sympy.Symbol:
        if id(root) in self._roots or source in self.root_alignments:
            raise UnsupportedCapture("Host-trace storage root has ambiguous ownership")
        pointer = sympy.Dummy(f"host_trace_{root.name}", integer=True, nonnegative=True)
        self._roots[id(root)] = root, source
        self.address_symbols[pointer] = source
        self.root_alignments[source] = alignment
        return pointer

    def root_source(self, root: _Root) -> InputSource | BufferSource:
        entry = self._roots.get(id(root))
        if entry is None or entry[0] is not root:
            raise UnsupportedCapture("Host-trace view has no recorded storage root")
        return entry[1]

    def translate(
        self, value: object, *, preserve_operations: bool = False
    ) -> sympy.Basic:
        expression = self._expression(value)
        if not expression.free_symbols.issubset(
            self.metadata_symbols.keys()
            | self.substitutions.keys()
            | self.opaque_symbols.keys()
        ):
            raise UnsupportedCapture(
                "Host-trace expression has an unbound symbolic source"
            )
        if not preserve_operations:
            return expression.xreplace(self.substitutions)

        @cache
        def replace(node: sympy.Basic) -> sympy.Basic:
            if node in self.substitutions:
                return self.substitutions[node]
            arguments = tuple(replace(argument) for argument in node.args)
            if arguments == node.args:
                return node
            if isinstance(node, (sympy.And, sympy.Or)):
                return node.func._from_args(arguments)
            if isinstance(node, ExprCondPair):
                return ExprCondPair(*arguments)
            try:
                return node.func(*arguments, evaluate=False)
            except (TypeError, ValueError) as error:
                raise UnsupportedCapture(
                    f"Host guard cannot preserve {type(node).__name__} during translation"
                ) from error

        return replace(expression)

    def pointer(
        self, value: object, lower_integer: Callable[[sympy.Expr], int | IntExpr]
    ) -> PointerSource:
        expression = self.translate(value)
        if not isinstance(expression, sympy.Expr) or expression.is_integer is not True:
            raise UnsupportedCapture(
                "Host-trace pointer requires an integer address expression"
            )
        addresses = expression.free_symbols.intersection(self.address_symbols)
        if len(addresses) != 1:
            raise UnsupportedCapture(
                "Host-trace pointer requires exactly one recorded storage root"
            )
        address = next(iter(addresses))
        if expression.coeff(address) != 1:
            raise UnsupportedCapture(
                "Host-trace pointer requires a unit storage-root coefficient"
            )
        displacement = expression - address
        if displacement.free_symbols.intersection(self.address_symbols):
            raise UnsupportedCapture(
                "Host-trace pointer displacement depends on a storage address"
            )
        offset = lower_integer(displacement)
        if type(offset) is int:
            offset = IntExpr("constant", offset)
        if type(offset) is not IntExpr:
            raise UnsupportedCapture(
                "Host-trace pointer displacement has no runtime integer source"
            )
        return PointerSource(self.address_symbols[address], offset)

    def view_pointer(
        self,
        root: _Root,
        offset: object,
        dtype: torch.dtype,
        lower_integer: Callable[[sympy.Expr], int | IntExpr],
    ) -> PointerSource:
        self.root_source(root)
        value = self._original(root.sym) + self._expression(offset) * dtype.itemsize
        return self.pointer(value, lower_integer)
