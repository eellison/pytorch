"""Import local host-tape sources using the canonical physical ABI lowering."""

from dataclasses import replace
from functools import cache
from typing import cast

import sympy
from sympy.functions.elementary.piecewise import ExprCondPair

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    ExpressionSource,
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture


def expression(value):
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return value.node._expr
    return sympy.sympify(value)


def recorded_guard_pins(tape):
    tape_guards = getattr(tape.shape_env, "tape_guards", None)
    if tape_guards is None:
        return {}
    guards, pins, *_ = tape_guards()
    return pins if guards == tape.guards else {}


def replace_operations(value, substitutions):
    @cache
    def visit(node):
        if node in substitutions:
            return substitutions[node]
        args = tuple(visit(arg) for arg in node.args)
        if args == node.args:
            return node
        if isinstance(node, (sympy.And, sympy.Or)):
            return node.func._from_args(args)
        if isinstance(node, ExprCondPair):
            return ExprCondPair(*args)
        try:
            return node.func(*args, evaluate=False)
        except (TypeError, ValueError) as error:
            raise UnsupportedCapture(
                f"Cannot preserve imported {type(node).__name__}"
            ) from error

    return visit(value)


class TapeSources:
    r"""Rebase a local CUDA tape's symbols and storage roots into the outer trace.

    Input metadata, allocation addresses and computed integers retain their
    recorded correspondence while views compose with the outer storage roots.
    """

    def __init__(
        self,
        tape,
        arguments,
        roots,
        addresses,
        lower_integer,
        *,
        mapping=None,
        computed=(),
    ):
        if getattr(tape, "constants", ()):
            raise UnsupportedCapture("CUDA host import only accepts tensor formals")
        if set(arguments) != set(range(tape.nargs)):
            raise UnsupportedCapture(
                "Imported arguments differ from the tape's formal positions"
            )
        if any(type(record) is not dict for record in getattr(tape, "opaque", ())):
            raise UnsupportedCapture("Imported opaque calls require canonical records")
        if any(type(slot) is not dict for slot in getattr(tape, "rng_slots", ())):
            raise UnsupportedCapture("Imported RNG slots require canonical records")
        if getattr(tape, "rng_increment", None) is not None and not getattr(
            tape, "rng_slots", ()
        ):
            raise UnsupportedCapture("CUDA host import requires recorded RNG slots")
        self.tape = tape
        self.mapping = (
            mapping
            if mapping is not None
            else HostTraceSymbolMapping(
                tape,
                input_indices={
                    rec.position: index for index, rec in enumerate(tape.inputs)
                },
            )
        )
        if self.mapping.tape is not tape or self.mapping.input_indices != {
            rec.position: index for index, rec in enumerate(tape.inputs)
        }:
            raise UnsupportedCapture(
                "Imported mapping differs from the local tape's compact input order"
            )
        self.roots = roots
        self.addresses = addresses
        self.lower_integer = lower_integer
        self.arguments = tuple(arguments[rec.position] for rec in tape.inputs)
        self.substitutions = {}
        self.pointers = {}
        self._integers = {}
        self.allocations = {}
        for index, (record, tensor) in enumerate(
            zip(tape.inputs, self.arguments, strict=True)
        ):
            resolution = roots(tensor)
            if (
                record.dtype != tensor.dtype
                or len(record.sizes) != tensor.dim()
                or tensor.fake_mode is not roots.mode
            ):
                raise UnsupportedCapture(
                    "Local formal differs from its outer tensor contract"
                )
            self.pointers[InputSource(index)] = PointerSource(
                resolution.root, self._lower(resolution.byte_offset)
            )
        for symbol, metadata in self.mapping.metadata_symbols.items():
            tensor = self.arguments[metadata.index]
            value = (
                tensor.storage_offset()
                if metadata.property == "storage_offset"
                else getattr(tensor, metadata.property)(metadata.dimension)
            )
            self.substitutions[symbol] = expression(value)
        for symbol, source in self.mapping.address_symbols.items():
            if type(source) is InputSource:
                resolution = roots(self.arguments[source.index])
                self.substitutions[symbol] = expression(
                    addresses(resolution)
                ) + expression(resolution.byte_offset)
        for record, value in computed:
            self.bind_computed(record, value)

    def bind_computed(self, record, value):
        symbol = self.mapping._original(record["sym"])
        if (
            self.mapping.opaque_symbols.get(symbol) is not record
            or symbol in self.substitutions
        ):
            raise UnsupportedCapture("Computed integer lost its unique local callback")
        if (
            type(value) is not torch.SymInt
            or value.node.shape_env is not self.roots.mode.shape_env
        ):
            raise UnsupportedCapture(
                "Computed integer belongs to another tracing environment"
            )
        self.substitutions[symbol] = expression(value)

    def _lower(self, value):
        value = self.lower_integer(value)
        return IntExpr("constant", value) if type(value) is int else value

    def bind_allocation(self, record, tensor):
        if not any(record is rec for rec in self.tape.allocs):
            raise UnsupportedCapture("Allocation does not belong to this local tape")
        source = BufferSource(record.name)
        if source in self.pointers:
            raise UnsupportedCapture("Local allocation was already imported")
        resolution = self.roots(tensor)
        if type(resolution.root) is not BufferSource or resolution.byte_offset != 0:
            raise UnsupportedCapture(
                "Local allocation requires a fresh outer allocation root"
            )
        if resolution.alignment != 256 or tensor.dtype != record.dtype:
            raise UnsupportedCapture(
                "Imported allocation lost its dtype or allocator contract"
            )
        for local, outer in zip(
            (*record.sizes, *record.strides),
            (*tensor.size(), *tensor.stride()),
            strict=True,
        ):
            if sympy.expand(self.translate(local) - expression(outer)) != 0:
                raise UnsupportedCapture(
                    "Imported allocation metadata differs symbolically"
                )
        self.pointers[source] = PointerSource(resolution.root, IntExpr("constant", 0))
        self.allocations[source] = tensor
        symbol = next(
            key
            for key, value in self.mapping.address_symbols.items()
            if value == source
        )
        self.substitutions[symbol] = expression(self.addresses(resolution))

    def translate(self, value):
        local = self.mapping.translate(value, preserve_operations=True)
        if not local.free_symbols.issubset(self.substitutions):
            raise UnsupportedCapture("Imported expression has an unbound local source")
        return replace_operations(local, self.substitutions)

    def integer(self, value):
        if type(value) is int:
            return value
        if type(value) is not IntExpr:
            raise UnsupportedCapture(
                "Imported numeric record is not the shared IntExpr"
            )
        old = self._integers.get(id(value))
        if old is not None:
            if old[0] is not value:
                raise UnsupportedCapture("Numeric source identity was reused")
            return old[1]
        if value.op in ("size", "stride"):
            (dimension,) = value.args
            if dimension.op != "constant":
                raise UnsupportedCapture(
                    "Metadata dimension must be the recorded constant"
                )
            tensor = self.arguments[cast(int, value.value)]
            result = self._lower(getattr(tensor, value.op)(dimension.value))
        elif value.op == "storage_offset":
            result = self._lower(
                self.arguments[cast(int, value.value)].storage_offset()
            )
        elif value.op in ("boxed", "input"):
            raise UnsupportedCapture("Local tape has no open scalar argument binding")
        else:
            result = IntExpr(
                value.op, value.value, tuple(self.integer(arg) for arg in value.args)
            )
        self._integers[id(value)] = value, result
        return result

    def source(self, value):
        if type(value) in (InputSource, BufferSource):
            return self.pointers[value]
        if type(value) is PointerSource:
            base = self.pointers[value.root]
            return PointerSource(
                base.root,
                IntExpr(
                    "add", args=(base.byte_offset, self.integer(value.byte_offset))
                ),
            )
        if type(value) is ExpressionSource:
            return ExpressionSource(self.integer(value.expression))
        raise UnsupportedCapture(
            f"Imported field has no source rule: {type(value).__name__}"
        )

    def table(self, value):
        return replace(
            value,
            elements=tuple(
                (
                    offset,
                    width,
                    self.source(source)
                    if type(source) is PointerSource
                    else self.integer(source),
                )
                for offset, width, source in value.elements
            ),
        )

    def call(self, value):
        if value.tensor_maps:
            raise UnsupportedCapture("CUDA host import has no tensor-map event splice")
        return replace(
            value,
            fields=tuple(
                replace(field, source=self.source(field.source))
                for field in value.fields
            ),
            storage_sources=tuple(
                self.source(source) for source in value.storage_sources
            ),
            grid=tuple(self.integer(item) for item in value.grid),
            block=None
            if value.block is None
            else tuple(self.integer(item) for item in value.block),
            shared=None if value.shared is None else self.integer(value.shared),
        )

    def rng_call(self, call, slots, prefix):
        from torch._inductor.runtime.cudagraph_boxed_replay import _PhysicalField

        fields, constants, guards = list(call.fields), list(call.constants), []
        for slot in slots:
            value = sympy.Add(prefix, self.translate(slot.prefix), evaluate=False)
            guards.extend(
                (sympy.Ge(value, 0), sympy.Lt(value, 2 ** min(63, 8 * slot.width)))
            )
            source = self._lower(value)
            if slot.width == 4:
                source = IntExpr(
                    "select",
                    args=(
                        IntExpr("ge", args=(source, IntExpr("constant", 2**31))),
                        IntExpr("add", args=(source, IntExpr("constant", -(2**32)))),
                        source,
                    ),
                )
            elif slot.width != 8:
                raise UnsupportedCapture("RNG slot has no supported scalar width")
            fields = [
                field
                for field in fields
                if not (
                    field.parameter == slot.parameter
                    and field.byte_offset == slot.byte_offset
                )
            ]
            fields.append(
                _PhysicalField(
                    slot.parameter,
                    slot.byte_offset,
                    "i32" if slot.width == 4 else "i64",
                    ExpressionSource(source),
                )
            )
            remaining = []
            for parameter, offset, data in constants:
                start, end = (
                    max(offset, slot.byte_offset),
                    min(offset + len(data), slot.byte_offset + slot.width),
                )
                if parameter != slot.parameter or start >= end:
                    remaining.append((parameter, offset, data))
                    continue
                if start > offset:
                    remaining.append((parameter, offset, data[: start - offset]))
                if end < offset + len(data):
                    remaining.append((parameter, end, data[end - offset :]))
            constants = remaining
        return replace(call, fields=tuple(fields), constants=tuple(constants)), tuple(
            guards
        )

    def guards(self, extra=()):
        guards = []
        for symbol, bounds in self.tape.shape_env.var_to_range.items():
            for bound, relation in ((bounds.lower, sympy.Ge), (bounds.upper, sympy.Le)):
                if isinstance(bound, sympy.Integer):
                    guards.append(
                        self.translate(relation(symbol, bound, evaluate=False))
                    )
        guards.extend(self.translate(value) for value in self.tape.guards)
        guards.extend(
            self.translate(sympy.Eq(key, value, evaluate=False))
            for key, value in self.tape.shape_env.replacements.items()
        )
        # Canonical lower_tape's extra guards already use its translated root symbols.
        for value in extra:
            if not value.free_symbols.issubset(self.substitutions):
                raise UnsupportedCapture(
                    "Imported extra guard has an unbound local source"
                )
            guards.append(replace_operations(value, self.substitutions))
        return tuple(guards)
