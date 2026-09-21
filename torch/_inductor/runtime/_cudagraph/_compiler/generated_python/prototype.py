"""Symbolic tensor and layout helpers for host lowering."""

from dataclasses import dataclass, field, replace

import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import Allocate, InputAssertion, Invoke, Normalize, Reinterpret
from torch._inductor.runtime._cudagraph._compiler.selected_kernel import SelectedKernel
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy,
    BufferSource,
    CallArgument,
    ExpressionSource,
    InputSource,
    IntegerSource,
    IntExpr,
    KernelCallRecord,
    pointwise_product,
)


class TraceDeclined(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class _Integer:
    expression: IntExpr

    def __mul__(self, other):
        other = other.expression if type(other) is _Integer else other
        expression = pointwise_product((self.expression, other))
        if type(expression) is not IntExpr:
            raise TraceDeclined("Unsupported symbolic product")
        return _Integer(expression)

    __rmul__ = __mul__

    def __bool__(self):
        raise TraceDeclined("Symbolic host dispatch needs an explicit local guard")

    def __index__(self):
        raise TraceDeclined("A boxed symbolic integer cannot be specialized to an example")

    def __eq__(self, other):
        raise TraceDeclined("Symbolic host dispatch needs an explicit local guard")

    __ne__ = __eq__
    __lt__ = __eq__
    __le__ = __eq__
    __gt__ = __eq__
    __ge__ = __eq__


@dataclass(eq=False)
class _Tensor:
    source: InputSource | BufferSource
    dtype: object = None
    size: tuple = ()
    stride: tuple = ()
    valid: bool = True
    used: bool = False

    def __bool__(self):
        raise TraceDeclined("Tensor-dependent host dispatch is outside this prototype")


def _dimensions(values):
    if type(values) is not tuple:
        raise TraceDeclined("Expected a literal tuple of symbolic dimensions")
    result = tuple(value.expression if type(value) is _Integer else value for value in values)
    if any(type(value) not in (int, IntExpr) for value in result):
        raise TraceDeclined("Unsupported dimension value")
    return result


@dataclass
class _Trace:
    device_index: int
    kernels: dict[str, SelectedKernel]
    events: list = field(default_factory=list)
    allocations: list[_Tensor] = field(default_factory=list)
    calls: list[KernelCallRecord] = field(default_factory=list)
    selected: list[SelectedKernel] = field(default_factory=list)
    copies: list[AlignmentCopy] = field(default_factory=list)
    assertions: list = field(default_factory=list)
    in_device: bool = False
    compiler_sources: dict[BufferSource, BufferSource] = field(default_factory=dict)

    def check_device(self, device):
        if type(device) is not int or device != self.device_index:
            raise TraceDeclined("Device dispatch changed the supplied compiler contract")

    def raw_stream(self, device):
        self.check_device(device)
        if not self.in_device:
            raise TraceDeclined("Stream lookup outside the recorded device scope")
        return self

    def allocate(self, size, stride, dtype):
        if not self.in_device or type(dtype) is not torch.dtype:
            raise TraceDeclined("Expected an allocation on the bound CUDA device")
        token = _Tensor(BufferSource(f"trace_{len(self.allocations)}"), dtype,
                        _dimensions(size), _dimensions(stride))
        self.allocations.append(token)
        self.events.append(Allocate(token.source.name, token.dtype, token.size, token.stride))
        return token

    def reinterpret(self, token, size, stride, offset):
        size, stride = _dimensions(size), _dimensions(stride)
        if (type(token) is not _Tensor or type(token.source) is not BufferSource
                or not token.valid or token.used or self.calls or type(offset) is not int or offset != 0
                or not self.events or type(self.events[-1]) is not Allocate
                or self.events[-1].value_id != token.source.name
                or size != token.size or len(stride) != len(token.stride)
                or any(before != after and dimension != 1
                       for dimension, before, after in zip(size, token.stride, stride))
                or stride != ((1,) if len(size) == 1 else (size[1], 1))):
            raise TraceDeclined("Only an adjacent fresh singleton-stride reinterpret is supported")
        token.valid = False
        result = _Tensor(BufferSource(f"view_{len(self.events)}"), token.dtype, size, stride)
        self.allocations[self.allocations.index(token)] = result
        self.events.append(Reinterpret(token.source.name, result.source.name, size, stride, offset))
        return result

    def check_layout(self, token, size, stride, label=None):
        if type(token) is not _Tensor or type(token.source) is not InputSource:
            raise TraceDeclined("Only compiler input assertions are supported")
        self.assertions.append(InputAssertion(token.source, _dimensions(size), _dimensions(stride), label))
        return True

    def normalize(self, token):
        if type(token) is not _Tensor or type(token.source) is not InputSource or token.used:
            raise TraceDeclined("Alignment normalization must precede the input's first use")
        self.copies.append(AlignmentCopy(token.source.index, len(self.calls)))
        self.events.append(Normalize(token.source.index, len(self.calls)))
        return token

    def launch(self, name, args, stream, kwargs):
        if not self.in_device or stream is not self or kwargs:
            raise TraceDeclined("Unexpected launch stream or keyword arguments")
        selected = self.kernels[name]
        try:
            selected.check()
        except ValueError as error:
            raise TraceDeclined(str(error)) from error
        original = selected.compiler_call
        if selected.dispatch is not None:
            if (original is None or original.source_record is None
                    or len(args) != len(original.source_record.sources)):
                raise TraceDeclined("Multi-kernel invocation lost its compiler union operands")
            args = tuple(args[index] for index in selected.dispatch.argument_indices)
        grid_type = "Grid1D" if original is None else original.record.grid_type
        launcher_grid, constexprs = None, ()
        if original is not None and original.record.grid_type == "FixedGrid":
            if len(args) < 3:
                raise TraceDeclined("User invocation lost its three grid arguments")
            args, suffix = args[:-3], args[-3:]
            grid_type, launcher_grid = "FixedGrid", _dimensions(suffix)
            constexprs = original.record.constexprs
        rows = sorted((row for row in selected.arguments if row.call_arg_index is not None),
                      key=lambda row: row.call_arg_index)
        if [row.call_arg_index for row in rows] != list(range(len(args))):
            raise TraceDeclined("Actual call differs from selected positional arguments")
        arguments = []
        writes = dict(selected.pointer_writes)
        for row, value in zip(rows, args):
            if type(value) is _Tensor:
                if not value.valid or type(writes.get(row.formal)) is not bool:
                    raise TraceDeclined("Missing pointer effect fact or a retired alias was used")
                if type(value.source) is InputSource and writes[row.formal] is not False:
                    raise TraceDeclined("The selected kernel writes a boxed input")
                source = value.source
                value.used = True
            elif type(value) is _Integer:
                source = ExpressionSource(value.expression)
            elif type(value) is int:
                source = IntegerSource(value)
            else:
                raise TraceDeclined("Unsupported actual launch argument")
            arguments.append(CallArgument(row.formal, row.source_arg_index, row.call_arg_index,
                                          row.triton_type, source))
        call = KernelCallRecord(len(self.calls), name, tuple(row.formal for row in selected.arguments),
                                tuple(arguments), grid_type, launcher_grid, constexprs,
                                original is not None and original.record.generated_template)
        try:
            grid = selected.bind_call(call, self.compiler_sources)
        except ValueError as error:
            raise TraceDeclined(str(error)) from error
        if call.generated_template:
            call = replace(call, launcher_grid=original.record.launcher_grid)
        self.calls.append(call)
        self.selected.append(selected)
        self.events.append(Invoke(call, grid))
