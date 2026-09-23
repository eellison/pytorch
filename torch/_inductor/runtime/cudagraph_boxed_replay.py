import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils._ordered_set import OrderedSet

from .cudagraph_arg_mapping import (
    BorrowedInputOutput,
    BufferSource,
    CallArgument,
    ExpressionSource,
    InputSource,
    IntegerInput,
    IntegerOutput,
    IntegerSource,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    storage_roots,
    TensorViewOutput,
)
from .cudagraph_launch_association import (
    associate_kernel_launches,
    RecordedGraphNode,
    RecordedKernelLaunch,
    UnsupportedCapture,
)
from .static_triton_launcher import StaticallyLaunchedCudaKernel


class _NumericInputs(Protocol):
    input_names: tuple[str, ...]
    integer_inputs: tuple[IntegerInput, ...]


class _NumericProgram:
    def __init__(self, records: _NumericInputs, example_inputs: Any) -> None:
        if type(example_inputs) not in (list, tuple) or len(example_inputs) != len(
            records.input_names
        ):
            raise UnsupportedCapture(
                "Dynamic replay requires the actual boxed example inputs"
            )
        self.integer_indices = tuple(row.boxed_index for row in records.integer_inputs)
        self.tensor_inputs = {
            index: value
            for index, value in enumerate(example_inputs)
            if index not in self.integer_indices
        }
        self.inputs: dict[int, int] = {}
        for index in self.integer_indices:
            value = example_inputs[index]
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise UnsupportedCapture(
                    "Dynamic replay requires an exact signed int64 input"
                )
            self.inputs[index] = value
        self.instructions: list[tuple] = []
        self.values: list[int] = []
        self.instruction_indices: dict[tuple, int] = {}
        self.expressions: dict[int, tuple[IntExpr, int]] = {}

    def prepared_value(self, expression: IntExpr) -> int:
        try:
            return self.values[self.expressions[id(expression)][1]]
        except KeyError:
            raise UnsupportedCapture(
                "Numeric expression was not registered before capture"
            ) from None

    def add(self, expression: IntExpr) -> int:
        pending = [(expression, False)]
        while pending:
            node, ready = pending.pop()
            if type(node) is not IntExpr or type(node.args) is not tuple:
                raise UnsupportedCapture("Unsupported compiler integer expression")
            if id(node) in self.expressions:
                continue
            if ready:
                self._add_node(node)
                continue
            pending.append((node, True))
            if node.op not in ("size", "stride"):
                pending.extend((arg, False) for arg in reversed(node.args))
        return self.expressions[id(expression)][1]

    def _add_node(self, expression: IntExpr) -> int:
        if type(expression) is not IntExpr or type(expression.args) is not tuple:
            raise UnsupportedCapture("Unsupported compiler integer expression")
        previous = self.expressions.get(id(expression))
        if previous is not None:
            return previous[1]
        instruction: tuple
        if (
            expression.op == "constant"
            and type(expression.value) is int
            and not expression.args
        ):
            value = expression.value
            instruction = ("constant", value)
        elif (
            expression.op == "fconst"
            and type(expression.value) is int
            and not expression.args
        ):
            value = expression.value
            instruction = ("fconst", value)
        elif expression.op in ("call", "pcall"):
            if (
                type(expression.value) is not tuple
                or len(expression.value) != 2
                or type(expression.value[0]) is not int
                or not 0 < expression.value[0] < 1 << 64
                or not callable(expression.value[1])
            ):
                raise UnsupportedCapture(
                    "Native calls require an address and callable owner"
                )
            address, owner = expression.value
            arguments = tuple(self.add(arg) for arg in expression.args)
            value = owner([self.values[index] for index in arguments])
            if type(value) is not int:
                raise UnsupportedCapture(
                    "Native call preparation must return an exact integer"
                )
            instruction = (expression.op, address, owner, *arguments)
        elif (
            expression.op
            in (
                "ffromint",
                "fneg",
                "fsqrt",
                "fround32",
                "ftobits32",
                "fadd",
                "fsub",
                "fmul",
                "fdiv",
                "fpow",
            )
            and expression.value is None
        ):
            from .cudagraph_compiled_evaluation import _evaluate_float

            arity = (
                1
                if expression.op
                in ("ffromint", "fneg", "fsqrt", "fround32", "ftobits32")
                else 2
            )
            if len(expression.args) != arity:
                raise UnsupportedCapture(
                    "Float expression has the wrong number of operands"
                )
            arguments = tuple(self.add(arg) for arg in expression.args)
            value = _evaluate_float(
                expression.op, *(self.values[index] for index in arguments)
            )
            instruction = (expression.op, *arguments)
        elif (
            expression.op == "boxed"
            and type(expression.value) is int
            and not expression.args
        ):
            if expression.value not in self.inputs:
                raise UnsupportedCapture(
                    "Integer expression has no compiler boxed origin"
                )
            value = self.inputs[expression.value]
            instruction = ("boxed", expression.value)
        elif (
            expression.op == "storage_offset"
            and type(expression.value) is int
            and not expression.args
        ):
            tensor = self.tensor_inputs.get(expression.value)
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                raise UnsupportedCapture("Storage offset has no original Tensor input")
            value = tensor.storage_offset()
            instruction = ("storage_offset", expression.value)
        elif (
            expression.op == "pointer"
            and type(expression.value) is int
            and not expression.args
        ):
            # a Tensor input's data pointer as an int64 value: the argument of a host
            # lookup keyed by an address (symmetric memory's handle from its buffer)
            tensor = self.tensor_inputs.get(expression.value)
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                raise UnsupportedCapture("Data pointer has no original Tensor input")
            value = tensor.data_ptr()
            instruction = ("pointer", expression.value)
        elif (
            expression.op in ("size", "stride")
            and type(expression.value) is int
            and len(expression.args) == 1
        ):
            tensor = self.tensor_inputs.get(expression.value)
            (dimension,) = expression.args
            if (
                type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or type(dimension) is not IntExpr
                or dimension.op != "constant"
                or type(dimension.value) is not int
                or dimension.args
                or not 0 <= dimension.value < tensor.dim()
            ):
                raise UnsupportedCapture(
                    "Tensor metadata load has no original input dimension"
                )
            value = (
                tensor.size(dimension.value)
                if expression.op == "size"
                else tensor.stride(dimension.value)
            )
            instruction = (expression.op, expression.value, dimension.value)
        elif (
            expression.op == "multiply"
            and expression.value is None
            and len(expression.args) == 2
        ):
            left, right = (self.add(arg) for arg in expression.args)
            value = self.values[left] * self.values[right]
            instruction = ("multiply", left, right)
        elif (
            expression.op == "add"
            and expression.value is None
            and len(expression.args) == 2
        ):
            left, right = (self.add(arg) for arg in expression.args)
            value = self.values[left] + self.values[right]
            instruction = ("add", left, right)
        elif (
            expression.op in ("ceildiv", "floordiv")
            and expression.value is None
            and len(expression.args) == 2
        ):
            left, right = (self.add(arg) for arg in expression.args)
            numerator, divisor = self.values[left], self.values[right]
            if numerator < 0 or divisor <= 0:
                raise UnsupportedCapture("Unsupported integer division domain")
            value = numerator // divisor
            if expression.op == "ceildiv":
                value += numerator % divisor != 0
            instruction = (expression.op, left, right)
        elif (
            expression.op in ("eq", "ne", "lt", "le", "gt", "ge", "and")
            and expression.value is None
            and len(expression.args) == 2
        ):
            left, right = (self.add(arg) for arg in expression.args)
            a, b = self.values[left], self.values[right]
            if expression.op == "and":
                if a not in (0, 1) or b not in (0, 1):
                    raise UnsupportedCapture(
                        "Boolean conjunction requires zero or one operands"
                    )
                value = a & b
            else:
                value = int(
                    {
                        "eq": a == b,
                        "ne": a != b,
                        "lt": a < b,
                        "le": a <= b,
                        "gt": a > b,
                        "ge": a >= b,
                    }[expression.op]
                )
            instruction = (expression.op, left, right)
        elif (
            expression.op in ("max", "min")
            and expression.value is None
            and len(expression.args) >= 2
        ):
            operands = tuple(self.add(arg) for arg in expression.args)
            reducer = max if expression.op == "max" else min
            value = reducer(self.values[operand] for operand in operands)
            instruction = (expression.op, *operands)
        elif (
            expression.op == "select"
            and expression.value is None
            and len(expression.args) == 3
        ):
            condition, when_true, when_false = (
                self.add(arg) for arg in expression.args
            )
            if self.values[condition] not in (0, 1):
                raise UnsupportedCapture(
                    "Integer selection requires a zero or one condition"
                )
            value = self.values[when_true if self.values[condition] else when_false]
            instruction = ("select", condition, when_true, when_false)
        else:
            raise UnsupportedCapture("Unresolved compiler integer expression")
        if not -(2**63) <= value < 2**63:
            raise UnsupportedCapture("Compiler integer expression exceeds int64")
        previous = self.instruction_indices.get(instruction)
        if previous is not None:
            index = previous
        else:
            index = len(self.instructions)
            self.instruction_indices[instruction] = index
            self.instructions.append(instruction)
            self.values.append(value)
        self.expressions[id(expression)] = expression, index
        return index


@dataclass(frozen=True)
class _BoundCall:
    arguments: tuple[CallArgument, ...]
    module: StaticallyLaunchedCudaKernel
    grid: tuple[IntExpr, IntExpr, IntExpr] | None
    scratch: tuple[PointerSource | ParameterSource, ...] = ()

    @property
    def launch_arguments(self):
        if not self.scratch:
            return self.arguments
        slots = tuple(
            size
            for present, size in (
                (self.module.has_global_scratch, self.module.global_scratch_size),
                (self.module.has_profile_scratch, self.module.profile_scratch_size),
            )
            if present
        )
        if len(self.scratch) != len(slots):
            raise UnsupportedCapture(
                "Scratch sources must cover every selected launcher slot"
            )
        for size, source in zip(slots, self.scratch):
            if size:
                if (
                    type(source) is not PointerSource
                    or type(source.root) is not BufferSource
                    or source.byte_offset != IntExpr("constant", 0)
                ):
                    raise UnsupportedCapture(
                        "Nonempty launcher scratch requires its own traced allocation"
                    )
            elif source != ParameterSource("constant", 64, 0):
                raise UnsupportedCapture(
                    "Empty launcher scratch must retain its null pointer"
                )
        return self.arguments + tuple(
            CallArgument(
                f"__launcher_scratch_{index}",
                -1,
                len(self.arguments) + index,
                "*i8",
                source,
            )
            for index, source in enumerate(self.scratch)
        )


class _KernelModule(ABC):
    @property
    @abstractmethod
    def function(self) -> int:
        pass

    @property
    @abstractmethod
    def parameter_layout(self) -> tuple[tuple[int, int], ...]:
        pass

    @property
    @abstractmethod
    def shared(self) -> int | None:
        """Fixed shared bytes, or None when the call supplies a numeric recipe."""

    @abstractmethod
    def check(self) -> None:
        pass

    @abstractmethod
    def _borrow_for_cudagraph(self) -> object:
        pass


@dataclass(frozen=True)
class _PhysicalField:
    parameter: int
    byte_offset: int
    kind: str
    source: (
        InputSource
        | BufferSource
        | PointerSource
        | IntegerSource
        | ExpressionSource
        | ParameterSource
    )


@dataclass(frozen=True)
class _TensorMapField:
    parameter: int
    pointer: PointerSource
    dimensions: tuple[IntExpr, ...]
    strides: tuple[IntExpr, ...]
    box_dimensions: tuple[int, ...]
    data_type: int
    swizzle: int
    nan_fill: bool

    def binding(self, numeric, input_count, buffer_indices, node=0):
        root = self.pointer.root
        if type(root) is InputSource and 0 <= root.index < input_count:
            index = root.index
        elif type(root) is BufferSource and root in buffer_indices:
            index = buffer_indices[root]
        else:
            raise UnsupportedCapture("Tensor map has no live input or owned allocation")
        return torch._C._CUDAGraphTensorMapBinding(
            node=node,
            argument=self.parameter,
            pointer_index=index,
            address_offset_value_index=numeric.add(self.pointer.byte_offset),
            dimensions=tuple(numeric.add(value) for value in self.dimensions),
            strides=tuple(numeric.add(value) for value in self.strides),
            box_dimensions=self.box_dimensions,
            data_type=self.data_type,
            swizzle=self.swizzle,
            nan_fill=self.nan_fill,
        )

    def encode(self, numeric, inputs, buffers, buffer_indices):
        binding = self.binding(numeric, len(inputs), buffer_indices)
        root = self.pointer.root
        tensor = inputs[root.index] if type(root) is InputSource else buffers[root]
        pointers = [0] * (len(inputs) + len(buffer_indices))
        pointers[binding.pointer_index] = tensor.data_ptr()
        return torch._C._cuda_encode_tensor_map(
            binding, tuple(numeric.values), tuple(pointers)
        )


@dataclass(frozen=True)
class _PhysicalCall:
    fields: tuple[_PhysicalField, ...]
    module: _KernelModule
    grid: tuple[IntExpr, IntExpr, IntExpr]
    padding: tuple[tuple[int, int, int], ...]
    constants: tuple[tuple[int, int, bytes], ...] = ()
    undefined: tuple[tuple[int, int, int], ...] = ()
    shared: IntExpr | None = None
    tensor_maps: tuple[_TensorMapField, ...] = ()
    block: tuple[IntExpr, IntExpr, IntExpr] | None = None
    storage_sources: tuple[PointerSource, ...] = ()


class _ParameterProgram:
    def __init__(self, numeric, input_count, buffer_indices):
        self.numeric = numeric
        self.input_count = input_count
        self.buffer_indices = buffer_indices
        self.rows = []
        self.indices = {}
        self.expressions = {}
        self.outputs = []
        self.output_indices = {}
        self.roots = {}

    def _node(self, source):
        if (
            type(source) is not ParameterSource
            or type(source.width) is not int
            or source.width not in (1, 32, 64)
            or type(source.args) is not tuple
            or type(source.flags) is not tuple
        ):
            raise UnsupportedCapture("Late parameter lost its typed expression")
        previous = self.expressions.get(id(source))
        if previous is not None:
            return previous[1]
        args = tuple(self._node(arg) for arg in source.args)
        op, width, value = source.op, source.width, source.value
        if op == "constant" and not args and not source.flags and type(value) is int:
            row = (op, width, value)
        elif op == "value" and not args and not source.flags and type(value) is IntExpr:
            row = (op, width, self.numeric.add(value))
        elif (
            op == "pointer"
            and width == 64
            and not args
            and not source.flags
            and type(value) is PointerSource
        ):
            root = value.root
            if type(root) is InputSource:
                index = root.index
                if (
                    type(index) is not int
                    or not 0 <= index < self.input_count
                    or index in self.numeric.integer_indices
                ):
                    raise UnsupportedCapture(
                        "Late pointer has no original Tensor input"
                    )
            elif type(root) is BufferSource and root in self.buffer_indices:
                index = self.buffer_indices[root]
            else:
                raise UnsupportedCapture(
                    "Late pointer has no allocated or input storage root"
                )
            self.roots[index] = root
            row = (op, width, index, self.numeric.add(value.byte_offset))
        elif (
            op == "icmp"
            and width == 1
            and type(value) is str
            and len(args) == 2
            and not source.flags
        ):
            row = (op, value, *args)
        elif (
            value is None
            and op in ("add", "sub", "mul", "shl", "udiv", "sdiv", "lshr", "ashr")
            and len(args) == 2
        ):
            row = (op, width, *args, source.flags)
        elif (
            value is None
            and not source.flags
            and (
                op in ("urem", "srem", "and", "or", "xor")
                and len(args) == 2
                or op in ("trunc", "zext", "sext")
                and len(args) == 1
                or op == "select"
                and len(args) == 3
            )
        ):
            row = (op, width, *args)
        else:
            raise UnsupportedCapture("Unsupported typed late parameter instruction")
        index = self.indices.get(row)
        if index is None:
            index = len(self.rows)
            self.indices[row] = index
            self.rows.append(row)
        self.expressions[id(source)] = (source, index)
        return index

    def add(self, source):
        node = self._node(source)
        if source.width not in (32, 64):
            raise UnsupportedCapture(
                "Late parameter outputs require i32/i64 scalar transport"
            )
        if node not in self.output_indices:
            self.output_indices[node] = len(self.outputs)
            self.outputs.append(node)
        return self.output_indices[node]

    def prepared_index(self, source):
        try:
            return self.output_indices[self.expressions[id(source)][1]]
        except KeyError:
            raise UnsupportedCapture(
                "Late parameter was not registered before capture"
            ) from None

    def plan(self):
        return ("parameter_v1", tuple(self.rows), tuple(self.outputs))

    def evaluate(self, inputs, buffers):
        if type(inputs) not in (list, tuple) or len(inputs) != self.input_count:
            raise UnsupportedCapture(
                "Late parameter evaluation requires the captured original inputs"
            )
        pointers = [0] * (self.input_count + len(self.buffer_indices))
        for index, root in self.roots.items():
            tensor = (
                inputs[root.index] if type(root) is InputSource else buffers.get(root)
            )
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                raise UnsupportedCapture(
                    "Late parameter lacks its live capture storage root"
                )
            pointers[index] = tensor.data_ptr()
        return torch._C._cuda_evaluate_parameter_program(
            self.plan(), tuple(self.numeric.values), tuple(pointers)
        )


_PHYSICAL_SCALAR_WIDTHS = {"i8": 1, "i16": 2, "i32": 4, "i64": 8}


def _physical_scalar_bytes(kind, value):
    width = _PHYSICAL_SCALAR_WIDTHS[kind]
    # The recorder's narrow kinds encode low bits without declaring signedness.
    if width < 4:
        return struct.pack("<q", value)[:width]
    return struct.pack("i" if width == 4 else "q", value)


def _pointer_value(pointer, displacement):
    value = pointer + displacement
    if not 0 <= value < 2 ** (8 * struct.calcsize("P")):
        raise UnsupportedCapture("Symbolic pointer address exceeds uintptr")
    return value


def _bind_physical_call(
    call,
    launch,
    captured,
    input_count,
    buffer_owners,
    buffer_indices,
    numeric,
    parameters=None,
    late_bindings=None,
    tensor_map_bindings=None,
    capture_inputs=None,
):
    if (
        not isinstance(call.module, _KernelModule)
        or type(call.fields) is not tuple
        or not call.fields
        or type(call.padding) is not tuple
        or numeric is None
    ):
        raise UnsupportedCapture(
            "Physical call requires an explicit owned module, fields and numeric plan"
        )
    call.module.check()
    layout = call.module.parameter_layout
    if (
        call.module.function != launch.function
        or type(layout) is not tuple
        or layout != tuple((offset, width) for offset, width, _ in captured.snapshot[8])
        or len(launch.argument_bytes) != len(layout)
    ):
        raise UnsupportedCapture(
            "Physical call lost its selected function or actual CUDA parameter layout"
        )
    coverage = [bytearray(width) for _, width in layout]
    pointers, scalars, inputs, buffers = [], [], OrderedSet(), OrderedSet()

    def span(parameter, offset, width):
        if (
            type(parameter) is not int
            or not 0 <= parameter < len(layout)
            or type(offset) is not int
            or type(width) is not int
            or offset < 0
            or width <= 0
            or offset + width > len(coverage[parameter])
        ):
            raise UnsupportedCapture("Physical field exceeds its actual CUDA parameter")
        covered = coverage[parameter]
        if any(covered[offset : offset + width]):
            raise UnsupportedCapture("Physical parameter fields overlap")
        covered[offset : offset + width] = b"\1" * width
        return launch.argument_bytes[parameter][offset : offset + width]

    for field in call.fields:
        if type(field) is not _PhysicalField or field.kind not in (
            "pointer",
            "i8",
            "i16",
            "i32",
            "i64",
        ):
            raise UnsupportedCapture("Unsupported physical parameter field")
        width = 8 if field.kind == "pointer" else _PHYSICAL_SCALAR_WIDTHS[field.kind]
        actual = span(field.parameter, field.byte_offset, width)
        source = field.source
        if type(source) is ParameterSource:
            if parameters is None or late_bindings is None or source.width != width * 8:
                raise UnsupportedCapture(
                    "Computed physical field lacks its typed late parameter program"
                )
            output = parameters.add(source)
            late_bindings.append(
                (
                    captured.snapshot[0],
                    field.parameter,
                    field.byte_offset,
                    width,
                    output,
                    actual,
                )
            )
            for root in storage_roots(source):
                if type(root) is InputSource:
                    inputs.add(root.index)
                else:
                    buffers.add(root)
        elif field.kind == "pointer":
            root = source.root if type(source) is PointerSource else source
            offset_index = (
                numeric.add(source.byte_offset)
                if type(source) is PointerSource
                else None
            )
            displacement = (
                numeric.values[offset_index] if offset_index is not None else 0
            )
            if type(root) is InputSource:
                if type(root.index) is not int or not 0 <= root.index < input_count:
                    raise UnsupportedCapture("Physical pointer input is out of range")
                pointer = root.index
                inputs.add(pointer)
            elif type(root) is BufferSource:
                value = buffer_owners.get(root)
                if (
                    type(value) is not torch.Tensor
                    or struct.pack("P", _pointer_value(value.data_ptr(), displacement))
                    != actual
                ):
                    raise UnsupportedCapture(
                        "Physical buffer field differs from its captured owner"
                    )
                pointer = buffer_indices.get(root)
                if pointer is None:
                    raise UnsupportedCapture(
                        "Physical buffer has no eager allocation slot"
                    )
                buffers.add(root)
            else:
                raise UnsupportedCapture(
                    "Physical pointer has no recorded Tensor source"
                )
            binding = (
                captured.snapshot[0],
                field.parameter,
                field.byte_offset,
                pointer,
            )
            pointers.append(
                (*binding, offset_index) if offset_index is not None else binding
            )
        else:
            if type(source) is IntegerSource and type(source.value) is int:
                value, index = source.value, None
            elif type(source) is ExpressionSource:
                index = numeric.add(source.expression)
                value = numeric.values[index]
            else:
                raise UnsupportedCapture(
                    "Physical integer has no exact symbolic or literal source"
                )
            bits = 32 if width == 4 else 64
            if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
                raise UnsupportedCapture(
                    "Physical integer exceeds its selected ABI width"
                )
            if _physical_scalar_bytes(field.kind, value) != actual:
                raise UnsupportedCapture(
                    "Physical integer field differs from its recorded source"
                )
            if index is not None:
                scalars.append(
                    (
                        captured.snapshot[0],
                        field.parameter,
                        field.byte_offset,
                        width,
                        index,
                    )
                )
    for field in call.tensor_maps:
        if (
            type(field) is not _TensorMapField
            or tensor_map_bindings is None
            or capture_inputs is None
        ):
            raise UnsupportedCapture(
                "Tensor map requires an explicit native binding and capture inputs"
            )
        actual = span(field.parameter, 0, 128)
        if actual != field.encode(
            numeric, capture_inputs, buffer_owners, buffer_indices
        ):
            raise UnsupportedCapture(
                "Captured tensor map differs from its traced CUDA encoder inputs"
            )
        tensor_map_bindings.append(
            field.binding(numeric, input_count, buffer_indices, captured.snapshot[0])
        )
        root = field.pointer.root
        if type(root) is InputSource:
            inputs.add(root.index)
        else:
            buffers.add(root)
    for constant in call.constants:
        if (
            type(constant) is not tuple
            or len(constant) != 3
            or type(constant[2]) is not bytes
        ):
            raise UnsupportedCapture("Physical constant requires exact compiler bytes")
        parameter, offset, data = constant
        if span(parameter, offset, len(data)) != data:
            raise UnsupportedCapture(
                "Physical constant differs from its exact compiler bytes"
            )
    for undefined in call.undefined:
        if type(undefined) is not tuple or len(undefined) != 3 or any(span(*undefined)):
            raise UnsupportedCapture(
                "Physical undefined field differs from its chosen zero image"
            )
    for padding in call.padding:
        if type(padding) is not tuple or len(padding) != 3:
            raise UnsupportedCapture("Physical padding requires an exact compiler span")
        if any(span(*padding)):
            raise UnsupportedCapture(
                "Physical padding differs from its declared zero bytes"
            )
    if any(not all(covered) for covered in coverage):
        raise UnsupportedCapture(
            "Physical fields do not cover every captured parameter byte"
        )
    return pointers, scalars, inputs, buffers


def _make_replay(
    graph: torch.cuda.CUDAGraph,
    input_count: int,
    allocations: tuple[OwnedBuffer, ...],
    output_indices: tuple[
        int
        | BorrowedInputOutput
        | TensorViewOutput
        | IntegerOutput
        | OutputReference
        | None,
        ...,
    ]
    | None,
    copies: tuple[int, ...],
    calls: tuple[_BoundCall | _PhysicalCall, ...],
    launches: tuple[RecordedKernelLaunch, ...],
    buffer_owners: dict[BufferSource, torch.Tensor],
    stream: torch.cuda.Stream,
    *,
    numeric: _NumericProgram | None,
    copy_if_misaligned: Any = None,
    resources: tuple[object, ...] = (),
    release_steps: tuple[tuple[str, int], ...] | None = None,
    capture_inputs: list[torch.Tensor | int]
    | tuple[torch.Tensor | int, ...]
    | None = None,
    memsets: tuple[tuple[int, PointerSource, IntExpr, int], ...] = (),
    host_tables: tuple[tuple[tuple[int, ...], int, tuple], ...] = (),
    memcpys: tuple[tuple[int, Any, PointerSource, IntExpr], ...] = (),
    rng: tuple[torch.Generator, IntExpr] | None = None,
    templates: tuple[
        tuple[tuple[int, ...], int, IntExpr, tuple[PointerSource, ...], int], ...
    ] = (),
    capture_events: tuple[RecordedKernelLaunch | RecordedGraphNode, ...] | None = None,
    pinned_positions: tuple[int, ...] = (),
    const_positions: tuple[int, ...] = (),
) -> Any:
    """`memsets`: (node, destination, byte count, value) rows for one-dimensional
    byte memset nodes the caller issued in the capture; their destination and
    byte count follow the replay's pointers and numeric plan.
    `host_tables`: (pinned slot addresses, byte count, elements) per host table the
    replay renders natively per call into the next slot; an element is
    (offset, width, PointerSource | IntExpr). `memcpys`: (node, source, destination,
    byte count) rows for one-dimensional host-to-device memcpy nodes the caller
    issued in the capture; the source is a host table index or a PointerSource
    over an input (a pinned CPU tensor). `pinned_positions` identifies those boxed
    inputs for ownership until rebinding and host-allocator submission tracking.
    `const_positions`: boxed inputs the replay only reads, whose address is taken
    through the const accessor per call (a copy-on-write tensor there stays lazy);
    every other input is read through `data_ptr()`, which materializes.
    `templates`: (nodes, site, variant,
    operands, workspace) rows for kernel template sites the caller launched in
    the capture (torch._C._cuda_kernel_template_register): the variant IntExpr
    selects the registered variant per call, the operand PointerSources fill its
    address slots."""
    if pinned_positions and capture_inputs is None:
        raise UnsupportedCapture("Pinned replay inputs require their captured values")
    if (
        len(launches) != len(calls)
        or type(resources) is not tuple
        or type(buffer_owners) is not dict
    ):
        raise UnsupportedCapture(
            "Replay requires complete bound calls and captured buffer owners"
        )
    if copies and copy_if_misaligned is None:
        raise UnsupportedCapture(
            "Recorded alignment copies require their native operation"
        )
    if release_steps is not None and output_indices is None:
        raise UnsupportedCapture(
            "Saved-input release requires eager-pool allocation outputs"
        )
    buffer_indices = {
        layout.source: input_count + index for index, layout in enumerate(allocations)
    }
    layouts = []
    for layout in allocations:
        dimensions = (layout.size, layout.stride)
        if numeric is not None:
            dimensions = tuple(
                tuple(
                    ("value", numeric.add(value)) if type(value) is IntExpr else value
                    for value in values
                )
                for values in dimensions
            )
        layouts.append((layout.dtype, *dimensions))
    layouts = tuple(layouts)
    if output_indices is not None:
        slots = []
        for output in output_indices:
            if type(output) is OutputReference:
                slots.append(("output", output.index))
            elif type(output) is BorrowedInputOutput:
                slots.append(("input", output.source.index))
            elif type(output) is TensorViewOutput:
                source = output.source
                if type(source) is InputSource:
                    pointer = source.index
                elif type(source) is BufferSource and source in buffer_indices:
                    pointer = buffer_indices[source]
                else:
                    raise UnsupportedCapture(
                        "Output view has no input or allocated root"
                    )
                fields = []
                for dimensions in (output.size, output.stride, (output.offset,)):
                    values = []
                    for value in dimensions:
                        if type(value) is IntExpr:
                            if numeric is None:
                                raise UnsupportedCapture(
                                    "Symbolic output views require a numeric plan"
                                )
                            values.append(("value", numeric.add(value)))
                        else:
                            values.append(value)
                    fields.append(tuple(values))
                slot = ("view", pointer, fields[0], fields[1], fields[2][0])
                if getattr(output, "dtype", None) is not None:
                    slot = (*slot, output.dtype)
                slots.append(slot)
            elif type(output) is IntegerOutput:
                if type(output.value) is int:
                    slots.append(("literal", output.value))
                elif numeric is not None:
                    slots.append(("value", numeric.add(output.value)))
                else:
                    raise UnsupportedCapture(
                        "Symbolic outputs require a numeric replay plan"
                    )
            else:
                slots.append(output)
        output_indices = tuple(slots)
    nodes = tuple(
        launch.after[3][0][0] for launch in launches if len(launch.after[3]) == 1
    )
    associated = associate_kernel_launches(
        launches,
        graph._inspect_captured_kernel_nodes(nodes),
        events=capture_events,
    )
    template_nodes = tuple(node for nodes, _, _, _, _ in templates for node in nodes)
    for kind, bound in (
        ("memset", tuple(node for node, _, _, _ in memsets)),
        ("memcpy", tuple(node for node, _, _, _ in memcpys)),
        ("template", template_nodes),
    ):
        recorded = OrderedSet(
            event.after[3][0][0]
            for event in capture_events or ()
            if type(event) is RecordedGraphNode and event.kind == kind
        )
        if recorded != OrderedSet(bound):
            raise UnsupportedCapture(
                f"Recorded {kind} events do not match their replay bindings"
            )
    bindings, scalar_bindings, grid_bindings, tensor_map_bindings = [], [], [], []
    parameters, late_bindings = None, []
    used_inputs, used_buffers = OrderedSet(), OrderedSet()
    modules = {}
    for index, (call, launch, captured) in enumerate(zip(calls, launches, associated)):
        if (
            type(call) not in (_BoundCall, _PhysicalCall)
            or captured.occurrence != index
        ):
            raise UnsupportedCapture("Bound and captured launch occurrences must agree")
        module = call.module
        shared_index = None
        block_indices = None
        if type(call) is _PhysicalCall:
            if call.block is not None:
                if (
                    numeric is None
                    or type(call.block) is not tuple
                    or len(call.block) != 3
                ):
                    raise UnsupportedCapture(
                        "Dynamic block requires three early expressions"
                    )
                block_indices = tuple(numeric.add(axis) for axis in call.block)
                block = tuple(numeric.values[index] for index in block_indices)
                if block != captured.snapshot[5] or any(value <= 0 for value in block):
                    raise UnsupportedCapture(
                        "Captured block differs from its symbolic recipe"
                    )
            shared = module.shared
            if call.shared is not None:
                if shared is not None or numeric is None:
                    raise UnsupportedCapture(
                        "Dynamic shared bytes require an early recipe and dynamic owner"
                    )
                shared_index = numeric.add(call.shared)
                shared = numeric.values[shared_index]
            if (
                type(shared) is not int
                or not 0 <= shared < 2**32
                or shared != captured.snapshot[6]
            ):
                raise UnsupportedCapture(
                    "Captured shared bytes differ from their exact request"
                )
            if parameters is None and any(
                type(field.source) is ParameterSource for field in call.fields
            ):
                parameters = _ParameterProgram(numeric, input_count, buffer_indices)
            pointers, scalars, inputs, buffers = _bind_physical_call(
                call,
                launch,
                captured,
                input_count,
                buffer_owners,
                buffer_indices,
                numeric,
                parameters,
                late_bindings,
                tensor_map_bindings,
                capture_inputs,
            )
            bindings.extend(pointers)
            scalar_bindings.extend(scalars)
            used_inputs.update(inputs)
            used_buffers.update(buffers)
        else:
            if (
                type(module) is not StaticallyLaunchedCudaKernel
                or module.function != launch.function
            ):
                raise UnsupportedCapture(
                    "Captured function does not match its selected module owner"
                )
            bound = call.launch_arguments
            scratch = []
            if not call.scratch:
                for present, size in (
                    (module.has_global_scratch, module.global_scratch_size),
                    (module.has_profile_scratch, module.profile_scratch_size),
                ):
                    if present:
                        if size != 0:
                            raise UnsupportedCapture(
                                "Nonempty launcher scratch requires a separate ownership contract"
                            )
                        scratch.append(bytes(struct.calcsize("P")))
            if len(launch.argument_bytes) != len(bound) + len(scratch):
                raise UnsupportedCapture(
                    "Bound arguments must cover the complete captured ABI"
                )
            if launch.argument_bytes[len(bound) :] != tuple(scratch):
                raise UnsupportedCapture(
                    "Launcher scratch fields differ from the selected constants"
                )
            for slot, argument in enumerate(bound):
                source = argument.source
                root = source.root if type(source) is PointerSource else source
                offset_index = None
                displacement = 0
                if type(source) is PointerSource:
                    if numeric is None:
                        raise UnsupportedCapture(
                            "Symbolic pointers require a numeric replay plan"
                        )
                    offset_index = numeric.add(source.byte_offset)
                    displacement = numeric.values[offset_index]
                actual = launch.argument_bytes[slot]
                if type(source) is ParameterSource:
                    pointer = argument.triton_type.startswith("*")
                    width = (
                        8 if pointer else {"i32": 4, "i64": 8}.get(argument.triton_type)
                    )
                    if (
                        numeric is None
                        or width is None
                        or source.width != width * 8
                        or len(actual) != width
                        or pointer
                        and source != ParameterSource("constant", 64, 0)
                    ):
                        raise UnsupportedCapture(
                            "Computed scalar differs from its captured flat ABI width"
                        )
                    if parameters is None:
                        parameters = _ParameterProgram(
                            numeric, input_count, buffer_indices
                        )
                    output = parameters.add(source)
                    late_bindings.append(
                        (captured.snapshot[0], slot, 0, width, output, actual)
                    )
                    for root in storage_roots(source):
                        if type(root) is InputSource:
                            used_inputs.add(root.index)
                        else:
                            used_buffers.add(root)
                    continue
                if type(root) is InputSource:
                    if type(root.index) is not int or not 0 <= root.index < input_count:
                        raise UnsupportedCapture("Input index is out of range")
                    used_inputs.add(root.index)
                    pointer_index = root.index
                elif type(root) is BufferSource:
                    tensor = buffer_owners.get(root)
                    if type(tensor) is not torch.Tensor:
                        raise UnsupportedCapture("Missing a captured buffer owner")
                    if (
                        struct.pack(
                            "P", _pointer_value(tensor.data_ptr(), displacement)
                        )
                        != actual
                    ):
                        raise UnsupportedCapture(
                            "Captured buffer bytes differ from their recorded source"
                        )
                    used_buffers.add(root)
                    pointer_index = buffer_indices.get(root)
                elif type(source) is IntegerSource:
                    if (
                        struct.pack(
                            {"i32": "i", "i64": "q"}[argument.triton_type], source.value
                        )
                        != actual
                    ):
                        raise UnsupportedCapture(
                            "Captured integer differs from its literal"
                        )
                    continue
                elif type(source) is ExpressionSource and numeric is not None:
                    value_index = numeric.add(source.expression)
                    value = numeric.values[value_index]
                    width = {"i32": 4, "i64": 8}[argument.triton_type]
                    if not -(2 ** (width * 8 - 1)) <= value < 2 ** (width * 8 - 1):
                        raise UnsupportedCapture(
                            "Integer exceeds its selected scalar type"
                        )
                    if struct.pack("i" if width == 4 else "q", value) != actual:
                        raise UnsupportedCapture(
                            "Captured integer differs from its symbolic expression"
                        )
                    scalar_bindings.append(
                        (captured.snapshot[0], slot, width, value_index)
                    )
                    continue
                else:
                    raise UnsupportedCapture("Unknown bound argument source")
                if len(actual) != struct.calcsize("P"):
                    raise UnsupportedCapture(
                        "A bound pointer does not have the native pointer width"
                    )
                if pointer_index is not None:
                    bindings.append(
                        (captured.snapshot[0], slot, pointer_index)
                        if offset_index is None
                        else (
                            captured.snapshot[0],
                            slot,
                            None,
                            pointer_index,
                            offset_index,
                        )
                    )
        if numeric is not None:
            if call.grid is None or len(call.grid) != 3:
                raise UnsupportedCapture("Bound call has no explicit three-axis grid")
            grid_indices = tuple(numeric.add(expression) for expression in call.grid)
            expected = tuple(numeric.values[index] for index in grid_indices)
            if expected != captured.snapshot[4] or any(
                not 0 < value < 2**31 for value in expected
            ):
                raise UnsupportedCapture(
                    "Captured grid differs from its symbolic recipe or supported domain"
                )
            row = (
                (captured.snapshot[0], *grid_indices)
                if shared_index is None
                else (captured.snapshot[0], *grid_indices, shared_index)
            )
            if block_indices is not None:
                row = (*row, *block_indices)
            grid_bindings.append(row)
        elif call.grid is not None:
            if (
                len(call.grid) != 3
                or any(
                    type(value) is not IntExpr
                    or value.op != "constant"
                    or type(value.value) is not int
                    or value.args
                    or not 0 < value.value < 2**31
                    for value in call.grid
                )
                or tuple(value.value for value in call.grid) != captured.snapshot[4]
            ):
                raise UnsupportedCapture(
                    "Captured grid differs from the recorded literal grid"
                )
        modules[id(module)] = module
    # buffers and inputs a host table element or a copy names are used too: a kernel
    # reaches them through the device copy of the table, or the copy writes them
    for _, _, elements in host_tables:
        for _, _, source in elements:
            if type(source) is PointerSource:
                if type(source.root) is InputSource:
                    used_inputs.add(source.root.index)
                elif source.root in buffer_indices:
                    used_buffers.add(source.root)
    for _, source, destination, _ in memcpys:
        for root in (
            (destination.root,)
            if type(source) is int
            else (source.root, destination.root)
        ):
            if type(root) is InputSource:
                used_inputs.add(root.index)
            elif root in buffer_indices:
                used_buffers.add(root)
    for _, _, _, operands, _ in templates:
        for operand in operands:
            if type(operand.root) is InputSource:
                used_inputs.add(operand.root.index)
            elif operand.root in buffer_indices:
                used_buffers.add(operand.root)
    for _, destination, _, _ in memsets:
        if type(destination.root) is InputSource:
            used_inputs.add(destination.root.index)
        elif destination.root in buffer_indices:
            used_buffers.add(destination.root)
    owned_buffers = set(buffer_owners)
    planned_buffers = set(buffer_indices)
    required_buffers = set(used_buffers) if output_indices is None else planned_buffers
    if (
        owned_buffers != required_buffers
        or not (set(used_buffers) | planned_buffers) <= owned_buffers
    ):
        raise UnsupportedCapture(
            "Captured owners must cover all bound buffers and returned outputs"
        )
    if numeric is None:
        if memsets or host_tables or memcpys or rng is not None:
            raise UnsupportedCapture("Non-kernel replay updates require a numeric plan")
        batch = graph._prepare_kernel_pointer_updates(
            tuple(bindings), input_count + len(layouts)
        )
    else:
        for index in numeric.integer_indices:
            numeric.add(IntExpr("boxed", index))
        keywords = (
            {"tensor_map_bindings": tuple(tensor_map_bindings)}
            if tensor_map_bindings
            else {}
        )
        if memsets:
            memset_bindings = []
            for node, destination, byte_count, _ in memsets:
                root = destination.root
                if type(root) is InputSource:
                    pointer_index = root.index
                elif type(root) is BufferSource and root in buffer_indices:
                    pointer_index = buffer_indices[root]
                    used_buffers.add(root)
                else:
                    raise UnsupportedCapture(
                        "A memset destination has no allocated or input storage root"
                    )
                memset_bindings.append(
                    (
                        node,
                        pointer_index,
                        numeric.add(destination.byte_offset),
                        numeric.add(byte_count),
                    )
                )
            keywords["memset_bindings"] = tuple(memset_bindings)

        def _root_pointer_index(root, what):
            if type(root) is InputSource:
                used_inputs.add(root.index)
                return root.index
            if type(root) is BufferSource and root in buffer_indices:
                used_buffers.add(root)
                return buffer_indices[root]
            raise UnsupportedCapture(f"{what} has no allocated or input storage root")

        if host_tables:
            table_bindings = []
            for slots, nbytes, elements in host_tables:
                rows = []
                for offset, width, source in elements:
                    if type(source) is PointerSource:
                        index = _root_pointer_index(
                            source.root, "A host table pointer element"
                        )
                        rows.append(
                            (
                                offset,
                                width,
                                "pointer",
                                index,
                                numeric.add(source.byte_offset),
                            )
                        )
                    else:
                        rows.append((offset, width, "value", numeric.add(source), None))
                table_bindings.append((tuple(slots), nbytes, tuple(rows)))
            keywords["host_table_bindings"] = tuple(table_bindings)
        if memcpys:
            memcpy_bindings = []
            for node, source, destination, byte_count in memcpys:
                if type(source) is int:
                    src = (source, 0, None)
                else:
                    src = (
                        None,
                        _root_pointer_index(source.root, "A memcpy source"),
                        numeric.add(source.byte_offset),
                    )
                memcpy_bindings.append(
                    (
                        node,
                        *src,
                        _root_pointer_index(destination.root, "A memcpy destination"),
                        numeric.add(destination.byte_offset),
                        numeric.add(byte_count),
                    )
                )
            keywords["memcpy_bindings"] = tuple(memcpy_bindings)
        if rng is not None:
            # the philox offsets one replay draws: set on the graph before each replay
            keywords["rng_bindings"] = ((rng[0], numeric.add(rng[1])),)
        early_count = len(numeric.instructions)
        if parameters is not None:
            values = parameters.evaluate(capture_inputs, buffer_owners)
            for node, parameter, offset, width, output, actual in late_bindings:
                if struct.pack("i" if width == 4 else "q", values[output]) != actual:
                    raise UnsupportedCapture(
                        "Computed physical field differs from its captured compiler expression"
                    )
                scalar_bindings.append(
                    (node, parameter, offset, width, early_count + output)
                )
        if templates:
            keywords["template_bindings"] = tuple(
                (
                    tuple(nodes),
                    site,
                    numeric.add(variant),
                    tuple(
                        (
                            _root_pointer_index(operand.root, "A template operand"),
                            numeric.add(operand.byte_offset),
                        )
                        for operand in operands
                    ),
                    workspace,
                )
                for nodes, site, variant, operands, workspace in templates
            )
        batch = graph._prepare_kernel_replay_updates(
            tuple(bindings),
            input_count + len(layouts),
            tuple(scalar_bindings),
            tuple(grid_bindings),
            early_count + (len(parameters.outputs) if parameters is not None else 0),
            **keywords,
        )
    borrows = tuple(module._borrow_for_cudagraph() for module in modules.values())
    retained_buffers = buffer_owners.values() if output_indices is None else ()
    retained = (*borrows, *retained_buffers, *resources)
    prologue = (tuple(copies), copy_if_misaligned) if copies else None
    arguments = (
        batch,
        stream,
        retained,
        input_count,
        tuple(sorted(used_inputs)),
        layouts,
        prologue,
    )
    numeric_plan = (
        None
        if numeric is None
        else (numeric.integer_indices, tuple(numeric.instructions))
    )
    if parameters is not None:
        if output_indices is None:
            raise UnsupportedCapture(
                "Late parameters require prepared eager allocations and outputs"
            )
        numeric_plan = (*numeric_plan, parameters.plan())
    compiled_evaluation = None
    if numeric is not None:
        from .cudagraph_compiled_evaluation import compile_evaluation

        compiled = compile_evaluation(
            numeric, parameters, pointer_count=input_count + len(layouts)
        )
        compiled_evaluation = torch._C._CUDAGraphCompiledEvaluation(
            **compiled.registration_kwargs
        )
    owner_keywords: dict[str, Any] = (
        {"pinned_positions": pinned_positions} if pinned_positions else {}
    )
    if pinned_positions:
        # The initial copy-node bindings must survive until the first replay.
        owner_keywords["pinned_examples"] = tuple(
            capture_inputs[position] for position in pinned_positions
        )
    if const_positions:
        owner_keywords["const_positions"] = const_positions
    if release_steps is not None:
        release_plan = (
            release_steps,
            tuple(captured.snapshot[0] for captured in associated),
        )
        return graph._make_boxed_replay(
            *arguments,
            output_indices,
            numeric_plan,
            release_plan,
            compiled_evaluation=compiled_evaluation,
            **owner_keywords,
        )
    if output_indices is None:
        return graph._make_boxed_replay(*arguments, **owner_keywords)
    if numeric is not None:
        return graph._make_boxed_replay(
            *arguments,
            output_indices,
            numeric_plan,
            compiled_evaluation=compiled_evaluation,
            **owner_keywords,
        )
    return graph._make_boxed_replay(*arguments, output_indices, **owner_keywords)
