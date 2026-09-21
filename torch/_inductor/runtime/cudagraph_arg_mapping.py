from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from .cudagraph_multikernel import MultiKernelCallRecord


FIXED_GRID_ARGUMENTS = ("_grid_0", "_grid_1", "_grid_2")


@dataclass(frozen=True)
class KernelArgument:
    formal: str
    source_arg_index: int
    triton_type: str
    abi_index: int | None
    constant: int | float | bool | str | None
    attributes: tuple[tuple[str, int | float | bool | str | None], ...] | None = None


@dataclass(frozen=True)
class LauncherArgument:
    formal: str
    source_arg_index: int
    call_arg_index: int | None
    triton_type: str
    abi_index: int | None
    constant: int | float | bool | str | None


@dataclass(frozen=True)
class InputSource:
    index: int


@dataclass(frozen=True)
class BufferSource:
    name: str


@dataclass(frozen=True)
class IntegerSource:
    value: int


@dataclass(frozen=True)
class IntegerInput:
    symbol: str
    boxed_index: int
    source_index: int | None = None


def integer_input_sources(
    rows: tuple[IntegerInput, ...], input_count: int
) -> dict[int, int] | None:
    if type(rows) is not tuple or type(input_count) is not int or input_count < 0:
        return None
    sources, names = {}, OrderedSet()
    for row in rows:
        if (
            type(row) is not IntegerInput
            or type(row.symbol) is not str
            or not row.symbol
            or row.symbol in names
            or type(row.boxed_index) is not int
            or not 0 <= row.boxed_index < input_count
            or row.boxed_index in sources
        ):
            return None
        index = row.boxed_index
        source = index if row.source_index is None else row.source_index
        if row.source_index is not None and (
            type(source) is not int or source >= index or sources.get(source) != source
        ):
            return None
        sources[index] = source
        names.add(row.symbol)
    return sources


@dataclass(frozen=True)
class IntExpr:
    op: str
    # Tensor metadata uses value=input and a constant dimension operand.
    value: int | str | tuple[int, object] | None = None
    args: tuple[IntExpr, ...] = ()
    # the structural hash, computed once: expressions are DAGs (a prefix sum shares
    # its predecessor), and the generated hash re-walked every path
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    def __hash__(self) -> int:
        h = self._hash
        if h is not None:
            return h
        # bottom-up over the DAG so a deep chain (a 257-input cat's n-ary max) does not
        # recurse once per level; a node is hashed only after all its operands are, and
        # each node's hash is cached the first time
        pending: list[tuple[IntExpr, bool]] = [(self, False)]
        while pending:
            node, ready = pending.pop()
            if node._hash is not None:
                continue
            missing = [a for a in node.args if type(a) is IntExpr and a._hash is None]
            if ready and not missing:
                object.__setattr__(
                    node, "_hash", hash((node.op, node.value, node.args))
                )
                continue
            pending.append((node, True))
            pending.extend((a, False) for a in missing)
        return self._hash  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        # Structural equality over the DAG, each (self, other) pair visited once. The
        # generated dataclass method compared the args as a tree, exponential on a
        # shared chain such as an n-ary min folded through select (two distinct but
        # equal 128-deep chains never finished comparing inside a dict lookup).
        if self is other:
            return True
        if type(other) is not IntExpr or hash(self) != hash(other):
            return False
        seen: OrderedSet[tuple[int, int]] = OrderedSet()
        stack = [(self, other)]
        while stack:
            a, b = stack.pop()
            if a is b:
                continue
            key = (id(a), id(b))
            if key in seen:
                continue
            seen.add(key)
            if a.op != b.op or a.value != b.value or len(a.args) != len(b.args):
                return False
            stack.extend(zip(a.args, b.args))
        return True

    def render_grid(self, mode: str) -> str | int:
        if mode not in ("python", "cpp"):
            raise ValueError("Unsupported grid rendering mode")
        if self.op == "constant" and type(self.value) is int and not self.args:
            return self.value
        if (
            self.op == "formal"
            and type(self.value) is str
            and self.value.isidentifier()
            and not self.args
        ):
            return self.value
        if (
            self.op in ("add", "multiply")
            and self.value is None
            and len(self.args) == 2
        ):
            if all(type(arg) is IntExpr for arg in self.args):
                left, right = (arg.render_grid(mode) for arg in self.args)
                if type(left) is int and type(right) is int:
                    return left + right if self.op == "add" else left * right
                operator = "+" if self.op == "add" else "*"
                return f"(({left}) {operator} ({right}))"
        if self.op == "ceildiv" and self.value is None and len(self.args) == 2:
            numerator, divisor = self.args
            if (
                type(numerator) is IntExpr
                and type(divisor) is IntExpr
                and divisor.op == "constant"
                and type(divisor.value) is int
                and divisor.value > 0
            ):
                numel, block = numerator.render_grid(mode), divisor.render_grid(mode)
                if type(numel) is int:
                    return -(numel // -block)
                if block == 1:
                    return numel
                if mode == "python":
                    return f"-(({numel}) // -({block}))"
                return f"(({numel} + ({block} - 1)) / ({block}))"
        raise ValueError("Unsupported grid expression")


def pointwise_product(size: tuple[int | IntExpr, ...]) -> int | IntExpr | None:
    if type(size) is not tuple or len(size) not in (1, 2):
        return None
    operands = []
    inputs = OrderedSet()
    constant_product = 1
    for value in size:
        if type(value) is int:
            if not 0 < value < 2**63:
                return None
            constant_product *= value
            if value != 1:
                operands.append(IntExpr("constant", value))
        elif (
            type(value) is IntExpr
            and value.op == "boxed"
            and type(value.value) is int
            and value.value >= 0
            and type(value.args) is tuple
            and not value.args
        ):
            if value.value in inputs:
                return None
            inputs.add(value.value)
            operands.append(value)
        else:
            return None
    if not inputs:
        return constant_product if constant_product < 2**63 else None
    if len(operands) == 1:
        return operands[0]
    operands.sort(key=lambda value: (value.op, value.value))
    return IntExpr("multiply", args=tuple(operands))


def pointwise_expression_inputs(expression: IntExpr) -> tuple[int, ...] | None:
    if type(expression) is not IntExpr or type(expression.args) is not tuple:
        return None
    if expression.op == "boxed":
        operands = (expression,)
    elif (
        expression.op == "multiply"
        and expression.value is None
        and len(expression.args) == 2
    ):
        operands = expression.args
    else:
        return None
    dimensions = []
    for operand in operands:
        if type(operand) is not IntExpr:
            return None
        if (
            operand.op == "constant"
            and type(operand.args) is tuple
            and not operand.args
        ):
            dimensions.append(operand.value)
        else:
            dimensions.append(operand)
    if pointwise_product(tuple(dimensions)) != expression:
        return None
    return tuple(operand.value for operand in operands if operand.op == "boxed")


@dataclass(frozen=True)
class ExpressionSource:
    expression: IntExpr


@dataclass(frozen=True)
class PointerSource:
    root: InputSource | BufferSource
    byte_offset: IntExpr


@dataclass(frozen=True)
class ParameterSource:
    op: str
    width: int
    value: int | IntExpr | PointerSource | str | None = None
    args: tuple[ParameterSource, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def pointers(self) -> tuple[PointerSource, ...]:
        found = {}
        pending = [self]
        while pending:
            value = pending.pop()
            if value.op == "pointer":
                found[value.value] = None
            pending.extend(reversed(value.args))
        return tuple(found)


def storage_roots(
    source: InputSource
    | BufferSource
    | PointerSource
    | IntegerSource
    | ExpressionSource
    | ParameterSource,
) -> tuple[InputSource | BufferSource, ...]:
    if type(source) is ParameterSource:
        roots = {}
        seen = set()
        pending = [source]
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if value.op == "pointer":
                roots[value.value.root] = None
            pending.extend(reversed(value.args))
        return tuple(roots)
    if type(source) is PointerSource:
        return (source.root,)
    if type(source) in (InputSource, BufferSource):
        return (source,)
    return ()


@dataclass(frozen=True)
class CallArgument:
    formal: str
    source_arg_index: int
    call_arg_index: int
    triton_type: str
    source: (
        InputSource
        | BufferSource
        | PointerSource
        | IntegerSource
        | ExpressionSource
        | ParameterSource
    )


@dataclass(frozen=True)
class KernelCallRecord:
    occurrence: int
    kernel_global: str
    formals: tuple[str, ...]
    arguments: tuple[CallArgument, ...]
    grid_type: str
    launcher_grid: tuple[int | IntExpr, int | IntExpr, int | IntExpr] | None = None
    constexprs: tuple[tuple[str, int | float | bool | str | None], ...] = ()
    generated_template: bool = False


@dataclass(frozen=True)
class AlignmentCopy:
    input_index: int
    before_call: int


@dataclass(frozen=True)
class OwnedBuffer:
    source: BufferSource
    dtype: torch.dtype
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]


OwnedOutput = OwnedBuffer


@dataclass(frozen=True)
class BorrowedInputOutput:
    source: InputSource


@dataclass(frozen=True)
class TensorViewOutput:
    source: InputSource | BufferSource
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]
    offset: int | IntExpr
    # the view's own dtype when it differs from its root's (a view_as_real /
    # view_as_complex output): `offset` is then the storage offset over the root's
    # storage in the view's element units (nothing of the root's offset is added)
    dtype: torch.dtype | None = None


@dataclass(frozen=True)
class OutputReference:
    index: int


@dataclass(frozen=True)
class IntegerOutput:
    value: int | IntExpr


@dataclass(frozen=True)
class WrapperCallRecords:
    version: int
    device_index: int
    input_names: tuple[str, ...]
    calls: tuple[KernelCallRecord | MultiKernelCallRecord, ...]
    alignment_copies: tuple[AlignmentCopy, ...]
    outputs: (
        tuple[OwnedBuffer | BorrowedInputOutput | IntegerOutput | None, ...] | None
    ) = None
    allocations: tuple[OwnedBuffer, ...] | None = None
    integer_inputs: tuple[IntegerInput, ...] = ()


def bind_alignment_copies(records: WrapperCallRecords) -> tuple[int, ...] | None:
    from .cudagraph_multikernel import call_sources

    if (
        type(records) is not WrapperCallRecords
        or type(records.version) is not int
        or records.version not in (2, 3, 4)
        or type(records.input_names) is not tuple
        or type(records.calls) is not tuple
        or type(records.alignment_copies) is not tuple
    ):
        return None
    input_count = len(records.input_names)
    integers = OrderedSet()
    if records.version == 4:
        sources = integer_input_sources(records.integer_inputs, input_count)
        if not sources:
            return None
        integers = OrderedSet(sources)
    first_use = {}
    for index, call in enumerate(records.calls):
        sources = call_sources(call)
        if sources is None:
            return None
        for source in sources:
            if type(source) is InputSource:
                if (
                    type(source.index) is not int
                    or not 0 <= source.index < input_count
                    or source.index in integers
                ):
                    return None
                first_use.setdefault(source.index, index)
    copies = []
    previous = 0
    for copy in records.alignment_copies:
        if (
            type(copy) is not AlignmentCopy
            or type(copy.input_index) is not int
            or not 0 <= copy.input_index < input_count
            or copy.input_index in copies
            or copy.input_index in integers
            or type(copy.before_call) is not int
            or copy.before_call < previous
            or (
                copy.before_call != 0
                and (
                    records.version not in (3, 4)
                    or copy.before_call != first_use.get(copy.input_index)
                )
            )
        ):
            return None
        copies.append(copy.input_index)
        previous = copy.before_call
    return tuple(copies)


def bind_output_slots(
    outputs: tuple[
        OwnedBuffer
        | BorrowedInputOutput
        | TensorViewOutput
        | IntegerOutput
        | OutputReference
        | None,
        ...,
    ],
    layouts: tuple[OwnedBuffer, ...],
    input_names: tuple[str, ...],
    integers: OrderedSet[int],
    *,
    symbolic: bool,
    roots: OrderedSet[int] | None = None,
) -> (
    tuple[
        int
        | BorrowedInputOutput
        | TensorViewOutput
        | IntegerOutput
        | OutputReference
        | None,
        ...,
    ]
    | None
):
    """Bind physical slots using validated layouts and canonical integer roots."""
    if roots is None:
        roots = integers
    if type(outputs) is not tuple or type(input_names) is not tuple:
        return None
    tensors = OrderedSet(range(len(input_names))) - integers
    if outputs and not any(
        type(output) in (OwnedBuffer, BorrowedInputOutput, TensorViewOutput)
        for output in outputs
    ):
        return None
    indices = {layout.source: index for index, layout in enumerate(layouts)}
    if len(indices) != len(layouts):
        return None

    def valid_integer(value):
        if type(value) is int:
            return -(2**63) <= value < 2**63
        if not symbolic:
            return False
        return (
            grid_expression_inputs(value, boxed_indices=roots, tensor_indices=tensors)
            is not None
        )

    returned = []
    for output in outputs:
        if output is None:
            returned.append(None)
        elif type(output) is OutputReference:
            if (
                type(output.index) is not int
                or not 0 <= output.index < len(returned)
                or type(outputs[output.index])
                not in (
                    OwnedBuffer,
                    BorrowedInputOutput,
                    TensorViewOutput,
                    OutputReference,
                )
            ):
                return None
            returned.append(output)
        elif type(output) is BorrowedInputOutput:
            source = output.source
            if (
                type(source) is not InputSource
                or type(source.index) is not int
                or not 0 <= source.index < len(input_names)
                or source.index in integers
            ):
                return None
            returned.append(output)
        elif type(output) is TensorViewOutput:
            source = output.source
            if type(source) is InputSource:
                if (
                    type(source.index) is not int
                    or not 0 <= source.index < len(input_names)
                    or source.index in integers
                ):
                    return None
            elif type(source) is not BufferSource or source not in indices:
                return None
            if (
                type(output.size) is not tuple
                or type(output.stride) is not tuple
                or len(output.size) != len(output.stride)
            ):
                return None
            for value in (*output.size, *output.stride, output.offset):
                if not valid_integer(value):
                    return None
            returned.append(output)
        elif type(output) is IntegerOutput:
            if not valid_integer(output.value):
                return None
            returned.append(output)
        elif type(output) is OwnedBuffer:
            if (
                type(output.source) is not BufferSource
                or type(output.source.name) is not str
            ):
                return None
            index = indices.get(output.source)
            if index is None or index in returned or output != layouts[index]:
                return None
            returned.append(index)
        else:
            return None
    return tuple(returned)


def bind_wrapper_allocations(
    records: WrapperCallRecords,
) -> (
    tuple[
        tuple[OwnedBuffer, ...],
        tuple[int | BorrowedInputOutput | IntegerOutput | None, ...] | None,
    ]
    | None
):
    from .cudagraph_multikernel import call_sources, MultiKernelCallRecord

    if (
        type(records) is not WrapperCallRecords
        or type(records.version) is not int
        or records.version not in (2, 3, 4)
    ):
        return None
    if type(records.calls) is tuple and any(
        type(call) is MultiKernelCallRecord for call in records.calls
    ):
        if any(call_sources(call) is None for call in records.calls):
            return None
        calls = tuple(
            child
            for call in records.calls
            for child in (
                tuple(choice.call for choice in call.alternatives)
                if type(call) is MultiKernelCallRecord
                else (call,)
            )
        )
        return bind_wrapper_allocations(replace(records, calls=calls))
    if type(records.outputs) is not tuple:
        return None
    layouts = records.outputs if records.version == 2 else records.allocations
    if type(layouts) is not tuple or (
        records.version == 2 and records.allocations is not None
    ):
        return None
    integers, roots = OrderedSet(), OrderedSet()
    if records.version == 4:
        if (
            type(records.input_names) is not tuple
            or len(records.input_names) < 2
            or any(type(name) is not str for name in records.input_names)
            or len(OrderedSet(records.input_names)) != len(records.input_names)
            or type(records.calls) is not tuple
            or not records.calls
            or not layouts
            or not records.outputs
        ):
            return None
        sources = integer_input_sources(
            records.integer_inputs, len(records.input_names)
        )
        if not sources:
            return None
        integers, roots = OrderedSet(sources), OrderedSet(sources.values())
    shape_inputs, output_inputs = OrderedSet(), OrderedSet()
    tensor_outputs = tuple(
        output for output in records.outputs if type(output) is OwnedBuffer
    )
    if records.outputs and not any(
        type(output) in (OwnedBuffer, BorrowedInputOutput) for output in records.outputs
    ):
        return None
    for layout in (*layouts, *tensor_outputs):
        if (
            type(layout) is not OwnedBuffer
            or type(layout.source) is not BufferSource
            or type(layout.source.name) is not str
            or type(layout.dtype) is not torch.dtype
            or type(layout.size) is not tuple
            or type(layout.stride) is not tuple
            or len(layout.size) != len(layout.stride)
        ):
            return None
        if records.version != 4:
            if any(
                type(value) is not int or value < 0
                for value in (*layout.size, *layout.stride)
            ):
                return None
        else:
            product = pointwise_product(layout.size)
            origins = (
                OrderedSet(pointwise_expression_inputs(product) or ())
                if type(product) is IntExpr
                else OrderedSet()
            )
            if product is None or not origins.issubset(roots):
                return None
            shape_inputs.update(origins)
            stride = (1,) if len(layout.size) == 1 else (layout.size[1], 1)
            if pointwise_product(layout.stride) is None or layout.stride != stride:
                return None
    indices = {}
    for index, layout in enumerate(layouts):
        if layout.source in indices:
            return None
        indices[layout.source] = index
    if records.version == 2:
        return layouts, None
    returned = bind_output_slots(
        records.outputs,
        layouts,
        records.input_names,
        integers,
        symbolic=records.version == 4,
        roots=roots,
    )
    if returned is None:
        return None
    for output in records.outputs:
        if type(output) is IntegerOutput and type(output.value) is IntExpr:
            origins = pointwise_expression_inputs(output.value)
            if origins is None:
                return None
            output_inputs.update(origins)
    if type(records.calls) is not tuple:
        return None
    used = OrderedSet()
    for call in records.calls:
        if type(call) is not KernelCallRecord or type(call.arguments) is not tuple:
            return None
        for argument in call.arguments:
            if type(argument) is not CallArgument:
                return None
            if type(argument.source) is BufferSource:
                if type(argument.source.name) is not str:
                    return None
                used.add(argument.source)
    if used != OrderedSet(indices):
        return None
    if records.version == 4:
        used_input = False
        argument_inputs = OrderedSet()
        for call in records.calls:
            scalars = [
                arg for arg in call.arguments if type(arg.source) is ExpressionSource
            ]
            inputs = [
                arg.source.index
                for arg in call.arguments
                if type(arg.source) is InputSource
            ]
            if call.grid_type not in (
                "Grid1D",
                "FixedGrid",
                "SequentialComboKernelGrid",
            ) or any(
                type(index) is not int
                or not 0 <= index < len(records.input_names)
                or index in integers
                for index in inputs
            ):
                return None
            for scalar in scalars:
                origins = pointwise_expression_inputs(scalar.source.expression)
                if origins is None or not OrderedSet(origins).issubset(roots):
                    return None
                argument_inputs.update(origins)
            if call.grid_type == "FixedGrid" and not call.generated_template:
                grid = bind_fixed_grid(call)
                if (
                    grid is None
                    or len(roots) != 1
                    or len(scalars) != 1
                    or OrderedSet(
                        pointwise_expression_inputs(scalars[0].source.expression)
                    )
                    != shape_inputs
                ):
                    return None
                for axis in grid:
                    expression = axis.args[0] if axis.op == "ceildiv" else axis
                    if expression.op != "constant" and not OrderedSet(
                        pointwise_expression_inputs(expression) or ()
                    ).issubset(roots):
                        return None
            elif call.generated_template:
                grid = bind_fixed_grid(call)
                if grid is None:
                    return None
                for axis in grid:
                    origins = grid_expression_inputs(axis)
                    if origins is None or not OrderedSet(origins).issubset(roots):
                        return None
                    argument_inputs.update(origins)
            used_input |= bool(inputs)
        if not used_input or roots != shape_inputs | argument_inputs | output_inputs:
            return None
    return layouts, tuple(returned)


def grid_expression_inputs(
    expression: IntExpr,
    *,
    boxed_indices: set[int] | None = None,
    tensor_indices: set[int] | None = None,
) -> tuple[int, ...] | None:
    def visit(value, depth):
        if type(value) is not IntExpr or type(value.args) is not tuple or depth > 16:
            return None
        if value.op == "constant" and type(value.value) is int and not value.args:
            return set() if -(2**63) <= value.value < 2**63 else None
        if (
            value.op in ("boxed", "storage_offset")
            and type(value.value) is int
            and not value.args
        ):
            allowed = boxed_indices if value.op == "boxed" else tensor_indices
            if allowed is not None and value.value not in allowed:
                return None
            return {value.value} if value.value >= 0 else None
        if (
            value.op in ("size", "stride")
            and type(value.value) is int
            and len(value.args) == 1
        ):
            (dimension,) = value.args
            if (
                value.value < 0
                or tensor_indices is not None
                and value.value not in tensor_indices
                or type(dimension) is not IntExpr
                or dimension.op != "constant"
                or dimension.args
                or type(dimension.value) is not int
                or dimension.value < 0
            ):
                return None
            return {value.value}
        binary = (
            "add",
            "multiply",
            "ceildiv",
            "floordiv",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "and",
        )
        arity = 3 if value.op == "select" else 2 if value.op in binary else None
        if value.op in ("call", "pcall"):
            if (
                type(value.value) is not tuple
                or len(value.value) != 2
                or type(value.value[0]) is not int
                or not 0 < value.value[0] < 2**64
                or not callable(value.value[1])
            ):
                return None
        elif value.op in ("max", "min"):
            if value.value is not None or len(value.args) < 2:
                return None
        elif value.value is not None or arity is None or len(value.args) != arity:
            return None
        sources = tuple(visit(arg, depth + 1) for arg in value.args)
        if any(source is None for source in sources):
            return None
        if (
            value.op in ("ceildiv", "floordiv")
            and value.args[1].op == "constant"
            and value.args[1].value <= 0
        ):
            return None
        return set().union(*sources)

    result = visit(expression, 0)
    return None if result is None else tuple(sorted(result))


def bind_fixed_grid(call: KernelCallRecord) -> tuple[IntExpr, IntExpr, IntExpr] | None:
    if (
        type(call) is not KernelCallRecord
        or call.grid_type != "FixedGrid"
        or type(call.generated_template) is not bool
        or type(call.launcher_grid) is not tuple
        or len(call.launcher_grid) != 3
    ):
        return None
    result = []
    for value in call.launcher_grid:
        if type(value) is int:
            if not 0 < value < 2**31:
                return None
            value = IntExpr("constant", value)
        elif call.generated_template and grid_expression_inputs(value) is not None:
            pass
        elif type(value) is IntExpr and pointwise_expression_inputs(value) is not None:
            pass
        elif (
            type(value) is IntExpr
            and value.op == "ceildiv"
            and value.value is None
            and type(value.args) is tuple
            and len(value.args) == 2
        ):
            numerator, divisor = value.args
            if (
                pointwise_expression_inputs(numerator) is None
                or type(divisor) is not IntExpr
                or divisor.op != "constant"
                or type(divisor.value) is not int
                or not 0 < divisor.value < 2**31
                or type(divisor.args) is not tuple
                or divisor.args
            ):
                return None
        else:
            return None
        result.append(value)
    return tuple(result)


def bind_launcher_arguments(
    arguments: tuple[KernelArgument, ...] | None,
    arg_names: list[str],
    def_args: list[str],
    call_args: list[str],
    *,
    grid_args: tuple[str, ...] = (),
) -> tuple[LauncherArgument, ...] | None:
    if type(grid_args) is not tuple:
        return None
    if grid_args:
        if (
            grid_args != FIXED_GRID_ARGUMENTS
            or tuple(def_args[-3:]) != grid_args
            or OrderedSet(grid_args).intersection(arg_names)
        ):
            return None
        def_args = def_args[:-3]
    if (
        type(arguments) is not tuple
        or len(arguments) != len(arg_names)
        or len(OrderedSet(arg_names)) != len(arg_names)
        or len(OrderedSet(def_args)) != len(def_args)
        or any(name not in arg_names for name in def_args)
    ):
        return None
    active = []
    result = []
    for index, (name, argument) in enumerate(zip(arg_names, arguments)):
        if (
            type(argument) is not KernelArgument
            or argument.formal != name
            or type(argument.source_arg_index) is not int
            or argument.source_arg_index != index
            or type(argument.triton_type) is not str
            or type(argument.constant) not in (int, float, bool, str, type(None))
        ):
            return None
        call_index = def_args.index(name) if name in def_args else None
        if argument.abi_index is not None:
            if (
                type(argument.abi_index) is not int
                or argument.abi_index != len(active)
                or argument.constant is not None
                or call_index is None
                or not (
                    argument.triton_type.startswith("*")
                    or argument.triton_type in ("i32", "i64")
                )
            ):
                return None
            active.append(name)
        result.append(
            LauncherArgument(
                name,
                index,
                call_index,
                argument.triton_type,
                argument.abi_index,
                argument.constant,
            )
        )
    if active != call_args:
        return None
    return tuple(result)


def bind_wrapper_arguments(
    call: KernelCallRecord,
    selected: tuple[LauncherArgument, ...] | None,
) -> tuple[CallArgument, ...] | None:
    if type(call) is KernelCallRecord and (
        type(call.generated_template) is not bool
        or call.generated_template
        and call.grid_type != "FixedGrid"
    ):
        return None
    if type(selected) is not tuple or len(selected) != len(call.formals):
        return None
    if len(OrderedSet(call.formals)) != len(call.formals):
        return None
    constants = {}
    fixed_grid = type(call) is KernelCallRecord and call.grid_type == "FixedGrid"
    if fixed_grid:
        if bind_fixed_grid(call) is None or type(call.constexprs) is not tuple:
            return None
        for item in call.constexprs:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or item[0] not in call.formals
                or item[0] in constants
                or type(item[1]) not in (int, float, bool, str, type(None))
            ):
                return None
            constants[item[0]] = item[1]
        if any(type(row) is not LauncherArgument for row in selected) or OrderedSet(
            constants
        ) != OrderedSet([row.formal for row in selected if row.call_arg_index is None]):
            return None
    elif type(call) is KernelCallRecord and (
        call.launcher_grid is not None
        or type(call.constexprs) is not tuple
        or call.constexprs
    ):
        return None
    by_source = {}
    for position, arg in enumerate(call.arguments):
        if (
            type(arg) is not CallArgument
            or type(arg.source_arg_index) is not int
            or not 0 <= arg.source_arg_index < len(call.formals)
            or arg.source_arg_index in by_source
            or type(arg.call_arg_index) is not int
            or arg.call_arg_index != position
            or type(arg.triton_type) is not str
            or arg.formal != call.formals[arg.source_arg_index]
        ):
            return None
        by_source[arg.source_arg_index] = arg

    result = []
    for index, (formal, row) in enumerate(zip(call.formals, selected)):
        if (
            type(row) is not LauncherArgument
            or row.formal != formal
            or type(row.source_arg_index) is not int
            or row.source_arg_index != index
        ):
            return None
        arg = by_source.get(index)
        if row.call_arg_index is None:
            if (
                arg is not None
                or row.abi_index is not None
                or row.triton_type != "constexpr"
            ):
                return None
            if fixed_grid and (
                type(row.constant) is not type(constants[formal])
                or row.constant != constants[formal]
            ):
                return None
            continue
        if (
            arg is None
            or type(row.call_arg_index) is not int
            or row.call_arg_index != arg.call_arg_index
            or row.triton_type != arg.triton_type
        ):
            return None
        source = arg.source
        if row.abi_index is None:
            if (
                type(source) is not IntegerSource
                or type(source.value) is not int
                or type(row.constant) is not int
                or row.constant != source.value
                or row.triton_type not in ("constexpr", "i32", "i64")
            ):
                return None
            continue
        if (
            type(row.abi_index) is not int
            or row.abi_index != len(result)
            or row.constant is not None
        ):
            return None
        if type(source) in (InputSource, BufferSource):
            if not row.triton_type.startswith("*"):
                return None
        elif type(source) is IntegerSource:
            if type(source.value) is not int or row.triton_type not in ("i32", "i64"):
                return None
            bits = 32 if row.triton_type == "i32" else 64
            if not -(2 ** (bits - 1)) <= source.value < 2 ** (bits - 1):
                return None
        elif type(source) is ExpressionSource:
            expression = source.expression
            if (
                row.triton_type not in ("i32", "i64")
                or pointwise_expression_inputs(expression) is None
            ):
                return None
        else:
            return None
        result.append(arg)
    return tuple(result)


def bind_grid_recipe(
    call: KernelCallRecord,
    selected: tuple[LauncherArgument, ...] | None,
    recipe: tuple[IntExpr, IntExpr, IntExpr] | None,
) -> tuple[IntExpr, IntExpr, IntExpr] | None:
    if (
        type(recipe) is not tuple
        or len(recipe) != 3
        or bind_wrapper_arguments(call, selected) is None
    ):
        return None
    scalars = {
        arg.formal: arg.source
        for arg in call.arguments
        if type(arg.source) in (IntegerSource, ExpressionSource)
    }

    def bind(expr: IntExpr, depth: int = 0) -> IntExpr | None:
        if type(expr) is not IntExpr or type(expr.args) is not tuple or depth > 8:
            return None
        if expr.op == "constant" and type(expr.value) is int and not expr.args:
            return expr
        if expr.op == "formal" and type(expr.value) is str and not expr.args:
            source = scalars.get(expr.value)
            if type(source) is IntegerSource:
                return IntExpr("constant", source.value)
            if type(source) is ExpressionSource:
                return source.expression
        if expr.op == "ceildiv" and expr.value is None and len(expr.args) == 2:
            numerator, divisor = (bind(arg, depth + 1) for arg in expr.args)
            if (
                numerator is not None
                and divisor is not None
                and divisor.op == "constant"
                and type(divisor.value) is int
                and divisor.value > 0
            ):
                return IntExpr("ceildiv", args=(numerator, divisor))
        if (
            expr.op in ("add", "multiply")
            and expr.value is None
            and len(expr.args) == 2
        ):
            operands = tuple(bind(arg, depth + 1) for arg in expr.args)
            if all(operand is not None for operand in operands):
                return IntExpr(expr.op, args=operands)
        return None

    bound = tuple(bind(expr) for expr in recipe)
    if any(expr is None for expr in bound):
        return None
    return bound
