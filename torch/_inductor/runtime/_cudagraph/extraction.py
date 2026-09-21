"""Extract generated host operations in a fresh local symbolic FX context."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from dis import get_instructions
from inspect import CO_VARARGS, CO_VARKEYWORDS
from types import FunctionType, MethodType, SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource, TensorProperty, TensorPropertySource
from torch._guards import detect_fake_mode, TracingContext
from torch._inductor.runtime._cudagraph._compiler.fx_adapter import invocation
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
    FXTrace,
    FXTraceDeclined,
    HostEvent,
    InputContract,
    IntegerRange,
    LayoutEvent,
    NormalizeEvent,
    ReinterpretEvent,
    TensorInput,
)
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import _config as fx_config
from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.overrides import TorchFunctionMode
from torch.utils._sympy.numbers import int_oo

from .address_trace import AddressBinding, SymbolicTensorAddresses, TensorAddressRoots


@dataclass(frozen=True)
class StorageOffsetBinding:
    index: int
    value: torch.SymInt
    source: TensorPropertySource
    symbol: sympy.Symbol


@dataclass(frozen=True)
class ComputedIntegerBinding:
    value: torch.SymInt
    source: LocalSource
    symbol: sympy.Symbol
    arguments: tuple[sympy.Expr, ...]
    address: int
    owner: object
    kind: str
    expected: int


@dataclass(frozen=True)
class TerminalTrace(FXTrace):
    integer_placeholders: tuple[torch.SymInt, ...] = ()
    integer_sources: tuple[LocalSource, ...] = ()
    address_bindings: tuple[AddressBinding, ...] = ()
    tensor_roots: TensorAddressRoots | None = None
    storage_offset_bindings: tuple[StorageOffsetBinding, ...] = ()
    computed_integer_bindings: tuple[ComputedIntegerBinding, ...] = ()


def _materialize(expression, integers):
    if type(expression) is int:
        return expression
    if type(expression) is not IntExpr or type(expression.args) is not tuple:
        raise FXTraceDeclined("Expected an exact compiler layout expression")
    if (
        expression.op == "constant"
        and type(expression.value) is int
        and not expression.args
    ):
        return expression.value
    if (
        expression.op == "boxed"
        and type(expression.value) is int
        and not expression.args
    ):
        if expression.value not in integers:
            raise FXTraceDeclined(
                "Layout expression references an undeclared boxed integer"
            )
        return integers[expression.value]
    if expression.value is None and len(expression.args) == 2:
        left, right = (_materialize(arg, integers) for arg in expression.args)
        if expression.op == "add":
            return left + right
        if expression.op == "multiply":
            return left * right
        if expression.op == "ceildiv" and type(right) is int and right > 0:
            return -((-left) // right)
    raise FXTraceDeclined("Unsupported compiler layout expression")


class _DeviceGuard:
    def __init__(self, state, index):
        self.state, self.index = state, index

    def __enter__(self):
        self.state.check_device(self.index)
        if self.state.in_device:
            raise FXTraceDeclined("Nested device scopes are outside the host contract")
        self.state.in_device = True

    def __exit__(self, *unused):
        self.state.in_device = False


class _InputPinning(TorchFunctionMode):
    def __init__(self, roots, rows):
        super().__init__()
        self.roots = roots
        self.rows = {row.index: row for row in rows}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.is_pinned:
            if len(args) != 1 or kwargs:
                raise FXTraceDeclined(
                    "Pinned input metadata requires the default device query"
                )
            tensor = args[0]
            if tensor.device.type != "cpu":
                return False
            root = self.roots(tensor).root
            if type(root) is InputSource:
                return self.rows[root.index].pinned
        return func(*args, **kwargs)


@dataclass
class _Extraction:
    mode: FakeTensorMode
    device: torch.device
    events: list[HostEvent] = field(default_factory=list)
    placeholders: tuple[object, ...] | None = None
    outputs: tuple[torch.Tensor | int | torch.SymInt | None, ...] | None = None
    in_device: bool = False
    tensor_roots: TensorAddressRoots = field(init=False)
    computed_integer_bindings: list[ComputedIntegerBinding] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.tensor_roots = TensorAddressRoots(self.mode)

    def record(self, event):
        self.tensor_roots.record(event)
        self.events.append(event)

    def check_device(self, index):
        if type(index) is not int or index != self.device.index:
            raise FXTraceDeclined(
                "Device dispatch differs from the supplied compiler contract"
            )

    def raw_stream(self, index):
        self.check_device(index)
        if not self.in_device:
            raise FXTraceDeclined("Stream lookup outside the recorded device scope")
        return self

    def check_tensor(self, tensor):
        if type(tensor) is not FakeTensor or tensor.fake_mode is not self.mode:
            raise FXTraceDeclined(
                "Host operation escaped the fresh local FakeTensorMode"
            )

    def dimensions(self, values):
        if type(values) not in (tuple, torch.Size):
            raise FXTraceDeclined("Expected a tuple of compiler layout dimensions")
        for value in values:
            if type(value) is int:
                continue
            if (
                type(value) is not torch.SymInt
                or value.node.shape_env is not self.mode.shape_env
            ):
                raise FXTraceDeclined("Layout dimension has no local integer source")
        return tuple(values)

    def allocate(self, size, stride, dtype):
        if not self.in_device or type(dtype) is not torch.dtype:
            raise FXTraceDeclined("Expected an allocation on the bound CUDA device")
        size, stride = self.dimensions(size), self.dimensions(stride)
        tensor = torch.ops.aten.empty_strided.default(
            size, stride, dtype=dtype, device=self.device
        )
        self.record(AllocateEvent(tensor, size, stride, dtype, self.device))
        return tensor

    def reinterpret(self, tensor, size, stride, offset):
        self.check_tensor(tensor)
        self.dimensions((offset,))
        size, stride = self.dimensions(size), self.dimensions(stride)
        result = torch.ops.aten.as_strided.default(
            tensor, size, stride, tensor.storage_offset() + offset
        )
        self.record(ReinterpretEvent(tensor, result, size, stride, offset))
        return result

    def assert_layout(self, tensor, size, stride, label=None):
        self.check_tensor(tensor)
        if label is not None and type(label) is not str:
            raise FXTraceDeclined("Expected a literal diagnostic assertion label")
        size, stride = self.dimensions(size), self.dimensions(stride)
        self.record(LayoutEvent(tensor, size, stride, label))
        invocation.assert_layout(tensor, size, stride, label)
        return True

    def assert_layout_grouped(self, items, sizes, strides, label=None):
        if any(type(values) not in (tuple, list) for values in (items, sizes, strides)):
            raise FXTraceDeclined(
                "Expected tuple or list containers for grouped layout assertions"
            )
        if len(items) != len(sizes) or len(items) != len(strides):
            raise FXTraceDeclined("Expected equal numbers of items, sizes, and strides")
        for tensor, size, stride in zip(items, sizes, strides):
            self.assert_layout(tensor, size, stride, label)
        return True

    def normalize(self, tensor):
        self.check_tensor(tensor)
        self.record(NormalizeEvent(tensor))
        invocation.normalize(tensor)
        return tensor


def trace_host(
    wrapper,
    contract,
    example_inputs,
    kernel_names: tuple[str, ...],
    kernel_factory,
    *,
    direct=False,
    context_factory=None,
):
    """Execute supported host operations with symbolic inputs and isolated kernel views.

    Examples supply integer hints; layouts and ranges come from the input contract.
    The caller owns wrapper provenance, selected providers, and replay eligibility.
    """
    if (
        TracingContext.try_get() is not None
        or get_proxy_mode() is not None
        or detect_fake_mode(example_inputs) is not None
    ):
        raise FXTraceDeclined(
            "Local extraction cannot borrow an ambient tracing/fake/proxy context"
        )
    body = wrapper.__func__ if type(wrapper) is MethodType else wrapper
    if (
        type(body) is not FunctionType
        or type(wrapper) not in (FunctionType, MethodType)
        or direct
        and type(wrapper) is MethodType
        or body.__closure__
        or body.__defaults__
        or body.__kwdefaults__
        or body.__code__.co_argcount != (2 if type(wrapper) is MethodType else 1)
        or body.__code__.co_kwonlyargcount
        or body.__code__.co_flags & (CO_VARARGS | CO_VARKEYWORDS)
    ):
        raise FXTraceDeclined("Expected an exact one-box generated function")
    if type(wrapper) is MethodType:
        from .metadata import check_terminal_attachment

        check_terminal_attachment(wrapper, wrapper._cudagraph_terminal_attachment)
    if (
        type(contract) is not InputContract
        or type(contract.kinds) is not tuple
        or not contract.kinds
        or any(kind not in ("integer", "tensor") for kind in contract.kinds)
        or type(contract.device_index) is not int
        or contract.device_index < 0
        or type(example_inputs) not in (list, tuple)
        or len(example_inputs) != len(contract.kinds)
        or type(contract.tensor_inputs) is not tuple
        or type(contract.integer_ranges) is not tuple
    ):
        raise FXTraceDeclined("Expected a complete boxed compiler input contract")
    tensor_indices = {
        index for index, kind in enumerate(contract.kinds) if kind == "tensor"
    }
    integer_indices = {
        index for index, kind in enumerate(contract.kinds) if kind == "integer"
    }
    tensor_types = (torch.Tensor, torch.nn.Parameter)
    if (
        not tensor_indices
        or any(
            type(row) is not TensorInput
            or type(row.index) is not int
            or type(row.dtype) is not torch.dtype
            or type(row.size) is not tuple
            or type(row.stride) is not tuple
            or len(row.size) != len(row.stride)
            or row.device is not None
            and type(row.device) is not torch.device
            or type(row.pinned) is not bool
            or row.pinned
            and (row.device is None or row.device.type != "cpu")
            for row in contract.tensor_inputs
        )
        or {row.index for row in contract.tensor_inputs} != tensor_indices
        or len(contract.tensor_inputs) != len(tensor_indices)
        or any(
            type(row) is not IntegerRange
            or type(row.index) is not int
            or type(row.lower) is not int
            or (
                row.upper is not None
                and (type(row.upper) is not int or row.upper < row.lower)
            )
            for row in contract.integer_ranges
        )
        or {row.index for row in contract.integer_ranges} != integer_indices
        or len(contract.integer_ranges) != len(integer_indices)
        or any(
            type(example_inputs[index]) not in tensor_types for index in tensor_indices
        )
    ):
        raise FXTraceDeclined(
            "Tensor layouts and integer ranges must cover their exact boxed slots"
        )
    if (
        type(kernel_names) is not tuple
        or any(
            type(name) is not str or not name.isidentifier() for name in kernel_names
        )
        or len(set(kernel_names)) != len(kernel_names)
        or (kernel_names and not callable(kernel_factory))
        or (not kernel_names and not (direct and callable(context_factory)))
    ):
        raise FXTraceDeclined("Expected kernel views or a direct invocation context")
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        raise FXTraceDeclined(
            "Ordinary CUDA execution must warm the device before extraction"
        )

    environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    integers, symbol_sources, integer_sources = {}, {}, []
    for row in contract.integer_ranges:
        hint = example_inputs[row.index]
        if (
            type(hint) is not int
            or hint < row.lower
            or (row.upper is not None and hint > row.upper)
        ):
            raise FXTraceDeclined(
                "Integer hint does not satisfy its inherited compiler domain"
            )
        source = LocalSource(f"boxed_{row.index}")
        integer_sources.append(source)
        symbol = environment.create_symbol(
            hint,
            source,
            dynamic_dim=DimDynamic.DYNAMIC,
            positive=None,
            do_not_specialize_zero_one=True,
        )
        if type(symbol) is not sympy.Symbol:
            raise FXTraceDeclined("A declared boxed integer lost its symbolic source")
        environment.constrain_symbol_range(
            symbol, row.lower, int_oo if row.upper is None else row.upper
        )
        integers[row.index] = environment.create_symintnode(
            symbol, hint=hint, source=source
        )
        symbol_sources[symbol] = IntExpr("boxed", row.index)

    input_hints = {
        InputSource(index): example_inputs[index].data_ptr() for index in tensor_indices
    }
    mode = FakeTensorMode(
        shape_env=environment,
        allow_fallback_kernels=False,
        allow_non_fake_inputs=False,
        static_shapes=False,
    )
    mode._allow_unsafe_data_ptr_access = False
    device = torch.device("cuda", contract.device_index)
    flat_inputs = [integers.get(index) for index in range(len(contract.kinds))]
    storage_offsets = []
    with mode:
        for row in contract.tensor_inputs:
            source = TensorPropertySource(
                LocalSource(f"boxed_{row.index}"), TensorProperty.STORAGE_OFFSET
            )
            hint = example_inputs[row.index].storage_offset()
            symbol = environment.create_symbol(
                hint,
                source,
                dynamic_dim=DimDynamic.DYNAMIC,
                positive=None,
                do_not_specialize_zero_one=True,
            )
            if type(symbol) is not sympy.Symbol:
                raise FXTraceDeclined(
                    "Original Tensor storage offset lost its symbolic source"
                )
            environment.constrain_symbol_range(symbol, 0, (1 << 63) - 1)
            offset = environment.create_symintnode(symbol, hint=hint, source=source)
            storage_offsets.append(
                StorageOffsetBinding(row.index, offset, source, symbol)
            )
            size = tuple(_materialize(value, integers) for value in row.size)
            stride = tuple(_materialize(value, integers) for value in row.stride)
            input_device = device if row.device is None else row.device
            layout = torch.ops.aten.empty_strided.default(
                size, stride, dtype=row.dtype, device=input_device
            )
            span = layout.untyped_storage().nbytes() // row.dtype.itemsize
            backing = torch.ops.aten.empty_strided.default(
                (offset + span,), (1,), dtype=row.dtype, device=input_device
            )
            flat_inputs[row.index] = torch.ops.aten.as_strided.default(
                backing,
                size,
                stride,
                offset,
            )

    state = _Extraction(mode, device)
    addresses = SymbolicTensorAddresses(mode, state.tensor_roots, input_hints)
    cuda = SimpleNamespace(
        _DeviceGuard=lambda index: _DeviceGuard(state, index),
        set_device=state.check_device,
    )
    dtype_names = (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    )
    namespace = {
        "__builtins__": {},
        "torch": SimpleNamespace(
            cuda=cuda, **{name: getattr(torch, name) for name in dtype_names}
        ),
        "assert_size_stride": state.assert_layout,
        "assert_size_stride_grouped": state.assert_layout_grouped,
        "copy_if_misaligned": state.normalize,
        "empty_strided_cuda": state.allocate,
        "reinterpret_tensor": state.reinterpret,
        "get_raw_stream": state.raw_stream,
    }
    if not direct and namespace.keys() & set(kernel_names):
        raise FXTraceDeclined(
            "Selected kernel name conflicts with a supported host operation"
        )
    if direct:
        normalize = torch._C._dynamo.guards.copy_if_misaligned
        namespace = {
            name: state.normalize if value is normalize else value
            for name, value in wrapper.__globals__.items()
        }
        state.in_device = True
    namespace.update({name: kernel_factory(state, name) for name in kernel_names})
    if not direct and any(
        op.opname == "LOAD_GLOBAL" and op.argval not in namespace
        for op in get_instructions(wrapper)
    ):
        raise FXTraceDeclined("Generated wrapper references an unsupported host global")
    cloned = FunctionType(body.__code__, namespace, body.__name__)
    if type(wrapper) is MethodType:
        cloned = MethodType(cloned, wrapper.__self__)

    def run_boxed(*args):
        if state.placeholders is not None:
            raise FXTraceDeclined(
                "A host extraction must execute its wrapper exactly once"
            )
        state.placeholders = args
        for row in contract.tensor_inputs:
            state.tensor_roots.bind_input(args[row.index], row.index)
        box = list(args)
        pinning = _InputPinning(state.tensor_roots, contract.tensor_inputs)
        with tensor_events, addresses, pinning:
            for row in contract.tensor_inputs:
                addresses.bind(state.tensor_roots.inputs[row.index])
            outputs = cloned(box)
        if box or type(outputs) is not tuple:
            raise FXTraceDeclined(
                "Generated wrapper must consume its input box and return a tuple"
            )
        for output in outputs:
            if type(output) in (int, torch.SymInt):
                continue
            if output is not None:
                state.check_tensor(output)
        state.outputs = outputs
        return outputs

    # Diagnostic contiguity queries use safe guard_or fallbacks. Explicit host
    # SymBool branches still retain their ordinary ShapeEnv guards.
    tensor_events = nullcontext()
    if direct:
        from .direct_host import HostTensorEvents

        tensor_events = HostTensorEvents(state)
    invocation_context = (
        nullcontext() if context_factory is None else context_factory(state)
    )
    with mode, invocation_context, fx_config.patch(backed_size_oblivious=True):
        graph_module = make_fx(
            run_boxed, tracing_mode="symbolic", _allow_non_fake_inputs=False
        )(*flat_inputs)
    if (
        state.placeholders is None
        or state.outputs is None
        or len(state.placeholders) != len(contract.kinds)
        or graph_module.shape_env is not environment
    ):
        raise FXTraceDeclined(
            "Host extraction escaped the explicit local input contract"
        )
    for index in tensor_indices:
        state.check_tensor(state.placeholders[index])
    return TerminalTrace(
        graph_module,
        contract,
        symbol_sources,
        {},
        environment,
        state.placeholders,
        tuple(state.events),
        state.outputs,
        integer_placeholders=tuple(integers.values()),
        integer_sources=tuple(integer_sources),
        address_bindings=tuple(addresses.bindings.values()),
        tensor_roots=state.tensor_roots,
        storage_offset_bindings=tuple(storage_offsets),
        computed_integer_bindings=tuple(state.computed_integer_bindings),
    )
