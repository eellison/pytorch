from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import cutlass
import cutlass.compiler as compiler
import cutlass.cute as cute
from cuda.bindings import driver
import torch
from cutlass.cute.metadata import build_function_metadata
from torch._inductor.runtime._cudagraph._compiler.python_entry import EntryCall, Operand
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node


if TYPE_CHECKING:
    from collections.abc import Mapping
    from torch._inductor.runtime._cudagraph.cute_adapter import _InvocationTrace


SIGNATURE_VERSION = 1
_DTYPES = {
    torch.bool: cutlass.Boolean, torch.uint8: cutlass.Uint8,
    torch.int8: cutlass.Int8, torch.int16: cutlass.Int16,
    torch.int32: cutlass.Int32, torch.int64: cutlass.Int64,
    torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32, torch.float64: cutlass.Float64,
}
_SCALARS = {cutlass.Int32: ("integer", 32), cutlass.Int64: ("integer", 64),
            cutlass.Float32: ("float", 32), cutlass.Float64: ("float", 64)}


@dataclass(frozen=True)
class SignaturePolicy:
    shape_bits: int
    stride_bits: int
    assumed_alignment: int
    stream_name: str | None = None

    def __post_init__(self) -> None:
        if any(type(bits) is not int or bits not in (32, 64) for bits in (self.shape_bits, self.stride_bits)):
            raise ValueError("Shape and stride widths must be explicitly 32 or 64")
        alignment = self.assumed_alignment
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Assumed alignment must be a positive power of two")
        if self.stream_name is not None and (type(self.stream_name) is not str or not self.stream_name):
            raise ValueError("The injected stream requires a formal name")


@dataclass(frozen=True, eq=False)
class ValueUse:
    path: tuple[Any, ...]
    value: Any
    expression: Any
    shape_env: Any
    fx_argument: Any


@dataclass(frozen=True, eq=False)
class TensorSpec:
    value: FakeTensor
    dtype: torch.dtype
    device: torch.device
    shape: tuple[ValueUse, ...]
    strides: tuple[ValueUse, ...]
    storage_offset: ValueUse
    assumed_alignment: int
    required_device: str = "cuda"


@dataclass(frozen=True, eq=False)
class ScalarSpec:
    annotation: Any
    kind: str
    bits: int
    use: ValueUse


@dataclass(frozen=True, eq=False)
class OperandBinding:
    formal_index: int
    name: str
    origin: str
    path: tuple[Any, ...]
    operand: Operand | None
    fx_argument: Any
    tensor: TensorSpec | None = None
    scalar: ScalarSpec | None = None


@dataclass(frozen=True, eq=False)
class SymbolBinding:
    marker: Any
    expression: Any
    shape_env: Any
    bits: int
    uses: tuple[ValueUse, ...]


@dataclass(frozen=True)
class RuntimeRequirement:
    kind: str
    path: tuple[Any, ...]
    minimum: int | None = None
    maximum: int | None = None
    alignment: int | None = None
    divisor: int | None = None


@dataclass(frozen=True)
class ParameterMetadata:
    kind: str
    name: str
    ir_arg_index: int | None
    abi_arg_index: int | None
    dtype: tuple[str, int, int] | None = None
    divisibility: int | None = None
    element_dtype: tuple[str, int, int] | None = None
    shape: tuple[tuple[str, int], ...] = ()
    strides: tuple[tuple[str, int], ...] = ()
    data_address_space: str | None = None
    device_kind: int | None = None
    data_alignment: int | None = None


@dataclass(frozen=True)
class MetadataSnapshot:
    symbol_name: str
    display_name: str
    abi: str
    symbols: tuple[tuple[str, int, int | None], ...]
    params: tuple[ParameterMetadata, ...]
    ret: ParameterMetadata


def _dtype(value: Any) -> tuple[str, int, int]:
    if type(value) is not compiler.DType or type(value.bits) is not int or type(value.lanes) is not int:
        raise ValueError("Unsupported metadata dtype")
    return str(value.code), value.bits, value.lanes


def _dimension(value: Any) -> tuple[str, int]:
    if type(value) is not compiler.Dim:
        raise ValueError("Unsupported metadata dimension")
    if value.is_const is True and value.is_symbol is False and type(value.value) is int:
        return "constant", value.value
    if (value.is_symbol is True and value.is_const is False
            and type(value.value) is compiler.SymbolId and type(value.value.value) is int):
        return "symbol", value.value.value
    raise ValueError("Ambiguous metadata dimension")


def _parameter(value: Any) -> ParameterMetadata:
    if type(value) not in (compiler.Tensor, compiler.Var, compiler.Stream, compiler.EnvStream, compiler.Unit):
        raise ValueError("Unsupported metadata parameter kind")
    if (type(value.name) is not str or any(index is not None and (type(index) is not int or index < 0)
                                         for index in (value.ir_arg_index, value.abi_arg_index))):
        raise ValueError("Invalid metadata name or source index")
    common = (type(value).__name__, value.name, value.ir_arg_index, value.abi_arg_index)
    if type(value) is compiler.Var:
        return ParameterMetadata(*common, dtype=_dtype(value.dtype), divisibility=value.divisibility)
    if type(value) is not compiler.Tensor:
        return ParameterMetadata(*common)
    return ParameterMetadata(
        *common, dtype=_dtype(value.dtype),
        element_dtype=None if value.element_dtype is None else _dtype(value.element_dtype),
        shape=tuple(_dimension(dim) for dim in value.shape),
        strides=tuple(_dimension(dim) for dim in value.strides),
        data_address_space=str(value.data_address_space), device_kind=value.device_kind,
        data_alignment=value.data_alignment,
    )


def snapshot_metadata(metadata: Any) -> MetadataSnapshot:
    if type(metadata) is not compiler.FunctionMetadata:
        raise TypeError("Expected the actual typed function metadata")
    symbols = []
    for symbol in metadata.dim_symbol_table:
        if (type(symbol) is not compiler.DimSymbol or type(symbol.name) is not str
                or type(symbol.bits) is not int or symbol.bits not in (32, 64)
                or symbol.divisibility is not None and type(symbol.divisibility) is not int):
            raise ValueError("Unsupported metadata symbol")
        symbols.append((symbol.name, symbol.bits, symbol.divisibility))
    return MetadataSnapshot(metadata.symbol_name, metadata.display_name, str(metadata.abi), tuple(symbols),
                            tuple(_parameter(param) for param in metadata.params), _parameter(metadata.ret))


def _number_state(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.SymInt):
        return "symbol", id(value.node.shape_env), value.node.expr
    if type(value) is int:
        return "int", value
    if type(value) is float and math.isfinite(value):
        return "float", value.hex()
    raise ValueError("Expected an integer expression or finite literal floating scalar")


def _tensor_state(value: FakeTensor) -> tuple[Any, ...]:
    return (id(value), id(value.fake_mode), value.dtype, value.device, value.layout,
            value.requires_grad, value.is_conj(), value.is_neg(),
            tuple(_number_state(item) for item in value.shape),
            tuple(_number_state(item) for item in value.stride()), _number_state(value.storage_offset()))


def _function_state(function: types.FunctionType) -> tuple[Any, ...]:
    if "__signature__" in vars(function):
        raise ValueError("Custom Python signatures are unsupported")
    return (id(function), function.__code__, id(getattr(function, "__wrapped__", None)),
            function.__name__, tuple((name, id(value)) for name, value in function.__annotations__.items()),
            id(function.__defaults__), tuple(id(value) for value in function.__defaults__ or ()),
            id(function.__kwdefaults__), tuple((name, id(value)) for name, value in (function.__kwdefaults__ or {}).items()))


def _fake_state(value: Any) -> tuple[Any, ...]:
    if type(value) is cute.runtime._FakeTensor:
        return ("tensor", id(value), value.element_type, value.memspace, value._assumed_align,
                value._typed_tensor.assumed_align, value._use_32bit_stride,
                tuple(_fake_state(item) for item in value.shape), tuple(_fake_state(item) for item in value.stride))
    if type(value) is cute.SymInt:
        return "symbol", id(value), value.width, value.divisibility, value.symbol
    if type(value) in _SCALARS:
        return "scalar", id(value), type(value), _number_state(value.value)
    if type(value) is cute.runtime._FakeStream:
        return "stream", id(value), value.use_tvm_ffi_env_stream
    return _number_state(value)


def _use_state(use: ValueUse) -> tuple[Any, ...]:
    return use.path, _number_state(use.value), use.expression, id(use.shape_env), id(use.fx_argument)


def _helper_state() -> tuple[Any, ...]:
    return tuple((value, getattr(value, "__code__", None)) for value in (
        cute.runtime.make_fake_tensor, cute.runtime.make_fake_stream, cute.sym_int, build_function_metadata,
    ))


@dataclass(frozen=True, eq=False)
class BoundParameter:
    source: OperandBinding
    metadata: ParameterMetadata


@dataclass(frozen=True, eq=False)
class BoundSignature:
    signature: EntrySignature
    metadata: MetadataSnapshot
    params: tuple[BoundParameter, ...]


@dataclass(frozen=True, eq=False)
class EntrySignature:
    trace: _InvocationTrace
    call: EntryCall
    target: types.FunctionType
    body: types.FunctionType
    signature: inspect.Signature
    policy: SignaturePolicy
    operands: tuple[OperandBinding, ...]
    symbols: tuple[SymbolBinding, ...]
    requirements: tuple[RuntimeRequirement, ...]
    fake_args: tuple[Any, ...]
    fake_kwargs: Mapping[str, Any]
    _expected: MetadataSnapshot = field(repr=False)
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        records = []
        for record in self.operands:
            tensor = record.tensor
            scalar = record.scalar
            records.append((id(record), record.formal_index, record.name, record.origin, record.path,
                            id(record.operand), id(record.fx_argument),
                            None if tensor is None else (
                                _tensor_state(tensor.value), tensor.dtype, tensor.device,
                                tuple(_use_state(use) for use in (*tensor.shape, *tensor.strides, tensor.storage_offset)),
                                tensor.assumed_alignment, tensor.required_device),
                            None if scalar is None else (scalar.annotation, scalar.kind, scalar.bits, _use_state(scalar.use))))
        call_operands = tuple((id(operand), operand.path, id(operand.fx_argument),
                               _tensor_state(operand.value) if isinstance(operand.value, FakeTensor)
                               else _number_state(operand.value)) for operand in self.call.operands)
        symbols = tuple((id(value), _fake_state(value.marker), value.expression, id(value.shape_env), value.bits,
                         tuple(_use_state(use) for use in value.uses)) for value in self.symbols)
        policy = self.policy.shape_bits, self.policy.stride_bits, self.policy.assumed_alignment, self.policy.stream_name
        requirements = tuple((value.kind, value.path, value.minimum, value.maximum, value.alignment, value.divisor)
                             for value in self.requirements)
        return (id(self.trace), id(self.call), id(self.signature), policy, requirements,
                self.call.entry_index, id(self.call.entry), id(self.call.node), self.call.config,
                _function_state(self.target), _function_state(self.body), tuple(records), call_operands, symbols,
                tuple(_fake_state(value) for value in self.fake_args),
                tuple((name, _fake_state(value)) for name, value in self.fake_kwargs.items()),
                self._expected, _helper_state())

    def check(self) -> None:
        self.trace.check()
        if (not any(call is self.call for call in self.trace.calls) or self.call.target is not self.target
                or self.call.entry.target is not self.target or self._state() != self._seal):
            raise RuntimeError("Entry signature or source association changed")
        expected = snapshot_metadata(build_function_metadata(
            function_name=self.body.__name__, signature=self.signature,
            args=self.fake_args, kwonlyargs=dict(self.fake_kwargs),
        ))
        if expected != self._expected:
            raise RuntimeError("Retained CuTe fake signature changed")

    def bind_metadata(self, metadata: Any) -> BoundSignature:
        self.check()
        actual = snapshot_metadata(metadata)
        if (actual.params != self._expected.params or actual.symbols != self._expected.symbols
                or actual.ret != self._expected.ret):
            raise ValueError("Actual CuTe metadata differs from the exact retained entry signature")
        return BoundSignature(self, actual, tuple(BoundParameter(source, parameter)
                                                for source, parameter in zip(self.operands, actual.params)))


def build_entry_signature(trace: _InvocationTrace, call: EntryCall, *, policy: SignaturePolicy,
                          metadata: MetadataSnapshot | None = None) -> EntrySignature:
    trace.check()
    if (type(policy) is not SignaturePolicy or type(call) is not EntryCall
            or not any(item is call for item in trace.calls) or call.config):
        raise ValueError("Expected an owned EntryCall without compile-time configuration")
    target = call.target
    body = getattr(target, "__wrapped__", target)
    if type(target) is not types.FunctionType or type(body) is not types.FunctionType:
        raise ValueError("Expected the pinned original Python target and body")
    _function_state(target)
    _function_state(body)
    signature = inspect.signature(body, follow_wrapped=False, eval_str=False)
    if signature.return_annotation not in (inspect.Signature.empty, None, type(None)):
        raise ValueError("The initial entry target must return no value")
    if any(parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
           for parameter in signature.parameters.values()):
        raise ValueError("Variadic entry signatures are unsupported")
    if metadata is not None and (type(metadata) is not MetadataSnapshot
            or tuple(parameter.name for parameter in metadata.params) != tuple(signature.parameters)):
        raise ValueError("Logical invocation metadata must cover the exact entry formals")
    by_path = {operand.path: operand for operand in call.operands}
    paths = tuple(("args", index) for index in range(len(call.arguments)))
    paths += tuple(("kwargs", name) for name, _ in call.keyword_arguments)
    if len(by_path) != len(call.operands) or set(by_path) != set(paths):
        raise ValueError("Nested, duplicate, or missing entry operand paths are unsupported")
    for path in paths:
        argument = call.arguments[path[1]] if path[0] == "args" else dict(call.keyword_arguments)[path[1]]
        original = by_path[path].fx_argument
        if original is not argument and (isinstance(original, Node) or _number_state(original) != _number_state(argument)):
            raise RuntimeError("Entry operand lost its exact FX argument")
    positional = tuple(by_path[("args", index)] for index in range(len(call.arguments)))
    keywords = {name: by_path[("kwargs", name)] for name, _ in call.keyword_arguments}
    bound = signature.bind_partial(*positional, **keywords)
    stream_token = object()
    if policy.stream_name is not None:
        if policy.stream_name not in signature.parameters or policy.stream_name in bound.arguments:
            raise ValueError("The injected environment stream must name an unsupplied formal")
        bound.arguments[policy.stream_name] = stream_token
    supplied = set(bound.arguments)
    bound.apply_defaults()
    if set(bound.arguments) != set(signature.parameters):
        raise ValueError("Missing required original target arguments")
    records, requirements, fake_values = [], [], {}
    marker_uses: dict[Any, Any] = {}

    def use(value: Any, path: tuple[Any, ...], fx_argument: Any) -> ValueUse:
        _number_state(value)
        environment = value.node.shape_env if isinstance(value, torch.SymInt) else None
        expression = value.node.expr if environment is not None else value
        if environment is not None and (environment is not trace.shape_env
                                         or not expression.free_symbols.issubset(environment.var_to_range)):
            raise ValueError("Symbolic entry value escaped the local ShapeEnv")
        return ValueUse(path, value, expression, environment, fx_argument)

    def integer(value: ValueUse, bits: int, minimum: int) -> Any:
        maximum = (1 << (bits - 1)) - 1
        if value.shape_env is None:
            if type(value.value) is not int or not minimum <= value.value <= maximum:
                raise ValueError("Static integer is outside the declared signature width/range")
            return value.value
        requirements.append(RuntimeRequirement("integer_range", value.path, minimum, maximum))
        key = (value.expression, bits)
        if key not in marker_uses:
            marker_uses[key] = (cute.sym_int(bits, divisibility=1), [])
        marker, uses = marker_uses[key]
        uses.append(value)
        return marker

    for index, (name, parameter) in enumerate(signature.parameters.items()):
        value = bound.arguments[name]
        if value is stream_token:
            if parameter.annotation is not inspect.Parameter.empty and parameter.annotation is not driver.CUstream:
                raise ValueError("The injected environment stream requires an unannotated or CUstream formal")
            path = ("environment_stream", name)
            records.append(OperandBinding(index, name, "environment_stream", path, None, None))
            fake_values[name] = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            continue
        operand = value if name in supplied else None
        if operand is not None and type(operand) is not Operand:
            raise ValueError("Unsupported supplied entry source")
        path = operand.path if operand is not None else ("default", name)
        source = operand.value if operand is not None else value
        fx_argument = operand.fx_argument if operand is not None else value
        origin = path[0]
        if isinstance(source, FakeTensor):
            if (operand is None or source.fake_mode is not trace.fake_mode or source.layout != torch.strided
                    or source.device.type != "cuda"
                    or source.dtype not in _DTYPES or source.is_conj() or source.is_neg()
                    or parameter.annotation not in (inspect.Parameter.empty, cute.Tensor)):
                raise ValueError("Expected a local CUDA Tensor with supported annotation and layout")
            shape = tuple(use(item, (*path, "shape", axis), fx_argument) for axis, item in enumerate(source.shape))
            strides = tuple(use(item, (*path, "stride", axis), fx_argument) for axis, item in enumerate(source.stride()))
            if metadata is not None:
                logical = metadata.params[index]
                if (logical.kind != "Tensor" or len(logical.shape) != len(shape)
                        or len(logical.strides) != len(strides)):
                    raise ValueError("Logical invocation metadata differs from the tensor formal")
                strides = tuple(use(0, step.path, fx_argument)
                                if size.shape_env is None and type(size.value) is int and size.value == 1
                                and logical.shape[axis] == ("constant", 1)
                                and logical.strides[axis] == ("constant", 0) else step
                                for axis, (size, step) in enumerate(zip(shape, strides, strict=True)))
            offset = use(source.storage_offset(), (*path, "storage_offset"), fx_argument)
            if offset.shape_env is None and offset.value < 0:
                raise ValueError("Negative tensor storage offset is unsupported")
            requirements.extend((RuntimeRequirement("effective_pointer_alignment", path, alignment=policy.assumed_alignment),
                                 RuntimeRequirement("storage_offset_nonnegative", offset.path, minimum=0)))
            fake_values[name] = cute.runtime.make_fake_tensor(
                _DTYPES[source.dtype], tuple(integer(item, policy.shape_bits, 0) for item in shape),
                stride=tuple(integer(item, policy.stride_bits, 0) for item in strides),
                memspace=cute.AddressSpace.gmem, assumed_align=policy.assumed_alignment,
            )
            tensor = TensorSpec(source, source.dtype, source.device, shape, strides, offset, policy.assumed_alignment)
            records.append(OperandBinding(index, name, origin, path, operand, fx_argument, tensor=tensor))
        else:
            if parameter.annotation not in _SCALARS:
                raise ValueError("Scalar parameters require an explicit supported CuTe numeric annotation")
            kind, bits = _SCALARS[parameter.annotation]
            scalar_use = use(source, path, fx_argument)
            if kind == "integer":
                minimum, maximum = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
                if scalar_use.shape_env is None:
                    if type(source) is not int or not minimum <= source <= maximum:
                        raise ValueError("Static integer is outside the declared signature width/range")
                    carrier = source
                else:
                    requirements.append(RuntimeRequirement("integer_range", path, minimum, maximum))
                    # Numeric arguments are reconstructed from incoming SSA; zero supplies only the type.
                    carrier = 0
                fake_values[name] = parameter.annotation(carrier)
            else:
                if type(source) is not float or not math.isfinite(source):
                    raise ValueError("Floating arguments currently require finite literal values")
                fake_values[name] = parameter.annotation(source)
            records.append(OperandBinding(index, name, origin, path, operand, fx_argument,
                                          scalar=ScalarSpec(parameter.annotation, kind, bits, scalar_use)))
    fake_bound = inspect.BoundArguments(signature, fake_values)
    fake_args, fake_kwargs = tuple(fake_bound.args), types.MappingProxyType(dict(fake_bound.kwargs))
    expected = snapshot_metadata(build_function_metadata(
        function_name=body.__name__, signature=signature, args=fake_args, kwonlyargs=dict(fake_kwargs),
    ))
    if len(expected.params) != len(records) or any(parameter.kind not in ("Tensor", "Var", "EnvStream")
                                                  for parameter in expected.params) or expected.ret.kind != "Unit":
        raise ValueError("The typed metadata dropped or changed an entry source")
    symbols = tuple(SymbolBinding(marker, expression, trace.shape_env, bits, tuple(uses))
                    for (expression, bits), (marker, uses) in marker_uses.items())
    result = EntrySignature(trace, call, target, body, signature, policy, tuple(records), symbols,
                            tuple(requirements), fake_args, fake_kwargs, expected, ())
    object.__setattr__(result, "_seal", result._state())
    result.check()
    return result
