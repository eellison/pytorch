"""Execute the same user CuTe conversion with ordinary and symbolic tensors."""

import struct
from dataclasses import dataclass, field
from dis import get_instructions
from types import CodeType, FunctionType, ModuleType

import cutlass
from cutlass import cute
from torch._inductor.runtime._cudagraph._compiler.entry_signature import _number_state, _tensor_state
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import ReinterpretEvent
import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import get_alias_info, TorchDispatchMode


class ConversionDeclined(ValueError):
    pass


_FROM_DLPACK = cute.runtime.from_dlpack
_RUNTIME = cute.runtime
_TENSOR = _RUNTIME._Tensor
_INTEGERS = (cutlass.Int32, cutlass.Int64)


@dataclass(frozen=True)
class LayoutMark:
    leading_dim: int | None


@dataclass(frozen=True)
class CompactShapeMark:
    mode: int
    stride_order: tuple[int, ...] | None
    divisibility: int


@dataclass(frozen=True, eq=False)
class TensorConversion:
    tensor: torch.Tensor
    assumed_align: int | None
    use_32bit_stride: bool
    enable_tvm_ffi: bool
    force_tf32: bool
    markings: tuple[LayoutMark | CompactShapeMark, ...]

    @property
    def value(self):
        return self.tensor


@dataclass(frozen=True, eq=False)
class ScalarConversion:
    value: int | torch.SymInt
    scalar_type: object = None

    def snapshot(self):
        return self


@dataclass(frozen=True, eq=False)
class OrdinaryConversion:
    arguments: tuple
    sources: tuple

    @property
    def tensors(self):
        return tuple(value for value in self.sources if isinstance(value, torch.Tensor))


@dataclass(frozen=True, eq=False)
class ConversionTrace:
    conversion: object
    operands: tuple
    outputs: tuple[TensorConversion | ScalarConversion, ...]
    events: tuple[ReinterpretEvent, ...]
    _seal: tuple = field(repr=False)

    def _state(self):
        return (id(self.conversion), tuple(_tensor_state(value) if isinstance(value, torch.Tensor) else _number_state(value)
                      for value in self.operands),
                tuple((_tensor_state(row.tensor), row.assumed_align, row.use_32bit_stride,
                       row.enable_tvm_ffi, row.force_tf32, row.markings) if type(row) is TensorConversion
                      else (_number_state(row.value), id(row.scalar_type)) for row in self.outputs),
                tuple((_tensor_state(event.source), _tensor_state(event.tensor),
                       tuple(_number_state(value) for value in (*event.size, *event.stride, event.offset)))
                      for event in self.events))

    def check(self):
        self.conversion.check()
        if self._state() != self._seal:
            raise RuntimeError("CuTe conversion lost its original tensors or recorded options")


class _SymbolicTensor:
    def __init__(self, tensor, assumed_align, use_32bit_stride, enable_tvm_ffi, force_tf32):
        self.tensor = tensor
        self.options = assumed_align, use_32bit_stride, enable_tvm_ffi, force_tf32
        self.markings = []

    def __getattr__(self, name):
        raise ConversionDeclined(f"Unsupported converted CuTe property or method: {name}")

    def mark_layout_dynamic(self, leading_dim=None):
        if leading_dim is not None and (type(leading_dim) is not int or not 0 <= leading_dim < self.tensor.ndim):
            raise ConversionDeclined("CuTe layout marking requires an explicit axis or None")
        self.markings.append(LayoutMark(leading_dim))
        return self

    def mark_compact_shape_dynamic(self, mode, stride_order=None, divisibility=1):
        if (type(mode) is not int or not 0 <= mode < self.tensor.ndim
                or type(divisibility) is not int or divisibility <= 0
                or stride_order is not None and (type(stride_order) is not tuple
                    or any(type(axis) is not int for axis in stride_order))):
            raise ConversionDeclined("CuTe compact marking requires explicit integer arguments")
        self.markings.append(CompactShapeMark(mode, stride_order, divisibility))
        return self

    def snapshot(self):
        return TensorConversion(self.tensor, *self.options, tuple(self.markings))


class _RecordTensorViews(TorchDispatchMode):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.prim.device.default:
            return func(*args, **kwargs)
        if type(func) is not OpOverload or func.namespace != "aten" or func._schema.is_mutable:
            raise ConversionDeclined("Only read-only ATen views have converter host event recording")
        aliases = get_alias_info(func).read_only_alias_match_indexes
        if len(func._schema.returns) != 1 or len(aliases) != 1:
            raise ConversionDeclined("The converter tensor operation is not an unambiguous view")
        index, output_index = aliases[0]
        name = func._schema.arguments[index].name
        source = args[index] if index < len(args) else kwargs.get(name)
        if output_index != 0 or type(source) is not FakeTensor or source.fake_mode is not self.mode:
            raise ConversionDeclined("The converter view lost its original local tensor")
        result = func(*args, **kwargs)
        if (type(result) is not FakeTensor or result.fake_mode is not self.mode
                or result.dtype != source.dtype or result.device != source.device
                or result.layout != torch.strided or source.layout != torch.strided
                or result.is_conj() != source.is_conj() or result.is_neg() != source.is_neg()
                or not torch._C._is_alias_of(source, result)):
            raise ConversionDeclined("The converter view cannot be represented by its original storage and layout")
        self.events.append(ReinterpretEvent(source, result, tuple(result.size()), tuple(result.stride()),
                                            result.storage_offset() - source.storage_offset()))
        return result


def _global_names(code):
    names = {op.argval for op in get_instructions(code) if op.opname == "LOAD_GLOBAL"}
    for value in code.co_consts:
        if type(value) is CodeType:
            names.update(_global_names(value))
    return names


def _literal(value):
    if type(value) is float:
        return float, struct.pack("<d", value)
    if value is None or type(value) in (int, bool, str, bytes):
        return type(value), value
    if type(value) is tuple:
        return tuple(_literal(item) for item in value)
    if any(value is kind for kind in _INTEGERS):
        return type(value), id(value)
    if type(value) is FunctionType or type(value) is ModuleType and value in (cutlass, cute, _RUNTIME, torch):
        return type(value), id(value)
    raise ConversionDeclined("Converter dependencies must be functions, canonical modules or immutable configuration")


def _user_function(value):
    if type(value) is not FunctionType or value is _FROM_DLPACK:
        return False
    module = value.__module__ or ""
    return (module.startswith("torch._inductor.runtime._cudagraph.")
            or not module.startswith(("torch.", "cutlass.")))


def _function_state(function):
    if function.__defaults__ or function.__kwdefaults__:
        raise ConversionDeclined("Converter functions with defaults require explicit alias interception")
    closure = () if function.__closure__ is None else tuple(cell.cell_contents for cell in function.__closure__)
    globals_used = tuple((name, function.__globals__[name]) for name in sorted(_global_names(function.__code__))
                         if name in function.__globals__)
    state = (id(function), function.__code__, _literal(function.__defaults__),
             tuple((name, _literal(value)) for name, value in (function.__kwdefaults__ or {}).items()),
             tuple(_literal(value) for value in closure),
             tuple((name, _literal(value)) for name, value in globals_used))
    return state, (*closure, *(value for _, value in globals_used))


def _dependencies(function):
    pending, seen, states = [function], set(), []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        state, values = _function_state(current)
        states.append(state)
        pending.extend(value for value in values if _user_function(value))
    helpers = (_FROM_DLPACK, _TENSOR.__init__, _TENSOR.mark_layout_dynamic,
               _TENSOR.mark_compact_shape_dynamic, cutlass.Int32.__init__)
    sdk = (id(cute.runtime), id(_RUNTIME.from_dlpack), id(_RUNTIME._Tensor),
           id(cutlass.Int32), id(cutlass.Int64),
           tuple((id(value), value.__code__, _literal(value.__defaults__)) for value in helpers))
    return tuple(states), sdk


class _ModuleView:
    def __init__(self, module, execution):
        self.module, self.execution = module, execution

    def __getattr__(self, name):
        return self.execution.replace(getattr(self.module, name))


class _Execution:
    def __init__(self, from_dlpack, integer):
        self.from_dlpack = from_dlpack
        self.integer = integer
        self.functions = {}
        self.modules = {}

    def replace(self, value):
        if value is _FROM_DLPACK:
            return self.from_dlpack
        if any(value is kind for kind in _INTEGERS):
            return lambda source: self.integer(value, source)
        if type(value) is ModuleType and value in (cutlass, cute, _RUNTIME):
            if value not in self.modules:
                self.modules[value] = _ModuleView(value, self)
            return self.modules[value]
        if _user_function(value):
            return self.clone(value)
        return value

    def clone(self, function):
        if function in self.functions:
            return self.functions[function]
        namespace = dict(function.__globals__)
        closure = None if function.__closure__ is None else tuple(
            type(cell)(self.replace(cell.cell_contents)) for cell in function.__closure__)
        cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, closure)
        cloned.__kwdefaults__ = function.__kwdefaults__
        self.functions[function] = cloned
        for name in _global_names(function.__code__):
            if name in namespace:
                namespace[name] = self.replace(namespace[name])
        return cloned


class ConversionFunction:
    def __init__(self, converter):
        if type(converter) is not FunctionType:
            raise ConversionDeclined("Expected the original user conversion function")
        self.function = converter
        self._function = converter
        self._dependencies = _dependencies(converter)

    def check(self):
        if self.function is not self._function or _dependencies(self.function) != self._dependencies:
            raise RuntimeError("User CuTe conversion code or its bindings changed")

    def _run(self, operands, from_dlpack, integer, converted):
        self.check()
        if type(operands) is not tuple or not operands:
            raise ConversionDeclined("Expected the ordered registered operands")
        outputs = _Execution(from_dlpack, integer).clone(self.function)(*operands)
        self.check()
        if type(outputs) is not tuple or len(outputs) != len(operands):
            raise ConversionDeclined("The user converter must return all ordered CuTe arguments")
        records = []
        for output in outputs:
            matches = [record for value, record in converted if value is output]
            if not matches and type(output) in (int, torch.SymInt):
                integer(None, output)
                matches = [converted[-1][1]]
            if len(matches) != 1:
                raise ConversionDeclined("A converted argument lost its actual tensor or scalar conversion source")
            records.append(matches[0])
        return outputs, tuple(records)

    def convert(self, operands):
        converted = []

        def observe(tensor_dlpack, assumed_align=None, use_32bit_stride=False, *,
                    enable_tvm_ffi=False, force_tf32=False):
            if not isinstance(tensor_dlpack, torch.Tensor) or isinstance(tensor_dlpack, FakeTensor):
                raise ConversionDeclined("Ordinary conversion requires an actual Torch tensor")
            result = _FROM_DLPACK(tensor_dlpack, assumed_align, use_32bit_stride,
                                 enable_tvm_ffi=enable_tvm_ffi, force_tf32=force_tf32)
            converted.append((result, tensor_dlpack))
            return result

        def integer(kind, value):
            if type(value) is not int:
                raise ConversionDeclined("Ordinary CuTe integer conversion requires an actual integer")
            result = value if kind is None else kind(value)
            converted.append((result, value))
            return result

        arguments, sources = self._run(operands, observe, integer, converted)
        return OrdinaryConversion(arguments, sources)

    def trace(self, operands):
        if type(operands) is not tuple or any(type(value) not in (FakeTensor, int, torch.SymInt) for value in operands):
            raise ConversionDeclined("Symbolic conversion requires local tensors and integer expressions")
        tensors = tuple(value for value in operands if type(value) is FakeTensor)
        if not tensors or any(value.fake_mode is not tensors[0].fake_mode for value in tensors):
            raise ConversionDeclined("Symbolic conversion requires the original local FakeTensors")
        converted = []

        def record(tensor_dlpack, assumed_align=None, use_32bit_stride=False, *,
                   enable_tvm_ffi=False, force_tf32=False):
            if (type(tensor_dlpack) is not FakeTensor or tensor_dlpack.fake_mode is not tensors[0].fake_mode
                    or assumed_align is not None and (type(assumed_align) is not int or assumed_align <= 0)
                    or any(type(value) is not bool for value in (use_32bit_stride, enable_tvm_ffi, force_tf32))):
                raise ConversionDeclined("CuTe conversion lost its tensor or explicit conversion options")
            result = _SymbolicTensor(tensor_dlpack, assumed_align, use_32bit_stride, enable_tvm_ffi, force_tf32)
            converted.append((result, result))
            return result

        def integer(kind, value):
            if type(value) not in (int, torch.SymInt):
                raise ConversionDeclined("CuTe integer conversion requires an exact integer expression")
            result = ScalarConversion(value, kind)
            converted.append((result, result))
            return result

        recorder = _RecordTensorViews(tensors[0].fake_mode)
        with recorder:
            _, carriers = self._run(operands, record, integer, converted)
        trace = ConversionTrace(self, operands, tuple(value.snapshot() for value in carriers),
                                tuple(recorder.events), ())
        object.__setattr__(trace, "_seal", trace._state())
        trace.check()
        return trace
