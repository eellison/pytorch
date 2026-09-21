"""Cold compiler-to-SDK conversion proof; not an invocation or replay entry."""

from dataclasses import dataclass, fields
import inspect

import cutlass.cute as cute
from cutlass import Int32
from cutlass.base_dsl.jit_executor import JitCompiledFunction, JitExecutor
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry, TensorPolicy, _metadata, _OwnedExecutor
from torch._inductor.runtime._cudagraph._compiler.entry_signature import _DTYPES, MetadataSnapshot, ParameterMetadata, SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, IntExpr, pointwise_expression_inputs, pointwise_product,
)

from .envelope import BoundMixedEnvelope
from .invocation import _CuTeProvider


class ConversionDeclined(ValueError):
    pass


@dataclass(frozen=True)
class TensorConversion:
    formal: str
    source: BufferSource
    dtype: torch.dtype
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]
    stride_order: tuple[int, ...]
    dynamic_axis: int | None
    shape_mask: tuple[int, ...]
    stride_mask: tuple[int, ...]


@dataclass(frozen=True)
class RangeCheck:
    operand_index: int
    kind: str
    axis: int
    expression: int | IntExpr
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ConversionPlan:
    entry_key: str
    selected_symbol: str
    device_index: int
    alignment: int
    stream_name: str
    stream_keyword: bool
    operands: tuple[TensorConversion, ...]
    residuals: tuple[RangeCheck, ...]


def _data_snapshot(value):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is tuple:
        return kind, tuple(_data_snapshot(item) for item in value)
    if kind in (MetadataSnapshot, ParameterMetadata, SignaturePolicy, TensorPolicy):
        return kind, tuple(_data_snapshot(getattr(value, field.name)) for field in fields(value))
    raise ConversionDeclined("Conversion metadata or policy contains noncanonical data")


def _dimension(value, ranges):
    if type(value) is int:
        return value, value, value
    origins = pointwise_expression_inputs(value)
    if origins is None or any(index not in ranges for index in origins):
        raise ConversionDeclined("Conversion layout lost its compiler integer source")
    if value.op == "boxed":
        row = ranges[value.value]
        return IntExpr("boxed", value.value), row.lower, row.upper
    dimensions = []
    lower, upper = 1, 1
    for arg in value.args:
        arg = arg.value if arg.op == "constant" else arg
        copied, lo, hi = _dimension(arg, ranges)
        dimensions.append(copied)
        lower *= lo
        upper = None if upper is None or hi is None else upper * hi
    return pointwise_product(tuple(dimensions)), lower, upper


def _check_signatures(owned, original, runtime):
    if any(type(signature) is not inspect.Signature for signature in (owned, original, runtime)):
        raise ConversionDeclined("Conversion requires canonical owned and selected signatures")
    expected = tuple(owned.parameters.values())
    for signature in (original, runtime):
        parameters = tuple(signature.parameters.values())
        if (len(parameters) != len(expected)
                or any(type(actual) is not inspect.Parameter or type(formal) is not inspect.Parameter
                       or actual.name != formal.name or actual.kind is not formal.kind
                       or actual.default is not formal.default or actual.annotation is not formal.annotation
                       for actual, formal in zip(parameters, expected))):
            raise ConversionDeclined("Selected original/runtime parameters differ from the owned call contract")
    if (original.return_annotation is not owned.return_annotation
            or runtime.return_annotation is not original.return_annotation and runtime.return_annotation is not Int32):
        raise ConversionDeclined("Selected return annotation differs from the owned call or CUDA status contract")


def build_conversion_plan(compiler, call_index=0):
    if type(compiler) is not BoundMixedEnvelope:
        raise ConversionDeclined("Expected the exact checked mixed compiler binding")
    compiler.check()
    if type(call_index) is not int or not 0 <= call_index < len(compiler.cute.calls):
        raise ConversionDeclined("Conversion requires an exact compiler call index")
    call = compiler.cute.calls[call_index]
    registration = compiler.binding.entries[call_index]
    provider = registration._seal[1]
    if type(provider) is not _CuTeProvider:
        raise ConversionDeclined("Conversion requires the registered CuTe provider")
    owner = provider._seal[0]
    if type(owner) is not ObservedOrdinaryEntry:
        raise ConversionDeclined("Conversion requires the original observed CuTe owner")
    if owner.conversion is not None:
        raise ConversionDeclined("Legacy compiler conversion does not consume user conversion callables")
    prepared = owner._owned_executor
    selected, metadata = owner.selected, owner.metadata
    capture = owner._capture
    if (type(prepared) is not _OwnedExecutor or not isinstance(selected, JitCompiledFunction)
            or type(metadata) is not MetadataSnapshot
            or prepared.selected is not selected or prepared.metadata is not metadata
            or type(prepared.executor) is not JitExecutor or prepared.executor.jit_module is not selected.jit_module
            or prepared.host is not provider._seal[2] or provider._seal[5].__func__ is not OrdinaryEntry._invoke_owned
            or prepared.identity is not owner._identity or prepared.device != owner._device
            or prepared.function_name != selected.function_name or capture is None or capture.name != selected.function_name
            or prepared.executor.jit_module.execution_args is not selected.execution_args
            or prepared.device.index != compiler.inputs.device_index):
        raise ConversionDeclined("Conversion requires the registered selected owned executor")
    signature, policy, policies = owner.signature, owner.policy, owner.tensor_policies
    execution_args = selected.execution_args
    original_signature, runtime_signature = execution_args.original_signature, execution_args.signature
    _check_signatures(signature, original_signature, runtime_signature)
    if (policy.shape_bits != 32 or policy.stride_bits != 64
            or type(policy.assumed_alignment) is not int or not 0 < policy.assumed_alignment <= 16
            or policy.assumed_alignment & (policy.assumed_alignment - 1)
            or tuple(signature.parameters) != (*call.formals, policy.stream_name)
            or set(policies) != set(call.formals)):
        raise ConversionDeclined("Conversion requires exact Tensor/stream formals and supported alignment")
    snapshot = _data_snapshot((metadata, policy, tuple(policies.items())))
    stream = signature.parameters[policy.stream_name]
    if stream.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY):
        raise ConversionDeclined("Unsupported stream parameter placement")
    stream_keyword = stream.kind is inspect.Parameter.KEYWORD_ONLY
    ranges = {row.index: row for row in compiler.inputs.integer_ranges}
    allocations = {row.source: row for row in compiler.records.allocations}
    operands, residuals, fake_arguments = [], [], []
    for index, (formal, fact) in enumerate(zip(call.formals, call.operands)):
        layout = policies[formal]
        if type(layout) is not TensorPolicy or len(layout.shape) != 2:
            raise ConversionDeclined("Conversion requires a declared rank-two Tensor policy")
        TensorPolicy(layout.shape, layout.stride_order)
        if layout.shape.count(None) > 1:
            raise ConversionDeclined("Legacy conversion supports at most one dynamic Tensor axis")
        allocation = allocations.get(fact.source)
        if (type(fact.source) is not BufferSource or allocation is None
                or (fact.dtype, fact.size, fact.stride) != (allocation.dtype, allocation.size, allocation.stride)
                or fact.dtype not in _DTYPES or len(fact.size) != len(layout.shape)
                or any(expected is not None and (type(value) is not int or value != expected)
                       for value, expected in zip(fact.size, layout.shape))):
            raise ConversionDeclined("Conversion operand differs from its compiler allocation or static shape")
        size = tuple(_dimension(value, ranges)[0] for value in fact.size)
        stride = tuple(_dimension(value, ranges)[0] for value in fact.stride)
        expected_stride, stride_mask = [None] * len(size), [0] * len(size)
        product, dynamic = 1, False
        for axis in reversed(layout.stride_order):
            expected_stride[axis], stride_mask[axis] = product, int(dynamic)
            product = size[axis] if product == 1 else pointwise_product((product, size[axis]))
            dynamic |= layout.shape[axis] is None
        if stride != tuple(expected_stride):
            raise ConversionDeclined("Compiler allocation is not compact in the declared stride order")
        shape_mask = tuple(int(value is None) for value in layout.shape)
        dynamic_axis = layout.shape.index(None) if None in layout.shape else None
        for kind, dimensions, bits, minimum in (("shape", size, 32, 1), ("stride", stride, 64, 0)):
            maximum = (1 << (bits - 1)) - 1
            for axis, value in enumerate(dimensions):
                expression, lower, upper = _dimension(value, ranges)
                if lower < minimum or lower > maximum:
                    raise ConversionDeclined("Compiler extent or stride violates the declared integer width")
                if upper is None or upper > maximum:
                    residuals.append(RangeCheck(index, kind, axis, expression, minimum, maximum))
        operands.append(TensorConversion(formal, BufferSource(fact.source.name), fact.dtype,
            size, stride, tuple(layout.stride_order), dynamic_axis, shape_mask, tuple(stride_mask)))
        fake_shape = tuple(cute.sym_int(32, divisibility=1) if dynamic else value
                           for value, dynamic in zip(layout.shape, shape_mask))
        fake_stride = tuple(cute.sym_int(64, divisibility=1) if dynamic else value
                            for value, dynamic in zip(stride, stride_mask))
        if any(type(value) is IntExpr for value in fake_stride):
            raise ConversionDeclined("Static SDK stride still has a compiler symbol")
        fake_arguments.append(cute.runtime.make_fake_tensor(_DTYPES[fact.dtype], fake_shape,
            stride=fake_stride, memspace=cute.AddressSpace.gmem, assumed_align=policy.assumed_alignment))
    fake_stream = cute.runtime.make_fake_stream()
    arguments = tuple(fake_arguments) if stream_keyword else (*fake_arguments, fake_stream)
    keywords = {policy.stream_name: fake_stream} if stream_keyword else {}
    expected = _metadata(selected.function_name, original_signature, arguments, keywords)
    if _data_snapshot(metadata) != _data_snapshot(expected):
        raise ConversionDeclined("Selected SDK metadata differs from the compiler conversion policy")
    plan = ConversionPlan(call.entry_key, selected.function_name, compiler.inputs.device_index,
        policy.assumed_alignment, policy.stream_name, stream_keyword, tuple(operands), tuple(residuals))
    compiler.check()
    if (owner._owned_executor is not prepared or owner.selected is not selected or owner.metadata is not metadata
            or owner.signature is not signature or owner.policy is not policy or owner.tensor_policies is not policies
            or _data_snapshot((metadata, policy, tuple(policies.items()))) != snapshot
            or owner._capture is not capture or capture.name != selected.function_name
            or prepared.function_name != selected.function_name
            or prepared.metadata is not metadata or prepared.selected is not selected
            or prepared.executor.jit_module is not selected.jit_module
            or prepared.executor.jit_module.execution_args is not selected.execution_args
            or selected.execution_args is not execution_args
            or execution_args.original_signature is not original_signature
            or execution_args.signature is not runtime_signature):
        raise ConversionDeclined("Compiler conversion source changed during cold construction")
    _check_signatures(signature, original_signature, runtime_signature)
    return plan
