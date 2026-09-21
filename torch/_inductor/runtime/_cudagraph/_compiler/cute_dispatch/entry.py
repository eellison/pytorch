"""Ordinary PythonEntry execution and a retained source bridge for later tracing."""

from __future__ import annotations

import inspect
import io
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from threading import Lock
from types import FunctionType, MappingProxyType

import cutlass
import cutlass.compiler as compiler
import cutlass.cute as cute
from cuda.bindings import driver
from cutlass.base_dsl.jit_executor import JitCompiledFunction, JitExecutor
from cutlass.cute.metadata import build_function_metadata
from torch._inductor.runtime._cudagraph._compiler.entry_signature import MetadataSnapshot, ParameterMetadata, SignaturePolicy, _parameter
from torch._inductor.runtime._cudagraph._compiler.launch_events import CloneBundle, _FunctionState
from torch._inductor.runtime._cudagraph._compiler import python_entry
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry, PythonHostUnsupported
import torch
from torch._guards import TracingContext, detect_fake_mode
from torch.fx.experimental.proxy_tensor import get_proxy_mode


@dataclass(frozen=True)
class TensorPolicy:
    shape: tuple[int | None, ...]
    stride_order: tuple[int, ...]

    def __post_init__(self):
        if (type(self.shape) is not tuple or not self.shape
                or any(value is not None and (type(value) is not int or value <= 0) for value in self.shape)
                or type(self.stride_order) is not tuple
                or any(type(axis) is not int for axis in self.stride_order)
                or sorted(self.stride_order) != list(range(len(self.shape)))):
            raise PythonHostUnsupported("Expected a declared compact layout with static or dynamic dimensions")


@dataclass(frozen=True, eq=False)
class OrdinaryCall:
    entry: PythonEntry
    arguments: tuple
    keyword_arguments: tuple
    operands: tuple[tuple[str, object], ...]
    runtime_arguments: tuple
    runtime_keywords: tuple
    selected: JitCompiledFunction
    metadata: MetadataSnapshot
    stream: torch.cuda.Stream


@dataclass(frozen=True, eq=False)
class OrdinaryRun:
    outputs: object
    calls: tuple[OrdinaryCall, ...]
    dependencies: python_entry._Dependencies
    owner: OrdinaryEntry

    def check(self):
        self.dependencies.check()
        self.owner.check()


@dataclass(frozen=True, eq=False)
class _OwnedExecutor:
    host: FunctionType
    selected: JitCompiledFunction
    executor: JitExecutor
    identity: tuple
    metadata: MetadataSnapshot
    device: torch.device
    function_name: str


def _function_states(functions):
    return tuple(_FunctionState(function, function.__code__, getattr(function, "__wrapped__", None),
                                tuple(function.__annotations__.items()),
                                tuple(cell.cell_contents for cell in function.__closure__ or ()))
                 for function in functions)


def _bytecode(module):
    stream = io.BytesIO()
    module.operation.write_bytecode(stream)
    return stream.getvalue()


def _metadata(name, signature, arguments, keywords):
    from cutlass._mlir import ir
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.logical_metadata import project_metadata

    value = build_function_metadata(function_name=name, signature=signature,
                                    args=arguments, kwonlyargs=keywords)
    bound = signature.bind(*arguments, **keywords)
    bound.apply_defaults()
    with ir.Context(), ir.Location.unknown():
        project_metadata(value, {parameter.name: bound.arguments[parameter.name].mlir_type
                                 for parameter in value.params if type(parameter) is compiler.Tensor})
    parameters = tuple(ParameterMetadata("Stream", p.name, p.ir_arg_index, p.abi_arg_index)
                       if type(p) is compiler.Stream else _parameter(p) for p in value.params)
    symbols = tuple((s.name, s.bits, s.divisibility) for s in value.dim_symbol_table)
    return MetadataSnapshot(value.symbol_name, value.display_name, str(value.abi), symbols,
                            parameters, _parameter(value.ret))


class _SourceCapture:
    def __init__(self):
        self.owner = None
        self.name = None
        self.bytecode = None

    def __call__(self, owner, module, function_name):
        if self.bytecode is not None:
            raise PythonHostUnsupported("Expected one ordinary CuTe source compilation")
        hooks = tuple(owner._trace_finalize_hooks) + tuple(owner._scoped_trace_finalize_hooks.get())
        if hooks != (self,):
            raise PythonHostUnsupported("Ordinary compilation requires an isolated source observer")
        self.owner, self.name, self.bytecode = owner, function_name, _bytecode(module)


class _RetainedBundle(CloneBundle):
    def __init__(self, owner, retained):
        super().__init__(**vars(retained))
        self.original_host = owner.entry.target
        self.original_kernel = owner.kernel
        self._owner = owner
        self._retained = retained

    def check_originals(self):
        self._owner._check()
        self._retained.check_originals()
        if (self.original_host is not self._owner.entry.target or self.original_kernel is not self._owner.kernel
                or any(getattr(self, name) is not value for name, value in vars(self._retained).items()
                       if name not in ("original_host", "original_kernel"))):
            raise RuntimeError("The compiler bundle lost its retained pre-warm source authority")


class _RetainedFactory:
    def __init__(self, owner):
        self.owner = owner

    def create(self, host, kernel):
        if host is not self.owner.entry.target or kernel is not self.owner.kernel:
            raise RuntimeError("The retained source factory received another target")
        self.owner._check()
        return _RetainedBundle(self.owner, CloneBundle.create(self.owner._cold.host, self.owner._cold.kernel))


class OrdinaryEntry:
    def __init__(self, entry, kernel, *, policy, tensor_policies=None, conversion=None):
        if (version("nvidia-cutlass-dsl") != "4.6.2"
                or type(entry) is not PythonEntry or entry.config or type(policy) is not SignaturePolicy
                or policy.shape_bits != 32 or policy.stride_bits != 64 or policy.stream_name is None
                or conversion is None and (type(tensor_policies) not in (dict, MappingProxyType)
                    or any(type(name) is not str or type(row) is not TensorPolicy
                           for name, row in tensor_policies.items()))
                or conversion is not None and tensor_policies is not None):
            raise PythonHostUnsupported("Ordinary CuTe requires explicit shape32/stride64 Tensor and stream policies")
        self.entry, self.kernel, self.policy = entry, kernel, policy
        self.tensor_policies = None if tensor_policies is None else MappingProxyType(dict(tensor_policies))
        self.conversion = None
        if conversion is not None:
            from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.conversion_trace import ConversionFunction

            self.conversion = ConversionFunction(conversion)
        self._initialize_source()
        self.signature = inspect.signature(entry.target.__wrapped__, follow_wrapped=False, eval_str=False)
        self.argument_names = tuple(name for name in self.signature.parameters if name != policy.stream_name)
        self.tensor_names = tuple(name for name in self.argument_names
                                  if self.signature.parameters[name].annotation in (inspect.Parameter.empty, cute.Tensor))
        self.argument_kinds = tuple("tensor" if name in self.tensor_names else "integer" for name in self.argument_names)
        if (self.signature.return_annotation not in (inspect.Signature.empty, None, type(None))
                or any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                       for p in self.signature.parameters.values())
                or policy.stream_name not in self.signature.parameters
                or tensor_policies is not None and set(self.tensor_names) != set(tensor_policies)
                or (self.signature.parameters[policy.stream_name].annotation is not inspect.Parameter.empty
                    and self.signature.parameters[policy.stream_name].annotation is not driver.CUstream)
                or not self.tensor_names
                or any(self.signature.parameters[name].annotation not in (inspect.Parameter.empty, cute.Tensor, cutlass.Int32, cutlass.Int64)
                       for name in self.argument_names)
                or self.conversion is None and self.argument_names != self.tensor_names):
            raise PythonHostUnsupported("The ordinary entry must have declared Tensor, integer and stream formals")
        self._lock = Lock()
        self._identity = (self.entry, self.kernel, self.policy, self.tensor_policies, self.signature,
                          self.tensor_names, self.conversion, self.argument_names, self.argument_kinds)
        self._selected = None
        self._selected_state = None
        self._metadata = None
        self._capture = None
        self._streams = {}
        self._calls = None
        self._device = None
        self._tensor_dtypes = None
        self._closed = False
        self._native_borrows = set()
        self._owned_executor = None

    def _initialize_source(self):
        self._cold = CloneBundle.create(self.entry.target, self.kernel)
        self._originals = (self.entry.target, self.entry.target.__wrapped__, self.kernel, self.kernel.__wrapped__)
        self._cold_states = _function_states((self._cold.host, self._cold.host_body,
                                             self._cold.kernel, self._cold.kernel_body))
        self._states = self._snapshot_source()
        self._globals = tuple((namespace, dict(values)) for namespace, values in self._cold._globals)

    def _snapshot_source(self):
        return _function_states(self._originals)

    @property
    def selected(self):
        return self._selected

    @property
    def metadata(self):
        return self._metadata

    @property
    def source_sha256(self):
        return None if self._capture is None else sha256(self._capture.bytecode).hexdigest()

    @property
    def compiled_sha256(self):
        return None if self._selected_state is None else self._selected_state[-1]

    @property
    def closed(self):
        return self._closed

    def _state(self):
        selected = self._selected
        return (selected, selected.ir_module, selected.engine, selected.capi_func, selected.execution_args,
                selected.jit_module, selected.function_name, selected.execution_args.original_signature,
                sha256(_bytecode(selected.ir_module)).hexdigest())

    def _check(self):
        values = (self.entry, self.kernel, self.policy, self.tensor_policies, self.signature,
                  self.tensor_names, self.conversion, self.argument_names, self.argument_kinds)
        if (self._closed or any(value is not old for value, old in zip(values, self._identity))
                or self.entry.target is not self._originals[0] or self.entry.config):
            raise RuntimeError("Ordinary CuTe entry is closed or its target changed")
        if self.conversion is not None:
            self.conversion.check()
        for state in (*self._cold_states, *self._states):
            state.check()
        for namespace, values in self._globals:
            if any(name not in namespace or namespace[name] is not value for name, value in values.items()):
                raise RuntimeError("Original CuTe source globals changed")
        if self._selected is not None and self._state() != self._selected_state:
            raise RuntimeError("Ordinary selected CuTe code or execution owner changed")

    def check(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            self._check()
        finally:
            self._lock.release()

    def _convert(self, name, tensor, device):
        layout = self.tensor_policies[name]
        if (type(tensor) is not torch.Tensor or tensor.device != device or tensor.layout != torch.strided
                or tensor.requires_grad or tensor.is_conj() or tensor.is_neg()
                or len(tensor.shape) != len(layout.shape)
                or any(type(size) is not int or not 0 < size < 1 << 31
                       or expected is not None and size != expected
                       for size, expected in zip(tensor.shape, layout.shape))
                or tensor.data_ptr() % self.policy.assumed_alignment):
            raise PythonHostUnsupported("Ordinary Tensor violates its declared dtype/device/layout/width policy")
        strides, product = [None] * len(layout.shape), 1
        for axis in reversed(layout.stride_order):
            strides[axis] = product
            product *= tensor.shape[axis]
        if tuple(tensor.stride()) != tuple(strides) or any(value >= 1 << 63 for value in strides):
            raise PythonHostUnsupported("Ordinary Tensor is not compact in the declared stride order")
        value = cute.runtime.from_dlpack(tensor, assumed_align=self.policy.assumed_alignment,
                                         use_32bit_stride=False)
        for axis, size in enumerate(layout.shape):
            if size is None:
                value.mark_compact_shape_dynamic(axis, stride_order=layout.stride_order, divisibility=1)
        if tuple(value.dynamic_shapes_mask) != tuple(int(size is None) for size in layout.shape):
            raise PythonHostUnsupported("DLPack conversion changed the declared dynamic shape policy")
        value.__c_pointers__()
        return value

    def _compilation_context(self):
        return nullcontext()

    def _make_source_capture(self, arguments, keywords):
        return _SourceCapture()

    def _arguments(self, args, kwargs):
        bound = self.signature.bind_partial(*args, **kwargs)
        if self.policy.stream_name in bound.arguments:
            raise PythonHostUnsupported("The current stream must be injected by the ordinary entry")
        bound.apply_defaults()
        if set(bound.arguments) != set(self.argument_names):
            raise PythonHostUnsupported("Ordinary entry arguments do not cover the declared Tensor formals")
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise PythonHostUnsupported("Ordinary CuTe execution cannot run inside CUDA graph capture")
        if self._device is not None and self._device != device:
            raise PythonHostUnsupported("The selected ordinary CuTe callable belongs to another device")
        self._device = device
        operands = tuple(bound.arguments.items())
        if self.conversion is None:
            converted = {name: self._convert(name, tensor, device) for name, tensor in operands}
        else:
            result = self.conversion.convert(tuple(bound.arguments[name] for name in self.argument_names))
            if any(type(tensor) is not torch.Tensor or tensor.device != device or tensor.layout != torch.strided
                   or tensor.requires_grad or tensor.is_conj() or tensor.is_neg() for tensor in result.tensors):
                raise PythonHostUnsupported("User conversion must retain ordinary CUDA Tensor operands")
            operands = tuple(zip(self.argument_names, result.sources, strict=True))
            converted = dict(zip(self.argument_names, result.arguments, strict=True))
            for name, source in operands:
                if name in self.tensor_names:
                    if type(source) is not torch.Tensor:
                        raise PythonHostUnsupported("Converted Tensor formal lost its actual tensor source")
                    converted[name].__c_pointers__()
                elif type(source) is not int or type(converted[name]) not in (int, self.signature.parameters[name].annotation):
                    raise PythonHostUnsupported("Converted integer formal changed its declared scalar type")
        converted[self.policy.stream_name] = driver.CUstream(stream.cuda_stream)
        runtime = inspect.BoundArguments(self.signature, converted)
        return device, stream, operands, runtime

    def _prepare_selected(self, operands, runtime):
        if self._selected is None:
            from cutlass._mlir import ir
            from torch._inductor.runtime._cudagraph._compiler.frontend import _TRACE_LOCK

            capture = self._make_source_capture(runtime.args, runtime.kwargs)
            if ir.Context.current is not None or not _TRACE_LOCK.acquire(blocking=False):
                raise PythonHostUnsupported("Ordinary compilation requires an isolated CuTe frontend context")
            try:
                with self._compilation_context():
                    selected = cute.compile(self.entry.target, *runtime.args, **runtime.kwargs,
                                            trace_finalize_hooks=capture)
            finally:
                _TRACE_LOCK.release()
            if (not isinstance(selected, JitCompiledFunction) or capture.bytecode is None
                    or capture.owner is not self.entry.target.__wrapped__._dsl_object
                    or selected.function_name != capture.name):
                raise PythonHostUnsupported("Ordinary compilation did not return its executable and source observation")
            self._selected, self._capture = selected, capture
            self._metadata = _metadata(capture.name, self.signature, runtime.args, runtime.kwargs)
            self._tensor_dtypes = {name: value.dtype for name, value in operands if name in self.tensor_names}
            self._states = self._snapshot_source()
            self._selected_state = self._state()
        else:
            if _metadata(self._capture.name, self.signature, runtime.args, runtime.kwargs) != self._metadata:
                raise PythonHostUnsupported("Ordinary arguments changed the selected runtime signature")

    def record(self, entry, args, kwargs):
        if entry is not self.entry or self._calls is None or self._calls:
            raise PythonHostUnsupported("Expected one ordinary invocation of the registered PythonEntry")
        device, stream, operands, runtime = self._arguments(args, kwargs)
        self._prepare_selected(operands, runtime)
        self._streams[(device.index, stream.cuda_stream)] = stream
        self._selected(*runtime.args, **runtime.kwargs)
        self._selected_state = self._state()
        self._calls.append(OrdinaryCall(entry, args, tuple(kwargs.items()), operands, tuple(runtime.args),
                                        tuple(runtime.kwargs.items()), self._selected, self._metadata, stream))

    def _run_locked(self, host, inputs):
        self._check()
        dependencies = python_entry._Dependencies(host, (self.entry,))
        dependencies.check()
        self._calls = []
        token = python_entry._ACTIVE.set(self)
        try:
            outputs = host(*inputs)
        finally:
            python_entry._ACTIVE.reset(token)
        if len(self._calls) != 1:
            raise PythonHostUnsupported("The ordinary host must invoke its registered entry exactly once")
        dependencies.check()
        self._check()
        return OrdinaryRun(outputs, tuple(self._calls), dependencies, self)

    def run(self, host, *inputs):
        if (type(host) is not FunctionType or python_entry._ACTIVE.get() is not None
                or TracingContext.try_get() is not None or get_proxy_mode() is not None
                or detect_fake_mode(inputs) is not None):
            raise PythonHostUnsupported("Ordinary CuTe execution requires an isolated real host invocation")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            return self._run_locked(host, inputs)
        finally:
            self._calls = None
            self._lock.release()

    def _invoke_owned(self, host, *inputs):
        if (type(host) is not FunctionType or python_entry._ACTIVE.get() is not None
                or TracingContext.try_get() is not None or get_proxy_mode() is not None
                or detect_fake_mode(inputs) is not None):
            raise PythonHostUnsupported("Ordinary CuTe execution requires an isolated real host invocation")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            prepared = self._owned_executor
            if prepared is None:
                result = self._run_locked(host, inputs)
                if result.outputs is not None:
                    raise PythonHostUnsupported("Owned invocation must return None")
                selected = self._selected
                executor = selected.to(self._device.index)
                if type(executor) is not JitExecutor or executor.jit_module is not selected.jit_module:
                    raise PythonHostUnsupported("Owned invocation requires the selected SDK executor")
                self._check()
                self._owned_executor = _OwnedExecutor(host, selected, executor, self._identity,
                    self._metadata, self._device, self._capture.name)
                return
            values = (self.entry, self.kernel, self.policy, self.tensor_policies, self.signature,
                      self.tensor_names, self.conversion, self.argument_names, self.argument_kinds)
            if (self._closed or host is not prepared.host or self._selected is not prepared.selected
                    or any(value is not expected for value, expected in zip(values, prepared.identity))
                    or self._metadata is not prepared.metadata or self._device != prepared.device):
                raise RuntimeError("Owned ordinary invocation is closed or its binding changed")
            device, stream, _, runtime = self._arguments(inputs, {})
            if _metadata(prepared.function_name, self.signature, runtime.args, runtime.kwargs) != prepared.metadata:
                raise PythonHostUnsupported("Ordinary arguments changed the selected runtime signature")
            self._streams[(device.index, stream.cuda_stream)] = stream
            prepared.executor(*runtime.args, **runtime.kwargs)
        finally:
            self._calls = None
            self._lock.release()

    def _invoke_compiler(self, prepared, plan, source, destination):
        inputs = source, destination
        if (python_entry._ACTIVE.get() is not None or TracingContext.try_get() is not None
                or get_proxy_mode() is not None or detect_fake_mode(inputs) is not None):
            raise PythonHostUnsupported("Ordinary CuTe execution requires an isolated real host invocation")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            values = (self.entry, self.kernel, self.policy, self.tensor_policies, self.signature,
                      self.tensor_names, self.conversion, self.argument_names, self.argument_kinds)
            if (self._closed or self._owned_executor is not prepared or self._selected is not prepared.selected
                    or any(value is not expected for value, expected in zip(values, prepared.identity))
                    or self._metadata is not prepared.metadata or self._device != prepared.device):
                raise RuntimeError("Compiler ordinary invocation is closed or its binding changed")
            device = torch.device("cuda", torch.cuda.current_device())
            stream = torch.cuda.current_stream(device)
            if torch.cuda.is_current_stream_capturing():
                raise PythonHostUnsupported("Ordinary CuTe execution cannot run inside CUDA graph capture")
            if device != prepared.device:
                raise PythonHostUnsupported("The selected ordinary CuTe callable belongs to another device")
            for check in plan.residuals:
                tensor = inputs[check.operand_index]
                value = tensor.size(check.axis) if check.kind == "shape" else tensor.stride(check.axis)
                if not check.minimum <= value <= check.maximum:
                    raise PythonHostUnsupported("Compiler operand exceeds the selected shape or stride width")
            converted = []
            for tensor, row in zip(inputs, plan.operands):
                value = cute.runtime.from_dlpack(tensor, assumed_align=plan.alignment, use_32bit_stride=False)
                if row.dynamic_axis is not None:
                    value.mark_compact_shape_dynamic(row.dynamic_axis, stride_order=row.stride_order, divisibility=1)
                converted.append(value)
            stream_argument = driver.CUstream(stream.cuda_stream)
            arguments = tuple(converted) if plan.stream_keyword else (*converted, stream_argument)
            keywords = {plan.stream_name: stream_argument} if plan.stream_keyword else {}
            self._streams[(device.index, stream.cuda_stream)] = stream
            prepared.executor(*arguments, **keywords)
        finally:
            self._lock.release()

    def join(self, signature, *, arch):
        if self._cold is None:
            raise PythonHostUnsupported("Observed compilation uses its captured ordinary artifact, not a cold clone")
        if self.conversion is not None:
            raise PythonHostUnsupported("User conversion requires the observed ordinary compiler binding")
        from torch._inductor.runtime._cudagraph._compiler.component_owner import ComponentCompilation
        from torch._inductor.runtime._cudagraph._compiler.components import compile_dispatch_components
        from torch._inductor.runtime._cudagraph._compiler.entry import DispatchEntry
        from torch._inductor.runtime._cudagraph._compiler.join_components import JoinedComponents, _bind_properties
        from torch._inductor.runtime._cudagraph._compiler.joined import JoinedInvocation, _compose

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            self._check()
            signature.check()
            if (self._selected is None or signature.call.entry is not self.entry
                    or signature.target is not self.entry.target or signature.policy != self.policy):
                raise PythonHostUnsupported("The traced signature does not belong to the warmed original entry")
            tensors = {row.name: row.tensor for row in signature.operands if row.tensor is not None}
            if set(tensors) != set(self.tensor_policies):
                raise PythonHostUnsupported("Traced Tensor formals differ from the ordinary entry")
            for name, layout in self.tensor_policies.items():
                tensor = tensors[name]
                if (tensor.dtype != self._tensor_dtypes[name] or tensor.device != self._device
                        or len(tensor.shape) != len(layout.shape) or any(
                    use.shape_env is None if expected is None else use.shape_env is not None or use.value != expected
                    for use, expected in zip(tensor.shape, layout.shape)
                )):
                    raise PythonHostUnsupported("Traced dimensions differ from the declared ordinary layout policy")
                strides, product = [None] * len(layout.shape), 1
                for axis in reversed(layout.stride_order):
                    strides[axis] = product
                    product *= tensor.shape[axis].expression
                if tuple(use.expression for use in tensor.strides) != tuple(strides):
                    raise PythonHostUnsupported("Traced strides differ from the declared ordinary layout policy")
            # Preserve the accepted compiler body and inject only its cold-source factory.
            namespace = dict(compile_dispatch_components.__globals__)
            namespace["CloneBundle"] = _RetainedFactory(self)
            compile_retained = FunctionType(compile_dispatch_components.__code__, namespace,
                                            compile_dispatch_components.__name__,
                                            compile_dispatch_components.__defaults__,
                                            compile_dispatch_components.__closure__)
            compile_retained.__kwdefaults__ = compile_dispatch_components.__kwdefaults__
            components = compile_retained(self.entry.target, self.kernel, *signature.fake_args,
                                           arch=arch, **signature.fake_kwargs)
            dispatch, mapping = components.dispatch, components.plan.mapping
            program = dispatch.program
            if program.bundle.original_host is not self.entry.target or program.bundle.original_kernel is not self.kernel:
                raise RuntimeError("Compiler continuation lost the retained cold source targets")
            formals, flow = dispatch.joined.mapping.formals, dispatch.joined.mapping.flow
            metadata = program.source_metadata[0]
            bound = signature.bind_metadata(metadata)
            invocation = JoinedInvocation(signature, program, bound, formals, flow, _compose(bound, formals, flow),
                                          (signature, program, bound, formals, flow))
            compilation = ComponentCompilation(program, mapping.specs, metadata, (program, mapping.specs, metadata))
            properties = _bind_properties(invocation, mapping)
            joined = JoinedComponents(invocation, compilation, mapping, properties,
                                      (invocation, compilation, mapping, properties))
            result = DispatchEntry(components, joined)
            result.check()
            self._check()
            return result
        finally:
            self._lock.release()

    def _acquire_native_borrow(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            self._check()
            token = object()
            self._native_borrows.add(token)
            return token
        finally:
            self._lock.release()

    def _check_native_borrow(self, token):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            if self._closed or token not in self._native_borrows:
                raise RuntimeError("Ordinary CuTe native borrow changed")
        finally:
            self._lock.release()

    def _release_native_borrow(self, token):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            if token not in self._native_borrows:
                raise RuntimeError("Ordinary CuTe native borrow changed")
            self._native_borrows.remove(token)
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            if self._closed:
                return
            if self._native_borrows:
                raise RuntimeError("Cannot close an ordinary CuTe entry borrowed by a native installation")
            for stream in self._streams.values():
                stream.synchronize()
            self._closed = True
            self._selected = None
            self._selected_state = None
            self._owned_executor = None
            self._streams.clear()
        finally:
            self._lock.release()


@dataclass(frozen=True, eq=False)
class _RetainedSourceGuard:
    original: object
    bundle: _RetainedBundle

    def check(self, signature):
        self.original.check(signature)
        self.bundle.check_originals()
        if self.bundle.original_host is not signature.target:
            raise RuntimeError("Artifact retained source lost its original target association")


def prepare_dispatch_artifact(entry, layout, stream_layout):
    """Create an artifact from the original warm target and its retained cold source."""
    from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime import factory

    from torch._inductor.runtime._cudagraph._compiler.entry import DispatchEntry
    from torch._inductor.runtime._cudagraph._compiler.fields_dispatch import derive_dispatch_fields
    from torch._inductor.runtime._cudagraph._compiler.prepared_dispatch import prepare_dispatch

    if type(entry) is not DispatchEntry:
        raise TypeError("Artifact preparation requires the genuine joined DispatchEntry")
    entry.check()
    prepared = prepare_dispatch(entry.components, layout, stream_layout)
    fields = derive_dispatch_fields(entry.components, layout)
    fields.check()
    dispatch = entry.components.dispatch
    invocation = entry.joined.invocation
    program, signature = dispatch.program, invocation.signature
    if (type(signature) is not factory.EntrySignature or invocation.program is not program
            or program.bundle.original_host is not signature.target or layout.query.host_target != stream_layout.host_target):
        raise ValueError("Artifact preparation requires one Python, compiler, and layout owner")
    formals = factory._copy_formals(entry, layout, stream_layout)
    metadata = dispatch.joined.admission.source.metadata
    symbols = tuple((name, bits, divisibility) for name, bits, divisibility in metadata.symbols)
    for name, bits, divisibility in symbols:
        if (type(name) is not str or type(bits) is not int or bits not in {32, 64}
                or divisibility is not None and (type(divisibility) is not int or divisibility <= 0)):
            raise ValueError("Unsupported compiler dimension symbol")
    for formal in formals:
        for dimensions in (formal.shape, formal.strides):
            for kind, value in dimensions:
                if (type(value) is not int or kind not in {"constant", "symbol"}
                        or kind == "symbol" and not 0 <= value < len(symbols)):
                    raise ValueError("Artifact tensor property lost its compiler dimension identity")
    source_sites = dispatch.joined.sites
    consumers = []
    for index, item in enumerate(prepared.consumers):
        bound = item.bound
        site_ids = [site_id for site_id, site in enumerate(source_sites) if bound.site is site.source]
        if bound.site is not None and len(site_ids) != 1:
            raise ValueError("Artifact helper lost its exact original launch consumer")
        site_id = None if bound.site is None else site_ids[0]
        if (item.numeric.source_order != bound.source_order or item.numeric.result_types != (bound.helper.result_type,)
                or item.numeric.argument_types != tuple(next(formal.llvm_type for formal in formals
                    if formal.source_arg_index == source_index) for source_index in bound.source_order)):
            raise ValueError("Artifact helper lost its actual source tags or scalar result type")
        consumers.append(factory.Consumer(index, bound.helper.symbol, site_id, bound.role, bound.index,
                                  bound.helper.result_type, item.numeric.source_order, item.numeric))
    consumers = tuple(consumers)
    predicates = [item for item in consumers if item.site_id is None]
    if len(predicates) != 1 or (predicates[0].role, predicates[0].index, predicates[0].result_type) != ("predicate", 0, "i1"):
        raise ValueError("Artifact requires its exact original root predicate")
    flow = dispatch.joined.mapping.flow
    binaries = tuple(factory.BinaryImage(image.library_slot, image.global_name, image.sha256, image.data) for image in flow.binaries)
    if (not binaries or len({image.library_slot for image in binaries}) != len(binaries)
            or any(type(image.data) is not bytes or sha256(image.data).hexdigest() != image.sha256 for image in binaries)):
        raise ValueError("Artifact embedded library bytes or identities changed")
    sites = factory._copy_sites(entry, fields, formals, consumers, binaries)
    stream_formal, = (formal for formal in formals if formal.kind == "EnvStream")
    stream = factory.Stream(stream_formal.source_arg_index, stream_formal.llvm_arg_index, stream_formal.operand_index,
                    stream_formal.llvm_type, stream_formal.size, stream_formal.alignment)
    if any(site.stream_source_index != stream.source_index for site in sites):
        raise ValueError("Artifact sites use different original environment streams")
    payload = factory._Payload(factory.ARTIFACT_VERSION, program.function_name, program.arch, layout.query.host_target,
        program.source_sha256, program.compiled_sha256, layout.object_sha256, stream_layout.object_sha256,
        tuple(flow.host_types), symbols, tuple((formal.source_arg_index, formal.operand_index) for formal in formals),
        formals, stream, binaries, consumers, sites)
    factory._immutable(payload)
    bundle = program.bundle
    if type(bundle) is not _RetainedBundle:
        raise TypeError("Ordinary artifact preparation requires its exact retained source bundle")
    bundle.check_originals()
    originals = (bundle.original_host, bundle.original_host.__wrapped__, bundle.original_kernel, bundle.original_kernel.__wrapped__)
    cold = bundle._owner._cold
    retained = (cold.host, cold.host_body, cold.kernel, cold.kernel_body)
    if (any(actual is not old for actual, old in zip(originals, bundle._owner._originals))
            or tuple(item.function for item in bundle._functions) != retained
            or retained[1].__code__ is not bundle.host_code or retained[3].__code__ is not bundle.kernel_code):
        raise ValueError("Artifact code pins lost the original warm target or retained cold compilation source")
    functions = tuple(factory._FunctionGuard(fn, fn.__code__, getattr(fn, "__wrapped__", None), tuple(fn.__annotations__.items()),
                                    tuple(cell.cell_contents for cell in fn.__closure__ or ()))
                      for fn in (*originals, *retained))
    namespaces = tuple((fn.__globals__, tuple(fn.__globals__.items()))
                       for fn in (originals[1], originals[3], retained[1], retained[3]))
    source_guard = _RetainedSourceGuard(factory.prepare_entry_source_guard(signature), bundle)
    guards = factory._LiveGuards(functions, namespaces, (retained[1], retained[3]), tuple(signature.operands), source_guard)
    entry.check()
    prepared.check()
    fields.check()
    factory._check_guards(signature, guards)
    result = object.__new__(factory.DispatchArtifact)
    object.__setattr__(result, "_payload", payload)
    object.__setattr__(result, "_signature", signature)
    object.__setattr__(result, "_guards", guards)
    object.__setattr__(result, "_seal", (result, payload, signature, guards))
    result.check()
    return result
