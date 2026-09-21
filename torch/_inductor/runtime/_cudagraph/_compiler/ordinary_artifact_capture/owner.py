"""Retain exact compiler metadata during the original ordinary CuTe compilation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from hashlib import sha256
from inspect import getattr_static
from threading import RLock
from types import CodeType, FunctionType, MethodType
from weakref import WeakKeyDictionary

import cutlass.compiler as compiler
from torch._inductor.runtime._cudagraph._sdk import require_active
from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _snapshot as _operation_snapshot, emit_accessors
from torch._inductor.runtime._cudagraph._compiler.compiler_boundary import _snapshot
from torch._inductor.runtime._cudagraph._compiler.compiler_owner import _TaggedCollector
from torch._inductor.runtime._cudagraph._compiler.continuation import _uses
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry
from cutlass._mlir import ir
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.metadata import build_function_metadata
from cutlass.cutlass_dsl.cutlass import CuTeDSL
from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata
from torch._inductor.runtime._cudagraph._compiler.helpers import ScalarRequest, emit_dispatch_helpers
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.logical_metadata import project_metadata
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import check_dispatch_source
from torch._inductor.runtime._cudagraph._compiler import python_entry
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonHostUnsupported
from torch._guards import TracingContext, detect_fake_mode
from torch.fx.experimental.proxy_tensor import get_proxy_mode


_SDK_PREPROCESSOR = getattr_static(BaseDSL, "_preprocess_and_replace_code")
_SDK_PREPROCESSOR_CODE = _SDK_PREPROCESSOR.__func__.__code__
_PREPROCESS_SCOPE: ContextVar[object | None] = ContextVar("cudagraph_cute_preprocessing", default=None)
_CODE_TRANSITIONS: WeakKeyDictionary[FunctionType, tuple[tuple[CodeType, CodeType], ...]] = WeakKeyDictionary()
_CODE_TRANSITIONS_LOCK = RLock()


@contextmanager
def _observe_preprocessing():
    if (getattr_static(BaseDSL, "_preprocess_and_replace_code") is not _SDK_PREPROCESSOR
            or _SDK_PREPROCESSOR.__func__.__code__ is not _SDK_PREPROCESSOR_CODE):
        raise RuntimeError("Ordinary compilation requires the original SDK preprocessor")
    scope = object()

    def preprocess(function):
        if _PREPROCESS_SCOPE.get() is not scope or type(function) is not FunctionType:
            return _SDK_PREPROCESSOR.__func__(function)
        with _CODE_TRANSITIONS_LOCK:
            before = function.__code__
            result = _SDK_PREPROCESSOR.__func__(function)
            after = function.__code__
            if after is not before:
                previous = _CODE_TRANSITIONS.get(function, ())
                _CODE_TRANSITIONS[function] = (*previous, (before, after))
            return result

    token = _PREPROCESS_SCOPE.set(scope)
    BaseDSL._preprocess_and_replace_code = staticmethod(preprocess)
    try:
        yield
    finally:
        BaseDSL._preprocess_and_replace_code = _SDK_PREPROCESSOR
        _PREPROCESS_SCOPE.reset(token)


def _callable_state(value):
    function = value.__func__ if type(value) is MethodType else value
    if type(function) is not FunctionType:
        raise TypeError("Observed CuTe source requires a Python function or bound method")
    receiver = value.__self__ if type(value) is MethodType else None
    return (id(receiver), type(receiver), id(function), function.__code__, id(getattr(function, "__wrapped__", None)),
            tuple((name, id(item)) for name, item in function.__annotations__.items()),
            id(function.__defaults__), tuple(id(item) for item in function.__defaults__ or ()),
            id(function.__kwdefaults__), tuple((name, id(item)) for name, item in (function.__kwdefaults__ or {}).items()),
            tuple(id(cell.cell_contents) for cell in function.__closure__ or ()))


@dataclass(frozen=True)
class _CallableState:
    value: object
    state: tuple

    def check(self):
        function = self.value.__func__ if type(self.value) is MethodType else self.value
        with _CODE_TRANSITIONS_LOCK:
            current = _callable_state(self.value)
            code = self.state[3]
            for before, after in _CODE_TRANSITIONS.get(function, ()):
                if code is before:
                    code = after
            if current[:3] != self.state[:3] or current[4:] != self.state[4:] or current[3] is not code:
                raise RuntimeError("Observed CuTe callable code, defaults or receiver changed")


@dataclass(frozen=True)
class _ReceiverState:
    value: object
    owner_type: type
    methods: tuple

    def check(self):
        if type(self.value) is not self.owner_type or any(
                getattr_static(self.value, name, None) is not method for name, method in self.methods):
            raise RuntimeError("Observed CuTe receiver method bindings changed")


def _source_states(originals):
    pending, seen, receivers, states = list(originals), set(), set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        states.append(_CallableState(value, _callable_state(value)))
        function = value.__func__ if type(value) is MethodType else value
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        pending.extend(item for item in (*(function.__defaults__ or ()),
                                         *(function.__kwdefaults__ or {}).values())
                       if type(item) in (FunctionType, MethodType))
        if type(value) is MethodType and id(value.__self__) not in receivers:
            receiver = value.__self__
            receivers.add(id(receiver))
            methods = tuple((name, method) for name, method in vars(type(receiver)).items()
                            if type(method) is FunctionType)
            states.append(_ReceiverState(receiver, type(receiver), methods))
            pending.extend(method for _, method in methods)
    return tuple(states)


@dataclass(frozen=True)
class CaptureDecline:
    stage: str
    reason: str


class CaptureIneligible(ValueError):
    """Ordinary execution succeeded, but this generation cannot provide replay metadata."""


class _OrdinaryCapture(_TaggedCollector):
    def __init__(self, owner, signature, arguments, keywords):
        super().__init__(owner)
        self.signature = signature
        self.arguments = arguments
        self.keywords = keywords
        self.name = None
        self.function_metadata = None
        self.metadata = None
        self.accessors = ()
        self.helpers = None
        self.original_text = ""
        self.original_bytecode = b""
        self.decline = None

    def __call__(self, owner, module, function_name):
        complete = False
        try:
            with module.context, ir.Location.unknown(), ir.raw_values():
                self.original_text, self.original_bytecode = _snapshot(module)
                original = tuple((view.operation, _operation_snapshot(view.operation))
                                 for view in module.body.operations)
                source = ir.Module.parse(self.original_bytecode)
                if (not module.operation.verify() or not source.operation.verify()
                        or _snapshot(source) != (self.original_text, self.original_bytecode)):
                    raise RuntimeError("Ordinary source snapshot differs from the original compiler module")
                metadata = build_function_metadata(function_name=function_name, signature=self.signature,
                                                   args=self.arguments, kwonlyargs=self.keywords)
                super().__call__(owner, source, function_name)
                stage = "metadata"
                try:
                    host = _function(source, function_name, "func.func")
                    arguments = host.regions[0].blocks[0].arguments
                    project_metadata(metadata, {parameter.name: arguments[parameter.ir_arg_index].type
                                               for parameter in metadata.params
                                               if type(parameter) is compiler.Tensor})
                    metadata_snapshot = snapshot_metadata(metadata)
                    stage = "accessors"
                    accessors = emit_accessors(source, function_name, metadata)
                    stage = "source"
                    admission = check_dispatch_source(source, function_name, metadata_snapshot)
                    prefix = "__cudagraph_dispatch_" + sha256(function_name.encode()).hexdigest()[:16]
                    requests = tuple(ScalarRequest(f"{prefix}_{number}", role, index, value, site)
                                     for number, (site, role, index, value) in enumerate(_uses(admission)))
                    stage = "helpers"
                    helpers = emit_dispatch_helpers(admission, requests)
                except ValueError as error:
                    if (tuple(view.operation for view in module.body.operations)
                            != tuple(operation for operation, _ in original)
                            or _snapshot(module) != (self.original_text, self.original_bytecode)):
                        raise RuntimeError("Declined analysis changed the original compiler module") from error
                    self.decline = CaptureDecline(stage, str(error))
                    self.text, self.bytecode = self.original_text, self.original_bytecode
                    self.name = function_name
                    return
                appended = tuple(view.operation for view in source.body.operations)[len(original):]
                expected = tuple(spec._function for spec in accessors) + tuple(row.operation for row in helpers.helpers)
                if appended != expected:
                    raise RuntimeError("Ordinary source emission appended unexpected compiler operations")
                host = _function(module, function_name, "func.func")
                source_host = _function(source, function_name, "func.func")
                old_attrs = host.attributes.get("arg_attrs")
                inserted = []
                try:
                    host.attributes["arg_attrs"] = source_host.attributes["arg_attrs"]
                    for operation in appended:
                        inserted.append(operation.clone(ip=ir.InsertionPoint(module.body)))
                    if (tuple(view.operation for view in module.body.operations)
                            != tuple(operation for operation, _ in original) + tuple(inserted)
                            or any(_operation_snapshot(operation) != before for operation, before in original
                                   if operation != host)
                            or not module.operation.verify() or _snapshot(module) != _snapshot(source)):
                        raise RuntimeError("Ordinary helper transfer changed the original compiler program")
                except BaseException:
                    try:
                        for operation in reversed(inserted):
                            operation.erase()
                        if old_attrs is None:
                            if "arg_attrs" in host.attributes:
                                del host.attributes["arg_attrs"]
                        else:
                            host.attributes["arg_attrs"] = old_attrs
                        if (tuple(view.operation for view in module.body.operations)
                                != tuple(operation for operation, _ in original)
                                or _snapshot(module) != (self.original_text, self.original_bytecode)):
                            raise RuntimeError("Original module was not restored")
                    except BaseException as error:
                        raise RuntimeError("Failed to restore the original module after capture commit failure") from error
                    raise
                # Ordinary lowering mutates its module; retained SSA owners use the separate snapshot.
                self.module, self.context = source, source.context
                self.text, self.bytecode = _snapshot(source)
                self.function_metadata, self.metadata = metadata, metadata_snapshot
                self.accessors, self.helpers, self.name = accessors, helpers, function_name
                complete = True
        finally:
            self.arguments = ()
            self.keywords = {}
            if not complete:
                self.module = self.context = None
                self.function_metadata = self.metadata = None
                self.accessors, self.helpers = (), None


@dataclass(frozen=True, eq=False)
class OrdinaryCompilation:
    entry_owner: ObservedOrdinaryEntry = field(repr=False)
    selected: object = field(repr=False)
    function_metadata: object = field(repr=False)
    metadata: object
    device: object
    tensor_dtypes: tuple
    source_module: object = field(repr=False)
    source_context: object = field(repr=False)
    source_text: str = field(repr=False)
    source_bytecode: bytes = field(repr=False)
    module: object = field(repr=False)
    context: object = field(repr=False)
    module_text: str = field(repr=False)
    module_bytecode: bytes = field(repr=False)
    function_name: str
    accessors: tuple
    helpers: object = field(repr=False)
    _capture: _OrdinaryCapture = field(repr=False)
    _owners: tuple = field(repr=False)

    def _state(self):
        return (self.entry_owner, self.selected, self.function_metadata, self.metadata, self.device, self.tensor_dtypes,
                self.source_module, self.source_context, self.source_text, self.source_bytecode,
                self.module, self.context, self.module_text, self.module_bytecode,
                self.function_name, self.accessors, self.helpers, self._capture)

    def check(self):
        if len(self._owners) != len(self._state()) or any(
                value is not expected for value, expected in zip(self._state(), self._owners)):
            raise RuntimeError("Ordinary compilation capsule ownership changed")
        owner, capture = self.entry_owner, self._capture
        if type(owner) is not ObservedOrdinaryEntry or type(capture) is not _OrdinaryCapture:
            raise TypeError("Expected the exact ordinary compiler capture owner")
        owner.check()
        if (owner._compilation is not self or owner._capture is not capture
                or owner.selected is not self.selected or self.selected.ir_module is not self.module
                or capture.module is not self.source_module or capture.context is not self.source_context
                or capture.function_metadata is not self.function_metadata or capture.metadata is not self.metadata
                or capture.accessors is not self.accessors or capture.helpers is not self.helpers
                or capture.calls != 1 or capture.arguments or capture.keywords or capture.decline is not None
                or capture.name != self.function_name or self.selected.function_name != self.function_name
                or capture.bytecode != self.source_bytecode or capture.text != self.source_text
                or owner._device != self.device or tuple(owner._tensor_dtypes.items()) != self.tensor_dtypes
                or snapshot_metadata(self.function_metadata) != self.metadata or owner.metadata != self.metadata):
            raise RuntimeError("Ordinary compilation lost its selected code, metadata or source generation")
        for module, context, expected in (
            (self.source_module, self.source_context, (self.source_text, self.source_bytecode)),
            (self.module, self.context, (self.module_text, self.module_bytecode)),
        ):
            if module.context != context:
                raise RuntimeError("Ordinary compilation module context changed")
            with context, ir.raw_values():
                if _snapshot(module) != expected:
                    raise RuntimeError("Ordinary compilation module changed")
        if self.helpers.source.module is not self.source_module:
            raise RuntimeError("Ordinary scalar helper source changed")
        self.helpers.check()
        for spec in self.accessors:
            if spec._source.module is not self.source_module:
                raise RuntimeError("Ordinary component accessor source changed")
            spec.check()


class ObservedOrdinaryEntry(OrdinaryEntry):
    def __init__(self, entry, kernel, *, policy, tensor_policies=None, conversion=None):
        require_active()
        super().__init__(entry, kernel, policy=policy, tensor_policies=tensor_policies, conversion=conversion)
        self._compilation = None

    def _initialize_source(self):
        self._cold = None
        self._originals = (self.entry.target, self.entry.target.__wrapped__, self.kernel, self.kernel.__wrapped__)
        self._cold_states = ()
        self._states = self._snapshot_source()
        self._globals = ()

    def _snapshot_source(self):
        return _source_states(self._originals)

    @contextmanager
    def _compilation_context(self):
        if not self._globals:
            namespaces = {}
            for state in self._states:
                if type(state) is _CallableState:
                    function = state.value.__func__ if type(state.value) is MethodType else state.value
                    namespaces[id(function.__globals__)] = function.__globals__
            self._globals = tuple((namespace, dict(namespace)) for namespace in namespaces.values())
        with _observe_preprocessing():
            yield

    def _make_source_capture(self, arguments, keywords):
        dsl = CuTeDSL._get_dsl()
        if (type(dsl) is not CuTeDSL or dsl._trace_finalize_hooks or dsl._scoped_trace_finalize_hooks.get()
                or dsl.envar.keep_ir_clean):
            raise RuntimeError("Observed ordinary compilation requires the standard isolated frontend")
        return _OrdinaryCapture(dsl, self.signature, arguments, keywords)

    def compile(self, *args, **kwargs):
        if (python_entry._ACTIVE.get() is not None or TracingContext.try_get() is not None
                or get_proxy_mode() is not None or detect_fake_mode((args, kwargs)) is not None):
            raise PythonHostUnsupported("Ordinary CuTe compilation requires isolated real arguments")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            self._check()
            _, _, operands, runtime = self._arguments(args, kwargs)
            self._prepare_selected(operands, runtime)
            self._check()
            return self._selected
        finally:
            self._lock.release()

    def compilation(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Ordinary CuTe entry is busy")
        try:
            self._check()
            if self._selected is None or type(self._capture) is not _OrdinaryCapture:
                raise RuntimeError("An observed ordinary call must precede compiler capsule access")
            if self._capture.decline is not None:
                decline = self._capture.decline
                raise CaptureIneligible(f"Ordinary capture declined at {decline.stage}: {decline.reason}") from None
            if self._compilation is None:
                capture, selected = self._capture, self._selected
                module = selected.ir_module
                with module.context, ir.raw_values():
                    text, bytecode = _snapshot(module)
                result = OrdinaryCompilation(self, selected, capture.function_metadata, capture.metadata,
                    self._device, tuple(self._tensor_dtypes.items()),
                    capture.module, capture.context, capture.text, capture.bytecode, module, module.context,
                    text, bytecode, capture.name, capture.accessors, capture.helpers, capture, ())
                self._compilation = replace(result, _owners=result._state())
            result = self._compilation
        finally:
            self._lock.release()
        result.check()
        return result

    def close(self):
        super().close()
        self._compilation = None
