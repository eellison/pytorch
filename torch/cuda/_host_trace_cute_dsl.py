"""CuTe DSL programs under a host trace, at the DSL's own launch level.

A CuTe DSL kernel reaches the GPU from Python through the callable
cute.compile returns (a JitCompiledFunction: the tvm-ffi form for torch._native's
overrides and the vendored QuACK, the plain form for a user's compiled host),
called with runtime arguments (torch tensors, cute.runtime.from_dlpack values,
integers, a stream). The runtime's registered entry (DirectCuTe over an
ObservedOrdinaryEntry) is one form of that call and _host_trace_cute records
it; this module takes the call at the DSL's level, registered or not, and gives
it the same record:

  - the compile is observed (CompileCallable.__call__, installed once per
    process): the jit callable and the compile-time arguments (fake tensors,
    scalars, the fake stream) are remembered by the compiled function;
  - trace()'s warm-up (the observe phase) meets each compiled function's call:
    an observed ordinary entry is synthesized for it (a generated @cute.jit
    host over the runtime formals, calling the jit callable; a conversion that
    marks the from_dlpack values as the compile-time arguments were), the
    entry's ordinary execution runs the call in place of the compiled function
    (the same kernel source, the same static parameters, under the runtime's
    source observation) and selects its compilation;
  - the symbolic run (the trace phase) meets the call again with traced
    operands (a from_dlpack of a traced tensor is a stand-in carrying it, a
    ReadOnlyTensorWrapper of one a read-only stand-in): it is recorded through
    _host_trace_cute._intercept as a registered invocation is, at its sequence
    position, with the same provenance; a read-only operand is not a written
    root.

Declined by name: a compile the recorder did not observe (before its import,
or a program loaded from a compiled module such as QuACK's on-disk cache), a
compile-time argument the synthesized entry cannot express, a runtime float or
constant that changed since the entry was synthesized, and what the runtime's
source observation or binder declines for the synthesized entry (their reason
quoted).
"""

from __future__ import annotations

import contextlib
import inspect
import linecache
import math
import struct
import sys
import threading
import types
import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.dlpack import ReadOnlyTensorWrapper


_state = threading.local()  # phase: None | "observe" | "trace"
_lock = threading.Lock()
_installed = False
# compiled function -> _Compile: what cute.compile was given for it
_compiles: Any = weakref.WeakKeyDictionary()
# compiled function -> _Synthesized | _Refused
_entries: Any = weakref.WeakKeyDictionary()
# a from_dlpack value made at the warm-up -> (its torch tensor, read-only)
_sources: Any = weakref.WeakKeyDictionary()
_counter = 0
# the last synthesized host's source, for diagnostics
_last_source = ""


def _host_trace() -> Any:
    from torch.cuda import _host_trace

    return _host_trace


@dataclass(frozen=True)
class _Compile:
    function: Any
    args: tuple
    kwargs: dict
    # compiled by the runtime's own observed entry (its trace_finalize_hooks):
    # the registered path records it, the hook leaves its calls alone
    owned: bool = False


@dataclass(frozen=True)
class _Formal:
    """One parameter of the compiled function's signature: a tensor, integer or
    stream formal of the synthesized host, or a value baked into it (a float
    scalar, a constexpr) or dropped (None)."""

    name: str
    kind: str  # "tensor" | "integer" | "stream" | "float" | "constexpr" | "none"
    runtime: bool  # passed at the compiled function's call
    constant: Any = None
    annotation: Any = None


@dataclass(frozen=True)
class _Synthesized:
    adapter: Any
    owner: Any
    formals: tuple
    name: str
    signature: inspect.Signature


@dataclass(frozen=True)
class _Refused:
    reason: str


class _Refusal(Exception):
    pass


class _TracedCuteValue:
    """What from_dlpack returns for a traced tensor under the trace phase: the
    traced tensor and whether it was exported read-only. A consumer other than
    the hooked compiled function's call (a program loaded from a compiled
    module reads it through DLPack) declines by name."""

    __slots__ = ("tensor", "read_only")

    def __init__(self, tensor: Any, read_only: bool) -> None:
        self.tensor, self.read_only = tensor, read_only

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        raise _host_trace().Declined(
            "host_trace: a traced tensor exported through DLPack to a CuTe DSL program the "
            "recorder does not hook (a program loaded from a compiled module, such as QuACK's "
            "on-disk cache, QUACK_CACHE_ENABLED=0 compiles in-process; or a @cute.jit function "
            "called directly, not the callable cute.compile returns): its launch is not "
            "recorded on the tape (declined)"
        )

    __dlpack_device__ = __dlpack__


class _ReadOnlyTraced:
    """ReadOnlyTensorWrapper of a traced tensor under the trace phase."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return _TracedCuteValue(self.tensor, True).__dlpack__()

    __dlpack_device__ = __dlpack__


def install() -> None:
    """The process-wide hooks, once the DSL is imported: cute.compile observed,
    the compiled functions' calls and from_dlpack's _Tensor phase-gated. Idle outside a
    trace's phases. Nothing to install without the DSL."""
    global _installed
    with _lock:
        if _installed:
            return
        if "cutlass" not in sys.modules:
            return
        import importlib

        compiler = importlib.import_module("cutlass.base_dsl.compiler")
        runtime = importlib.import_module("cutlass.cute.runtime")
        executor = importlib.import_module("cutlass.base_dsl.jit_executor")
        provider = importlib.import_module("cutlass.cutlass_dsl.tvm_ffi_provider")
        _installed = True
        compile_call = compiler.CompileCallable.__call__

        def compile_observed(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = compile_call(self, *args, **kwargs)
            if args:
                owned = kwargs.get("trace_finalize_hooks") is not None
                try:
                    _compiles[result] = _Compile(
                        args[0], tuple(args[1:]), dict(kwargs), owned
                    )
                except TypeError:
                    pass
            return result

        compiler.CompileCallable.__call__ = compile_observed
        tensor_init = runtime._Tensor.__init__

        def tensor_init_hooked(
            self: Any,
            tensor: Any,
            assumed_align: Any = None,
            use_32bit_stride: bool = False,
            *,
            enable_tvm_ffi: bool = False,
        ) -> None:
            # cute.runtime.from_dlpack constructs a _Tensor: of a traced tensor
            # under the trace phase, one that carries the traced tensor and no
            # DLPack data (the hooked call reads it, the marks are no-ops); of a
            # real tensor at the warm-up, the ordinary one, its source remembered
            phase = getattr(_state, "phase", None)
            source, read_only = _source_of(tensor) if phase else (None, False)
            if phase == "trace" and isinstance(source, _host_trace()._TracedTensor):
                self._dlpack_data = None
                self._dltensor_wrapper = None
                self._assumed_align = assumed_align
                self._is_dynamic = False
                self._memref_desc = None
                self._dtype = None
                self._use_32bit_stride = use_32bit_stride
                self._c_pointers_cache = None
                self._host_trace_value = _TracedCuteValue(source, read_only)
                return
            tensor_init(
                self,
                tensor,
                assumed_align,
                use_32bit_stride,
                enable_tvm_ffi=enable_tvm_ffi,
            )
            if phase is not None and source is not None:
                try:
                    _sources[self] = (source, read_only)
                except TypeError:
                    pass

        runtime._Tensor.__init__ = tensor_init_hooked
        for reader in ("__tvm_ffi_object__", "__c_pointers__", "load_dltensor"):
            original_reader = getattr(runtime._Tensor, reader)

            def reader_hooked(
                self: Any, *args: Any, _reader: Any = original_reader, **kwargs: Any
            ) -> Any:
                # a program the hook does not see (one loaded from a compiled
                # module) reading a traced tensor's stand-in declines by name
                carried = getattr(self, "_host_trace_value", None)
                if carried is not None:
                    return carried.__dlpack__()
                return _reader(self, *args, **kwargs)

            setattr(runtime._Tensor, reader, reader_hooked)
        for mark in ("mark_layout_dynamic", "mark_compact_shape_dynamic"):
            original_mark = getattr(runtime._Tensor, mark)

            def mark_hooked(
                self: Any, *args: Any, _mark: Any = original_mark, **kwargs: Any
            ) -> Any:
                if getattr(self, "_host_trace_value", None) is not None:
                    return self
                return _mark(self, *args, **kwargs)

            setattr(runtime._Tensor, mark, mark_hooked)
        classes = [
            executor.JitCompiledFunction,
            provider.TVMFFIJitCompiledFunction,
            provider.TVMFFIJitCompiledFunctionWithKwargs,
        ]
        for cls in classes:
            if "__call__" in vars(cls):
                cls.__call__ = _hooked_call(vars(cls)["__call__"])
        from torch.utils.dlpack import ReadOnlyTensorWrapper

        wrapper_new = ReadOnlyTensorWrapper.__new__

        def wrapper_new_hooked(cls: Any, tensor: Any) -> Any:
            if getattr(_state, "phase", None) == "trace" and isinstance(
                tensor, _host_trace()._TracedTensor
            ):
                return _ReadOnlyTraced(tensor)
            return wrapper_new(cls, tensor)

        ReadOnlyTensorWrapper.__new__ = staticmethod(wrapper_new_hooked)  # type: ignore[assignment]


def _source_of(value: Any) -> tuple:
    """A from_dlpack argument's torch tensor and whether the export is read-only."""
    from torch.utils.dlpack import ReadOnlyTensorWrapper

    if type(value) is _ReadOnlyTraced:
        return value.tensor, True
    if type(value) is ReadOnlyTensorWrapper:
        with torch._C.DisableTorchFunctionSubclass():
            return value.as_subclass(torch.Tensor), True
    if isinstance(value, torch.Tensor):
        return value, False
    return None, False


def _hooked_call(original: Any) -> Any:
    signature = inspect.signature(original)

    def __call__(self: Any, *args: Any, **kwargs: Any) -> Any:
        phase = getattr(_state, "phase", None)
        # an observed entry's own ordinary execution calls its selected program
        # (a registered entry's, or a synthesized one's under `inside`)
        record = _compiles.get(self)
        if (
            phase is None
            or getattr(_state, "inside", False)
            or (record is not None and record.owned)
        ):
            return original(self, *args, **kwargs)
        signature.bind(self, *args, **kwargs)
        if phase == "observe":
            return _observe(self, original, args, kwargs)
        return _trace(self, args, kwargs)

    return __call__


@contextlib.contextmanager
def hooked() -> Any:
    """Installs the hooks if the DSL is imported by now; the phases below do the
    work. Kept for symmetry with the Triton hook's scope."""
    install()
    yield


@contextlib.contextmanager
def observing() -> Any:
    """The warm-up: each compiled function's call on this thread runs through
    its synthesized observed entry (or as written, the reason remembered); the
    names of the programs called are yielded, in order."""
    previous = (getattr(_state, "phase", None), getattr(_state, "observations", None))
    observations: list = []
    _state.phase, _state.observations = "observe", observations
    try:
        yield observations
    finally:
        _state.phase, _state.observations = previous


@contextlib.contextmanager
def tracing(observations: list | None = None) -> Any:
    """The symbolic run: each compiled function's call on this thread is
    recorded on the active trace; the programs met (and the first failure of
    a recording) are yielded. `observations` are the warm-up's program names
    (None without a warm-up)."""
    previous = (
        getattr(_state, "phase", None),
        getattr(_state, "met", None),
        getattr(_state, "observations", None),
    )
    met = _Met()
    _state.phase, _state.met, _state.observations = "trace", met, observations
    try:
        yield met
    finally:
        _state.phase, _state.met, _state.observations = previous


@dataclass(frozen=True)
class _Observed:
    """A compiled program's call at the warm-up: its display name and the module
    of its jit callable (a torch._native override's own, or a user's)."""

    name: str
    module: str


@dataclass
class _Met:
    """The symbolic run's side of the warm-up's observations: the programs met,
    in order, and the first failure of a program's recording after its
    observation was consumed (a decline is a RuntimeError a host may catch;
    the run is then refused at the publication boundary, never published
    without the launch)."""

    names: list = field(default_factory=list)
    failure: BaseException | None = None


def _next_unmet() -> _Observed | None:
    observations = getattr(_state, "observations", None)
    met = getattr(_state, "met", None)
    if getattr(_state, "phase", None) != "trace" or not observations or met is None:
        return None
    n = len(met.names)
    return observations[n] if n < len(observations) else None


def unmet_hint() -> str:
    """For a decline of ATen's route during the symbolic run: the CuTe DSL program
    the warm-up called at this point, if eager's override launched one here."""
    unmet = _next_unmet()
    if unmet is None:
        return ""
    return (
        f"; the warm-up called the CuTe DSL program {unmet.name} here, which eager's "
        "torch._native override launched: the override's condition answered otherwise on the "
        "traced tensors (one analysing with a TensorIterator, which takes no symbolic shapes)"
    )


def claim(programs: tuple) -> None:
    """A torch._native override recorded as a closed region at this point of
    the symbolic run: the programs its warm-up call launched here (those of
    the override's modules, in order) count as met."""
    while True:
        unmet = _next_unmet()
        if unmet is None or not unmet.module.startswith(programs):
            return
        _state.met.names.append(unmet.name)


def check_met(observations: list | None, met: _Met) -> None:
    """The publication boundary. A recording that failed after its observation
    was consumed (the host caught the error and ran on) is refused: the tape
    would omit the launch. A warm-up that called a program the symbolic run
    did not meet (or the reverse) means eager's route and the traced route
    differ: a torch._native override's condition answered otherwise on the
    traced tensors (one that analyses with a TensorIterator, which takes no
    symbolic shapes), so the trace went to ATen's route while eager launched
    the program. Declined."""
    if met.failure is not None:
        raise _host_trace().Declined(
            f"host_trace: a CuTe DSL program's recording failed under the trace: {met.failure}"
        ) from met.failure
    if observations is None:
        return
    names = [o.name for o in observations]
    if names == met.names:
        return
    raise _host_trace().Declined(
        f"host_trace: the warm-up called the CuTe DSL programs {names} and the symbolic "
        f"run met {met}: eager's route differs from the traced one (a torch._native override's "
        "condition answered otherwise on the traced tensors, such as one analysing with a "
        "TensorIterator, which takes no symbolic shapes) (declined)"
    )


def _display_name(compiled: Any, record: _Compile | None) -> str:
    if record is not None:
        function = record.function
        name = getattr(function, "__qualname__", None)
        if name is None:
            name = type(function).__qualname__
        return name
    return str(getattr(compiled, "function_name", type(compiled).__name__))[:80]


def _module_name(record: _Compile | None) -> str:
    if record is None:
        return ""
    function = record.function
    if not isinstance(function, types.FunctionType):
        function = type(function)
    return getattr(function, "__module__", None) or ""


# the modules of torch._native's overrides and the libraries they vendor: their
# programs are recorded through the overrides' closed regions
# (torch/cuda/_host_trace_native.py), never re-selected here
_NATIVE_MODULES = ("torch._native.", "torch._vendor.")


def _observe(compiled: Any, original: Any, args: tuple, kwargs: dict) -> Any:
    record = _compiles.get(compiled)
    module = _module_name(record)
    _state.observations.append(_Observed(_display_name(compiled, record), module))
    if module.startswith(_NATIVE_MODULES):
        return original(compiled, *args, **kwargs)
    synth = _entries.get(compiled)
    if synth is None:
        record = _compiles.get(compiled)
        try:
            if record is None:
                raise _Refusal(
                    "its compile was not observed by the recorder (compiled before "
                    "torch.cuda._host_trace was imported, or loaded from a compiled module: "
                    "QuACK's on-disk cache, QUACK_CACHE_ENABLED=0 compiles in-process); the "
                    "recorder re-selects a program from its jit callable and compile-time arguments"
                )
            synth = _synthesize(compiled, record, args, kwargs)
        except _Refusal as refusal:
            synth = _Refused(str(refusal))
        _remember(compiled, synth)
    if type(synth) is _Refused:
        return original(compiled, *args, **kwargs)
    try:
        arguments, _ = _arguments(synth, args, kwargs, _real_operand)
    except _Refusal as error:
        first = str(error).splitlines()[0] if str(error) else type(error).__name__
        _remember(
            compiled,
            _Refused(
                f"its synthesized observed entry did not run at the warm-up "
                f"({type(error).__name__}: {first})"
            ),
        )
        return original(compiled, *args, **kwargs)
    previous = getattr(_state, "inside", False)
    _state.inside = True
    try:
        synth.adapter(*arguments)
    finally:
        _state.inside = previous
    return None


def _remember(compiled: Any, synth: Any) -> None:
    try:
        _entries[compiled] = synth
    except TypeError:
        pass


def _trace(compiled: Any, args: tuple, kwargs: dict) -> Any:
    ht = _host_trace()
    tr = getattr(ht._active, "trace", None)
    record = _compiles.get(compiled)
    name = _display_name(compiled, record)

    def decline(why: str) -> Any:
        raise ht.Declined(f"host_trace: CuTe DSL kernel {name}: {why} (declined)")

    if tr is None:
        decline("called in the trace phase without a trace on this thread")
    met = _state.met
    met.names.append(name)
    try:
        return _record(tr, compiled, args, kwargs)
    except BaseException as error:
        if met.failure is None:
            met.failure = error
        raise


def _record(tr: Any, compiled: Any, args: tuple, kwargs: dict) -> Any:
    record = _compiles.get(compiled)
    name = _display_name(compiled, record)

    def decline(why: str) -> Any:
        raise _host_trace().Declined(
            f"host_trace: CuTe DSL kernel {name}: {why} (declined)"
        )

    module = _module_name(record)
    if module.startswith(_NATIVE_MODULES):
        decline(
            f"a program of torch._native's override ({module}) reached the symbolic run; an "
            "override is recorded as a closed region when it is on the list "
            "(torch/cuda/_host_trace_native.py), never through its program"
        )
    synth = _entries.get(compiled)
    if synth is None:
        decline(
            "its compiled program was not met at the warm-up; trace(warm_up=True) runs the "
            "call as written first, where the recorder synthesizes the program's observed entry"
        )
    if type(synth) is _Refused:
        decline(synth.reason)
    try:
        arguments, read_only = _arguments(synth, args, kwargs, _traced_operand)
    except _Refusal as refusal:
        decline(str(refusal))
    from torch.cuda import _host_trace_cute

    return _host_trace_cute._intercept(
        tr, synth.adapter, arguments, read_only=read_only, name=synth.name
    )


def _real_operand(value: Any) -> tuple:
    """A runtime argument at the warm-up as the synthesized entry takes it."""
    from torch.utils.dlpack import ReadOnlyTensorWrapper

    known = _sources.get(value) if _weakable(value) else None
    if known is not None:
        return known
    if type(value) is ReadOnlyTensorWrapper:
        with torch._C.DisableTorchFunctionSubclass():
            return value.as_subclass(torch.Tensor), True
    if isinstance(value, torch.Tensor):
        return value, False
    raise _Refusal(
        f"a tensor argument is a {type(value).__name__} whose torch tensor the recorder does "
        "not know (a from_dlpack value made outside the warm-up)"
    )


def _weakable(value: Any) -> bool:
    try:
        weakref.ref(value)
    except TypeError:
        return False
    return True


def _traced_operand(value: Any) -> tuple:
    carried = getattr(value, "_host_trace_value", None)
    if carried is not None:
        value = carried
    elif _weakable(value) and value in _sources:
        # a from_dlpack value of a real tensor made under the trace (a parameter)
        tensor, read_only = _sources[value]
        value = ReadOnlyTensorWrapper(tensor) if read_only else tensor
    if type(value) is _TracedCuteValue:
        return value.tensor, value.read_only
    if type(value) is _ReadOnlyTraced:
        return value.tensor, True
    ht = _host_trace()
    if isinstance(value, ht._TracedTensor):
        return value, False
    read_only = type(value) is ReadOnlyTensorWrapper
    if read_only:
        with torch._C.DisableTorchFunctionSubclass():
            value = value.as_subclass(torch.Tensor)
    if type(value) is torch.Tensor and value.is_cuda:
        # the recorder's regions take inputs and allocations only; a CuTe operand
        # follows (a module traces through functional_call)
        raise _Refusal(
            "a tensor argument is a tensor the trace does not own (captured by the host's "
            "closure); pass it as an argument of the traced call"
        )
    raise _Refusal(
        f"tensor argument is a {type(value).__name__}, not a tensor of the trace"
    )


def _bind_arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise _Refusal(
            f"arguments do not match the compiled signature: {error}"
        ) from error
    bound.apply_defaults()
    return bound.arguments


def _float32_bits(value: Any) -> bytes:
    value = value.value if hasattr(value, "value") else value
    try:
        return struct.pack("=f", float(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise _Refusal(f"value {value!r} is not an f32 runtime scalar") from error


def _arguments(synth: _Synthesized, args: tuple, kwargs: dict, operand: Any) -> tuple:
    """The compiled function's runtime arguments as the synthesized entry's
    torch-level arguments (tensors and integers, in the host's formal order),
    and the names of the read-only tensor formals."""
    values = _bind_arguments(synth.signature, args, kwargs)
    arguments: list = []
    read_only: list = []
    for formal in synth.formals:
        if not formal.runtime:
            continue
        value = values[formal.name]
        if formal.kind == "tensor":
            tensor, ro = operand(value)
            arguments.append(tensor)
            if ro:
                read_only.append(formal.name)
        elif formal.kind == "integer":
            if hasattr(value, "value") and not isinstance(value, (int, torch.SymInt)):
                value = value.value  # a cutlass.Int32(v)
            arguments.append(value)
        elif formal.kind == "stream":
            if int(value) != torch.cuda.current_stream().cuda_stream:
                raise _Refusal(
                    "invoked on a stream other than the trace's capturing stream"
                )
        elif formal.kind in ("float", "constexpr"):
            changed = (
                _float32_bits(value) != _float32_bits(formal.constant)
                if formal.kind == "float"
                else value != formal.constant
            )
            if changed:
                raise _Refusal(
                    f"argument {formal.name} is {value!r} at this call; the entry was synthesized "
                    f"with {formal.constant!r} (a value baked into the compiled program)"
                )
        elif formal.kind == "none":
            if value is not None:
                raise _Refusal(
                    f"argument {formal.name} was None at the compile and is not now"
                )
    return tuple(arguments), frozenset(read_only)


_HOST_SOURCE = """\
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as driver
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.conversion_trace import _FROM_DLPACK as from_dlpack


@cute.jit
def host({host_formals}):
    OP({call_arguments})


def convert({convert_formals}):
    return ({conversions},)
"""


def _synthesize(
    compiled: Any, record: _Compile, args: tuple, kwargs: dict
) -> _Synthesized:
    """The observed ordinary entry that stands for the compiled function: a
    generated @cute.jit host with the runtime formals (tensors as cute.Tensor,
    integers with their compile-time scalar type, the stream), calling the jit
    callable with the compile-time Nones, floats and constexprs baked in; a
    conversion marking each tensor as the compile-time fake tensor was."""
    global _counter, _last_source
    import cutlass
    from cutlass.cute.runtime import _FakeTensor

    from torch._inductor.runtime._cudagraph._compiler.python_entry import (
        PythonHostUnsupported,
    )
    from torch._inductor.runtime._cudagraph._sdk import require_active
    from torch._inductor.runtime._cudagraph.api import (
        DirectCuTe,
        ObservedOrdinaryEntry,
        PythonEntry,
        SignaturePolicy,
    )

    try:
        require_active()
    except RuntimeError as error:
        raise _Refusal(f"the runtime's CuTe SDK is not active ({error})") from error
    signature = getattr(
        getattr(compiled, "execution_args", None), "original_signature", None
    )
    if signature is None:
        raise _Refusal("its compiled program carries no signature")
    names = list(signature.parameters)
    # a jit callable that is an instance (its __call__ the jit function): the
    # DSL's signature keeps self, the compile arguments do not
    if (
        names
        and names[0] == "self"
        and not isinstance(record.function, types.FunctionType)
    ):
        names = names[1:]
    signature = signature.replace(parameters=[signature.parameters[n] for n in names])
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in signature.parameters.values()
    ):
        raise _Refusal("variadic compiled signatures are not supported")
    compile_args = _bind_arguments(
        signature,
        record.args,
        {n: record.kwargs[n] for n in names if n in record.kwargs},
    )
    runtime_signature = compiled.execution_args.signature.replace(
        parameters=[
            p
            for p in compiled.execution_args.signature.parameters.values()
            if p.name in compile_args
            and not (
                type(compile_args[p.name]).__name__ == "_FakeStream"
                and getattr(compile_args[p.name], "use_tvm_ffi_env_stream", False)
            )
        ]
    )
    runtime = _bind_arguments(runtime_signature, args, kwargs)
    runtime_names = runtime_signature.parameters
    formals: list = []
    stream_name = None
    specs: list = []
    for name, value in compile_args.items():
        passed = name in runtime_names
        if isinstance(value, _FakeTensor):
            if not passed:
                raise _Refusal(f"tensor formal {name} is not a runtime argument")
            formals.append(_Formal(name, "tensor", True))
            specs.append((name, value))
        elif type(value).__name__ == "_FakeStream":
            if stream_name is not None:
                raise _Refusal("two stream formals")
            stream_name = name
            # the tvm-ffi environment stream is not an argument of the call
            passed = passed and not getattr(value, "use_tvm_ffi_env_stream", False)
            formals.append(_Formal(name, "stream", passed))
        elif value is None:
            formals.append(_Formal(name, "none", passed))
        elif type(value) in (cutlass.Int32, cutlass.Int64):
            formals.append(
                _Formal(name, "integer", passed, annotation=type(value).__name__)
            )
            if not passed:
                raise _Refusal(f"integer formal {name} is not a runtime argument")
        elif type(value) in (cutlass.Float32,) or type(value) is float:
            if name not in runtime:
                raise _Refusal(f"float formal {name} has no runtime value to bake")
            constant = runtime[name]
            constant = (
                float(constant.value) if hasattr(constant, "value") else float(constant)
            )
            if not math.isfinite(constant):
                raise _Refusal(f"float formal {name} has a non-finite baked constant")
            formals.append(_Formal(name, "float", passed, constant=constant))
        elif type(value) in (int, bool, str):
            constant = runtime.get(name, value)
            formals.append(_Formal(name, "constexpr", passed, constant=constant))
        else:
            raise _Refusal(
                f"compile-time argument {name} is a {type(value).__name__}, which the synthesized "
                "entry does not express"
            )
    if not specs:
        raise _Refusal("its signature has no tensor formal")
    if stream_name is None:
        stream_name = "stream"
        formals.append(_Formal(stream_name, "stream", False))
    host_formals = []
    call_arguments = []
    convert_formals = []
    conversions = []
    alignments = []
    for formal in formals:
        if formal.kind == "tensor":
            host_formals.append(f"{formal.name}: cute.Tensor")
            call_arguments.append(formal.name)
            convert_formals.append(formal.name)
            tensor, _ = _real_operand(runtime[formal.name])
            fake = dict(specs)[formal.name]
            conversions.append(_conversion(formal.name, fake, tensor))
            alignments.append(int(fake._assumed_align or 1))
        elif formal.kind == "integer":
            host_formals.append(f"{formal.name}: cutlass.{formal.annotation}")
            call_arguments.append(formal.name)
            convert_formals.append(formal.name)
            conversions.append(f"cutlass.{formal.annotation}({formal.name})")
        elif formal.kind == "stream":
            if formal.name in signature.parameters:
                call_arguments.append(formal.name)
        elif formal.kind == "float":
            call_arguments.append(f"cutlass.Float32({formal.constant!r})")
        elif formal.kind == "constexpr":
            call_arguments.append(repr(formal.constant))
        else:
            call_arguments.append("None")
        parameter = signature.parameters.get(formal.name)
        if parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            call_arguments[-1] = f"{formal.name}={call_arguments[-1]}"
    # the stream formal last: the runtime's entry injects it by name
    host_formals.append(f"{stream_name}: driver.CUstream")
    with _lock:
        _counter += 1
        tag = _counter
    source = _HOST_SOURCE.format(
        host_formals=", ".join(host_formals),
        call_arguments=", ".join(call_arguments),
        convert_formals=", ".join(convert_formals),
        conversions=", ".join(conversions),
    )
    _last_source = source
    filename = f"<host_trace_cute_dsl:{tag}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(f"torch.cuda._host_trace_cute_dsl_{tag}")
    module.__file__ = filename
    module.OP = record.function  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    # dont_inherit: this module's future annotations must not turn the host's
    # annotations into strings (the runtime's entry reads them as objects)
    exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
    function = record.function
    kernel = (
        function
        if isinstance(function, types.FunctionType)
        else type(function).__call__.__get__(function)
    )
    if not hasattr(kernel, "__wrapped__"):
        raise _Refusal(
            f"its jit callable {type(function).__name__} is not a @cute.jit function"
        )
    alignment = min(alignments)
    alignment = (
        1 << (alignment.bit_length() - 1) if alignment & (alignment - 1) else alignment
    )
    try:
        owner = ObservedOrdinaryEntry(
            PythonEntry(module.host),  # type: ignore[attr-defined]
            kernel,
            policy=SignaturePolicy(32, 64, max(alignment, 1), stream_name),
            conversion=module.convert,  # type: ignore[attr-defined]
        )
        adapter = DirectCuTe(owner)
    except (PythonHostUnsupported, TypeError, ValueError, RuntimeError) as error:
        import traceback

        where = traceback.extract_tb(error.__traceback__)[-1]
        raise _Refusal(
            f"the runtime does not take its synthesized entry ({type(error).__name__}: {error}; "
            f"at {where.filename.rsplit('/', 1)[-1]}:{where.lineno})"
        ) from error
    return _Synthesized(
        adapter,
        owner,
        tuple(formals),
        _display_name(compiled, record),
        runtime_signature,
    )


def _conversion(name: str, fake: Any, tensor: torch.Tensor) -> str:
    """The from_dlpack call and marks reproducing the compile-time fake tensor:
    every axis dynamic -> mark_layout_dynamic at the static unit-stride axis;
    some axes static -> mark_compact_shape_dynamic per dynamic axis with the
    tensor's dim order (the strides then follow the shape, which is the closest
    the marking API comes to a fake tensor's own stride symbols)."""
    shape, stride = tuple(fake.shape), tuple(fake.stride)
    dynamic = [type(s) is not int for s in shape]
    align = int(fake._assumed_align or 1)
    call = f"from_dlpack({name}, assumed_align={align}, use_32bit_stride=False)"
    if all(dynamic):
        unit = [i for i, s in enumerate(stride) if type(s) is int and s == 1]
        leading = f"leading_dim={unit[0]}" if len(unit) == 1 else "leading_dim=None"
        return f"{call}.mark_layout_dynamic({leading})"
    order = tuple(int(d) for d in tensor.dim_order())
    for axis, is_dynamic in enumerate(dynamic):
        if is_dynamic:
            div = int(getattr(shape[axis], "divisibility", 1) or 1)
            call += f".mark_compact_shape_dynamic({axis}, stride_order={order!r}, divisibility={div})"
    return call


install()
