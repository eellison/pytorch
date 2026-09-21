"""Live, per-wrapper authority for ordinary CuTe conversion."""

from dataclasses import dataclass
from threading import Lock

from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

from .conversion import build_conversion_plan, ConversionDeclined
from .envelope import BoundMixedEnvelope
from .invocation import _CuTeProvider, InvocationEntry


@dataclass(frozen=True, eq=False)
class _CallSource:
    compiler: object
    function: object
    code: object
    attachment: object
    records: object
    call_index: int
    call: object
    registration: object
    provider: object
    owner: object
    run: object
    run_code: object
    invoke: object
    invoke_code: object


class _CompilerInvocation:
    def __init__(self, compiler, call_index=0):
        if type(compiler) is not BoundMixedEnvelope:
            raise ConversionDeclined("Compiler invocation requires the exact mixed binding")
        compiler.check()
        if type(call_index) is not int or not 0 <= call_index < len(compiler.cute.calls):
            raise ConversionDeclined("Compiler invocation has no exact call index")
        call = compiler.cute.calls[call_index]
        registration = compiler.binding.entries[call_index]
        provider = registration._seal[1]
        if (type(provider) is not _CuTeProvider
                or provider._seal[5].__func__ is not OrdinaryEntry._invoke_owned):
            raise ConversionDeclined("Compiler invocation requires the owned CuTe executor route")
        function, owner = compiler.function(), provider._seal[0]
        if function is None or function.__globals__.get(call.entry_global) is not registration:
            raise ConversionDeclined("Compiler invocation lost its original emitted registration")
        self._source = _CallSource(compiler, compiler.function, function.__code__, compiler.binding.attachment,
            compiler.cute, call_index, call, registration, provider, owner,
            OrdinaryEntry._invoke_compiler.__get__(owner, type(owner)),
            OrdinaryEntry._invoke_compiler.__code__,
            _CompilerInvocation.invoke, _CompilerInvocation.invoke.__code__)
        self._seal = self._source
        self._lock = Lock()
        self._status = "cold"
        self._decline = None
        self._plan = None
        self._prepared = None

    @property
    def status(self):
        return self._status

    @property
    def decline(self):
        return self._decline

    @property
    def plan(self):
        return self._plan

    def check_binding(self, function, call):
        source = self._seal
        if (self._status == "closed" or self._source is not source
                or function is None or source.function() is not function or function.__code__ is not source.code
                or function.__dict__.get("_cute_invocation_attachment") is not source.attachment
                or function.__dict__.get("_cute_invocation_descriptors") is not source.records
                or call is not source.call or type(source.records.calls) is not tuple
                or type(source.call_index) is not int or not 0 <= source.call_index < len(source.records.calls)
                or source.records.calls[source.call_index] is not call
                or function.__globals__.get(call.entry_global) is not self
                or source.registration._seal[1] is not source.provider
                or source.provider._seal[0] is not source.owner
                or source.registration.key != call.entry_key or source.registration.formals != call.formals
                or getattr(self.invoke, "__func__", None) is not source.invoke
                or source.invoke.__code__ is not source.invoke_code
                or source.run.__func__ is not OrdinaryEntry._invoke_compiler
                or source.run.__func__.__code__ is not source.run_code):
            raise ConversionDeclined("Compiler invocation association changed or expired")
        return source.registration

    def invoke(self, source, destination):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Compiler CuTe invocation is busy")
        try:
            binding = self._seal
            if self._status == "ready":
                binding.run(self._prepared, self._plan, source, destination)
                return
            if self._status == "ordinary":
                InvocationEntry.invoke(binding.registration, source, destination)
                return
            if self._status != "cold":
                raise RuntimeError("Compiler CuTe invocation is closed or its cold join failed")
            binding.compiler.check()
            InvocationEntry.invoke(binding.registration, source, destination)
            self._status = "failed"
            prepared = binding.owner._owned_executor
            try:
                plan = build_conversion_plan(binding.compiler, call_index=binding.call_index)
            except ConversionDeclined as error:
                self._decline = str(error)
                self._status = "ordinary"
                return
            if binding.owner._owned_executor is not prepared:
                raise RuntimeError("Compiler CuTe executor changed during cold join")
            self._plan, self._prepared = plan, prepared
            self._status = "ready"
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Compiler CuTe invocation is busy")
        try:
            self._status = "closed"
            self._plan = self._prepared = None
        finally:
            self._lock.release()

    def __reduce_ex__(self, protocol):
        raise TypeError("Live compiler invocations cannot be serialized")
