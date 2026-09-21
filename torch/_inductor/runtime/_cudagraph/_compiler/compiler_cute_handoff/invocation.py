"""Owned, ordinary-only CuTe invocation with explicit destination mutation."""

from inspect import signature
from types import FunctionType
from uuid import uuid4
from weakref import WeakValueDictionary

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing, ProxyTorchDispatchMode, track_tensor_tree,
)
from torch.fx.node import has_side_effect


class InvocationDeclined(ValueError):
    pass


class _CuTeProvider:
    kind = "cute"
    device_type = "cuda"

    def __init__(self, owner, *, owned_executor=False):
        from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        if type(owned_executor) is not bool:
            raise InvocationDeclined("Owned executor selection must be a bool")
        if type(owner) is not ObservedOrdinaryEntry:
            raise InvocationDeclined("Expected the actual observed ordinary CuTe owner")
        OrdinaryEntry.check(owner)
        formals = tuple(name for name in owner.signature.parameters if name != owner.policy.stream_name)
        if (len(formals) < 2 or tuple(owner.signature.parameters)[:len(formals)] != formals
                or tuple(formals) != owner.tensor_names
                or any(owner.signature.parameters[name].kind not in (
                    owner.signature.parameters[name].POSITIONAL_ONLY,
                    owner.signature.parameters[name].POSITIONAL_OR_KEYWORD,
                ) for name in formals)):
            raise InvocationDeclined("The invocation requires Tensor sources followed by a destination Tensor")
        namespace = {"ENTRY": owner.entry}
        arguments = ", ".join(f"operand{index}" for index in range(len(formals)))
        exec(f"def invoke({arguments}):\n    ENTRY({arguments})\n", namespace)
        self.owner, self.host, self.formals = owner, namespace["invoke"], formals
        method = OrdinaryEntry._invoke_owned if owned_executor else OrdinaryEntry.run
        self.run = method.__get__(owner, type(owner))
        self._seal = owner, owner.entry, self.host, self.host.__code__, formals, self.run, method.__code__

    def check(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        owner, entry, host, code, formals, run, run_code = self._seal
        if (self.owner is not owner or owner.entry is not entry or self.host is not host
                or host.__code__ is not code or host.__globals__.get("ENTRY") is not entry
                or self.formals is not formals or self.run is not run or run.__self__ is not owner
                or run.__func__ not in (OrdinaryEntry.run, OrdinaryEntry._invoke_owned)
                or run.__func__.__code__ is not run_code):
            raise RuntimeError("The owned CuTe invocation changed")
        OrdinaryEntry.check(owner)

    def invoke(self, *operands):
        self._seal[5](self._seal[2], *operands)

    def close(self):
        from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry

        ObservedOrdinaryEntry.close(self._seal[0])


class _CPUStandIn:
    kind = "cpu_standin"
    device_type = "cpu"
    def __init__(self, function):
        if type(function) is not FunctionType:
            raise InvocationDeclined("CPU stand-in requires a Python function")
        parameters = signature(function).parameters
        if len(parameters) < 2 or any(parameter.kind not in (
                parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,
        ) for parameter in parameters.values()):
            raise InvocationDeclined("CPU stand-in requires Tensor sources followed by a destination Tensor")
        self.formals = tuple(parameters)
        self.function, self.code = function, function.__code__
        self._function = function

    def check(self):
        if self.function is not self._function or self.function.__code__ is not self.code:
            raise RuntimeError("CPU stand-in implementation changed")

    def invoke(self, *operands):
        if any(value.device.type != "cpu" for value in operands):
            raise InvocationDeclined("The explicit CPU stand-in cannot execute CUDA operands")
        if self._function(*operands) is not None:
            raise RuntimeError("An invocation implementation must return None")

    def close(self):
        pass


_entries = WeakValueDictionary()


class _NativeBorrow:
    def __init__(self, entry, owner):
        self._seal = entry, owner
        self._token = None

    @property
    def entry(self):
        return None if self._seal is None else self._seal[0]

    @property
    def owner(self):
        return None if self._seal is None else self._seal[1]

    def check(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        if self._token is None:
            raise RuntimeError("Invocation native borrow is closed")
        InvocationEntry.check(self.entry)
        if self.entry._seal[1]._seal[0] is not self.owner:
            raise RuntimeError("Invocation native borrow lost its registered owner")
        OrdinaryEntry._check_native_borrow(self.owner, self._token)

    def close(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        if self._token is not None:
            OrdinaryEntry._release_native_borrow(self.owner, self._token)
            self._token = None
            self._seal = None

    def __reduce_ex__(self, protocol):
        raise TypeError("Invocation native borrows cannot be serialized")


class InvocationEntry:
    def __init__(self, provider):
        if type(provider) not in (_CuTeProvider, _CPUStandIn):
            raise InvocationDeclined("Unsupported invocation provider")
        self.key = uuid4().hex
        self.provider = provider
        self.closed = False
        self._seal = self.key, provider, provider.kind, provider.device_type, provider.formals
        _entries[self.key] = self

    @property
    def kind(self):
        return self.provider.kind

    @property
    def formals(self):
        return self.provider.formals

    def check(self):
        key, provider, kind, device, formals = self._seal
        if self.closed or _entries.get(key) is not self:
            raise InvocationDeclined("Invocation entry is missing, stale or closed")
        if (self.key != key or self.provider is not provider or provider.kind != kind
                or provider.device_type != device or provider.formals is not formals
                or getattr(self.invoke, "__func__", None) is not InvocationEntry.invoke):
            raise RuntimeError("Invocation entry ownership changed")
        type(provider).check(provider)

    def invoke(self, *operands):
        if self.closed:
            raise InvocationDeclined("Invocation entry is closed")
        if len(operands) != len(self.formals) or any(not isinstance(value, torch.Tensor) for value in operands):
            raise InvocationDeclined("CuTe invocation operands must match the registered Tensor formals")
        provider = self._seal[1]
        type(provider).invoke(provider, *operands)

    def borrow_native(self):
        from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry

        InvocationEntry.check(self)
        provider = self._seal[1]
        if type(provider) is not _CuTeProvider:
            raise InvocationDeclined("Native replay requires an actual owned CuTe entry")
        owner = provider._seal[0]
        borrow = _NativeBorrow(self, owner)
        borrow._token = OrdinaryEntry._acquire_native_borrow(owner)
        return borrow

    def close(self):
        if self.closed:
            return
        self._seal[1].close()
        self.closed = True
        key = self._seal[0]
        if _entries.get(key) is self:
            del _entries[key]

    def __reduce_ex__(self, protocol):
        raise TypeError("Live invocation entries are not cache transport")


def register_cute_entry(owner, *, owned_executor=False):
    """Own an inference entry: sources are read-only, the final destination is distinct and mutable.

    Sources may alias one another, but the destination must not alias a source.
    Stride-changing views are outside this invocation contract.
    Registration does not infer or certify arbitrary device effects.
    An owned executor fixes the compiled program until close; run() remains audited.
    """
    return InvocationEntry(_CuTeProvider(owner, owned_executor=owned_executor))


def register_cpu_standin(function):
    """Explicit CPU test provider; never a certificate for a CuTe artifact."""
    return InvocationEntry(_CPUStandIn(function))


def resolve_entry(key):
    if type(key) is not str or (entry := _entries.get(key)) is None:
        raise InvocationDeclined("Unknown invocation entry")
    InvocationEntry.check(entry)
    return entry


class _Invocation(HigherOrderOperator):
    def __init__(self, name):
        super().__init__(name, cacheable=False)

    def __call__(self, key, *operands):
        if type(key) is not str or any(not isinstance(value, torch.Tensor) for value in operands):
            raise InvocationDeclined("CuTe invocation accepts an entry key and Tensor operands; scalar host operands are unsupported")
        if len(operands) != len(resolve_entry(key).formals):
            raise InvocationDeclined("CuTe invocation operands must match the registered Tensor formals")
        return super().__call__(key, *operands)


invoke_cute = has_side_effect(_Invocation("invoke_cute"))
invoke_cute_functional = _Invocation("invoke_cute_functional")


@invoke_cute.py_impl(DispatchKey.CompositeExplicitAutograd)
def _invoke(key, *operands):
    resolve_entry(key).invoke(*operands)


@invoke_cute_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def _invoke_functional(key, *operands):
    output = operands[-1].clone(memory_format=torch.preserve_format)
    invoke_cute(key, *operands[:-1], output)
    return output


@register_fake(invoke_cute, skip_cache=True)
def _fake(key, *operands):
    resolve_entry(key)
    return None


@register_fake(invoke_cute_functional, skip_cache=True)
def _fake_functional(key, *operands):
    resolve_entry(key)
    return operands[-1].clone(memory_format=torch.preserve_format)


def _proxy(mode, hop, key, *operands):
    with disable_proxy_modes_tracing():
        output = hop(key, *operands)
    args = tuple(mode.tracer.unwrap_proxy(value) for value in (key, *operands))
    proxy = mode.tracer.create_proxy("call_function", hop, args, {})
    return track_tensor_tree(output, proxy, constant=None, tracer=mode.tracer)


@invoke_cute.py_impl(ProxyTorchDispatchMode)
def _invoke_proxy(mode, key, *operands):
    return _proxy(mode, invoke_cute, key, *operands)


@invoke_cute_functional.py_impl(ProxyTorchDispatchMode)
def _functional_proxy(mode, key, *operands):
    return _proxy(mode, invoke_cute_functional, key, *operands)


@invoke_cute.py_functionalize_impl
def _functionalize(ctx, key, *operands):
    values = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next():
        output = invoke_cute_functional(key, *values)
    destination = operands[-1]
    ctx.replace(destination, output)
    ctx.mark_mutation_hidden_from_autograd(destination)
    ctx.commit_update(destination)
    ctx.sync(destination)


@invoke_cute_functional.py_functionalize_impl
def _functionalize_functional(ctx, key, *operands):
    values = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next():
        output = invoke_cute_functional(key, *values)
    return ctx.wrap_tensors(output)


@invoke_cute.py_autograd_impl
@invoke_cute_functional.py_autograd_impl
def _autograd(*args, **kwargs):
    raise InvocationDeclined("CuTe invocation currently supports inference only")


_operators = invoke_cute, invoke_cute_functional
_implementations = tuple((operator, tuple(operator.py_kernels.items()),
                         tuple(operator.python_key_table.items()), tuple(operator.functorch_table.items()))
                        for operator in _operators)
_methods = tuple((kind, name, getattr(kind, name)) for kind, name in (
    (_Invocation, "__call__"), (InvocationEntry, "invoke"), (InvocationEntry, "check"),
    (_CuTeProvider, "invoke"), (_CuTeProvider, "check"), (_CuTeProvider, "close"),
    (_CPUStandIn, "invoke"), (_CPUStandIn, "check"),
))
_source_functions = tuple((function, function.__code__) for function in (
    *(function for _, _, function in _methods),
    _invoke, _invoke_functional, _fake, _fake_functional, _functionalize, _functionalize_functional,
))


def check_implementation():
    from torch._higher_order_ops.utils import registered_hop_fake_fns

    if (invoke_cute is not _operators[0] or invoke_cute_functional is not _operators[1]
            or registered_hop_fake_fns.get(invoke_cute) is not _fake
            or registered_hop_fake_fns.get(invoke_cute_functional) is not _fake_functional):
        raise RuntimeError("CuTe invocation implementation changed")
    for operator, kernels, modes, transforms in _implementations:
        if (tuple(operator.py_kernels.items()) != kernels or tuple(operator.python_key_table.items()) != modes
                or tuple(operator.functorch_table.items()) != transforms or operator.cacheable()):
            raise RuntimeError("CuTe invocation dispatch implementation changed")
    if (any(getattr(kind, name) is not function for kind, name, function in _methods)
            or any(function.__code__ is not code for function, code in _source_functions)):
        raise RuntimeError("CuTe invocation source changed")
