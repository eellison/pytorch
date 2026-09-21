"""Trace an ordinary boxed Python composition into the shared replay program."""

from contextlib import ExitStack
from dataclasses import replace
from dis import get_instructions
from threading import Lock

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter import invocation
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
    FXTraceDeclined,
    ReinterpretEvent,
)

from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch._ops import OpOverload
from torch.utils._python_dispatch import get_alias_info, TorchDispatchMode
from torch.utils._pytree import tree_leaves

from .cute_types import CuteInvokeEvent
from .direct_cuda_host import CudaInvocation, CudaInvocationDeclined
from .direct_cute import DirectCuTe
from .direct_invocation import activate
from .direct_triton import DirectTritonDeclined, DirectTritonInvokeEvent
from .extraction import trace_host
from .frontend import DirectOrigin, lower_terminal
from .guard_export import prepare_guard
from .prepared import PreparedVariant
from .replay import prepare_terminal


class HostTensorEvents(TorchDispatchMode):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.prim.device.default:
            return func(*args, **kwargs)
        allocations = (
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_strided.default,
            torch.ops.aten.empty_like.default,
            torch.ops.aten.new_empty.default,
            torch.ops.aten.new_empty_strided.default,
        )
        if func in allocations:
            result = func(*args, **kwargs)
            self.state.check_tensor(result)
            self.state.record(
                AllocateEvent(
                    result,
                    tuple(result.size()),
                    tuple(result.stride()),
                    result.dtype,
                    result.device,
                )
            )
            return result
        if type(func) is not OpOverload or func._schema.is_mutable:
            raise FXTraceDeclined(
                f"Host CUDA operation has no invocation adapter: {func}"
            )
        aliases = get_alias_info(func).read_only_alias_match_indexes
        if not aliases or sorted(output for _, output in aliases) != list(
            range(len(func._schema.returns))
        ):
            raise FXTraceDeclined(
                f"Host operation has no allocation, view or invocation lowering: {func}"
            )
        sources = []
        for index, output_index in aliases:
            name = func._schema.arguments[index].name
            source = args[index] if index < len(args) else kwargs[name]
            self.state.check_tensor(source)
            sources.append((source, output_index))
        result = func(*args, **kwargs)
        outputs = (result,) if len(func._schema.returns) == 1 else result
        for source, output_index in sources:
            for tensor in tree_leaves(outputs[output_index]):
                self.state.check_tensor(tensor)
                if (
                    tensor.dtype != source.dtype
                    or tensor.device != source.device
                    or tensor.is_conj() != source.is_conj()
                    or tensor.is_neg() != source.is_neg()
                    or not torch._C._is_alias_of(source, tensor)
                ):
                    raise FXTraceDeclined(
                        "Host view lost its original storage representation"
                    )
                self.state.record(
                    ReinterpretEvent(
                        source,
                        tensor,
                        tuple(tensor.size()),
                        tuple(tensor.stride()),
                        tensor.storage_offset() - source.storage_offset(),
                    )
                )
        return result


def _direct_origin(host, contract):
    names = tuple(
        dict.fromkeys(
            op.argval for op in get_instructions(host) if op.opname == "LOAD_GLOBAL"
        )
    )
    bindings = tuple(
        (name, host.__globals__[name]) for name in names if name in host.__globals__
    )
    origin = DirectOrigin(host, host.__code__, contract, bindings)
    origin.check()
    return origin, []


class _ObservedInvocations:
    def __init__(self, calls, stack):
        self.calls = calls
        self.stack = stack
        self.triton_adapters = set()

    def cute(self, adapter, arguments):
        self.calls.append(adapter)
        return adapter._invoke_ordinary(*arguments)

    def cuda(self, adapter, arguments):
        from torch.cuda import _host_trace

        from .direct_hosttrace import lower_tape

        result = adapter.entry(*arguments)
        output_kind = torch.Tensor if isinstance(result, torch.Tensor) else type(result)
        try:
            if output_kind not in (torch.Tensor, tuple, list):
                raise UnsupportedCapture("Converted CUDA host requires tensor outputs")
            tape = _host_trace.trace(adapter.entry, arguments, warm_up=False)
            lowered = lower_tape(tape)
        except (_host_trace.Declined, UnsupportedCapture) as error:
            self.calls.append(CudaInvocationDeclined(str(error)))
            return result
        outputs = (result,) if output_kind is torch.Tensor else tuple(result)
        identities = []
        for position, output in enumerate(outputs):
            identity = next(
                (
                    ("argument", index)
                    for index, value in enumerate(arguments)
                    if output is value
                ),
                None,
            )
            if identity is None:
                identity = next(
                    (
                        ("output", index)
                        for index, value in enumerate(outputs[:position])
                        if output is value
                    ),
                    None,
                )
            identities.append(identity)
        self.calls.append(
            CudaInvocation(adapter, tape, lowered, output_kind, tuple(identities))
        )
        return result

    def triton(self, adapter, arguments, *, grid, warmup, kwargs):
        if id(adapter) not in self.triton_adapters:
            self.stack.enter_context(adapter.observe())
            self.triton_adapters.add(id(adapter))
        self.calls.append(adapter)
        return adapter._run_ordinary(*arguments, grid=grid, warmup=warmup, **kwargs)


def _observe_direct(origin, kernels, box):
    origin.check()
    kernels.clear()
    with ExitStack() as stack, activate(_ObservedInvocations(kernels, stack)):
        result = origin.wrapper(box)
    origin.check()
    return result


class _TracedInvocations:
    def __init__(self, state, calls):
        self.state = state
        self.calls = tuple(calls)
        self.position = 0
        self.views = {}

    def _view(self, adapter):
        if self.position >= len(self.calls) or self.calls[self.position] is not adapter:
            raise FXTraceDeclined(
                "Symbolic invocation order differs from ordinary execution"
            )
        self.position += 1
        key = id(adapter)
        if key in self.views:
            return self.views[key]
        state = self.state
        name = f"direct_{len(self.views)}"

        def record(event):
            if type(event) is ReinterpretEvent:
                if not any(
                    type(old) is ReinterpretEvent and old.tensor is event.tensor
                    for old in state.events
                ):
                    state.record(event)
                return
            if type(event) is CuteInvokeEvent:
                state.record(event)
                return
            if type(event) is not DirectTritonInvokeEvent:
                raise FXTraceDeclined("Direct kernel emitted an unsupported host event")
            from .triton_tma import DescriptorValue

            state.record(event)
            arguments = []
            for value in event.arguments:
                if type(value) is DescriptorValue:
                    arguments.extend((value.base, *value.shape, *value.strides))
                else:
                    arguments.append(value)
            invocation.invoke(name, (*arguments, *event.grid))

        if type(adapter) is DirectCuTe:
            from .cute_adapter import make_cute_trace_view

            view = make_cute_trace_view(adapter, record, state.mode)
        else:
            view = adapter.trace_view(record)
        self.views[key] = view
        return view

    def cute(self, adapter, arguments):
        return self._view(adapter).invoke(*arguments)

    def cuda(self, adapter, arguments):
        if self.position >= len(self.calls):
            raise FXTraceDeclined("Symbolic host added an unobserved CUDA invocation")
        call = self.calls[self.position]
        if type(call) is not CudaInvocation or call.adapter is not adapter:
            raise FXTraceDeclined(
                "Symbolic CUDA invocation differs from ordinary execution"
            )
        self.views[("cuda", self.position)] = call
        self.position += 1
        return call.trace(self.state, arguments)

    def triton(self, adapter, arguments, *, grid, warmup, kwargs):
        return self._view(adapter).run(*arguments, grid=grid, warmup=warmup, **kwargs)


def _prepare_observed(origin, kernels, example_inputs):
    origin.check()
    for call in kernels:
        if type(call) is CudaInvocationDeclined:
            raise FXTraceDeclined(call.reason)
    handler = None

    def context(state):
        nonlocal handler
        handler = _TracedInvocations(state, kernels)
        return activate(handler)

    trace = trace_host(
        origin.wrapper,
        origin.contract,
        example_inputs,
        (),
        None,
        direct=True,
        context_factory=context,
    )
    if handler.position != len(handler.calls):
        raise FXTraceDeclined("Symbolic host omitted an ordinary invocation")
    for view in handler.views.values():
        view.check()
        if hasattr(view, "finish"):
            view.finish()
    trace = replace(trace, compiler_binding=origin)
    program = lower_terminal(trace, tuple(handler.views.values()))
    entry = None
    try:
        guard = prepare_guard(program, example_inputs)
        entry = prepare_terminal(program, example_inputs)
        for view in handler.views.values():
            if type(view) is CudaInvocation:
                view.tape.args = None
        return PreparedVariant(entry, guard, program)
    except BaseException:
        if entry is not None:
            entry.close()
        program.close()
        raise


def prepare_direct(host, contract, example_inputs):
    origin, kernels = _direct_origin(host, contract)
    inputs = tuple(example_inputs)
    _observe_direct(origin, kernels, list(inputs))
    return _prepare_observed(origin, kernels, inputs)


class DirectHost:
    """Publish native local variants after each observed ordinary invocation."""

    def __init__(self, host, contract):
        self.origin, self._kernels = _direct_origin(host, contract)
        self.entry = None
        self.variants = []
        self.closed = False
        self._lock = Lock()

    def __call__(self, box):
        if self.closed:
            raise RuntimeError("Direct host is closed")
        if self.entry is None:
            return self._miss(box)
        return self.entry(box)

    def _miss(self, box):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Direct host preparation is busy")
        try:
            if self.closed or type(box) is not list:
                raise RuntimeError("Direct host requires an open boxed invocation")
            inputs = tuple(box)
            result = _observe_direct(self.origin, self._kernels, box)
            try:
                variant = _prepare_observed(self.origin, self._kernels, inputs)
            except (FXTraceDeclined, UnsupportedCapture, DirectTritonDeclined):
                return result
            try:
                self.origin.check()
                entry = self.entry
                if entry is None:
                    entry = variant.entry
                    if variant.guard is not None:
                        entry = torch._C._cuda_make_boxed_dispatch(
                            ((entry, variant.guard.registration),),
                            self._miss,
                        )
                else:
                    if variant.guard is None:
                        variant = replace(
                            variant,
                            guard=prepare_guard(variant.program, inputs, required=True),
                        )
                    entry.append(variant.entry, variant.guard.registration)
            except BaseException as error:
                variant.abort(error)
                raise
            self.variants.append(variant)
            self.entry = entry
            return result
        finally:
            self._kernels.clear()
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Direct host preparation is busy")
        try:
            if self.closed:
                return
            if self.entry is not None:
                self.entry.close()
            for variant in self.variants:
                variant.close()
            self.closed = True
        finally:
            self._lock.release()
