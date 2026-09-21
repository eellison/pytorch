"""Lower executed wrapper and selected launcher computations to native replay inputs."""

from dataclasses import dataclass, replace
from dis import get_instructions

import sympy

import torch
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import (
    InvocationEntry,
)
from torch._inductor.runtime._cudagraph._compiler.fx_adapter import invocation
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
    FXTraceDeclined,
    LayoutEvent,
    NormalizeEvent,
    ReinterpretEvent,
)
from torch._inductor.runtime._cudagraph._compiler.generated_python.prototype import (
    _Tensor,
    _Trace,
)
from torch._inductor.runtime._cudagraph._compiler.host_program import (
    Allocate,
    Normalize,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BorrowedInputOutput,
    BufferSource,
    CallArgument,
    ExpressionSource,
    grid_expression_inputs,
    InputSource,
    IntegerInput,
    IntegerOutput,
    IntegerSource,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    pointwise_product,
    storage_roots,
    TensorViewOutput,
)
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.utils._sympy.functions import CeilDiv, FloorDiv

from .address_scalars import lower_address_scalar, pointer_alignment_guard
from .address_trace import TensorAddressRoots
from .cuda_tape_import import TapeSources
from .cute_types import CuTeCall, CuteInvokeEvent
from .direct_cuda_host import (
    CudaHostCompleteEvent,
    CudaHostInvokeEvent,
    CudaHostMemcpyEvent,
    CudaHostMemsetEvent,
    CudaHostTableEvent,
)
from .direct_triton import DirectTritonInvokeEvent
from .extraction import trace_host
from .guard_export import export_guards, GuardExpressions
from .provider_facts import read_provider_facts
from .trace_views import make_trace_view
from .triton_scratch import scratch_specs, TritonScratchDeclined
from .triton_tma import DescriptorValue, lower_descriptor, TritonTmaModule


@dataclass(frozen=True)
class TerminalInvokeEvent:
    provider: object
    arguments: tuple
    grid: tuple


@dataclass(frozen=True)
class TerminalCall:
    provider: object
    arguments: tuple[CallArgument, ...]
    grid: tuple[IntExpr, IntExpr, IntExpr]
    scratch: tuple[PointerSource | ParameterSource, ...] = ()


@dataclass(frozen=True)
class DirectKernelCall:
    owner: object
    arguments: tuple[CallArgument, ...]
    grid: tuple[IntExpr, IntExpr, IntExpr]
    scratch: tuple[PointerSource | ParameterSource, ...] = ()


@dataclass(frozen=True)
class DirectPhysicalCall:
    owner: object
    bound: object
    rng_fields: tuple = ()


@dataclass(frozen=True)
class DirectMemset:
    owner: object
    destination: PointerSource
    byte_count: IntExpr
    value: int


@dataclass(frozen=True, eq=False)
class DirectHostTable:
    owner: object
    index: int
    table: object


@dataclass(frozen=True)
class DirectMemcpy:
    owner: object
    source: DirectHostTable | PointerSource
    destination: PointerSource
    byte_count: IntExpr


@dataclass(frozen=True)
class DirectOrigin:
    wrapper: object
    code: object
    contract: object
    bindings: tuple

    def check(self):
        if self.wrapper.__code__ is not self.code or any(
            self.wrapper.__globals__.get(name) is not value
            for name, value in self.bindings
        ):
            raise FXTraceDeclined("Direct host callable changed during preparation")


@dataclass(frozen=True)
class _WrapperOrigin:
    wrapper: object
    code: object
    attachment: object
    contract: object
    bindings: tuple

    def check(self):
        from .metadata import check_terminal_attachment

        if (
            self.wrapper.__code__ is not self.code
            or check_terminal_attachment(self.wrapper, self.attachment)
            is not self.attachment.metadata
            or self.attachment.metadata.inputs is not self.contract
            or any(
                self.wrapper.__globals__.get(name) is not provider
                for name, provider in self.bindings
            )
        ):
            raise FXTraceDeclined(
                "Original wrapper or inherited input contract changed during preparation"
            )


@dataclass(frozen=True)
class TerminalProgram:
    contract: object
    events: tuple
    outputs: tuple
    allocations: tuple[OwnedBuffer, ...]
    integer_inputs: tuple[IntegerInput, ...]
    views: tuple
    effects: tuple
    origin: _WrapperOrigin | DirectOrigin
    guards: GuardExpressions
    saved_input_indices: tuple[int, ...] = ()
    rng: IntExpr | None = None

    @property
    def input_names(self):
        return tuple(f"boxed_{index}" for index in range(len(self.contract.kinds)))

    def check(self):
        self.origin.check()
        for view in self.views:
            view.check()
        for receipt in self.effects:
            receipt.check()
        for event in self.events:
            if type(event) is CuTeCall:
                event.receipt.check()

    def close(self):
        failure = None
        for event in reversed(self.events):
            if type(event) is CuTeCall:
                try:
                    event.receipt.close()
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        failure.add_note(
                            f"Additional CuTe receipt cleanup failed: {error}"
                        )
        if failure is not None:
            raise failure


def trace_warmed_wrapper(wrapper, contract, example_inputs):
    views = []
    globals_used = tuple(
        dict.fromkeys(
            op.argval for op in get_instructions(wrapper) if op.opname == "LOAD_GLOBAL"
        )
    )
    names = tuple(
        name
        for name in globals_used
        if isinstance(
            wrapper.__globals__.get(name),
            (CachingAutotuner, MultiKernelCall, InvocationEntry),
        )
    )
    guards = torch._C._dynamo.guards
    helpers = {
        "torch": torch,
        "assert_size_stride": guards.assert_size_stride,
        "assert_size_stride_grouped": guards.assert_size_stride_grouped,
        "copy_if_misaligned": guards.copy_if_misaligned,
        "empty_strided_cuda": guards._empty_strided_cuda,
        "reinterpret_tensor": guards._reinterpret_tensor,
        "get_raw_stream": torch._C._cuda_getCurrentRawStream,
    }
    if any(
        name in helpers and wrapper.__globals__.get(name) is not helpers[name]
        for name in globals_used
    ):
        raise FXTraceDeclined(
            "Generated host helper differs from its canonical tracing operation"
        )
    origin = _WrapperOrigin(
        wrapper,
        wrapper.__code__,
        wrapper._cudagraph_terminal_attachment,
        contract,
        tuple((name, wrapper.__globals__.get(name)) for name in globals_used),
    )
    origin.check()

    def factory(state, name):
        carrier = wrapper.__globals__[name]
        if type(carrier) is InvocationEntry:
            from .cute_adapter import make_cute_trace_view

            def record_cute(event):
                if type(event) is ReinterpretEvent and state.in_device:
                    state.check_tensor(event.source)
                    state.check_tensor(event.tensor)
                    state.record(event)
                    return
                if (
                    type(event) is not CuteInvokeEvent
                    or event.entry is not carrier
                    or not state.in_device
                ):
                    raise FXTraceDeclined(
                        "CuTe invocation escaped its owned traced call site"
                    )
                for value in event.operands:
                    if isinstance(value, torch.Tensor):
                        state.check_tensor(value)
                    else:
                        state.dimensions((value,))
                state.record(event)
                invocation.invoke(name, event.operands)

            view = make_cute_trace_view(carrier, record_cute, state.mode)
            views.append(view)
            return view

        def record(provider, grid, stream, arguments):
            if stream is not state or not state.in_device:
                raise FXTraceDeclined(
                    "Terminal invocation escaped the traced device scope"
                )
            for value in arguments:
                if isinstance(value, torch.Tensor):
                    state.check_tensor(value)
                else:
                    state.dimensions((value,))
            state.dimensions(grid)
            state.record(TerminalInvokeEvent(provider, arguments, grid))
            invocation.invoke(name, (*arguments, *grid))

        view = make_trace_view(wrapper.__globals__[name], record)
        views.append(view)
        return view

    trace = trace_host(wrapper, contract, example_inputs, names, factory)
    for view in views:
        view.check()
    origin.check()
    return replace(trace, compiler_binding=origin), tuple(views)


def lower_terminal(trace, views, *, extra_guards=()):
    calls = []
    try:
        return _lower_terminal(trace, views, calls, extra_guards)
    except BaseException as error:
        for call in reversed(calls):
            try:
                call.receipt.close()
            except BaseException as cleanup:
                error.add_note(f"CuTe receipt retained after cleanup failed: {cleanup}")
        raise


def _lower_terminal(trace, views, cute_calls, extra_guards):
    from .direct_hosttrace import _Lowering

    if type(trace.compiler_binding) not in (_WrapperOrigin, DirectOrigin):
        raise FXTraceDeclined("Terminal trace lost its original wrapper")
    trace.compiler_binding.check()
    environment = trace.shape_env
    cute_preparation = None
    integer_sources = dict(trace.symbol_sources)
    integer_sources.update(
        (binding.symbol, IntExpr("storage_offset", binding.index))
        for binding in trace.storage_offset_bindings
    )
    host_integers = _Lowering(None, integer_sources=integer_sources)
    normalized_roots = {
        trace.tensor_roots.event(event).root
        for event in trace.events
        if type(event) is NormalizeEvent
    }
    normalized_offsets = {
        binding.symbol
        for binding in trace.storage_offset_bindings
        if InputSource(binding.index) in normalized_roots
    }

    def expression(value):
        if type(value) is int:
            return value
        if type(value) is torch.SymInt:
            if value.node.shape_env is not environment:
                raise FXTraceDeclined(
                    "Symbolic value belongs to another tracing environment"
                )
            value = value.node.expr
        if not isinstance(value, sympy.Expr) or not value.free_symbols.issubset(
            integer_sources
        ):
            raise FXTraceDeclined("Integer computation has no traced boxed source")
        if value.free_symbols.intersection(normalized_offsets):
            raise FXTraceDeclined(
                "Storage-offset use crosses an unrepresented normalization generation"
            )
        value = environment.simplify(value)
        value = value.replace(
            lambda term: isinstance(term, FloorDiv)
            and isinstance(term.args[1], sympy.Integer)
            and term.args[1] < 0,
            lambda term: -CeilDiv(term.args[0], -term.args[1]),
        )
        if isinstance(value, sympy.Integer):
            return int(value)
        if isinstance(value, sympy.Symbol):
            return integer_sources[value]
        if isinstance(value, sympy.Mul):
            result = pointwise_product(tuple(expression(arg) for arg in value.args))
            if result is not None:
                return result
        if isinstance(value, (sympy.Mul, sympy.Add)):
            operands = tuple(numeric(arg) for arg in value.args)
            result = operands[0]
            for operand in operands[1:]:
                result = IntExpr(
                    "multiply" if isinstance(value, sympy.Mul) else "add",
                    args=(result, operand),
                )
            if grid_expression_inputs(result) is not None:
                return result
        if isinstance(value, (CeilDiv, FloorDiv)):
            numerator, denominator = value.args
            if isinstance(denominator, sympy.Integer) and denominator > 0:
                floor = isinstance(value, FloorDiv)
                numerator = expression(numerator)
                if type(numerator) is int:
                    return (
                        numerator // int(denominator)
                        if floor
                        else -(-numerator // int(denominator))
                    )
                return IntExpr(
                    "floordiv" if floor else "ceildiv",
                    args=(numerator, IntExpr("constant", int(denominator))),
                )
        return host_integers.lower(value)

    def numeric(value):
        value = expression(value)
        return IntExpr("constant", value) if type(value) is int else value

    for binding in trace.computed_integer_bindings:
        if binding.symbol in integer_sources:
            raise FXTraceDeclined("Computed integer reuses an existing source")
        arguments = tuple(numeric(arg) for arg in binding.arguments)
        integer_sources[binding.symbol] = (
            IntExpr("constant", binding.expected)
            if binding.kind == "guard"
            else IntExpr("call", (binding.address, binding.owner), arguments)
        )

    state = _Trace(trace.contract.device_index, {})
    state.in_device = True
    roots = trace.tensor_roots
    if (
        type(roots) is not TensorAddressRoots
        or roots.mode.shape_env is not trace.shape_env
    ):
        raise FXTraceDeclined("Terminal trace lost its shared storage roots")
    tensors, offsets, aliases, root_alignments = {}, {}, set(), {}
    for row in trace.contract.tensor_inputs:
        resolution = roots.inputs[row.index]
        tensors[id(trace.placeholders[row.index])] = _Tensor(
            resolution.root, row.dtype, row.size, row.stride
        )
        offsets[id(trace.placeholders[row.index])] = resolution.byte_offset
        root_alignments[resolution.root] = resolution.alignment
    events, effects, extra_guards = [], {}, list(extra_guards)
    providers = {
        id(provider): provider for view in views for provider in view.providers
    }
    normalized, used = set(), set()
    host_tables = {}
    rng_prefix, has_rng = sympy.Integer(0), False

    def mark_storage_used(source, label):
        for root in storage_roots(source):
            tokens = tuple(
                token
                for token in tensors.values()
                if token.source == root and token.valid
            )
            if not tokens:
                raise FXTraceDeclined(f"{label} has no preceding traced storage root")
            for token in tokens:
                token.used = True
            used.add(root)

    for event in trace.events:
        if type(event) is AllocateEvent:
            if event.device != torch.device("cuda", trace.contract.device_index):
                raise FXTraceDeclined("Allocation escaped the inherited device")
            resolution = roots.event(event)
            token = _Tensor(
                resolution.root,
                event.dtype,
                tuple(expression(value) for value in event.size),
                tuple(expression(value) for value in event.stride),
            )
            tensors[id(event.tensor)] = token
            state.allocations.append(token)
            offsets[id(event.tensor)] = resolution.byte_offset
            root_alignments[resolution.root] = resolution.alignment
            events.append(
                Allocate(token.source.name, token.dtype, token.size, token.stride)
            )
        elif type(event) is LayoutEvent:
            state.check_layout(
                tensors[id(event.tensor)],
                tuple(expression(value) for value in event.size),
                tuple(expression(value) for value in event.stride),
                event.label,
            )
        elif type(event) is NormalizeEvent:
            token = tensors[id(event.tensor)]
            if token.source in used:
                raise FXTraceDeclined(
                    "Input normalization must precede the root tensor's first use"
                )
            state.normalize(token)
            if token.source in normalized:
                raise FXTraceDeclined(
                    "Input normalization must occur once before its first use"
                )
            normalized.add(token.source)
            root_alignments[token.source] = roots.event(event).alignment
            events.append(
                Normalize(
                    state.events[-1].input_index,
                    sum(
                        type(step)
                        in (
                            TerminalCall,
                            CuTeCall,
                            DirectKernelCall,
                            DirectPhysicalCall,
                        )
                        for step in events
                    ),
                )
            )
        elif type(event) in (
            CudaHostInvokeEvent,
            CudaHostCompleteEvent,
            CudaHostMemsetEvent,
            CudaHostMemcpyEvent,
            CudaHostTableEvent,
        ):
            owner = event.owner
            owner.check()
            imported = TapeSources(
                owner.tape,
                dict(enumerate(event.arguments)),
                roots,
                lambda resolution: event.addresses[resolution.root],
                numeric,
                mapping=owner.lowered.symbols.mapping,
                computed=event.computed,
            )
            for record, tensor in event.allocations:
                imported.bind_allocation(record, tensor)
            if type(event) is CudaHostCompleteEvent:
                extra_guards.extend(imported.guards(owner.lowered.extra_guards))
                if getattr(owner.lowered, "rng", None) is not None:
                    increment = imported.translate(owner.tape.rng_increment)
                    rng_prefix = sympy.Add(rng_prefix, increment, evaluate=False)
                    extra_guards.extend(
                        (
                            sympy.Ge(increment, 0),
                            sympy.Eq(sympy.Mod(increment, 4), 0),
                            sympy.Ge(rng_prefix, 0),
                            sympy.Lt(rng_prefix, 2**63),
                        )
                    )
                    has_rng = True
            elif type(event) is CudaHostTableEvent:
                table = imported.table(owner.lowered.host_tables[event.index])
                key = (id(owner), event.index)
                if key in host_tables:
                    raise FXTraceDeclined("Imported host table repeats its completion")
                table_event = DirectHostTable(owner, event.index, table)
                host_tables[key] = table_event
                for _, _, source in table.elements:
                    if type(source) is PointerSource:
                        mark_storage_used(source, "Converted CUDA host table")
                events.append(table_event)
            elif type(event) is CudaHostMemcpyEvent:
                _, source, destination, count = owner.lowered.memcpys[event.index]
                if type(source) is int:
                    source = host_tables.get((id(owner), source))
                    if source is None:
                        raise FXTraceDeclined("Host copy precedes its table image")
                else:
                    source = imported.source(source)
                    row = next(
                        (
                            row
                            for row in trace.contract.tensor_inputs
                            if InputSource(row.index) == source.root
                        ),
                        None,
                    )
                    if (
                        row is None
                        or row.device != torch.device("cpu")
                        or not row.pinned
                    ):
                        raise FXTraceDeclined(
                            "Host copy requires a pinned CPU input contract"
                        )
                    mark_storage_used(source, "Converted CUDA host copy source")
                destination = imported.source(destination)
                mark_storage_used(destination, "Converted CUDA host copy destination")
                events.append(
                    DirectMemcpy(owner, source, destination, imported.integer(count))
                )
            elif type(event) is CudaHostMemsetEvent:
                _, destination, count, value = owner.lowered.memsets[event.index]
                destination = imported.source(destination)
                mark_storage_used(destination, "Converted CUDA host memset")
                events.append(
                    DirectMemset(owner, destination, imported.integer(count), value)
                )
            else:
                bound = imported.call(owner.lowered.calls[event.call_index])
                slots = tuple(
                    slot
                    for slot in getattr(owner.lowered, "rng_slots", ())
                    if slot.call_index == event.call_index
                )
                if slots:
                    bound, rng_guards = imported.rng_call(bound, slots, rng_prefix)
                    extra_guards.extend(rng_guards)
                rng_fields = tuple(
                    (parameter, offset)
                    for index, parameter, offset in getattr(
                        owner.lowered, "rng_fields", ()
                    )
                    if index == event.call_index
                )
                for field in bound.fields:
                    mark_storage_used(field.source, "Converted CUDA host argument")
                events.append(DirectPhysicalCall(owner, bound, rng_fields))
            effects[id(owner)] = owner
        elif type(event) is CuteInvokeEvent:
            from .cute_adapter import CuTePreparation, lower_cute_calls

            if cute_preparation is None:
                cute_preparation = CuTePreparation()

            values = tuple(
                value for value in event.operands if isinstance(value, torch.Tensor)
            )
            operands = tuple(tensors.get(id(value)) for value in values)
            if any(
                token is None
                or not token.valid
                or type(token.source) not in (InputSource, BufferSource)
                for token in operands
            ):
                raise FXTraceDeclined(
                    "CuTe tensor arguments require traced storage roots"
                )
            calls = lower_cute_calls(
                trace,
                event,
                {
                    id(value): PointerSource(token.source, numeric(offsets[id(value)]))
                    for value, token in zip(values, operands)
                },
                expression,
                stream=torch.cuda.current_stream(
                    trace.contract.device_index
                ).cuda_stream,
                preparation=cute_preparation,
                root_alignments=root_alignments,
            )
            cute_calls.extend(calls)
            for call in calls:
                if tuple(pointer.root for pointer in call.pointers) != tuple(
                    token.source for token in operands
                ):
                    raise FXTraceDeclined(
                        "CuTe physical call lost its actual traced operands"
                    )
                for field in call.bound.fields:
                    if type(field.source) is ParameterSource:
                        mark_storage_used(field.source, "CuTe scalar")
                events.append(call)
                extra_guards.extend(call.guards)
            for token in operands:
                token.used = True
                used.add(token.source)
        elif type(event) in (TerminalInvokeEvent, DirectTritonInvokeEvent):
            descriptors = {}
            if type(event) is DirectTritonInvokeEvent:
                receipt = event.owner
                receipt.check()
                effects[id(receipt)] = receipt
                rows = tuple(
                    sorted(
                        (row for row in receipt.formals if row.abi_index is not None),
                        key=lambda row: row.abi_index,
                    )
                )
                descriptors = dict(receipt.descriptors)
                abi_index = 0
                for row in rows:
                    if row.abi_index != abi_index:
                        raise FXTraceDeclined(
                            "Direct Triton arguments lost their packed ABI order"
                        )
                    spec = descriptors.get(row.source_arg_index)
                    abi_index += len(spec.abi_types) if spec is not None else 1
                if abi_index != len(receipt.module.arg_tys):
                    raise FXTraceDeclined(
                        "Direct Triton arguments do not cover the expanded ABI"
                    )
                values = tuple(event.arguments[row.source_arg_index] for row in rows)
                extra_guards.extend(event.guards)
            else:
                provider = providers.get(id(event.provider))
                if provider is not event.provider:
                    raise FXTraceDeclined("Terminal call lost its warmed provider")
                receipt = effects.get(id(provider))
                if receipt is None:
                    receipt = read_provider_facts(provider)
                    effects[id(provider)] = receipt
                rows = receipt.arguments
                values = event.arguments
                if len(rows) != len(values):
                    raise FXTraceDeclined(
                        "Terminal arguments do not cover the selected ABI"
                    )
            arguments = []
            physical_fields, tensor_maps = [], []
            for slot, (row, value) in enumerate(zip(rows, values, strict=True)):
                if row.source_arg_index in descriptors:
                    if type(value) is not DescriptorValue:
                        raise FXTraceDeclined(
                            "Host tensor map lost its traced descriptor snapshot"
                        )
                    token = tensors.get(id(value.base))
                    if token is None or not token.valid or value.base.numel() == 0:
                        raise FXTraceDeclined(
                            "Host tensor map requires a nonempty traced base"
                        )
                    if roots(value.base).root_tensor.numel() == 0:
                        raise FXTraceDeclined(
                            "A nonempty view cannot use an empty input's null data pointer as its root"
                        )
                    token.used = True
                    used.add(token.source)
                    pointer = PointerSource(
                        token.source, numeric(offsets[id(value.base)])
                    )
                    extra_guards.append(
                        pointer_alignment_guard(pointer, 16, root_alignments, trace)
                    )
                    fields, tensor_map = lower_descriptor(
                        row.abi_index,
                        value,
                        descriptors[row.source_arg_index],
                        pointer,
                        numeric,
                    )
                    physical_fields.extend(fields)
                    tensor_maps.append(tensor_map)
                    continue
                if row.triton_type.startswith("*"):
                    token = tensors.get(id(value))
                    if (
                        token is None
                        or not token.valid
                        or type(token.source) not in (InputSource, BufferSource)
                    ):
                        raise FXTraceDeclined("Pointer argument has no traced source")
                    root = token.source
                    (alignment,) = (
                        pointer.alignment
                        for pointer in receipt.pointers
                        if pointer.formal == row.formal
                    )
                    token.used = True
                    used.add(root)
                    if value.numel() == 0:
                        source = ParameterSource("constant", 64, 0)
                    else:
                        if roots(value).root_tensor.numel() == 0:
                            raise FXTraceDeclined(
                                "A nonempty view cannot use an empty input's null data pointer as its root"
                            )
                        offset = offsets[id(value)]
                        source = PointerSource(root, numeric(offset))
                        extra_guards.append(
                            pointer_alignment_guard(
                                source, alignment, root_alignments, trace
                            )
                        )
                else:
                    late = lower_address_scalar(
                        value, row.triton_type, trace, expression
                    )
                    if late is None:
                        value = expression(value)
                        source = (
                            IntegerSource(value)
                            if type(value) is int
                            else ExpressionSource(value)
                        )
                    else:
                        source, guards = late
                        extra_guards.extend(guards)
                        mark_storage_used(source, "Address scalar")
                arguments.append(
                    CallArgument(
                        row.formal,
                        row.source_arg_index,
                        row.abi_index if descriptors else slot,
                        row.triton_type,
                        source,
                    )
                )
            grid = tuple(numeric(axis) for axis in event.grid)
            module = (
                event.owner.module
                if type(event) is DirectTritonInvokeEvent
                else provider.launchers[0].__globals__["runner"].__self__
            )
            try:
                specs = scratch_specs(module)
            except TritonScratchDeclined as error:
                raise FXTraceDeclined(str(error)) from error
            scratch = []
            for spec in specs:
                if spec.size == 0:
                    scratch.append(ParameterSource("constant", 64, 0))
                    continue
                size = IntExpr("constant", spec.size)
                for axis in grid:
                    size = IntExpr("multiply", None, (size, axis))
                source = BufferSource(f"scratch_{len(state.allocations)}")
                token = _Tensor(source, torch.uint8, (size,), (1,), used=True)
                state.allocations.append(token)
                root_alignments[source] = 256
                used.add(source)
                events.append(
                    Allocate(source.name, token.dtype, token.size, token.stride)
                )
                scratch.append(PointerSource(source, IntExpr("constant", 0)))
            if descriptors:
                from torch._inductor.runtime.cudagraph_boxed_replay import (
                    _PhysicalCall,
                    _PhysicalField,
                )

                physical_fields.extend(
                    _PhysicalField(
                        argument.call_arg_index,
                        0,
                        "pointer"
                        if argument.triton_type.startswith("*")
                        else argument.triton_type,
                        argument.source,
                    )
                    for argument in arguments
                )
                physical_fields.extend(
                    _PhysicalField(len(module.arg_tys) + index, 0, "pointer", source)
                    for index, source in enumerate(scratch)
                )
                bound = _PhysicalCall(
                    tuple(physical_fields),
                    TritonTmaModule(event.owner),
                    grid,
                    (),
                    tensor_maps=tuple(tensor_maps),
                )
                events.append(DirectPhysicalCall(event.owner, bound))
            else:
                events.append(
                    DirectKernelCall(
                        event.owner, tuple(arguments), grid, tuple(scratch)
                    )
                    if type(event) is DirectTritonInvokeEvent
                    else TerminalCall(provider, tuple(arguments), grid, tuple(scratch))
                )
        elif type(event) is ReinterpretEvent:
            token = tensors.get(id(event.source))
            if (
                token is None
                or not token.valid
                or event.tensor.dtype != token.dtype
                or event.tensor.device != event.source.device
            ):
                raise FXTraceDeclined(
                    "Tensor view has no traced allocation or input root"
                )
            resolution = roots.event(event)
            if resolution.root != token.source:
                raise FXTraceDeclined("Tensor view changed its recorded storage root")
            offsets[id(event.tensor)] = resolution.byte_offset
            tensors[id(event.tensor)] = _Tensor(
                resolution.root,
                token.dtype,
                tuple(expression(value) for value in event.size),
                tuple(expression(value) for value in event.stride),
            )
            aliases.add(id(event.tensor))
        else:
            raise FXTraceDeclined("Unsupported event in the terminal host program")

    allocations = tuple(
        OwnedBuffer(token.source, token.dtype, token.size, token.stride)
        for token in state.allocations
    )
    by_source = {layout.source: layout for layout in allocations}
    outputs, output_indices = [], {}
    for value in trace.outputs:
        if value is None:
            output = None
        elif type(value) in (int, torch.SymInt):
            output = IntegerOutput(expression(value))
        elif id(value) in output_indices:
            output = OutputReference(output_indices[id(value)])
        else:
            output_indices[id(value)] = len(outputs)
            token = tensors[id(value)]
            if id(value) in aliases:
                output = TensorViewOutput(
                    token.source,
                    token.size,
                    token.stride,
                    expression(offsets[id(value)] // token.dtype.itemsize),
                )
            elif type(token.source) is InputSource:
                output = BorrowedInputOutput(token.source)
            elif token.valid and token.source in by_source:
                output = by_source[token.source]
            else:
                raise FXTraceDeclined("Output allocation has no live traced source")
        outputs.append(output)
    rng = numeric(rng_prefix) if has_rng else None
    program = TerminalProgram(
        trace.contract,
        tuple(events),
        tuple(outputs),
        allocations,
        tuple(
            IntegerInput(f"boxed_{row.index}", row.index)
            for row in trace.contract.integer_ranges
        ),
        views,
        tuple(effects.values()),
        trace.compiler_binding,
        export_guards(trace, (*extra_guards, *host_integers.guards)),
        trace.compiler_binding.attachment.metadata.saved_input_indices
        if type(trace.compiler_binding) is _WrapperOrigin
        else (),
        rng=rng,
    )
    program.check()
    return program
