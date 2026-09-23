"""Capture traced terminal calls and prepare their native boxed replay."""

import math
import struct
from dataclasses import dataclass, replace
from types import MethodType

import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import (
    Allocate,
    Normalize,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots,
    BorrowedInputOutput,
    BufferSource,
    CallArgument,
    ExpressionSource,
    InputSource,
    IntegerOutput,
    IntegerSource,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    storage_roots,
    TensorViewOutput,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _BoundCall,
    _make_replay,
    _NumericProgram,
    _ParameterProgram,
    _physical_scalar_bytes,
    _PHYSICAL_SCALAR_WIDTHS,
    _PhysicalCall,
    _pointer_value,
    _TensorMapField,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedKernelLaunch,
    UnsupportedCapture,
)
from torch._inductor.runtime.cudagraph_preparation import _Preparation
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_compat import autograd_profiler
from torch.cuda._utils import _check_cuda_bindings
from torch.utils._debug_mode import get_active_debug_mode
from torch.utils._ordered_set import OrderedSet


@dataclass(frozen=True)
class _CaptureCall:
    bound: _BoundCall | _PhysicalCall
    function: int
    grid: tuple[int, int, int]
    scratch: tuple[bytes, ...]
    authorities: tuple[object, ...]
    parameters: _ParameterProgram | None = None
    shared: int | None = None
    block: tuple[int, int, int] | None = None
    rng_fields: tuple = ()

    def check(self):
        for authority in self.authorities:
            authority.check()


def _release_steps(program, calls):
    from .frontend import DirectHostTable, DirectMemcpy, DirectMemset

    if not program.saved_input_indices or not program.allocations or not calls:
        return None
    # The release plan names kernel uses only. Keep inputs through submission
    # until it can represent non-kernel uses at their exact positions.
    if any(
        type(event) in (DirectMemset, DirectMemcpy, DirectHostTable)
        for event in program.events
    ):
        return None
    returned = OrderedSet(
        output.source.index
        for output in program.outputs
        if type(output) in (BorrowedInputOutput, TensorViewOutput)
        and type(output.source) is InputSource
    )
    candidates = OrderedSet(program.saved_input_indices) - returned
    if not candidates:
        return None
    last_use = dict.fromkeys(candidates, -1)
    for index, call in enumerate(calls):
        bound = call.bound
        sources = [
            argument.source
            for argument in (
                bound.fields if type(bound) is _PhysicalCall else bound.launch_arguments
            )
        ]
        if type(bound) is _PhysicalCall:
            sources.extend(bound.storage_sources)
            sources.extend(field.pointer for field in bound.tensor_maps)
        for source in sources:
            for root in storage_roots(source):
                if type(root) is InputSource and root.index in last_use:
                    last_use[root.index] = index
    drops = {}
    for index in sorted(candidates):
        drops.setdefault(last_use[index], []).append(index)
    steps = [("drop", index) for index in drops.get(-1, ())]
    allocations = {
        layout.source: index for index, layout in enumerate(program.allocations)
    }
    call_index = 0
    for event in program.events:
        if type(event) is Normalize:
            continue
        if type(event) is Allocate:
            steps.append(("allocate", allocations[BufferSource(event.value_id)]))
        else:
            steps.append(("kernel", call_index))
            steps.extend(("drop", index) for index in drops.get(call_index, ()))
            call_index += 1
    return tuple(steps)


def _capture_outputs(output_specs, inputs, buffers, numeric):
    outputs = []
    for output in output_specs:
        if output is None:
            outputs.append(None)
        elif type(output) is OutputReference:
            outputs.append(outputs[output.index])
        elif type(output) is BorrowedInputOutput:
            outputs.append(inputs[output.source.index])
        elif type(output) is IntegerOutput:
            outputs.append(
                output.value
                if type(output.value) is int
                else numeric.prepared_value(output.value)
            )
        elif type(output) is TensorViewOutput:
            source = output.source
            root = (
                inputs[source.index] if type(source) is InputSource else buffers[source]
            )
            size, stride, offset = tuple(
                tuple(
                    numeric.prepared_value(value) if type(value) is IntExpr else value
                    for value in values
                )
                for values in (output.size, output.stride, (output.offset,))
            )
            dtype = getattr(output, "dtype", None)
            if dtype is None:
                outputs.append(
                    torch._C._dynamo.guards._reinterpret_tensor(
                        root, size, stride, offset[0]
                    )
                )
            else:
                # a view with its own dtype: the offset is over the root's storage in
                # the view's units
                outputs.append(
                    torch.empty((0,), dtype=dtype, device=root.device).set_(
                        root.untyped_storage(), offset[0], size, stride
                    )
                )
        else:
            outputs.append(buffers[output.source])
    return tuple(outputs)


def prepare_terminal(program, example_inputs):
    from cuda.bindings import runtime

    from torch.cuda import graphs

    from .cute_types import CuTeCall
    from .direct_hosttrace import (
        _bind_capture_rng,
        _capture_byte_memset,
        _capture_h2d_copy,
        _prepare_host_tables,
        _render_host_table,
    )
    from .frontend import (
        DirectHostTable,
        DirectKernelCall,
        DirectMemcpy,
        DirectMemset,
        DirectPhysicalCall,
        TerminalCall,
    )

    program.check()
    if (
        type(example_inputs) not in (list, tuple)
        or len(example_inputs) != len(program.contract.kinds)
        or len(program.input_names) != len(example_inputs)
    ):
        raise UnsupportedCapture(
            "Terminal preparation requires the complete boxed inputs"
        )
    integers = {row.boxed_index for row in program.integer_inputs}
    if integers != {
        index for index, kind in enumerate(program.contract.kinds) if kind == "integer"
    }:
        raise UnsupportedCapture(
            "Terminal integer sources differ from the inherited input contract"
        )
    if any(
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        for index, value in enumerate(example_inputs)
        if index not in integers
    ):
        raise UnsupportedCapture(
            "Terminal Tensor inputs must be ordinary tensors or parameters"
        )
    if (
        not torch.cuda.is_initialized()
        or autograd_profiler._is_profiler_enabled
        or get_active_debug_mode()
        or any(
            (
                graphs._global_capture_start_hooks,
                graphs._global_capture_end_hooks,
                graphs._global_instantiate_hooks,
                graphs._global_replay_start_hooks,
                graphs._global_replay_end_hooks,
                graphs._global_destroy_hooks,
            )
        )
    ):
        raise UnsupportedCapture(
            "Terminal preparation requires ordinary warmup without instrumentation"
        )
    if (
        torch.version.hip is not None
        or torch.version.cuda is None
        or tuple(map(int, torch.version.cuda.split("."))) < (12, 8)
    ):
        raise UnsupportedCapture(
            "Native terminal preparation requires NVIDIA CUDA 12.8 or later"
        )

    numeric = _NumericProgram(program, example_inputs)
    output_slots = bind_output_slots(
        program.outputs,
        program.allocations,
        program.input_names,
        integers,
        symbolic=True,
    )
    if output_slots is None:
        raise UnsupportedCapture("Terminal outputs lost their traced sources")
    dimensions, allocations, calls, copies = {}, [], [], []
    memset_values, memcpy_values, host_tables = {}, {}, {}
    pinned_positions = set()
    buffer_indices = {
        layout.source: len(example_inputs) + index
        for index, layout in enumerate(program.allocations)
    }
    active, used_inputs = set(), set()
    for event in program.events:
        if type(event) is Normalize:
            index = event.input_index
            if (
                type(index) is not int
                or not 0 <= index < len(example_inputs)
                or index in integers
                or index in used_inputs
                or index in copies
                or type(event.before_call) is not int
                or event.before_call != len(calls)
            ):
                raise UnsupportedCapture(
                    "Terminal normalization must precede the input's first use"
                )
            copies.append(index)
            continue
        if type(event) is Allocate:
            source = BufferSource(event.value_id)
            if (
                source in active
                or type(event.size) is not tuple
                or type(event.stride) is not tuple
            ):
                raise UnsupportedCapture(
                    "Terminal allocations require distinct sources and explicit layouts"
                )
            size, stride = tuple(
                tuple(
                    numeric.values[numeric.add(value)]
                    if type(value) is IntExpr
                    else value
                    for value in values
                )
                for values in (event.size, event.stride)
            )
            if (
                len(size) != len(stride)
                or type(event.dtype) is not torch.dtype
                or any(
                    type(value) is not int or not 0 <= value < 2**63 for value in size
                )
                or any(
                    type(value) is not int or not 0 <= value < 2**63 for value in stride
                )
            ):
                raise UnsupportedCapture(
                    "Terminal allocation dimensions exceed the supported integer domain"
                )
            span = (
                0
                if 0 in size
                else 1 + sum((extent - 1) * step for extent, step in zip(size, stride))
            )
            if math.prod(size) >= 2**63 or span * event.dtype.itemsize >= 2**63:
                raise UnsupportedCapture(
                    "Terminal allocation storage size exceeds int64"
                )
            dimensions[source] = size, stride
            allocations.append(
                OwnedBuffer(source, event.dtype, event.size, event.stride)
            )
            active.add(source)
            continue
        if type(event) is DirectHostTable:
            event.owner.check()
            host_tables[event] = len(host_tables)
            for _, _, source in event.table.elements:
                if type(source) is PointerSource:
                    root = source.root
                    if type(root) is InputSource:
                        used_inputs.add(root.index)
                    elif root not in active:
                        raise UnsupportedCapture(
                            "Host table pointer precedes its allocation"
                        )
            continue
        if type(event) is DirectMemcpy:
            event.owner.check()
            for source in (event.source, event.destination):
                if type(source) is DirectHostTable:
                    if source not in host_tables:
                        raise UnsupportedCapture("Copy precedes its host table")
                    continue
                root = source.root
                if type(root) is InputSource:
                    used_inputs.add(root.index)
                elif root not in active:
                    raise UnsupportedCapture(
                        "Host copy pointer precedes its allocation"
                    )
            if type(event.source) is PointerSource:
                pinned_positions.add(event.source.root.index)
            memcpy_values[id(event)] = (
                None
                if type(event.source) is DirectHostTable
                else numeric.add(event.source.byte_offset),
                numeric.add(event.destination.byte_offset),
                numeric.add(event.byte_count),
            )
            continue
        if type(event) is DirectMemset:
            event.owner.check()
            if type(event.destination) is not PointerSource:
                raise UnsupportedCapture("Memset destination lost its traced pointer")
            root = event.destination.root
            if type(root) is InputSource:
                if not 0 <= root.index < len(example_inputs) or root.index in integers:
                    raise UnsupportedCapture("Memset destination is not a Tensor input")
                used_inputs.add(root.index)
            elif type(root) is not BufferSource or root not in active:
                raise UnsupportedCapture(
                    "Memset destination has no preceding allocation"
                )
            memset_values[id(event)] = (
                numeric.add(event.destination.byte_offset),
                numeric.add(event.byte_count),
            )
            continue
        if type(event) in (CuTeCall, DirectPhysicalCall):
            if type(event) is CuTeCall:
                event.receipt.check()
                entry = event.receipt.trace.event.entry
                views = tuple(
                    view
                    for view in program.views
                    if getattr(view, "entry", None) is entry
                )
                if not views:
                    raise UnsupportedCapture(
                        "CuTe call lost its original invocation view"
                    )
                authorities = (program.origin, *views, event.receipt)
            else:
                event.owner.check()
                authorities = (program.origin, event.owner)
            bound = event.bound
            if type(bound) is not _PhysicalCall:
                raise UnsupportedCapture("CuTe call lost its physical compiler binding")
            used_inputs.update(
                source.root.index
                for source in bound.storage_sources
                if type(source.root) is InputSource
            )
            parameters = None
            for field in bound.fields:
                source = field.source
                if type(source) is ParameterSource:
                    if source.width != {"pointer": 64, "i32": 32, "i64": 64}.get(
                        field.kind
                    ):
                        raise UnsupportedCapture(
                            "CuTe computed field differs from its physical ABI width"
                        )
                    if parameters is None:
                        parameters = _ParameterProgram(
                            numeric, len(example_inputs), buffer_indices
                        )
                    parameters.add(source)
                    for root in storage_roots(source):
                        if type(root) is InputSource:
                            used_inputs.add(root.index)
                        elif type(root) is not BufferSource or root not in active:
                            raise UnsupportedCapture(
                                "CuTe encoded pointer has no preceding traced allocation"
                            )
                    continue
                root = source.root if type(source) is PointerSource else source
                if type(source) is PointerSource:
                    numeric.add(source.byte_offset)
                if type(root) is InputSource:
                    if (
                        type(root.index) is not int
                        or not 0 <= root.index < len(example_inputs)
                        or root.index in integers
                        or field.kind != "pointer"
                    ):
                        raise UnsupportedCapture(
                            "CuTe pointer has no original Tensor input"
                        )
                    used_inputs.add(root.index)
                elif type(root) is BufferSource:
                    if root not in active or field.kind != "pointer":
                        raise UnsupportedCapture(
                            "CuTe pointer has no preceding traced allocation"
                        )
                elif type(source) in (IntegerSource, ExpressionSource):
                    if field.kind not in _PHYSICAL_SCALAR_WIDTHS:
                        raise UnsupportedCapture(
                            "CuTe scalar has an unsupported physical type"
                        )
                    value = (
                        source.value
                        if type(source) is IntegerSource
                        else numeric.values[numeric.add(source.expression)]
                    )
                    bits = 32 if field.kind == "i32" else 64
                    if type(value) is not int or not -(
                        2 ** (bits - 1)
                    ) <= value < 2 ** (bits - 1):
                        raise UnsupportedCapture(
                            "CuTe scalar exceeds its selected ABI type"
                        )
                else:
                    raise UnsupportedCapture(
                        "CuTe field has an unsupported traced source"
                    )
            for field in bound.tensor_maps:
                if (
                    type(field) is not _TensorMapField
                    or type(field.pointer) is not PointerSource
                    or type(field.parameter) is not int
                    or not 0 <= field.parameter < len(bound.module.parameter_sizes)
                    or bound.module.parameter_sizes[field.parameter] != 128
                ):
                    raise UnsupportedCapture(
                        "Tensor map must cover its complete CUDA descriptor parameter"
                    )
                root = field.pointer.root
                if type(root) is InputSource and root.index not in integers:
                    used_inputs.add(root.index)
                elif type(root) is not BufferSource or root not in active:
                    raise UnsupportedCapture(
                        "Tensor map pointer has no preceding live storage root"
                    )
                field.binding(numeric, len(example_inputs), buffer_indices)
            if type(bound.grid) is not tuple or len(bound.grid) != 3:
                raise UnsupportedCapture("CuTe grid requires three traced expressions")
            grid = tuple(numeric.values[numeric.add(axis)] for axis in bound.grid)
            if not (
                0 < grid[0] < 2**31 and 0 < grid[1] <= 65535 and 0 < grid[2] <= 65535
            ):
                raise UnsupportedCapture("CuTe grid exceeds CUDA launch bounds")
            shared = None
            if bound.shared is not None:
                if bound.module.shared is not None:
                    raise UnsupportedCapture(
                        "Dynamic shared recipe differs from its module ownership"
                    )
                shared = numeric.values[numeric.add(bound.shared)]
            elif bound.module.shared is None:
                raise UnsupportedCapture(
                    "Dynamic shared owner lacks its traced request"
                )
            block = None
            if bound.block is not None:
                if type(bound.block) is not tuple or len(bound.block) != 3:
                    raise UnsupportedCapture("Block requires three traced expressions")
                block = tuple(numeric.values[numeric.add(axis)] for axis in bound.block)
                if any(value <= 0 for value in block):
                    raise UnsupportedCapture("Block dimensions must be positive")
            calls.append(
                _CaptureCall(
                    bound,
                    bound.module.function,
                    grid,
                    (),
                    authorities,
                    parameters,
                    shared,
                    block,
                    event.rng_fields if type(event) is DirectPhysicalCall else (),
                )
            )
            continue
        if type(event) not in (TerminalCall, DirectKernelCall):
            raise UnsupportedCapture("Unsupported terminal host event")
        if type(event) is DirectKernelCall:
            event.owner.check()
            authorities = (program.origin, event.owner)
            module = event.owner.module
        else:
            views = tuple(
                view
                for view in program.views
                if any(provider is event.provider for provider in view.providers)
            )
            facts = tuple(
                receipt
                for receipt in program.effects
                if receipt.provider is event.provider
            )
            if not views or len(facts) != 1:
                raise UnsupportedCapture(
                    "Terminal call lost its selected view or unique compiler facts"
                )
            authorities = (program.origin, *views, facts[0])
            (launcher,) = event.provider.launchers
            runner = launcher.__globals__.get("runner")
            module = runner.__self__ if type(runner) is MethodType else None
        if (
            type(module) is not StaticallyLaunchedCudaKernel
            or module.device_agnostic
            or module.module is None
            or type(module.function) is not int
            or module.function <= 0
            or module._has_tensordesc
            or type(event.arguments) is not tuple
        ):
            raise UnsupportedCapture(
                "Terminal call has no supported selected CUDA module"
            )
        abi = []
        parameters = None
        bound = _BoundCall(event.arguments, module, event.grid, event.scratch)
        for index, argument in enumerate(bound.launch_arguments):
            if type(argument) is not CallArgument or argument.call_arg_index != index:
                raise UnsupportedCapture(
                    "Terminal arguments must retain their exact ABI order"
                )
            source = argument.source
            root = source.root if type(source) is PointerSource else source
            if type(source) is PointerSource:
                numeric.add(source.byte_offset)
            if type(source) is ParameterSource:
                pointer = argument.triton_type.startswith("*")
                bits = (
                    64 if pointer else {"i32": 32, "i64": 64}.get(argument.triton_type)
                )
                if (
                    bits is None
                    or source.width != bits
                    or pointer
                    and source != ParameterSource("constant", 64, 0)
                ):
                    raise UnsupportedCapture(
                        "Terminal computed scalar differs from its selected ABI width"
                    )
                if parameters is None:
                    parameters = _ParameterProgram(
                        numeric, len(example_inputs), buffer_indices
                    )
                parameters.add(source)
                for root in storage_roots(source):
                    if type(root) is InputSource:
                        used_inputs.add(root.index)
                    elif type(root) is not BufferSource or root not in active:
                        raise UnsupportedCapture(
                            "Terminal scalar pointer has no preceding traced allocation"
                        )
                abi.append("O" if pointer else "i" if bits == 32 else "l")
            elif type(root) is InputSource:
                if (
                    type(root.index) is not int
                    or not 0 <= root.index < len(example_inputs)
                    or root.index in integers
                    or not argument.triton_type.startswith("*")
                ):
                    raise UnsupportedCapture(
                        "Terminal pointer has no original Tensor input"
                    )
                used_inputs.add(root.index)
                abi.append("O")
            elif type(root) is BufferSource:
                if root not in active or not argument.triton_type.startswith("*"):
                    raise UnsupportedCapture(
                        "Terminal pointer has no preceding traced allocation"
                    )
                abi.append("O")
            elif type(source) in (IntegerSource, ExpressionSource):
                if argument.triton_type not in ("i32", "i64"):
                    raise UnsupportedCapture(
                        "Terminal scalar has an unsupported ABI type"
                    )
                value = (
                    source.value
                    if type(source) is IntegerSource
                    else numeric.values[numeric.add(source.expression)]
                )
                bits = 32 if argument.triton_type == "i32" else 64
                if type(value) is not int or not -(2 ** (bits - 1)) <= value < 2 ** (
                    bits - 1
                ):
                    raise UnsupportedCapture(
                        "Terminal scalar exceeds its selected ABI type"
                    )
                abi.append("i" if bits == 32 else "l")
            else:
                raise UnsupportedCapture(
                    "Terminal argument has an unsupported traced source"
                )
        if "".join(abi) != module.arg_tys + "O" * len(event.scratch):
            raise UnsupportedCapture(
                "Terminal arguments do not cover the selected module ABI"
            )
        if type(event.grid) is not tuple or len(event.grid) != 3:
            raise UnsupportedCapture("Terminal grid requires three traced expressions")
        grid = tuple(numeric.values[numeric.add(axis)] for axis in event.grid)
        if not (0 < grid[0] < 2**31 and 0 < grid[1] <= 65535 and 0 < grid[2] <= 65535):
            raise UnsupportedCapture("Terminal grid exceeds CUDA launch bounds")
        scratch = []
        if not event.scratch:
            for present, size in (
                (module.has_global_scratch, module.global_scratch_size),
                (module.has_profile_scratch, module.profile_scratch_size),
            ):
                if (
                    type(present) is not bool
                    or present
                    and (type(size) is not int or size != 0)
                ):
                    raise UnsupportedCapture(
                        "Terminal capture requires explicit launcher-owned scratch"
                    )
                if present:
                    scratch.append(bytes(struct.calcsize("P")))
        calls.append(
            _CaptureCall(
                bound, module.function, grid, tuple(scratch), authorities, parameters
            )
        )
    if tuple(allocations) != program.allocations:
        raise UnsupportedCapture(
            "Terminal events do not cover the traced allocations and calls"
        )
    for output in program.outputs:
        if type(output) is IntegerOutput and type(output.value) is IntExpr:
            numeric.add(output.value)
        elif type(output) is TensorViewOutput:
            for value in (*output.size, *output.stride, output.offset):
                if type(value) is IntExpr:
                    numeric.add(value)
    program.check()

    rng = getattr(program, "rng", None)
    if rng is not None:
        numeric.add(rng)
    release_steps = _release_steps(program, calls)
    table_slots, table_values = _prepare_host_tables(
        tuple(event.table for event in host_tables), numeric
    )
    capture_events, recorded_memsets, recorded_memcpys = [], [], []
    normalize = torch._C._dynamo.guards.copy_if_misaligned
    state = _Preparation(program, (), tuple(example_inputs), list(example_inputs))
    try:
        modules = {id(call.bound.module): call.bound.module for call in calls}
        for module in modules.values():
            state.borrows.append(module._borrow_for_cudagraph())
        with torch.cuda.device(program.contract.device_index):
            state.stream = torch.cuda.current_stream()
            state.capture_stream = torch.cuda.Stream()
            for index in copies:
                state.normalized[index] = normalize(state.originals[index])
            state.graph = torch.cuda.CUDAGraph(keep_graph=True)
            program.check()

            def address_of(source, index):
                root = source.root
                base = (
                    state.normalized[root.index]
                    if type(root) is InputSource
                    else state.buffers[root]
                )
                return _pointer_value(base.data_ptr(), numeric.values[index])

            generator = (
                None
                if rng is None
                else torch.cuda.default_generators[program.contract.device_index]
            )
            with torch.cuda.graph(state.graph, stream=state.capture_stream):
                philox = None
                if generator is not None:
                    philox = torch._C._host_trace_generator_capture_pointers(generator)
                    if philox[2] != 0:
                        raise UnsupportedCapture(
                            "Mixed RNG capture has a preceding unrecorded draw"
                        )
                for event in program.events:
                    if type(event) is Allocate:
                        source = BufferSource(event.value_id)
                        size, stride = dimensions[source]
                        state.buffers[source] = (
                            torch._C._dynamo.guards._empty_strided_cuda(
                                size, stride, event.dtype
                            )
                        )
                        continue
                    if type(event) is Normalize:
                        value = state.normalized[event.input_index]
                        if normalize(value) is not value:
                            raise UnsupportedCapture(
                                "Pre-normalized terminal input changed during capture"
                            )
                        continue
                    if type(event) is DirectHostTable:
                        event.owner.check()
                        index = host_tables[event]
                        _render_host_table(
                            table_slots[index][0],
                            table_values[index],
                            numeric,
                            address_of,
                        )
                        continue
                    if type(event) is DirectMemcpy:
                        event.owner.check()
                        src_index, dst_index, count_index = memcpy_values[id(event)]
                        source = (
                            host_tables[event.source]
                            if type(event.source) is DirectHostTable
                            else event.source
                        )
                        address = (
                            table_slots[source][0].data_ptr()
                            if type(source) is int
                            else address_of(source, src_index)
                        )
                        node, recorded = _capture_h2d_copy(
                            address_of(event.destination, dst_index),
                            address,
                            numeric.values[count_index],
                            state.capture_stream.cuda_stream,
                        )
                        recorded_memcpys.append(
                            (node, source, event.destination, event.byte_count)
                        )
                        capture_events.append(recorded)
                        continue
                    if type(event) is DirectMemset:
                        event.owner.check()
                        root = event.destination.root
                        base = (
                            state.normalized[root.index]
                            if type(root) is InputSource
                            else state.buffers[root]
                        )
                        offset_index, bytes_index = memset_values[id(event)]
                        address = _pointer_value(
                            base.data_ptr(), numeric.values[offset_index]
                        )
                        node, recorded = _capture_byte_memset(
                            address,
                            numeric.values[bytes_index],
                            event.value,
                            state.capture_stream.cuda_stream,
                        )
                        recorded_memsets.append(
                            (node, event.destination, event.byte_count, event.value)
                        )
                        capture_events.append(recorded)
                        continue
                    call = calls[len(state.launches)]
                    call.check()
                    module = call.bound.module
                    if module.function != call.function:
                        raise UnsupportedCapture(
                            "Selected terminal function changed before capture"
                        )
                    physical = type(call.bound) is _PhysicalCall
                    arguments = []
                    images = (
                        [bytearray(size) for size in module.parameter_sizes]
                        if physical
                        else []
                    )
                    parameter_values = (
                        ()
                        if call.parameters is None
                        else call.parameters.evaluate(state.normalized, state.buffers)
                    )
                    if physical:
                        for parameter, offset, data in call.bound.constants:
                            images[parameter][offset : offset + len(data)] = data
                        if call.rng_fields:
                            if philox is None:
                                raise UnsupportedCapture(
                                    "RNG fields lost their generator increment"
                                )
                            prepared_bound = _bind_capture_rng(
                                call.bound, images, call.rng_fields, philox
                            )
                            call = replace(call, bound=prepared_bound)
                            calls[len(state.launches)] = call
                    for argument in (
                        call.bound.fields if physical else call.bound.launch_arguments
                    ):
                        source = argument.source
                        root = source.root if type(source) is PointerSource else source
                        if type(source) is ParameterSource:
                            value = parameter_values[
                                call.parameters.prepared_index(source)
                            ]
                        elif type(root) is InputSource:
                            value = state.normalized[root.index]
                        elif type(root) is BufferSource:
                            value = state.buffers[root]
                        elif type(source) is IntegerSource:
                            value = source.value
                        else:
                            value = numeric.prepared_value(source.expression)
                        if type(source) is PointerSource:
                            value = _pointer_value(
                                value.data_ptr(),
                                numeric.prepared_value(source.byte_offset),
                            )
                        arguments.append(value)
                        kind = argument.kind if physical else argument.triton_type
                        payload = (
                            struct.pack("i" if source.width == 32 else "q", value)
                            if type(source) is ParameterSource
                            else struct.pack("P", value)
                            if type(source) is PointerSource
                            else struct.pack("P", value.data_ptr())
                            if type(source) in (InputSource, BufferSource)
                            else _physical_scalar_bytes(kind, value)
                        )
                        if physical:
                            images[argument.parameter][
                                argument.byte_offset : argument.byte_offset
                                + len(payload)
                            ] = payload
                        else:
                            images.append(payload)
                    if physical:
                        for field in call.bound.tensor_maps:
                            images[field.parameter][:] = field.encode(
                                numeric,
                                state.normalized,
                                state.buffers,
                                buffer_indices,
                            )
                    images = tuple(bytes(image) for image in images)
                    stream = state.capture_stream.cuda_stream
                    before = torch._C._cuda_get_capture_frontier(stream)
                    if physical:
                        options = {}
                        if call.shared is not None:
                            options["shared"] = call.shared
                        if call.block is not None:
                            options["block"] = call.block
                        module.launch(images, call.grid, stream, **options)
                    elif call.bound.scratch:
                        module.C_impl._launch_kernel(
                            module.function,
                            *call.grid,
                            module.num_warps,
                            module.shared,
                            module.arg_tys + "O" * len(call.bound.scratch),
                            tuple(arguments),
                            stream,
                        )
                    else:
                        module.run(*call.grid, stream, *arguments)
                    after = torch._C._cuda_get_capture_frontier(stream)
                    call.check()
                    state.launches.append(
                        RecordedKernelLaunch(
                            stream,
                            before,
                            after,
                            call.function,
                            (*images, *call.scratch),
                        )
                    )
                    capture_events.append(state.launches[-1])
                state.outputs = _capture_outputs(
                    program.outputs, state.normalized, state.buffers, numeric
                )
            program.check()
            if len(state.launches) != len(calls) or set(state.buffers) != set(
                dimensions
            ):
                raise UnsupportedCapture(
                    "Captured terminal calls or allocation owners are incomplete"
                )
            for layout in program.allocations:
                value = state.buffers[layout.source]
                size, stride = dimensions[layout.source]
                if (
                    type(value) is not torch.Tensor
                    or value._base is not None
                    or value.storage_offset() != 0
                    or value.dtype != layout.dtype
                    or value.device
                    != torch.device("cuda", program.contract.device_index)
                    or tuple(value.size()) != size
                    or value.stride() != stride
                ):
                    raise UnsupportedCapture(
                        "Captured terminal allocation differs from its traced layout"
                    )
            state.graph.instantiate()
            state.instantiated = True
            _check_cuda_bindings(
                runtime.cudaGraphUpload(
                    state.graph.raw_cuda_graph_exec(), state.stream.cuda_stream
                )
            )
            state.stream.synchronize()
            state.capture_stream.synchronize()
            program.check()
            state.entry = _make_replay(
                state.graph,
                len(program.input_names),
                program.allocations,
                output_slots,
                tuple(copies),
                tuple(call.bound for call in calls),
                tuple(state.launches),
                state.buffers,
                state.stream,
                numeric=numeric,
                copy_if_misaligned=normalize,
                resources=(program, *table_slots),
                capture_inputs=state.normalized,
                release_steps=release_steps,
                memsets=tuple(recorded_memsets),
                capture_events=tuple(capture_events),
                host_tables=tuple(
                    (
                        tuple(slot.data_ptr() for slot in slots),
                        event.table.nbytes,
                        tuple(
                            (offset, width, source)
                            for offset, width, source, _ in values
                        ),
                    )
                    for event, slots, values in zip(
                        host_tables, table_slots, table_values, strict=True
                    )
                ),
                memcpys=tuple(recorded_memcpys),
                rng=None if rng is None else (generator, rng),
                **(
                    {"pinned_positions": tuple(sorted(pinned_positions))}
                    if pinned_positions
                    else {}
                ),
            )
            if rng is None:
                state.entry._retire_capture_pool()
            program.check()
            state.borrows.clear()
            return state.entry
    except BaseException:
        state.abort()
        raise
