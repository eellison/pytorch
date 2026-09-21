"""A complete mixed host program read directly from final typed wrapper IR."""

from dataclasses import fields
import json

from torch._inductor.runtime._cudagraph._compiler.compiler_input_contract.collector import compiler_inputs, CompilerInputDeclined
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph._compiler.retained_ir.prototype import _decode, _encode, RetentionDeclined
import torch
from torch._inductor import config, ir
from torch._inductor.codegen.wrapper import (
    AllocateLine, AssertAlignmentLine, AssertSizeStrideLine, CommentLine, CuTeCallLine,
    DEFAULT_STREAM_IDX, EnterDeviceContextManagerLine, ExitDeviceContextManagerLine,
    FreeIfNotReusedLine, FreeLine, GroupedAssertSizeStrideLine, InputAlignmentLine,
    KernelCallLine, KernelDefinitionLine, LineContext, NullLine, PythonWrapperCodegen,
    SymbolicCallArgLine, _coor_enabled,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, bind_output_slots, BorrowedInputOutput, BufferSource, CallArgument, ExpressionSource, InputSource, IntegerInput,
    IntegerSource, IntExpr, KernelCallRecord, OwnedBuffer, pointwise_expression_inputs,
)
from torch._inductor.virtualized import V

from .descriptor import _check_descriptors, collect_descriptors, CuTeCall, CuTeDescriptors, DescriptorDeclined, TensorFact
from .envelope import _owned_call_sequence, Release
from .program import MixedHostProgram


def retain_mixed_wrapper(wrapper):
    """Read compiler instructions; initialization and selected-device proof remain separate."""
    graph = V.graph
    if (type(wrapper) is not PythonWrapperCodegen or graph.cpp_wrapper or graph.aot_mode
            or graph.partition_maps or config.graph_partition or _coor_enabled()
            or graph.effectful_ops or graph.mutated_inputs or graph.constants
            or wrapper._multistream_alignment_copies or torch.version.hip is not None
            or config.cuda_backend != "triton" or not config.use_static_triton_launcher
            or config.generate_intermediate_hooks or config.profiler_mark_wrapper_call
            or config.profile_bandwidth or config.nan_asserts or config.annotate_training
            or config.incremental_autotune or config.triton.debug_sync_graph
            or config.triton.debug_sync_kernel or config.triton.proton_profiling or config.triton.store_cubin
            or config.aot_inductor.debug_intermediate_value_printer != "0"):
        raise RetentionDeclined("Wrapper has effects outside the retained mixed program")
    guards = tuple(graph.sizevars.shape_env.guards)
    try:
        try:
            inputs = compiler_inputs(wrapper)
            descriptors = collect_descriptors(wrapper)
        except (CompilerInputDeclined, DescriptorDeclined) as error:
            raise RetentionDeclined("Retained inputs or CuTe sources lack compiler authority") from error
        if not _owned_call_sequence(descriptors.calls):
            raise RetentionDeclined("Expected distinct owned CuTe invocations")
        names, graph_inputs = tuple(wrapper.get_graph_input_names()), wrapper.get_graph_inputs()
        symbols, live, operations, required = {}, {}, {}, set()
        device = torch.device("cuda", inputs.device_index)
        for index, name in enumerate(names):
            node = graph_inputs[name]
            if inputs.kinds[index] == "integer":
                symbols[node] = symbols[graph.sizevars.simplify(node)] = IntExpr("boxed", index)
            else:
                while type(node) in (ir.TensorBox, ir.StorageBox):
                    node = node.data
                live[name] = node, InputSource(index)
        for node in graph.operations:
            if type(node) is ir.UserDefinedCuTeKernel:
                continue
            if (type(node) is not ir.ComputedBuffer or type(node.data) not in (ir.Pointwise, ir.Reduction)
                    or node.get_name() in operations or node.get_device() != device
                    or node.get_inputs_that_alias_output() or node.get_mutation_names()):
                raise RetentionDeclined("Unsupported compiler operation in retained prefix")
            operations[node.get_name()] = node
            if not any(wrapper._cudagraph_literal(value) == 0 for value in node.data.ranges):
                required.add(id(node))
        if sum(type(node) is ir.UserDefinedCuTeKernel for node in graph.operations) != len(descriptors.calls):
            raise RetentionDeclined("CuTe descriptors lost the complete typed operation sequence")
        allocations, events, seen, used, copied = {}, [], set(), set(), set()
        depth, count, cute_count = 0, 0, 0
        for line in wrapper.lines:
            kind = type(line)
            if kind is EnterDeviceContextManagerLine:
                if depth or line.device_idx != device.index:
                    raise RetentionDeclined("Retained program requires one fixed CUDA stream")
                depth = 1
            elif kind is ExitDeviceContextManagerLine:
                if depth != 1:
                    raise RetentionDeclined("Unbalanced retained device scope")
                depth = 0
            elif kind in (FreeLine, FreeIfNotReusedLine):
                pair = live.get(line.node.get_name())
                if (pair is None or pair[0] is not line.node
                        or kind is FreeIfNotReusedLine and (line.is_reused or line.comm_buffer)):
                    raise RetentionDeclined("Release lost its exact live compiler source")
                events.append(Release(pair[1]))
                del live[line.node.get_name()]
            elif kind is AllocateLine:
                node = line.node
                if (depth != 1 or line.comm_buffer or operations.get(node.get_name()) is not node
                        or node.get_name() in live or node.get_name() in allocations):
                    raise RetentionDeclined("Allocation lost its original compiler identity")
                layout = wrapper._cudagraph_owned_buffer(node, device, symbols)
                if layout is None:
                    raise RetentionDeclined("Retained allocation has no complete layout")
                allocations[node.get_name()] = node
                live[node.get_name()] = node, layout.source
                events.append(layout)
            elif kind is InputAlignmentLine:
                pair = live.get(line.name)
                if (pair is None or type(pair[1]) is not InputSource
                        or line.name in used or line.name in copied):
                    raise RetentionDeclined("Normalization moved after its input's first use")
                copied.add(line.name)
                events.append(AlignmentCopy(pair[1].index, count))
            elif kind is KernelCallLine:
                group = line.cudagraph_computations
                if (depth != 1 or line.wrapper is not wrapper or line.device != device
                        or line.current_stream_idx not in (None, DEFAULT_STREAM_IDX)
                        or line.cudagraph_user is not None or type(group) is not tuple or not group):
                    raise RetentionDeclined("Call lost its typed compiler occurrence")
                for node in group:
                    if (type(node) is not ir.ComputedBuffer or operations.get(node.get_name()) is not node
                            or id(node) not in required or id(node) in seen):
                        raise RetentionDeclined("Retained call changed or duplicated its compiler operation")
                    seen.add(id(node))
                call = wrapper._cudagraph_call_record(line, count, {name: pair[1] for name, pair in live.items()}, symbols)
                if type(call) is not KernelCallRecord:
                    raise RetentionDeclined("Retained call has incomplete formal, scalar or grid sources")
                events.append(call)
                used.update(names[arg.source.index] for arg in call.arguments if type(arg.source) is InputSource)
                count += 1
            elif kind is CuTeCallLine:
                if cute_count == len(descriptors.calls):
                    raise RetentionDeclined("Retained program added a CuTe invocation")
                call = descriptors.calls[cute_count]
                if (depth != 1 or not count or type(line.node) is not ir.UserDefinedCuTeKernel
                        or line.entry_global != call.entry_global
                        or any(live.get(node.get_name(), (None, None))[0] is not node
                               or live[node.get_name()][1] != fact.source
                               for node, fact in zip(line.node.inputs, call.operands))
                        or any(type(row.source) is not BufferSource for row in call.operands)):
                    raise RetentionDeclined("CuTe invocation lost its owned compiler operands")
                events.append(call)
                cute_count += 1
            elif kind is SymbolicCallArgLine:
                if wrapper._cudagraph_integer(line.arg, symbols) is None:
                    raise RetentionDeclined("Unsupported retained scalar expression")
            elif kind not in (CommentLine, LineContext, KernelDefinitionLine, NullLine,
                              AssertSizeStrideLine, GroupedAssertSizeStrideLine, AssertAlignmentLine):
                raise RetentionDeclined("Unsupported interleaving in retained wrapper")
        if depth or cute_count != len(descriptors.calls) or seen != required:
            raise RetentionDeclined("Retained wrapper omitted a compiler operation")
        expected = graph.inputs_to_check
        if (type(expected) not in (tuple, list) or any(type(index) is not int or not 0 <= index < len(names)
                or inputs.kinds[index] != "tensor" for index in expected)
                or copied != {names[index] for index in expected if names[index] in used}):
            raise RetentionDeclined("Retained program omitted required normalization")
        outputs = wrapper._cudagraph_owned_outputs(allocations, {name: pair[1] for name, pair in live.items()}, device, symbols)
        if outputs is None:
            raise RetentionDeclined("Retained outputs lost their actual compiler sources")
        returned = {row.source for row in outputs if type(row) in (OwnedBuffer, BorrowedInputOutput)}
        events.extend(Release(pair[1]) for pair in live.values() if pair[1] not in returned)
        program = MixedHostProgram(1, inputs, tuple(events), outputs)
        validate_mixed_ir(program)
        try:
            current_descriptors, current_inputs = collect_descriptors(wrapper), compiler_inputs(wrapper)
        except (CompilerInputDeclined, DescriptorDeclined) as error:
            raise RetentionDeclined("Compiler sources became unsupported during retention") from error
        if current_descriptors != descriptors or current_inputs != inputs:
            raise RetentionDeclined("Compiler sources changed during retention")
        return program
    finally:
        if tuple(graph.sizevars.shape_env.guards) != guards:
            raise RuntimeError("Mixed retention changed compiler guards")


_CARRIERS = {kind.__name__: kind for kind in (
    MixedHostProgram, InputContract, IntegerRange, TensorInput, CuTeCall, TensorFact, Release,
)}


def _pack(value):
    if type(value) in _CARRIERS.values():
        return ("mixed", type(value).__name__, tuple(_pack(getattr(value, row.name)) for row in fields(value)))
    if type(value) is tuple:
        return tuple(_pack(item) for item in value)
    return value


def _unpack(value):
    if type(value) is not tuple:
        return value
    if value and value[0] == "mixed":
        if (len(value) != 3 or type(value[1]) is not str or value[1] not in _CARRIERS
                or type(value[2]) is not tuple or len(value[2]) != len(fields(_CARRIERS[value[1]]))):
            raise RetentionDeclined("Unknown mixed retained carrier")
        return _CARRIERS[value[1]](*(_unpack(item) for item in value[2]))
    return tuple(_unpack(item) for item in value)


def _tensor_source(source):
    return (type(source) is InputSource and type(source.index) is int and source.index >= 0
            or type(source) is BufferSource and type(source.name) is str and source.name.isidentifier())


def validate_mixed_ir(program):
    """Validate transport and source order, without inventing initialization effects."""
    _encode(_pack(program))
    if (type(program) is not MixedHostProgram or type(program.version) is not int or program.version != 1
            or type(program.inputs) is not InputContract or type(program.events) is not tuple
            or type(program.outputs) is not tuple or not program.outputs):
        raise RetentionDeclined("Unknown retained mixed program")
    inputs = program.inputs
    if (type(inputs.kinds) is not tuple or not inputs.kinds or type(inputs.device_index) is not int
            or inputs.device_index < 0 or any(kind not in ("tensor", "integer") for kind in inputs.kinds)
            or type(inputs.tensor_inputs) is not tuple or type(inputs.integer_ranges) is not tuple
            or any(type(row) is not TensorInput or type(row.index) is not int for row in inputs.tensor_inputs)
            or any(type(row) is not IntegerRange or type(row.index) is not int or type(row.lower) is not int
                   or row.lower < 1 or row.upper is not None and (type(row.upper) is not int or row.upper < row.lower)
                   for row in inputs.integer_ranges)):
        raise RetentionDeclined("Malformed retained input contract")
    tensor_indices = {index for index, kind in enumerate(inputs.kinds) if kind == "tensor"}
    integers = {index for index, kind in enumerate(inputs.kinds) if kind == "integer"}
    if (not tensor_indices or {row.index for row in inputs.tensor_inputs} != tensor_indices
            or len(inputs.tensor_inputs) != len(tensor_indices)
            or {row.index for row in inputs.integer_ranges} != integers or len(inputs.integer_ranges) != len(integers)):
        raise RetentionDeclined("Retained contract lost complete boxed slots or outputs")
    layouts = tuple(row for row in program.events if type(row) is OwnedBuffer)
    calls = tuple(row for row in program.events if type(row) is CuTeCall)
    tensor_outputs = tuple(row for row in program.outputs if type(row) in (OwnedBuffer, BorrowedInputOutput))
    if (any(not _tensor_source(row.source) for row in (*layouts, *tensor_outputs))
            or any(type(call.operands) is not tuple or any(type(row) is not TensorFact
                or not _tensor_source(row.source) for row in call.operands) for call in calls)):
        raise RetentionDeclined("Malformed retained Tensor source")
    facts = tuple(TensorFact(InputSource(row.index), row.dtype, f"cuda:{inputs.device_index}", row.size, row.stride)
                  for row in inputs.tensor_inputs)
    facts += tuple(TensorFact(row.source, row.dtype, f"cuda:{inputs.device_index}", row.size, row.stride) for row in layouts)
    descriptors = CuTeDescriptors(1, tuple(f"boxed_{index}" for index in range(len(inputs.kinds))),
        tuple(IntegerInput(f"boxed_{index}", index) for index in sorted(integers)), facts, calls,
        tuple(row.source for row in tensor_outputs))
    try:
        _check_descriptors(descriptors)
    except DescriptorDeclined as error:
        raise RetentionDeclined("Malformed retained Tensor or CuTe facts") from error
    if bind_output_slots(program.outputs, layouts, descriptors.input_names, integers, symbolic=bool(integers)) is None:
        raise RetentionDeclined("Retained outputs lost their owning, boxed or integer sources")
    if not _owned_call_sequence(calls):
        raise RetentionDeclined("Expected distinct retained owned CuTe invocations")
    live, allocated, used, copied = {InputSource(index) for index in tensor_indices}, {}, set(), set()
    count, cute_count = 0, 0
    for event in program.events:
        kind = type(event)
        if kind is Release:
            if not _tensor_source(event.source) or event.source not in live:
                raise RetentionDeclined("Retained release has no live source")
            live.remove(event.source)
        elif kind is OwnedBuffer:
            allocated[event.source] = event
            live.add(event.source)
        elif kind is AlignmentCopy:
            if (type(event.input_index) is not int or type(event.before_call) is not int
                    or event.before_call != count or InputSource(event.input_index) not in live
                    or event.input_index not in tensor_indices or event.input_index in copied
                    or InputSource(event.input_index) in used):
                raise RetentionDeclined("Retained normalization lost its ordered input")
            copied.add(event.input_index)
        elif kind is KernelCallRecord:
            if (type(event.occurrence) is not int or event.occurrence != count
                    or type(event.kernel_global) is not str or not event.kernel_global.isidentifier()
                    or type(event.formals) is not tuple or any(type(name) is not str for name in event.formals)
                    or len(set(event.formals)) != len(event.formals) or type(event.arguments) is not tuple
                    or event.grid_type not in ("Grid1D", "Grid2D", "Grid3D")
                    or event.launcher_grid is not None or type(event.constexprs) is not tuple or event.constexprs):
                raise RetentionDeclined("Malformed retained generated occurrence")
            for index, arg in enumerate(event.arguments):
                if (type(arg) is not CallArgument or type(arg.call_arg_index) is not int or arg.call_arg_index != index
                        or type(arg.source_arg_index) is not int or not 0 <= arg.source_arg_index < len(event.formals)
                        or arg.formal != event.formals[arg.source_arg_index] or type(arg.triton_type) is not str):
                    raise RetentionDeclined("Malformed retained formal correspondence")
                if type(arg.source) in (InputSource, BufferSource):
                    if not _tensor_source(arg.source) or arg.source not in live or not arg.triton_type.startswith("*"):
                        raise RetentionDeclined("Retained pointer has no live Tensor source")
                    used.add(arg.source)
                elif type(arg.source) is IntegerSource:
                    if type(arg.source.value) is not int or arg.triton_type not in ("i32", "i64", "constexpr"):
                        raise RetentionDeclined("Malformed retained literal scalar")
                elif type(arg.source) is ExpressionSource:
                    origins = pointwise_expression_inputs(arg.source.expression)
                    if arg.triton_type not in ("i32", "i64") or origins is None or not set(origins).issubset(integers):
                        raise RetentionDeclined("Retained scalar lost its boxed origin")
                else:
                    raise RetentionDeclined("Unsupported retained argument source")
            count += 1
        elif kind is CuTeCall:
            if not count or any(type(row.source) is not BufferSource or row.source not in live for row in event.operands):
                raise RetentionDeclined("Retained CuTe operands require owned live sources")
            cute_count += 1
        else:
            raise RetentionDeclined("Unknown retained mixed instruction")
    if (not cute_count or live != {row.source for row in tensor_outputs}
            or any(allocated.get(row.source) != row for row in program.outputs if type(row) is OwnedBuffer)
            or calls[-1].operands[1].source not in live):
        raise RetentionDeclined("Retained program lost its full output or release projection")


def export_mixed_ir(program):
    validate_mixed_ir(program)
    return json.dumps({"mixed_schema": 1, "program": _encode(_pack(program))}, sort_keys=True)


def reconstruct_mixed_ir(data):
    if type(data) is not str:
        raise RetentionDeclined("Expected retained mixed JSON text")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise RetentionDeclined("Malformed retained mixed JSON") from error
    if (type(payload) is not dict or set(payload) != {"mixed_schema", "program"}
            or type(payload["mixed_schema"]) is not int or payload["mixed_schema"] != 1):
        raise RetentionDeclined("Unknown retained mixed schema")
    program = _unpack(_decode(payload["program"]))
    validate_mixed_ir(program)
    return program
