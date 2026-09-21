"""Complete compiler order for interleaved generated and owned CuTe calls."""

from dataclasses import dataclass
from types import FunctionType
from weakref import ref

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, bind_output_slots, BorrowedInputOutput, BufferSource, CallArgument, ExpressionSource, InputSource,
    IntegerOutput, IntegerSource, IntExpr, KernelCallRecord, OwnedBuffer, pointwise_expression_inputs,
)

from .descriptor import (
    _check_descriptors, bind_descriptors, BoundCuTeCalls, collect_descriptors,
    CuTeCall, CuTeDescriptors, DescriptorDeclined,
)
from .invocation import InvocationDeclined


class EnvelopeDeclined(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedCall:
    record: KernelCallRecord
    operations: tuple[str, ...]
    reads: tuple[InputSource | BufferSource, ...]
    writes: tuple[BufferSource, ...]


@dataclass(frozen=True)
class Release:
    source: InputSource | BufferSource


@dataclass(frozen=True)
class MixedEnvelope:
    version: int
    inputs: InputContract
    cute: CuTeDescriptors
    events: tuple[OwnedBuffer | AlignmentCopy | GeneratedCall | CuTeCall | Release, ...]
    outputs: tuple[OwnedBuffer | BorrowedInputOutput | IntegerOutput | None, ...]

    @property
    def allocations(self):
        return tuple(event for event in self.events if type(event) is OwnedBuffer)

    @property
    def generated_calls(self):
        return tuple(event.record for event in self.events if type(event) is GeneratedCall)


def _owned_call_sequence(calls):
    return (bool(calls) and all(call.provider_kind == "cute" for call in calls)
            and len({call.entry_global for call in calls}) == len(calls)
            and len({call.operands[1].source for call in calls}) == len(calls))


def _check_envelope(records):
    if (type(records) is not MixedEnvelope or type(records.version) is not int or records.version != 1
            or type(records.inputs) is not InputContract or type(records.events) is not tuple
            or type(records.outputs) is not tuple):
        raise EnvelopeDeclined("Unknown mixed compiler envelope schema")
    try:
        _check_descriptors(records.cute)
    except DescriptorDeclined as error:
        raise EnvelopeDeclined("Mixed envelope has unavailable CuTe descriptors") from error
    contract, cute = records.inputs, records.cute
    if (type(contract.kinds) is not tuple or len(contract.kinds) != len(cute.input_names)
            or any(kind not in ("tensor", "integer") for kind in contract.kinds)
            or type(contract.device_index) is not int or contract.device_index < 0
            or type(contract.tensor_inputs) is not tuple or type(contract.integer_ranges) is not tuple
            or not _owned_call_sequence(cute.calls)):
        raise EnvelopeDeclined("Expected complete CUDA inputs and distinct owned CuTe calls")
    tensor_slots = {index for index, kind in enumerate(contract.kinds) if kind == "tensor"}
    integer_slots = {index for index, kind in enumerate(contract.kinds) if kind == "integer"}
    facts = {row.source: row for row in cute.tensors}
    if (not tensor_slots or {row.boxed_index for row in cute.integer_inputs} != integer_slots
            or len(contract.tensor_inputs) != len(tensor_slots)
            or len(contract.integer_ranges) != len(integer_slots)):
        raise EnvelopeDeclined("Compiler input facts lost their boxed order")
    seen = set()
    for row in contract.tensor_inputs:
        if (type(row) is not TensorInput or type(row.index) is not int or row.index not in tensor_slots or row.index in seen
                or type(row.dtype) is not torch.dtype or type(row.size) is not tuple or type(row.stride) is not tuple
                or len(row.size) != len(row.stride) or any(type(value) not in (int, IntExpr) for value in (*row.size, *row.stride))):
            raise EnvelopeDeclined("Malformed compiler Tensor input")
        fact = facts.get(InputSource(row.index))
        if (fact is None or fact.dtype != row.dtype or fact.size != row.size or fact.stride != row.stride
                or fact.device != f"cuda:{contract.device_index}"):
            raise EnvelopeDeclined("InputContract and CuTe descriptors name different Tensor sources")
        seen.add(row.index)
    seen = set()
    for row in contract.integer_ranges:
        if (type(row) is not IntegerRange or type(row.index) is not int or row.index not in integer_slots
                or row.index in seen or type(row.lower) is not int or row.lower < 1
                or row.upper is not None and (type(row.upper) is not int or row.upper < row.lower)):
            raise EnvelopeDeclined("Malformed inherited integer range")
        seen.add(row.index)

    live = {InputSource(index) for index in tensor_slots}
    initialized, allocations, copied, used, operations = set(live), {}, set(), set(), set()
    count, cute_count = 0, 0
    for event in records.events:
        kind = type(event)
        if kind is Release:
            if type(event.source) not in (InputSource, BufferSource) or event.source not in live:
                raise EnvelopeDeclined("Release lost its live compiler source")
            live.remove(event.source)
        elif kind is OwnedBuffer:
            fact = facts.get(event.source)
            if (type(event.source) is not BufferSource or event.source in allocations or fact is None
                    or event.dtype != fact.dtype or event.size != fact.size or event.stride != fact.stride
                    or fact.device != f"cuda:{contract.device_index}"):
                raise EnvelopeDeclined("Allocation differs from its complete compiler layout")
            allocations[event.source] = event
            live.add(event.source)
        elif kind is AlignmentCopy:
            source = InputSource(event.input_index)
            if (type(event.input_index) is not int or type(event.before_call) is not int
                    or event.before_call != count or source not in live or source in used or source in copied):
                raise EnvelopeDeclined("Alignment copy moved after its first use or lost its source")
            copied.add(source)
        elif kind is GeneratedCall:
            call = event.record
            if (type(call) is not KernelCallRecord or type(call.occurrence) is not int or call.occurrence != count
                    or type(call.kernel_global) is not str or not call.kernel_global.isidentifier()
                    or call.grid_type not in ("Grid1D", "Grid2D", "Grid3D")
                    or call.launcher_grid is not None or call.constexprs != ()
                    or type(call.arguments) is not tuple or type(call.formals) is not tuple
                    or any(type(name) is not str for name in call.formals) or len(set(call.formals)) != len(call.formals)
                    or type(event.operations) is not tuple or not event.operations
                    or any(type(name) is not str or name in operations for name in event.operations)
                    or len(set(event.operations)) != len(event.operations)
                    or type(event.reads) is not tuple or type(event.writes) is not tuple
                    or len(set(event.reads)) != len(event.reads) or len(set(event.writes)) != len(event.writes)
                    or any(type(source) not in (InputSource, BufferSource) for source in event.reads)
                    or any(type(source) is not BufferSource for source in event.writes)
                    or not set(event.reads).issubset(live & initialized)
                    or not set(event.writes).issubset(live)):
                raise EnvelopeDeclined("Generated call lost its ordered compiler initialization facts")
            pointers = set()
            for index, arg in enumerate(call.arguments):
                if (type(arg) is not CallArgument or type(arg.call_arg_index) is not int or arg.call_arg_index != index
                        or type(arg.source_arg_index) is not int or not 0 <= arg.source_arg_index < len(call.formals)
                        or call.formals[arg.source_arg_index] != arg.formal or type(arg.triton_type) is not str):
                    raise EnvelopeDeclined("Malformed generated argument record")
                if type(arg.source) in (InputSource, BufferSource):
                    if not arg.triton_type.startswith("*"):
                        raise EnvelopeDeclined("Pointer source has a scalar compiler formal")
                    pointers.add(arg.source)
                elif type(arg.source) is IntegerSource:
                    if type(arg.source.value) is not int:
                        raise EnvelopeDeclined("Malformed literal compiler scalar")
                elif type(arg.source) is ExpressionSource:
                    origins = pointwise_expression_inputs(arg.source.expression)
                    if origins is None or not set(origins).issubset(integer_slots):
                        raise EnvelopeDeclined("Generated scalar lost its boxed source")
                else:
                    raise EnvelopeDeclined("Unsupported generated source")
            if pointers != set(event.reads) | set(event.writes):
                raise EnvelopeDeclined("Generated ABI pointers differ from compiler reads and writes")
            used.update(event.reads)
            initialized.update(event.writes)
            operations.update(event.operations)
            count += 1
        elif kind is CuTeCall:
            if cute_count == len(cute.calls) or event != cute.calls[cute_count]:
                raise EnvelopeDeclined("CuTe event differs from its ordered descriptor")
            source, destination = (row.source for row in event.operands)
            if (not count or source not in live & initialized
                    or type(source) is not BufferSource or destination not in live & initialized
                    or type(destination) is not BufferSource):
                raise EnvelopeDeclined("CuTe operands require initialized owned sources and fresh destinations")
            used.add(source)
            cute_count += 1
        else:
            raise EnvelopeDeclined("Unsupported mixed compiler event")
    outputs = bind_output_slots(records.outputs, tuple(allocations.values()), cute.input_names,
                                integer_slots, symbolic=bool(integer_slots))
    tensor_outputs = tuple(row.source for row in records.outputs if type(row) in (OwnedBuffer, BorrowedInputOutput))
    if (cute_count != len(cute.calls) or not records.outputs or outputs is None
            or tensor_outputs != cute.outputs
            or any(source not in live & initialized for source in tensor_outputs)
            or cute.calls[-1].operands[1].source not in cute.outputs
            or set(allocations) != {row.source for row in cute.tensors if type(row.source) is BufferSource}):
        raise EnvelopeDeclined("Mixed envelope lost its actual compiler output projection")


def collect_envelope(wrapper, descriptors):
    from torch._inductor.runtime._cudagraph._compiler.compiler_input_contract.collector import compiler_inputs, CompilerInputDeclined
    from torch._inductor import config, ir
    from torch._inductor.codegen.wrapper import (
        AllocateLine, AssertAlignmentLine, AssertSizeStrideLine, CommentLine, CuTeCallLine,
        DEFAULT_STREAM_IDX, EnterDeviceContextManagerLine, ExitDeviceContextManagerLine,
        FreeIfNotReusedLine, FreeLine, GroupedAssertSizeStrideLine, InputAlignmentLine,
        KernelCallLine, KernelDefinitionLine, LineContext, NullLine, PythonWrapperCodegen,
        SymbolicCallArgLine, _coor_enabled,
    )
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.virtualized import V

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
        raise EnvelopeDeclined("Wrapper has effects outside the mixed compiler envelope")
    try:
        inputs = compiler_inputs(wrapper)
        if collect_descriptors(wrapper) != descriptors:
            raise EnvelopeDeclined("Passed CuTe descriptors differ from live compiler sources")
    except (CompilerInputDeclined, DescriptorDeclined) as error:
        raise EnvelopeDeclined("Complete mixed compiler inputs or CuTe sources are unavailable") from error
    if not _owned_call_sequence(descriptors.calls):
        raise EnvelopeDeclined("Expected distinct owned CuTe invocations")
    device = torch.device("cuda", inputs.device_index)
    operations = tuple(graph.operations)
    by_name = {}
    required = set()
    for node in operations:
        if type(node) is ir.UserDefinedCuTeKernel:
            continue
        if (type(node) is not ir.ComputedBuffer or type(node.data) not in (ir.Pointwise, ir.Reduction)
                or node.get_name() in by_name or node.get_device() != device
                or node.get_inputs_that_alias_output() or node.get_mutation_names()):
            raise EnvelopeDeclined("Unsupported compiler operation in generated prefix")
        by_name[node.get_name()] = node
        if not any(wrapper._cudagraph_literal(value) == 0 for value in node.data.ranges):
            required.add(id(node))
    if sum(type(node) is ir.UserDefinedCuTeKernel for node in operations) != len(descriptors.calls):
        raise EnvelopeDeclined("CuTe descriptors lost the complete typed compiler operation sequence")
    names = descriptors.input_names
    graph_inputs = wrapper.get_graph_inputs()
    symbols = {}
    for row in descriptors.integer_inputs:
        value = graph_inputs[names[row.boxed_index]]
        symbols[value] = symbols[graph.sizevars.simplify(value)] = IntExpr("boxed", row.boxed_index)
    live = {}
    for index, name in enumerate(names):
        if inputs.kinds[index] != "tensor":
            continue
        node = graph_inputs[name]
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        live[name] = node, InputSource(index)
    initialized = set(live)
    allocations, events, seen, used, copied = {}, [], set(), set(), set()
    depth, count, cute_count = 0, 0, 0
    guards = tuple(graph.sizevars.shape_env.guards)
    try:
        for line in wrapper.lines:
            kind = type(line)
            if kind is EnterDeviceContextManagerLine:
                if depth or line.device_idx != device.index:
                    raise EnvelopeDeclined("Mixed envelope requires one ordinary CUDA stream")
                depth = 1
            elif kind is ExitDeviceContextManagerLine:
                if depth != 1:
                    raise EnvelopeDeclined("Unbalanced compiler device scope")
                depth = 0
            elif kind in (FreeLine, FreeIfNotReusedLine):
                pair = live.get(line.node.get_name())
                if (pair is None or pair[0] is not line.node
                        or kind is FreeIfNotReusedLine and (line.is_reused or line.comm_buffer)):
                    raise EnvelopeDeclined("Compiler release lost its exact live source")
                events.append(Release(pair[1]))
                del live[line.node.get_name()]
            elif kind is AllocateLine:
                node = line.node
                if (depth != 1 or line.comm_buffer or by_name.get(node.get_name()) is not node
                        or node.get_name() in live or node.get_name() in allocations):
                    raise EnvelopeDeclined("Unsupported, repeated or noncompiler allocation")
                owned = wrapper._cudagraph_owned_buffer(node, device, symbols)
                if owned is None:
                    raise EnvelopeDeclined("Allocation lacks complete compiler layout facts")
                allocations[node.get_name()] = node
                live[node.get_name()] = node, owned.source
                events.append(owned)
            elif kind is InputAlignmentLine:
                pair = live.get(line.name)
                if (pair is None or type(pair[1]) is not InputSource
                        or line.name in used or line.name in copied):
                    raise EnvelopeDeclined("Compiler normalization moved after its first use")
                copied.add(line.name)
                events.append(AlignmentCopy(pair[1].index, count))
            elif kind is KernelCallLine:
                group = line.cudagraph_computations
                if (depth != 1 or line.wrapper is not wrapper or line.device != device
                        or line.current_stream_idx not in (None, DEFAULT_STREAM_IDX)
                        or line.cudagraph_user is not None or type(group) is not tuple or not group):
                    raise EnvelopeDeclined("Generated call lacks its exact typed scheduler group")
                record = wrapper._cudagraph_call_record(line, count, {name: pair[1] for name, pair in live.items()}, symbols)
                if type(record) is not KernelCallRecord:
                    raise EnvelopeDeclined("Generated prefix call has incomplete argument or grid facts")
                reads, writes, internal = set(), set(), set()
                for node in group:
                    if (type(node) is not ir.ComputedBuffer or by_name.get(node.get_name()) is not node
                            or id(node) not in required or id(node) in seen
                            or tuple(node.data.get_pointwise_size()) != tuple(node.get_size())):
                        raise EnvelopeDeclined("Generated call changed its compiler computation identity or extent")
                    dependencies = node.get_read_writes()
                    if (any(type(dep) is not MemoryDep or dep.mode is not None for dep in (*dependencies.reads, *dependencies.writes))
                            or {dep.name for dep in dependencies.writes} != {node.get_name()}):
                        raise EnvelopeDeclined("Unsupported compiler read/write dependency")
                    for dependency in dependencies.reads:
                        name = dependency.name
                        if name not in internal:
                            if name not in live or name not in initialized:
                                raise EnvelopeDeclined("Generated prefix reads an uninitialized compiler source")
                            reads.add(name)
                    internal.add(node.get_name())
                    seen.add(id(node))
                    if node.get_name() in live:
                        if live[node.get_name()][0] is not node:
                            raise EnvelopeDeclined("Generated output changed its allocated compiler identity")
                        writes.add(node.get_name())
                pointer_names = set()
                for argument in record.arguments:
                    if type(argument.source) not in (InputSource, BufferSource):
                        continue
                    name, written, alignment = line.inductor_meta["cudagraph_parameter_provenance"][argument.formal]
                    if (type(written) is not bool or written != (name in writes)
                            or type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1)):
                        raise EnvelopeDeclined("Generated pointer facts differ from typed compiler writes")
                    pointer_names.add(name)
                if pointer_names != reads | writes:
                    raise EnvelopeDeclined("Generated call is missing a typed compiler pointer source")
                events.append(GeneratedCall(record, tuple(node.get_name() for node in group),
                    tuple(live[name][1] for name in sorted(reads)), tuple(live[name][1] for name in sorted(writes))))
                initialized.update(internal)
                used.update(reads)
                count += 1
            elif kind is CuTeCallLine:
                if cute_count == len(descriptors.calls) or depth != 1 or not count:
                    raise EnvelopeDeclined("Expected generated initialization before ordered CuTe calls")
                call = descriptors.calls[cute_count]
                source, destination = line.node.inputs
                if (line.entry_global != call.entry_global
                        or any(node.get_name() not in initialized
                            or live.get(node.get_name(), (None, None))[0] is not node
                            or live[node.get_name()][1] != fact.source
                            for node, fact in zip((source, destination), call.operands))):
                    raise EnvelopeDeclined("CuTe operands lack earlier compiler initialization")
                events.append(call)
                used.add(source.get_name())
                cute_count += 1
            elif kind is SymbolicCallArgLine:
                if wrapper._cudagraph_integer(line.arg, symbols) is None:
                    raise EnvelopeDeclined("Unsupported compiler scalar expression")
            elif kind not in (CommentLine, LineContext, KernelDefinitionLine, NullLine,
                              AssertSizeStrideLine, GroupedAssertSizeStrideLine, AssertAlignmentLine):
                raise EnvelopeDeclined("Unsupported interleaving in mixed compiler wrapper")
        if depth or cute_count != len(descriptors.calls) or seen != required:
            raise EnvelopeDeclined("Mixed envelope is missing a generated prefix computation")
        expected_copies = graph.inputs_to_check
        if (type(expected_copies) not in (list, tuple)
                or any(type(index) is not int or not 0 <= index < len(names)
                       or inputs.kinds[index] != "tensor" for index in expected_copies)
                or copied != {names[index] for index in expected_copies if names[index] in used}):
            raise EnvelopeDeclined("Mixed envelope lost compiler-required alignment copies")
        outputs = wrapper._cudagraph_owned_outputs(allocations, {name: pair[1] for name, pair in live.items()}, device, symbols)
        if outputs is None:
            raise EnvelopeDeclined("Mixed wrapper has unsupported output projection")
        records = MixedEnvelope(1, inputs, descriptors, tuple(events), outputs)
        _check_envelope(records)
        return records
    finally:
        if tuple(graph.sizevars.shape_env.guards) != guards:
            raise RuntimeError("Mixed envelope collection changed compiler guards")


@dataclass(frozen=True, eq=False)
class _Attachment:
    function: object
    code: object
    records: MixedEnvelope
    descriptors: object
    kernels: tuple[tuple[str, object], ...]
    snapshot: str


def attach_envelope(function, records):
    if (type(function) is not FunctionType or type(records) is not MixedEnvelope
            or function.__dict__.get("_cute_invocation_descriptors") is not records.cute
            or "_cute_mixed_attachment" in function.__dict__):
        raise EnvelopeDeclined("Expected the generated function's exact CuTe descriptor attachment")
    kernels = tuple((row.kernel_global, function.__globals__.get(row.kernel_global)) for row in records.generated_calls)
    function._cute_mixed_envelope = records
    function._cute_mixed_attachment = _Attachment(ref(function), function.__code__, records,
        function.__dict__.get("_cute_invocation_attachment"), kernels, repr(records))


class BoundMixedEnvelope:
    def __init__(self, function, attachment, binding):
        self.function = ref(function)
        self.records = attachment.records
        self.inputs = self.records.inputs
        self.cute = self.records.cute
        self.binding = binding
        self._seal = attachment, self.records, self.inputs, self.cute, binding

    def check(self):
        attachment, records, inputs, cute, binding = self._seal
        function = self.function()
        if (function is None or self.records is not records or self.inputs is not inputs
                or self.cute is not cute or self.binding is not binding or type(binding) is not BoundCuTeCalls
                or binding.function() is not function or binding.records is not cute):
            raise EnvelopeDeclined("Mixed compiler binding changed or expired")
        try:
            binding.check()
            _check_envelope(records)
            snapshot = repr(records)
            binding.check()
        except (DescriptorDeclined, InvocationDeclined) as error:
            raise EnvelopeDeclined("Associated CuTe compiler binding is unavailable") from error
        if (snapshot != attachment.snapshot or function.__code__ is not attachment.code
                or attachment.function() is not function
                or function.__dict__.get("_cute_mixed_attachment") is not attachment
                or function.__dict__.get("_cute_mixed_envelope") is not records
                or function.__dict__.get("_cute_invocation_attachment") is not attachment.descriptors
                or any(kernel is None or function.__globals__.get(name) is not kernel for name, kernel in attachment.kernels)):
            raise EnvelopeDeclined("Mixed compiler source or generated kernel globals changed")

    def __reduce_ex__(self, protocol):
        raise TypeError("Live mixed compiler bindings are not cache transport")


def bind_envelope(function):
    if type(function) is not FunctionType:
        raise EnvelopeDeclined("Expected the original generated function")
    attachment = function.__dict__.get("_cute_mixed_attachment")
    if type(attachment) is not _Attachment:
        raise EnvelopeDeclined("Mixed compiler envelope is unavailable")
    _check_envelope(attachment.records)
    try:
        binding = BoundMixedEnvelope(function, attachment, bind_descriptors(function))
        binding.check()
    except DescriptorDeclined as error:
        raise EnvelopeDeclined("Associated CuTe compiler binding is unavailable") from error
    return binding


def emit_envelope(result, function_name, records):
    _check_envelope(records)
    if type(function_name) is not str or not function_name.isidentifier():
        raise ValueError("Expected the generated function's exact global name")
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.envelope import GeneratedCall, MixedEnvelope, Release, attach_envelope')
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput')
    result.writeline("from torch._inductor.runtime.cudagraph_arg_mapping import AlignmentCopy, BorrowedInputOutput, CallArgument, ExpressionSource, IntegerOutput, IntegerSource, KernelCallRecord, OwnedBuffer")
    result.writeline(f"attach_envelope({function_name}, MixedEnvelope({records.version!r}, {records.inputs!r}, "
                     f"{function_name}._cute_invocation_descriptors, {records.events!r}, {records.outputs!r}))")
