"""Retained compiler metadata and its serialization."""

from dataclasses import dataclass, fields
import json

import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import Allocate, InputAssertion, Normalize, Reinterpret
from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    AllocateLine, AssertAlignmentLine, AssertSizeStrideLine, GroupedAssertSizeStrideLine,
    InputAlignmentLine, KernelCallLine, PythonWrapperCodegen, ReuseLine,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, bind_alignment_copies, bind_wrapper_allocations,
    BorrowedInputOutput, BufferSource, CallArgument, ExpressionSource, InputSource,
    IntegerInput, IntegerOutput, IntegerSource, IntExpr, KernelCallRecord, OwnedBuffer, WrapperCallRecords,
)
from torch._inductor.runtime.cudagraph_multikernel import (
    call_sources, KernelCallAlternative, MultiKernelCallRecord,
)


class RetentionDeclined(ValueError):
    pass


@dataclass(frozen=True)
class CallSite:
    call: KernelCallRecord
    pointer_writes: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class MultiCallSite:
    call: MultiKernelCallRecord
    alternatives: tuple[CallSite, ...]


@dataclass(frozen=True)
class UserCallSite:
    call: KernelCallRecord
    pointer_effects: tuple[tuple[str, int, bool, bool], ...]


@dataclass(frozen=True)
class RetainedIR:
    records: WrapperCallRecords
    events: tuple[Allocate | Reinterpret | Normalize | CallSite | MultiCallSite | UserCallSite, ...]
    input_assertions: tuple[InputAssertion, ...]


def _retain_user_call(wrapper, line, call, records, symbols, allocated_nodes):
    from torch._higher_order_ops.triton_kernel_wrap import TensorAccesses
    from torch._inductor.dependencies import ReadWrites, StarDep

    op = line.cudagraph_user
    if (type(op) is not ir.UserDefinedTritonKernel or type(op.arg_accesses) is not TensorAccesses
            or type(op.arg_accesses.read_writes) is not ReadWrites or call.grid_type != "FixedGrid"):
        raise RetentionDeclined("User call has no exact compiler HOP effects for selected-receipt binding")
    input_names, inputs = tuple(wrapper.get_graph_input_names()), wrapper.get_graph_inputs()
    if input_names != records.input_names:
        raise RetentionDeclined("User call lost its original compiler boxed order")
    sources, nodes = {}, {}
    for index, name in enumerate(input_names):
        node = inputs[name]
        if isinstance(node, sympy.Symbol):
            continue
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        if type(node) is not ir.InputBuffer or node.get_name() != name:
            raise RetentionDeclined("User call has no exact compiler input source")
        sources[name], nodes[name] = InputSource(index), node
    for name, node in allocated_nodes.items():
        if name in sources:
            raise RetentionDeclined("Compiler allocation collides with an input source")
        sources[name], nodes[name] = BufferSource(name), node
    for node in line.raw_args[:-3]:
        if type(node) in (ir.InputBuffer, ir.ComputedBuffer) and nodes.get(node.get_name()) is not node:
            raise RetentionDeclined("User pointer differs from its live compiler allocation or input")
    if wrapper._cudagraph_call_record(line, call.occurrence, sources, symbols) != call:
        raise RetentionDeclined("User effects lost their exact compiler call correspondence")
    pointers = tuple(arg for arg in call.arguments if type(arg.source) in (InputSource, BufferSource))
    reads, writes = op.arg_accesses.read_writes.reads, op.arg_accesses.read_writes.writes
    names = {arg.formal for arg in pointers}
    if any(type(dep) is not StarDep or type(dep.name) is not str or dep.name not in names
           for dep in (*reads, *writes)):
        raise RetentionDeclined("User HOP effects have an unsupported dependency")
    reads, writes = {dep.name for dep in reads}, {dep.name for dep in writes}
    return UserCallSite(call, tuple((arg.formal, arg.source_arg_index, arg.formal in reads, arg.formal in writes)
                                   for arg in pointers))


def retain_wrapper(wrapper: PythonWrapperCodegen, records: WrapperCallRecords | None = None) -> RetainedIR:
    """Retain the compiler hook's exact records while its typed IR remains alive."""
    if type(wrapper) is not PythonWrapperCodegen:
        raise RetentionDeclined("Expected the final Python wrapper IR")
    if records is None:
        records = wrapper.collect_cudagraph_call_records()
    if (type(records) is not WrapperCallRecords or type(records.version) is not int
            or records.version not in (3, 4)):
        raise RetentionDeclined("The compiler must prove a complete V3 or V4 contract")
    inputs = wrapper.get_graph_inputs()
    bindings = wrapper._cudagraph_integer_bindings()
    if bindings is None or bindings[0] != records.integer_inputs:
        raise RetentionDeclined("Integer origin changed while retaining compiler IR")
    _, symbols = bindings
    device = torch.device("cuda", records.device_index)
    events, assertions, allocated_nodes = [], [], {}
    call_index = 0

    def retain_assertion(name, size, stride, label):
        if name not in records.input_names:
            raise RetentionDeclined("Only compiler input layout assertions are retained")
        index = records.input_names.index(name)
        node = inputs[name]
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        if type(node) not in (ir.InputBuffer, ir.DonatedBuffer) or type(node.get_layout()) is not ir.FixedLayout:
            raise RetentionDeclined("Assertion has no exact input layout")
        layout = node.get_layout()
        if (size != wrapper.codegen_python_shape_tuple(layout.size)
                or stride != wrapper.codegen_python_shape_tuple(layout.stride)):
            raise RetentionDeclined("Rendered assertion differs from the retained typed input layout")
        retained_size = tuple(wrapper._cudagraph_integer(value, symbols) for value in layout.size)
        retained_stride = tuple(wrapper._cudagraph_integer(value, symbols) for value in layout.stride)
        if any(value is None for value in (*retained_size, *retained_stride)):
            raise RetentionDeclined("Assertion expression exceeds the current numeric recipe")
        assertions.append(InputAssertion(InputSource(index), retained_size, retained_stride, label))

    for line in wrapper.lines:
        if type(line) is AllocateLine:
            layout = wrapper._cudagraph_owned_buffer(line.node, device, symbols)
            if layout is None:
                raise RetentionDeclined("Allocation lost its exact compiler layout")
            events.append(Allocate(layout.source.name, layout.dtype, layout.size, layout.stride))
            allocated_nodes[line.node.get_name()] = line.node
        elif type(line) is ReuseLine:
            layout = wrapper._cudagraph_owned_buffer(line.reused_as, device, symbols)
            if layout is None:
                raise RetentionDeclined("Reuse destination lost its exact compiler layout")
            events.append(Reinterpret(line.node.get_name(), layout.source.name, layout.size, layout.stride, 0))
            allocated_nodes.pop(line.node.get_name(), None)
            allocated_nodes[line.reused_as.get_name()] = line.reused_as
        elif type(line) is InputAlignmentLine:
            events.append(Normalize(records.input_names.index(line.name), call_index))
        elif type(line) is KernelCallLine:
            call = records.calls[call_index]
            if call.kernel_global != line.kernel_name:
                raise RetentionDeclined("Compiler call occurrence changed during retention")
            if line.cudagraph_user is not None:
                events.append(_retain_user_call(wrapper, line, call, records, symbols, allocated_nodes))
            elif type(call) is MultiKernelCallRecord:
                metadata = line.cudagraph_alternatives
                if type(metadata) is not tuple or len(metadata) != len(call.alternatives):
                    raise RetentionDeclined("Compiler alternatives changed during retention")
                alternatives = []
                for alternative, source in zip(call.alternatives, metadata):
                    name, indices, _triton_meta, facts, _typed = source
                    if name != alternative.call.kernel_global or indices != alternative.argument_indices:
                        raise RetentionDeclined("Retained alternative lost its compiler argument projection")
                    provenance = facts["cudagraph_parameter_provenance"]
                    writes = tuple((argument.formal, provenance[argument.formal][1])
                                   for argument in alternative.call.arguments
                                   if type(argument.source) in (InputSource, BufferSource))
                    alternatives.append(CallSite(alternative.call, writes))
                events.append(MultiCallSite(call, tuple(alternatives)))
            else:
                provenance = line.inductor_meta["cudagraph_parameter_provenance"]
                writes = tuple((argument.formal, provenance[argument.formal][1])
                               for argument in call.arguments
                               if type(argument.source) in (InputSource, BufferSource))
                events.append(CallSite(call, writes))
            call_index += 1
        elif type(line) is AssertSizeStrideLine:
            retain_assertion(line.name, line.size, line.stride, line.op_name)
        elif type(line) is GroupedAssertSizeStrideLine:
            for name, size, stride in line.asserts:
                retain_assertion(name, size, stride, line.op_name)
        elif type(line) is AssertAlignmentLine:
            raise RetentionDeclined("Alignment assertions need a shared host assertion instruction")
    output_sources = {name: InputSource(index) for index, name in enumerate(records.input_names)
                      if index not in {row.boxed_index for row in records.integer_inputs}}
    output_sources.update((name, BufferSource(name)) for name in allocated_nodes)
    if wrapper._cudagraph_owned_outputs(allocated_nodes, output_sources, device, symbols) != records.outputs:
        raise RetentionDeclined("Retained outputs differ from the final compiler output slots")
    result = RetainedIR(records, tuple(events), tuple(assertions))
    validate(result)
    return result


def validate(retained: RetainedIR) -> None:
    if (type(retained) is not RetainedIR or type(retained.events) is not tuple
            or type(retained.input_assertions) is not tuple
            or type(retained.records) is not WrapperCallRecords or type(retained.records.version) is not int
            or retained.records.version not in (3, 4)
            or bind_wrapper_allocations(retained.records) is None
            or bind_alignment_copies(retained.records) is None):
        raise RetentionDeclined("Retained program exceeds the current runtime-record contract")
    if retained.records.version == 3:
        from torch._inductor.runtime._cudagraph._compiler.compiler_metadata_handoff.metadata import _check_v3_records, MetadataDeclined

        try:
            _check_v3_records(retained.records)
        except MetadataDeclined as error:
            raise RetentionDeclined("V3 retention requires literal compiler records") from error
    calls, copies, allocations = [], [], {}
    used_names = set()
    previous = None
    for event in retained.events:
        if type(event) is Allocate:
            if event.value_id in used_names:
                raise RetentionDeclined("Retained allocation identities must be distinct")
            used_names.add(event.value_id)
            allocations[event.value_id] = OwnedBuffer(BufferSource(event.value_id), event.dtype,
                                                       event.size, event.stride)
        elif type(event) is Reinterpret:
            old = allocations.get(event.source_id)
            if (type(previous) is not Allocate or previous.value_id != event.source_id
                    or old is None or event.result_id in used_names or calls or type(event.offset) is not int
                    or event.offset != 0 or event.size != old.size or len(event.stride) != len(old.stride)
                    or any(before != after and dimension != 1
                           for dimension, before, after in zip(old.size, old.stride, event.stride))):
                raise RetentionDeclined("Retained reinterpret is not the proved fresh singleton rename")
            used_names.add(event.result_id)
            allocations = {event.result_id if name == event.source_id else name:
                           OwnedBuffer(BufferSource(event.result_id), old.dtype, event.size, event.stride)
                           if name == event.source_id else layout for name, layout in allocations.items()}
        elif type(event) is Normalize:
            if event.before_call != len(calls):
                raise RetentionDeclined("Normalization moved across a call")
            copies.append(AlignmentCopy(event.input_index, event.before_call))
        elif type(event) in (CallSite, MultiCallSite, UserCallSite):
            if event.call.occurrence != len(calls):
                raise RetentionDeclined("Call occurrences must remain ordered")
            sites = (event,)
            if type(event) is MultiCallSite:
                if (type(event.call) is not MultiKernelCallRecord or call_sources(event.call) is None
                        or type(event.alternatives) is not tuple
                        or len(event.alternatives) != len(event.call.alternatives)
                        or any(type(site) is not CallSite or site.call != alternative.call
                               for site, alternative in zip(event.alternatives, event.call.alternatives))):
                    raise RetentionDeclined("Retained alternatives differ from the compiler dispatch")
                sites = event.alternatives
            for site in sites:
                if type(site.call) is not KernelCallRecord:
                    raise RetentionDeclined("Retained call has no complete kernel argument contract")
                pointer_arguments = [arg for arg in site.call.arguments
                                     if type(arg.source) in (InputSource, BufferSource)]
                if type(site) is UserCallSite:
                    effects = site.pointer_effects
                    if (site.call.grid_type != "FixedGrid" or type(effects) is not tuple
                            or len(effects) != len(pointer_arguments)
                            or any(type(row) is not tuple or len(row) != 4 or type(row[0]) is not str
                                   or type(row[1]) is not int or type(row[2]) is not bool or type(row[3]) is not bool
                                   for row in effects)
                            or tuple((row[0], row[1]) for row in effects)
                            != tuple((arg.formal, arg.source_arg_index) for arg in pointer_arguments)):
                        raise RetentionDeclined("User effects lost their complete formal correspondence")
                    writes = tuple((row[0], row[3]) for row in effects)
                else:
                    if site.call.grid_type == "FixedGrid" and not site.call.generated_template:
                        raise RetentionDeclined("User call requires its earlier HOP read/write facts")
                    writes = site.pointer_writes
                if (len(writes) != len(pointer_arguments)
                        or tuple(name for name, value in writes)
                        != tuple(arg.formal for arg in pointer_arguments)
                        or any(type(value) is not bool for name, value in writes)):
                    raise RetentionDeclined("Pointer effects lost their exact formal correspondence")
                for argument in pointer_arguments:
                    if type(argument.source) is BufferSource and argument.source.name not in allocations:
                        raise RetentionDeclined("Call refers to an undefined or retired buffer")
                    if type(argument.source) is InputSource and dict(writes)[argument.formal]:
                        raise RetentionDeclined("Input writes remain outside the current contract")
            calls.append(event.call)
        else:
            raise RetentionDeclined("Unknown retained instruction")
        previous = event
    if (tuple(calls) != retained.records.calls or tuple(copies) != retained.records.alignment_copies
            or tuple(allocations.values()) != retained.records.allocations):
        raise RetentionDeclined("Host instructions differ from the compiler's normalized records")
    integer_indices = {value.boxed_index for value in retained.records.integer_inputs}
    for assertion in retained.input_assertions:
        if (type(assertion) is not InputAssertion or type(assertion.source) is not InputSource
                or assertion.source.index in integer_indices
                or not 0 <= assertion.source.index < len(retained.records.input_names)
                or len(assertion.size) != len(assertion.stride)):
            raise RetentionDeclined("Input assertion lost its boxed Tensor origin")
        if retained.records.version == 3 and any(type(value) is not int or value < 0
                                                 for value in (*assertion.size, *assertion.stride)):
            raise RetentionDeclined("V3 retention requires literal input assertions")


_TYPES = {value.__name__: value for value in (
    RetainedIR, CallSite, MultiCallSite, UserCallSite, Allocate, Reinterpret, Normalize, InputAssertion, AlignmentCopy,
    BufferSource, InputSource, IntegerInput, IntegerSource, IntExpr, ExpressionSource,
    CallArgument, KernelCallRecord, KernelCallAlternative, MultiKernelCallRecord,
    OwnedBuffer, BorrowedInputOutput, IntegerOutput, WrapperCallRecords,
)}
_DTYPES = {str(getattr(torch, name)): getattr(torch, name)
           for name in ("float16", "bfloat16", "float32", "float64", "int32", "int64", "bool")}


def _encode(value):
    if value is None or type(value) in (int, str, bool):
        return value
    if type(value) is tuple:
        return {"tuple": [_encode(item) for item in value]}
    if type(value) is torch.dtype and str(value) in _DTYPES:
        return {"dtype": str(value)}
    if type(value).__name__ in _TYPES and type(value) is _TYPES[type(value).__name__]:
        return {"type": type(value).__name__, "fields": {
            row.name: _encode(getattr(value, row.name)) for row in fields(value)}}
    raise RetentionDeclined("Export contains a value outside immutable compiler metadata")


def _decode(value):
    if value is None or type(value) in (int, str, bool):
        return value
    if type(value) is not dict:
        raise RetentionDeclined("Malformed retained metadata")
    if set(value) == {"tuple"} and type(value["tuple"]) is list:
        return tuple(_decode(item) for item in value["tuple"])
    if set(value) == {"dtype"} and type(value["dtype"]) is str and value["dtype"] in _DTYPES:
        return _DTYPES[value["dtype"]]
    if set(value) == {"type", "fields"} and type(value["type"]) is str and value["type"] in _TYPES:
        kind, values = _TYPES[value["type"]], value["fields"]
        if kind is IntegerInput and type(values) is dict and set(values) == {"symbol", "boxed_index"}:
            values = {**values, "source_index": None}
        if kind is KernelCallRecord and type(values) is dict and set(values) == {
            "occurrence", "kernel_global", "formals", "arguments", "grid_type", "launcher_grid", "constexprs"
        }:
            values = {**values, "generated_template": False}
        if type(values) is dict and set(values) == {row.name for row in fields(kind)}:
            return kind(**{name: _decode(item) for name, item in values.items()})
    raise RetentionDeclined("Unknown retained metadata schema")


def export_ir(retained: RetainedIR) -> str:
    validate(retained)
    return json.dumps({"schema": 1, "program": _encode(retained)}, sort_keys=True)


def reconstruct_ir(data: str) -> RetainedIR:
    payload = json.loads(data)
    if type(payload) is not dict or set(payload) != {"schema", "program"} or type(payload["schema"]) is not int or payload["schema"] != 1:
        raise RetentionDeclined("Unknown retained host schema")
    retained = _decode(payload["program"])
    validate(retained)
    return retained
