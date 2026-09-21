"""Cold compiler facts attached to one original generated Python function."""

from dataclasses import dataclass, replace
from inspect import CO_VARARGS, CO_VARKEYWORDS
import json
from types import CodeType, FunctionType
from typing import TYPE_CHECKING
from weakref import ref, ReferenceType

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, bind_alignment_copies, bind_wrapper_allocations, BorrowedInputOutput, BufferSource, InputSource,
    IntegerInput, IntegerSource, IntExpr, integer_input_sources, KernelCallRecord, OwnedBuffer, WrapperCallRecords,
)
from torch._inductor.runtime.cudagraph_multikernel import call_sources, MultiKernelCallRecord

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.schedule import ReleaseSchedule


class MetadataDeclined(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedMetadata:
    version: int
    inputs: InputContract
    retained_ir: str | None
    release_schedule: "ReleaseSchedule | None" = None


@dataclass(frozen=True, eq=False)
class _Attachment:
    function: ReferenceType
    code: CodeType
    records: WrapperCallRecords
    metadata: GeneratedMetadata


def _check_v3_records(records):
    if (type(records.integer_inputs) is not tuple or records.integer_inputs
            or type(records.calls) is not tuple or not records.calls
            or type(records.allocations) is not tuple or not records.allocations
            or type(records.outputs) is not tuple or not records.outputs
            or bind_wrapper_allocations(records) is None or bind_alignment_copies(records) is None):
        raise MetadataDeclined("V3 metadata requires complete literal allocation and call facts")
    for call in records.calls:
        sources = call_sources(call)
        if sources is None:
            raise MetadataDeclined("V3 metadata requires exact compiler call operands")
        for source in sources:
            if (type(source) not in (InputSource, BufferSource, IntegerSource)
                    or type(source) is IntegerSource and type(source.value) is not int):
                raise MetadataDeclined("V3 metadata requires pointer or literal integer arguments")


def _check_metadata(metadata, records):
    if (type(metadata) is not GeneratedMetadata or type(metadata.version) is not int
            or metadata.version != 1 or type(metadata.inputs) is not InputContract
            or (metadata.retained_ir is not None and type(metadata.retained_ir) is not str)):
        raise MetadataDeclined("Unknown generated compiler metadata schema")
    inputs = metadata.inputs
    if (type(inputs.kinds) is not tuple or not inputs.kinds
            or any(type(kind) is not str or kind not in ("integer", "tensor") for kind in inputs.kinds)
            or type(inputs.tensor_inputs) is not tuple or type(inputs.integer_ranges) is not tuple
            or type(inputs.device_index) is not int or inputs.device_index < 0):
        raise MetadataDeclined("Malformed generated input contract")
    integers = tuple(index for index, kind in enumerate(inputs.kinds) if kind == "integer")
    tensors = tuple(index for index, kind in enumerate(inputs.kinds) if kind == "tensor")
    if (not tensors
            or any(type(row) is not IntegerRange or type(row.index) is not int
                   or type(row.lower) is not int or row.lower < 1
                   or (row.upper is not None and (type(row.upper) is not int or row.upper < row.lower))
                   for row in inputs.integer_ranges)
            or tuple(row.index for row in inputs.integer_ranges) != integers
            or any(type(row) is not TensorInput or type(row.index) is not int
                   or type(row.dtype) is not torch.dtype or type(row.size) is not tuple
                   or type(row.stride) is not tuple or len(row.size) != len(row.stride)
                   for row in inputs.tensor_inputs)
            or tuple(row.index for row in inputs.tensor_inputs) != tensors):
        raise MetadataDeclined("Compiler input facts do not cover exact boxed slots")
    sources = integer_input_sources(records.integer_inputs, len(inputs.kinds)) if type(records) is WrapperCallRecords else None
    if sources is None or tuple(sources) != integers:
        raise MetadataDeclined("Compiler integer bindings differ from their physical boxed slots")
    ranges = {row.index: (row.lower, row.upper) for row in inputs.integer_ranges}
    if any(ranges[index] != ranges[source] for index, source in sources.items()):
        raise MetadataDeclined("Equivalent compiler integers have different inherited domains")
    for tensor in inputs.tensor_inputs:
        for value in (*tensor.size, *tensor.stride):
            if type(value) is int:
                continue
            if (type(value) is not IntExpr or value.op != "boxed" or type(value.value) is not int
                    or sources.get(value.value) != value.value or type(value.args) is not tuple or value.args):
                raise MetadataDeclined("Input layout lost its literal or boxed integer origin")
    if (type(records) is not WrapperCallRecords or type(records.version) is not int or records.version not in (3, 4)
            or type(records.input_names) is not tuple or len(records.input_names) != len(inputs.kinds)
            or any(type(name) is not str for name in records.input_names)
            or len(set(records.input_names)) != len(records.input_names)
            or type(records.device_index) is not int or records.device_index != inputs.device_index
            or type(records.integer_inputs) is not tuple
            or any(type(row) is not IntegerInput or type(row.boxed_index) is not int
                   for row in records.integer_inputs)
            or tuple(row.boxed_index for row in records.integer_inputs) != integers):
        raise MetadataDeclined("Compiler input facts differ from emitted call records")
    if records.version == 3:
        if (integers
                or any(type(value) is not int or value < 0 for tensor in inputs.tensor_inputs
                       for value in (*tensor.size, *tensor.stride))):
            raise MetadataDeclined("V3 metadata requires literal Tensor inputs")
        _check_v3_records(records)
    if metadata.retained_ir is not None:
        from torch._inductor.runtime._cudagraph._compiler.retained_ir.prototype import reconstruct_ir, RetentionDeclined

        try:
            retained = reconstruct_ir(metadata.retained_ir)
        except (json.JSONDecodeError, RetentionDeclined) as error:
            raise MetadataDeclined("Malformed retained compiler metadata") from error
        if retained.records != records:
            raise MetadataDeclined("Retained transport differs from emitted call records")
    if metadata.release_schedule is not None:
        _check_release_schedule(metadata.release_schedule, records)


def _check_release_schedule(schedule, records):
    from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.schedule import DropInput, ReleaseSchedule

    if (type(schedule) is not ReleaseSchedule or type(records.version) is not int or records.version not in (3, 4)
            or schedule.input_names != records.input_names or schedule.outputs != records.outputs
            or type(schedule.input_uses) is not tuple or type(schedule.steps) is not tuple):
        raise MetadataDeclined("Release schedule differs from its generated wrapper")
    if bind_wrapper_allocations(records) is None or bind_alignment_copies(records) is None:
        raise MetadataDeclined(f"Release schedule requires complete V{records.version} compiler records")
    integer_indices = {row.boxed_index for row in records.integer_inputs}
    returned_inputs = {output.source.index for output in records.outputs if type(output) is BorrowedInputOutput}
    uses = [[] for _ in records.input_names]
    for index, call in enumerate(records.calls):
        sources = call_sources(call)
        if sources is None:
            raise MetadataDeclined("Release schedule lost its compiler call operands")
        for source in sources:
            if type(source) is InputSource and index not in uses[source.index]:
                uses[source.index].append(index)
    if (schedule.input_uses != tuple(tuple(row) for row in uses)
            or any(type(step) not in (OwnedBuffer, AlignmentCopy, KernelCallRecord, MultiKernelCallRecord, DropInput)
                   for step in schedule.steps)
            or tuple(step for step in schedule.steps if type(step) is OwnedBuffer) != records.allocations
            or tuple(step for step in schedule.steps if type(step) is AlignmentCopy) != records.alignment_copies
            or tuple(step for step in schedule.steps
                     if type(step) in (KernelCallRecord, MultiKernelCallRecord)) != records.calls):
        raise MetadataDeclined("Release schedule lost its exact operations and uses")
    allocated, dropped, call_index = set(), set(), 0
    for step in schedule.steps:
        if type(step) is OwnedBuffer:
            allocated.add(step.source)
        elif type(step) is AlignmentCopy:
            if step.before_call != call_index:
                raise MetadataDeclined("Input normalization moved away from its first use")
        elif type(step) in (KernelCallRecord, MultiKernelCallRecord):
            if any(type(source) is BufferSource and source not in allocated for source in call_sources(step)):
                raise MetadataDeclined("Kernel use precedes its buffer allocation")
            call_index += 1
        elif type(step) is DropInput:
            index = step.input_index
            if type(index) is int and index in returned_inputs:
                raise MetadataDeclined("Input reference drop cannot target a returned input")
            if type(index) is int and index in integer_indices:
                raise MetadataDeclined("Input reference drop cannot target a boxed integer")
            if (type(index) is not int or not 0 <= index < len(uses) or index in dropped
                    or type(step.after_call) is not int
                    or step.after_call != (uses[index][-1] if uses[index] else -1)
                    or step.after_call >= call_index):
                raise MetadataDeclined("Input reference drop precedes its last use")
            dropped.add(index)
    if not dropped:
        raise MetadataDeclined("Release schedule has no saved-input reference drops")


def collect_metadata(wrapper, records):
    if type(records) is not WrapperCallRecords or type(records.version) is not int or records.version not in (3, 4):
        return None
    if records.version == 3:
        try:
            _check_v3_records(records)
        except MetadataDeclined:
            return None
    from torch._inductor.runtime._cudagraph._compiler.compiler_input_contract.collector import compiler_inputs, CompilerInputDeclined

    try:
        inputs = compiler_inputs(wrapper)
    except CompilerInputDeclined:
        return None
    from torch._inductor.runtime._cudagraph._compiler.retained_ir.prototype import export_ir, retain_wrapper, RetentionDeclined

    try:
        retained = export_ir(retain_wrapper(wrapper, records))
    except RetentionDeclined:
        retained = None
    metadata = GeneratedMetadata(1, inputs, retained)
    _check_metadata(metadata, records)
    from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.transport import collect_optional_schedule

    schedule = collect_optional_schedule(wrapper, records, metadata)
    if schedule is not None:
        metadata = replace(metadata, release_schedule=schedule)
        _check_metadata(metadata, records)
    return metadata


def emit_metadata(result, function_name, metadata):
    if metadata is None:
        return
    if type(function_name) is not str or not function_name.isidentifier():
        raise ValueError("Expected the generated function's exact global name")
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.compiler_metadata_handoff.metadata import GeneratedMetadata, attach_metadata')
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput')
    if metadata.release_schedule is not None:
        result.writeline('from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.schedule import DropInput, ReleaseSchedule')
        result.writeline("from torch._inductor.runtime.cudagraph_arg_mapping import BorrowedInputOutput, IntegerOutput")
    result.writeline(f"attach_metadata({function_name}, {metadata!r})")


def attach_metadata(function, metadata):
    if (type(function) is not FunctionType or function.__closure__ or function.__defaults__
            or function.__kwdefaults__ or function.__code__.co_argcount != 1
            or function.__code__.co_kwonlyargcount
            or function.__code__.co_flags & (CO_VARARGS | CO_VARKEYWORDS)
            or function.__globals__.get(function.__name__) is not function
            or "_cudagraph_frontend_metadata" in function.__dict__
            or "_cudagraph_frontend_attachment" in function.__dict__):
        raise MetadataDeclined("Expected an unattached original one-box generated function")
    records = function.__dict__.get("_cudagraph_call_records")
    attachment = _Attachment(ref(function), function.__code__, records, metadata)
    function._cudagraph_frontend_metadata = metadata
    function._cudagraph_frontend_attachment = attachment


def _attachment_matches(function, attachment):
    return (type(function) is FunctionType and type(attachment) is _Attachment
            and function.__dict__.get("_cudagraph_frontend_attachment") is attachment
            and attachment.function() is function
            and function.__code__ is attachment.code
            and function.__globals__.get(function.__name__) is function
            and function.__dict__.get("_cudagraph_frontend_metadata") is attachment.metadata
            and function.__dict__.get("_cudagraph_call_records") is attachment.records
            and not function.__defaults__ and not function.__kwdefaults__)


def _check_attachment(function, attachment):
    if not _attachment_matches(function, attachment):
        raise MetadataDeclined("Generated callable lost its original metadata attachment")
    _check_metadata(attachment.metadata, attachment.records)
    if not _attachment_matches(function, attachment):
        raise MetadataDeclined("Generated callable changed during metadata validation")
    return attachment.metadata


def read_metadata(artifact):
    from torch._inductor.output_code import CompiledFxGraph

    if type(artifact) is not CompiledFxGraph:
        raise MetadataDeclined("Expected the original compiler output artifact")
    function = artifact.current_callable
    if (type(function) is not FunctionType
            or function is not artifact._cudagraph_original_callable):
        raise MetadataDeclined("Active callable is not this artifact's original generated function")
    attachment = function.__dict__.get("_cudagraph_frontend_attachment")
    result = function, _check_attachment(function, attachment)
    if (artifact.current_callable is not function
            or artifact._cudagraph_original_callable is not function):
        raise MetadataDeclined("Active callable changed during metadata validation")
    return result
