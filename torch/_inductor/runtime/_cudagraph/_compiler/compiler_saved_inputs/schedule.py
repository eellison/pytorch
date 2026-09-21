"""Cold reference-release order from the final generated wrapper IR."""

from dataclasses import dataclass

from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    AllocateLine, FreeIfNotReusedLine, FreeLine, InputAlignmentLine, KernelCallLine, PythonWrapperCodegen, ReuseLine,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, bind_alignment_copies, bind_wrapper_allocations, BorrowedInputOutput, BufferSource, InputSource,
    IntegerOutput, KernelCallRecord, OwnedBuffer, WrapperCallRecords,
)
from torch._inductor.runtime.cudagraph_multikernel import call_sources, MultiKernelCallRecord


@dataclass(frozen=True)
class DropInput:
    input_index: int
    after_call: int


@dataclass(frozen=True)
class ReleaseSchedule:
    input_names: tuple[str, ...]
    input_uses: tuple[tuple[int, ...], ...]
    steps: tuple[OwnedBuffer | AlignmentCopy | KernelCallRecord | MultiKernelCallRecord | DropInput, ...]
    outputs: tuple[OwnedBuffer | BorrowedInputOutput | IntegerOutput | None, ...]


def collect_release_schedule(saved, wrapper, records):
    """Use the live collector boundary after collect_saved_inputs succeeds.

    DropInput removes this invocation's original and normalized references, not
    external ownership. No step authorizes reuse before pending-use protection.
    """
    from .handoff import FinalSavedInputs

    if (type(saved) is not FinalSavedInputs or type(wrapper) is not PythonWrapperCodegen
            or type(records) is not WrapperCallRecords or type(records.version) is not int
            or records.version not in (3, 4)
            or not records.allocations or not records.outputs
            or saved.input_names != records.input_names):
        return None
    if bind_wrapper_allocations(records) is None or bind_alignment_copies(records) is None:
        return None
    candidates = saved.candidate_indices
    returned_inputs = {output.source.index for output in records.outputs if type(output) is BorrowedInputOutput}
    if (not candidates or any(type(index) is not int or not 0 <= index < len(records.input_names)
                              for index in candidates) or len(set(candidates)) != len(candidates)
            or returned_inputs.intersection(candidates)):
        return None
    # Records alone omit allocation/free positions. Recheck against their live source.
    if wrapper.collect_cudagraph_call_records() != records:
        return None
    integers = {row.boxed_index: row for row in records.integer_inputs}
    if any(index in integers for index in candidates):
        return None
    inputs = wrapper.get_graph_inputs()
    bindings = wrapper._cudagraph_integer_bindings()
    if bindings is None or bindings[0] != records.integer_inputs:
        return None
    _, symbols = bindings
    nodes, sources = {}, {}
    for index, name in enumerate(records.input_names):
        node = inputs[name]
        if index in integers:
            continue
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        if type(node) not in (ir.InputBuffer, ir.DonatedBuffer) or node.get_name() != name:
            return None
        nodes[name], sources[name] = node, InputSource(index)
    uses = [[] for _ in records.input_names]
    for index, call in enumerate(records.calls):
        if call.occurrence != index:
            return None
        operands = call_sources(call)
        if operands is None:
            return None
        for source in operands:
            if type(source) is InputSource:
                if index not in uses[source.index]:
                    uses[source.index].append(index)
    layouts = {layout.source.name: layout for layout in records.allocations}
    steps, allocated, dropped, normalized = [], [], set(), []
    call_index = 0
    for line in wrapper.lines:
        kind = type(line)
        if kind is ReuseLine:
            return None
        if kind is InputAlignmentLine:
            source = sources.get(line.name)
            if (type(source) is not InputSource or not uses[source.index]
                    or uses[source.index][0] != call_index):
                return None
            copy = AlignmentCopy(source.index, call_index)
            if (len(normalized) == len(records.alignment_copies)
                    or records.alignment_copies[len(normalized)] != copy
                    or any(previous.input_index == source.index for previous in normalized)):
                return None
            normalized.append(copy)
            steps.append(copy)
        elif kind is AllocateLine:
            node, name = line.node, line.node.get_name()
            if (line.wrapper is not wrapper or name in nodes or name not in layouts
                    or type(node) is not ir.ComputedBuffer or line.comm_buffer):
                return None
            layout = wrapper._cudagraph_owned_buffer(node, node.get_device(), symbols)
            if layout != layouts[name]:
                return None
            layout = layouts[name]
            nodes[name], sources[name] = node, BufferSource(name)
            allocated.append(layout)
            steps.append(layout)
        elif kind is KernelCallLine:
            if (line.wrapper is not wrapper or line.cudagraph_user is not None
                    or call_index == len(records.calls)):
                return None
            call = wrapper._cudagraph_call_record(line, call_index, sources, symbols)
            if call != records.calls[call_index]:
                return None
            call = records.calls[call_index]
            if type(call) is KernelCallRecord:
                for argument in call.arguments:
                    if type(argument.source) is InputSource:
                        if line.inductor_meta["cudagraph_parameter_provenance"][argument.formal][1] is not False:
                            return None
            steps.append(call)
            call_index += 1
        elif kind in (FreeLine, FreeIfNotReusedLine):
            name = line.node.get_name()
            if (line.wrapper is not wrapper or nodes.get(name) is not line.node or name not in sources
                    or kind is FreeIfNotReusedLine and (line.is_reused or line.comm_buffer)):
                return None
            source = sources.pop(name)
            if type(source) is InputSource and source.index in candidates:
                index = source.index
                if index in dropped or (uses[index] and uses[index][-1] >= call_index):
                    return None
                dropped.add(index)
                steps.append(DropInput(index, uses[index][-1] if uses[index] else -1))
    if (call_index != len(records.calls) or tuple(allocated) != records.allocations
            or tuple(normalized) != records.alignment_copies or dropped != set(candidates)):
        return None
    return ReleaseSchedule(records.input_names, tuple(tuple(row) for row in uses), tuple(steps), records.outputs)
