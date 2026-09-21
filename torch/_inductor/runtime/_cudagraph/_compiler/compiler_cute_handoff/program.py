"""Full reader programs bound to their original mixed compiler authority."""

from dataclasses import dataclass, fields, is_dataclass, replace

from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract
from torch._inductor.runtime.cudagraph_arg_mapping import (
    AlignmentCopy, BorrowedInputOutput, BufferSource, InputSource, IntegerOutput, KernelCallRecord, OwnedBuffer,
)

from .descriptor import CuTeCall
from .envelope import (
    _check_envelope, BoundMixedEnvelope, EnvelopeDeclined, GeneratedCall,
    MixedEnvelope, Release,
)


class ProgramDeclined(ValueError):
    pass


@dataclass(frozen=True)
class MixedHostProgram:
    version: int
    inputs: InputContract
    events: tuple[OwnedBuffer | AlignmentCopy | KernelCallRecord | CuTeCall | Release, ...]
    outputs: tuple[OwnedBuffer | BorrowedInputOutput | IntegerOutput | None, ...]


def _same(expected, actual):
    if type(actual) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(actual) == len(expected) and all(_same(left, right) for left, right in zip(expected, actual))
    if is_dataclass(expected):
        return all(_same(getattr(expected, row.name), getattr(actual, row.name)) for row in fields(expected))
    return expected == actual


def _check_origin(checked):
    if type(checked) is not BoundMixedEnvelope:
        raise ProgramDeclined("Expected the original bound mixed compiler envelope")
    try:
        checked.check()
    except EnvelopeDeclined as error:
        raise ProgramDeclined("Original mixed compiler authority is unavailable") from error


def _project_source(source, renamed):
    if type(source) is InputSource:
        return source
    if type(source) is not BufferSource or source not in renamed:
        raise ProgramDeclined("Compiler source has no preceding reader allocation")
    return renamed[source]


def _bind_program(checked, program):
    _check_origin(checked)
    origin = checked.records
    if (type(program) is not MixedHostProgram or type(program.version) is not int or program.version != 1
            or type(program.events) is not tuple or type(program.outputs) is not tuple
            or not _same(origin.inputs, program.inputs)):
        raise ProgramDeclined("Reader program changed its schema or compiler input contract")
    expected, release_bounds = [], {}
    for event in origin.events:
        if type(event) is Release:
            release_bounds[event.source] = len(expected)
        else:
            expected.append(event)
    inputs = {InputSource(index) for index, kind in enumerate(origin.inputs.kinds) if kind == "tensor"}
    renamed, originals, released, events = {}, {}, set(), []
    cursor, calls = 0, []
    for event in program.events:
        kind = type(event)
        if kind is Release:
            source = event.source
            if type(source) is InputSource and type(source.index) is int and source in inputs:
                original = source
            elif type(source) is BufferSource and type(source.name) is str and source in originals:
                original = originals[source]
            else:
                raise ProgramDeclined("Reader release has no original live source")
            if source in released or cursor < release_bounds.get(original, len(expected)):
                raise ProgramDeclined("Reader release is duplicated or precedes the original release boundary")
            released.add(source)
            events.append(event)
            continue
        if cursor == len(expected):
            raise ProgramDeclined("Reader added a computation or allocation")
        original = expected[cursor]
        if type(original) is OwnedBuffer:
            if (kind is not OwnedBuffer or type(event.source) is not BufferSource
                    or type(event.source.name) is not str or not event.source.name
                    or event.source in originals):
                raise ProgramDeclined("Reader allocations must remain distinct owned sources")
            renamed[original.source] = event.source
            originals[event.source] = original.source
            if not _same(replace(original, source=event.source), event):
                raise ProgramDeclined("Reader allocation changed its compiler layout")
            events.append(event)
        elif type(original) is AlignmentCopy:
            if not _same(original, event):
                raise ProgramDeclined("Reader changed compiler normalization or its order")
            events.append(event)
        elif type(original) is GeneratedCall:
            arguments = tuple(replace(arg, source=_project_source(arg.source, renamed))
                if type(arg.source) in (InputSource, BufferSource) else arg for arg in original.record.arguments)
            if not _same(replace(original.record, arguments=arguments), event):
                raise ProgramDeclined("Reader generated call changed its ordered formal or source correspondence")
            events.append(GeneratedCall(event, original.operations,
                tuple(_project_source(source, renamed) for source in original.reads),
                tuple(_project_source(source, renamed) for source in original.writes)))
        elif type(original) is CuTeCall:
            operands = tuple(replace(row, source=_project_source(row.source, renamed)) for row in original.operands)
            if not _same(replace(original, operands=operands), event):
                raise ProgramDeclined("Reader CuTe call changed its original entry or operand correspondence")
            calls.append(event)
            events.append(event)
        else:
            raise ProgramDeclined("Unsupported original compiler event")
        cursor += 1
    if cursor != len(expected) or not calls:
        raise ProgramDeclined("Reader omitted an allocation, normalization or computation")
    outputs = tuple(replace(row, source=_project_source(row.source, renamed)) if type(row) is OwnedBuffer else row
                    for row in origin.outputs)
    if not _same(outputs, program.outputs):
        raise ProgramDeclined("Reader outputs changed their complete compiler projection")
    tensor_outputs = tuple(row.source for row in outputs if type(row) in (OwnedBuffer, BorrowedInputOutput))
    if released != (inputs | set(originals)) - set(tensor_outputs):
        raise ProgramDeclined("Reader must release every non-output source exactly once")
    descriptors = replace(origin.cute,
        tensors=tuple(replace(row, source=_project_source(row.source, renamed)) for row in origin.cute.tensors),
        calls=tuple(calls), outputs=tensor_outputs)
    records = MixedEnvelope(origin.version, program.inputs, descriptors, tuple(events), program.outputs)
    try:
        _check_envelope(records)
    except EnvelopeDeclined as error:
        raise ProgramDeclined("Reader program violates compiler initialization or lifetime requirements") from error
    result = records, tuple(renamed.items())
    _check_origin(checked)
    return result


class BoundMixedProgram:
    def __init__(self, binding, program, records, buffer_sources):
        self.binding = binding
        self.program = program
        self.records = records
        self.buffer_sources = buffer_sources
        self._seal = binding, program, records, buffer_sources
        self._snapshot = repr((program, records, buffer_sources))

    @property
    def inputs(self):
        return self.binding.inputs

    def check(self):
        binding, program, records, sources = self._seal
        if (self.binding is not binding or self.program is not program or self.records is not records
                or self.buffer_sources is not sources):
            raise ProgramDeclined("Bound reader program replaced its source or projection")
        current, renamed = _bind_program(binding, program)
        if (not _same(records, current) or not _same(sources, renamed)
                or repr((program, records, sources)) != self._snapshot):
            raise ProgramDeclined("Bound reader program changed after correspondence checking")
        _check_origin(binding)
        if (self.binding is not binding or self.program is not program or self.records is not records
                or self.buffer_sources is not sources or repr((program, records, sources)) != self._snapshot):
            raise ProgramDeclined("Bound reader program changed during source validation")

    def __reduce_ex__(self, protocol):
        raise TypeError("Live mixed reader bindings are not cache transport")


def bind_mixed_program(checked, program):
    records, sources = _bind_program(checked, program)
    result = BoundMixedProgram(checked, program, records, sources)
    result.check()
    return result
