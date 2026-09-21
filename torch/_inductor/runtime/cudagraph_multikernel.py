"""Compiler dispatch operands and their source mapping."""

from dataclasses import dataclass

from .cudagraph_arg_mapping import (
    bind_fixed_grid, BufferSource, CallArgument, ExpressionSource, InputSource,
    IntegerSource, KernelCallRecord,
)


@dataclass(frozen=True)
class KernelCallAlternative:
    argument_indices: tuple[int, ...]
    call: KernelCallRecord


@dataclass(frozen=True)
class MultiKernelCallRecord:
    occurrence: int
    kernel_global: str
    sources: tuple[InputSource | BufferSource | IntegerSource | ExpressionSource, ...]
    alternatives: tuple[KernelCallAlternative, ...]


def call_sources(call):
    if type(call) is KernelCallRecord:
        if type(call.arguments) is not tuple or any(type(arg) is not CallArgument for arg in call.arguments):
            return None
        return tuple(arg.source for arg in call.arguments)
    if (type(call) is not MultiKernelCallRecord or type(call.occurrence) is not int or call.occurrence < 0
            or type(call.kernel_global) is not str or not call.kernel_global.isidentifier()
            or type(call.sources) is not tuple or not call.sources
            or any(type(source) not in (InputSource, BufferSource, IntegerSource, ExpressionSource)
                   for source in call.sources)
            or type(call.alternatives) is not tuple or len(call.alternatives) < 2):
        return None
    used, names = set(), set()
    for alternative in call.alternatives:
        if type(alternative) is not KernelCallAlternative:
            return None
        child, indices = alternative.call, alternative.argument_indices
        if (type(child) is not KernelCallRecord or child.occurrence != call.occurrence
                or type(child.kernel_global) is not str or not child.kernel_global.isidentifier()
                or child.kernel_global == call.kernel_global or child.kernel_global in names
                or type(indices) is not tuple
                or any(type(index) is not int or not 0 <= index < len(call.sources) for index in indices)):
            return None
        sources = call_sources(child)
        if sources is None:
            return None
        if child.grid_type == "FixedGrid":
            if bind_fixed_grid(child) is None:
                return None
            sources += tuple(IntegerSource(value) if type(value) is int else ExpressionSource(value)
                             for value in child.launcher_grid)
        if len(sources) != len(indices) or sources != tuple(call.sources[index] for index in indices):
            return None
        if tuple(arg.call_arg_index for arg in child.arguments) != tuple(range(len(child.arguments))):
            return None
        used.update(indices)
        names.add(child.kernel_global)
    return call.sources if used == set(range(len(call.sources))) else None
