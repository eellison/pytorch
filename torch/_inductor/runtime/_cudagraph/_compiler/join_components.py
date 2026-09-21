from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.accessors import Component, ComponentMapping
    from torch._inductor.runtime._cudagraph._compiler.component_owner import ComponentCompilation
    from torch._inductor.runtime._cudagraph._compiler.entry_signature import OperandBinding, ValueUse
    from torch._inductor.runtime._cudagraph._compiler.joined import JoinedInvocation


@dataclass(frozen=True)
class SymbolicComponent:
    component: Component
    source: OperandBinding
    use: ValueUse | None


def _bind_properties(invocation: JoinedInvocation, mapping: ComponentMapping) -> tuple[SymbolicComponent, ...]:
    by_index = {parameter.metadata.ir_arg_index: parameter.source for parameter in invocation.bound.params
                if parameter.metadata.ir_arg_index is not None}
    result = []
    for component in mapping.components:
        spec = component.spec
        source = by_index[spec.source_arg_index]
        if source.tensor is None:
            raise ValueError("Tensor component refers to a non-Tensor source")
        if spec.property == "pointer" and not spec.property_path:
            use = None
        elif spec.property in ("shape", "stride") and len(spec.property_path) == 1:
            values = source.tensor.shape if spec.property == "shape" else source.tensor.strides
            index = spec.property_path[0]
            if not 0 <= index < len(values):
                raise ValueError("Tensor component property path is outside the original source")
            use = values[index]
        else:
            raise ValueError("Unsupported tensor component property")
        result.append(SymbolicComponent(component, source, use))
    return tuple(result)


@dataclass(frozen=True)
class JoinedComponents:
    invocation: JoinedInvocation
    compilation: ComponentCompilation
    mapping: ComponentMapping
    properties: tuple[SymbolicComponent, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        values = self.invocation, self.compilation, self.mapping, self.properties
        if any(value is not owner for value, owner in zip(values, self._owners)):
            raise RuntimeError("Joined component ownership changed")
        self.invocation.check()
        self.compilation.check()
        self.mapping.check()
        if (self.invocation.program is not self.compilation.program
                or self.mapping.program is not self.compilation.program
                or self.mapping.specs is not self.compilation.accessors
                or _bind_properties(self.invocation, self.mapping) != self.properties):
            raise RuntimeError("Tensor component lost its original symbolic source")
