from dataclasses import dataclass, field
from typing import Any
from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import AggregatePlan
from torch._inductor.runtime._cudagraph._compiler.continuation import DispatchCompilation
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import MetadataAggregate, UnavailablePointer
from torch._inductor.runtime._cudagraph._compiler.stream_layout import StreamLayout
from torch._inductor.runtime._cudagraph._compiler.target_layout import TargetLayout
from torch._inductor.runtime._cudagraph._compiler.values import ScalarValue


@dataclass(frozen=True)
class TensorProperties:
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _owners(components: Any, layout: TargetLayout, stream_layout: StreamLayout) -> None:
    components.check()
    if (type(components.dispatch) is not DispatchCompilation or type(components.plan) is not AggregatePlan
            or type(layout) is not TargetLayout or type(stream_layout) is not StreamLayout):
        raise TypeError("Expected the original dispatch, component and compiler layout owners")
    layout.check()
    stream_layout.check()
    dispatch = components.dispatch
    if (components.plan.mapping.program is not dispatch.program or layout.query.plan is not components.plan
            or stream_layout.formals is not dispatch.joined.mapping.formals):
        raise ValueError("Dispatch metadata layouts belong to another compiler program")
    if any(site.source.stream_source_arg_index != stream_layout.source_index for site in dispatch.joined.sites):
        raise ValueError("Dispatch sites must use the same original environment stream")


@dataclass(frozen=True)
class DispatchInputs:
    components: Any = field(repr=False)
    layout: TargetLayout = field(repr=False)
    stream_layout: StreamLayout = field(repr=False)
    properties: tuple[tuple[int, TensorProperties], ...]
    values: tuple[tuple[int, MetadataAggregate | UnavailablePointer], ...] = field(repr=False)
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.components, self.layout, self.stream_layout, self.properties, self.values
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Dispatch metadata values changed their original owners")
        _owners(self.components, self.layout, self.stream_layout)


@dataclass(frozen=True)
class DispatchSelection:
    inputs: DispatchInputs = field(repr=False)
    site: Any = field(repr=False)
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared: int
    kernel_smem: int
    evaluated: tuple[str, ...]
    values: tuple[tuple[str, int, ScalarValue], ...] = field(repr=False)
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.inputs, self.site, self.grid, self.block, self.shared, self.kernel_smem, self.evaluated, self.values
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Selected dispatch changed its original metadata or site")
        self.inputs.check()
        if not any(self.site is site for site in self.inputs.components.dispatch.joined.sites):
            raise ValueError("Selected dispatch lost its exact source and registered kernel")
