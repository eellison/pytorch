from dataclasses import dataclass, field
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.continuation import BoundDispatchHelper
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_dispatch_evaluation.selection import DispatchInputs, DispatchSelection, _owners as _legacy_owners
from torch._inductor.runtime._cudagraph._compiler.dispatch_policy import select_with_policy
from torch._inductor.runtime._cudagraph._compiler.decoded_values import DecodedValues, prepare_decodings
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import PropertyCFG, bind_properties as bind_arithmetic_properties
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import OwnedNumeric, evaluate_owned, freeze_numeric
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_adapter.invocation import ArtifactInvocation
from torch._inductor.runtime._cudagraph._compiler.stream_layout import StreamLayout
from torch._inductor.runtime._cudagraph._compiler.target_layout import TargetLayout


def _owners(components, layout, stream_layout):
    if type(components) is ArtifactInvocation:
        components.check_layout(layout, stream_layout)
    else:
        _legacy_owners(components, layout, stream_layout)


@dataclass(frozen=True)
class PreparedConsumer:
    bound: BoundDispatchHelper
    properties: PropertyCFG
    decoded: DecodedValues
    numeric: OwnedNumeric


@dataclass(frozen=True)
class PreparedDispatch:
    components: Any
    layout: TargetLayout
    stream_layout: StreamLayout
    consumers: tuple[PreparedConsumer, ...]
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        return (id(self.components), id(self.layout), id(self.stream_layout), id(self.consumers),
                tuple((id(item), id(item.bound), id(item.properties), id(item.properties.cfg),
                       item.properties.flags, item.properties._owners, id(item.decoded),
                       id(item.numeric), id(item.numeric.source_order)) for item in self.consumers))

    def check(self) -> None:
        if (type(self.consumers) is not tuple or not self.consumers
                or any(type(item) is not PreparedConsumer or type(item.bound) is not BoundDispatchHelper
                       or type(item.properties) is not PropertyCFG or type(item.decoded) is not DecodedValues
                       or type(item.numeric) is not OwnedNumeric
                       for item in self.consumers)
                or self._state() != self._seal):
            raise RuntimeError("Prepared dispatch ownership or arithmetic properties changed")
        _owners(self.components, self.layout, self.stream_layout)
        dispatch = self.components if type(self.components) is ArtifactInvocation else self.components.dispatch
        actual = dispatch.consumers
        if (len(actual) != len(self.consumers) or any(item.bound is not bound or item.properties.cfg is not bound.cfg
                or item.decoded.cfg is not bound.cfg
                or item.numeric.source_order != bound.source_order
                for item, bound in zip(self.consumers, actual))):
            raise RuntimeError("Prepared properties lost an exact original dispatch consumer")
        for item in self.consumers:
            item.decoded.check(item.bound.cfg.context)
            item.numeric.check()

    def select(self, inputs: DispatchInputs) -> DispatchSelection:
        self.check()
        if type(self.components) is ArtifactInvocation:
            raise TypeError("Normalized invocations use DispatchArtifact selection, not legacy DispatchInputs")
        if type(inputs) is not DispatchInputs:
            raise TypeError("Expected the existing compiler-derived dispatch metadata inputs")
        inputs.check()
        if (inputs.components is not self.components or inputs.layout is not self.layout
                or inputs.stream_layout is not self.stream_layout):
            raise ValueError("Prepared dispatch inputs belong to another compiler or layout owner")
        source_values = dict(inputs.values)
        plans = {id(item.bound): item for item in self.consumers}

        def compute(bound):
            item = plans[id(bound)]
            values = tuple(source_values[index] for index in item.numeric.source_order)
            result = evaluate_owned(item.numeric, values)
            if len(result) != 1:
                raise ValueError("Prepared dispatch helper must return one original scalar")
            return result[0]

        return select_with_policy(inputs, compute)


def prepare_dispatch(components: Any, layout: TargetLayout, stream_layout: StreamLayout) -> PreparedDispatch:
    _owners(components, layout, stream_layout)
    consumers = []
    dispatch = components if type(components) is ArtifactInvocation else components.dispatch
    for bound in dispatch.consumers:
        properties = bind_arithmetic_properties(bound.cfg)
        decoded = prepare_decodings(bound.cfg)
        consumers.append(PreparedConsumer(bound, properties, decoded, freeze_numeric(properties, decoded, bound.source_order)))
    consumers = tuple(consumers)
    result = PreparedDispatch(components, layout, stream_layout, consumers, ())
    object.__setattr__(result, "_seal", result._state())
    result.check()
    return result
