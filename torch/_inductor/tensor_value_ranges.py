from __future__ import annotations

import operator
from typing import Any

import torch
from torch.utils._sympy.value_ranges import ValueRanges


aten = torch.ops.aten

_VALUE_PRESERVING_VIEW_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)


def _get_arg_value(
    node: torch.fx.Node,
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    if index < len(node.args):
        return node.args[index]
    return node.kwargs.get(name, default)


class TensorValueRangeAnalysis:
    """Prove tensor value ranges on demand from FX semantics."""

    def __init__(self) -> None:
        self._cache: dict[torch.fx.Node, ValueRanges[Any]] = {}

    def get(self, node: torch.fx.Node) -> ValueRanges[Any]:
        if node not in self._cache:
            self._cache[node] = self._get_uncached(node)
        return self._cache[node]

    def _get_uncached(self, node: torch.fx.Node) -> ValueRanges[Any]:
        if node.op != "call_function":
            return ValueRanges.unknown()

        if node.target in _VALUE_PRESERVING_VIEW_OPS:
            source = _get_arg_value(node, 0, "self")
            if isinstance(source, torch.fx.Node):
                return self.get(source)
            return ValueRanges.unknown()

        if node.target is not operator.getitem:
            return ValueRanges.unknown()

        producer = _get_arg_value(node, 0, "self")
        output_index = _get_arg_value(node, 1, "index")
        if not (
            isinstance(producer, torch.fx.Node)
            and producer.op == "call_function"
            and producer.target is aten.topk.default
            and output_index == 1
        ):
            return ValueRanges.unknown()

        topk_input = _get_arg_value(producer, 0, "self")
        if not isinstance(topk_input, torch.fx.Node):
            return ValueRanges.unknown()
        topk_input_val = topk_input.meta.get("val")
        if not isinstance(topk_input_val, torch.Tensor):
            return ValueRanges.unknown()

        dim = _get_arg_value(producer, 2, "dim", -1)
        if not isinstance(dim, int):
            return ValueRanges.unknown()
        if dim < 0:
            dim += topk_input_val.ndim
        if not 0 <= dim < topk_input_val.ndim:
            return ValueRanges.unknown()

        dim_size = topk_input_val.shape[dim]
        if not isinstance(dim_size, int) or dim_size < 1:
            return ValueRanges.unknown()
        return ValueRanges(0, dim_size - 1)
