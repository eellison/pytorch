"""Eliminate expensive live row sums selected by a one-hot mask.

The analysis proves that equality with an iota selects at most one value per
row, then tracks that value through multiplication and floating-point casts.
The sum can then be replayed at row shape instead of reduction shape.
"""

import dataclasses
import math
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..utils import is_view


aten = torch.ops.aten
prims = torch.ops.prims
_RESHAPE_OPS = (aten.reshape.default, aten.view.default)
_MAX_ANALYSIS_NODES = 64
_MAX_REPLAY_STEPS = 8
_MIN_LIVE_REDUCTION_ROW_BYTES = 16 * 1024
_Scalar = int | float


@dataclasses.dataclass(frozen=True)
class _Uniform:
    expr: torch.fx.Node
    scalar: _Scalar | None = None


@dataclasses.dataclass(frozen=True)
class _Iota:
    length: int


@dataclasses.dataclass(frozen=True)
class _OneHotMask:
    index: torch.fx.Node
    length: int


@dataclasses.dataclass(frozen=True)
class _OneHotRow:
    index: torch.fx.Node
    length: int
    select: torch.fx.Node
    hit: torch.fx.Node
    miss: torch.fx.Node
    hit_scalar: _Scalar | None
    miss_scalar: _Scalar | None
    steps: tuple[torch.fx.Node, ...] = ()
    has_bf16_rounding: bool = False

    @property
    def tail(self) -> torch.fx.Node:
        return self.steps[-1] if self.steps else self.select


_Value = _Uniform | _Iota | _OneHotMask | _OneHotRow | None


def _tensor_val(node: torch.fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


class _RowAnalyzer:
    def __init__(self, length: int) -> None:
        self.length = length
        self.memo: dict[torch.fx.Node, _Value] = {}

    def _is_uniform_shape(self, node: torch.fx.Node) -> bool:
        value = _tensor_val(node)
        if value is None or value.dim() > 2:
            return False
        return value.dim() == 0 or statically_known_true(
            value.shape[-1] == 1
        )

    @staticmethod
    def _constructor_scalar(node: torch.fx.Node) -> _Scalar | None:
        if node.target is aten.full.default and len(node.args) >= 2:
            scalar = node.args[1]
        elif node.target is aten.scalar_tensor.default and node.args:
            scalar = node.args[0]
        else:
            return None
        value = _tensor_val(node)
        if type(scalar) not in (int, float) or value is None:
            return None
        try:
            return torch.tensor(scalar, dtype=value.dtype, device="cpu").item()
        except (RuntimeError, TypeError, ValueError, OverflowError):
            return None

    def _uniform(self, node: torch.fx.Node) -> _Uniform:
        return _Uniform(node, self._constructor_scalar(node))

    def _view_iota(self, node: torch.fx.Node, iota: _Iota) -> _Iota | None:
        value = _tensor_val(node)
        if (
            value is not None
            and value.dim() == 2
            and statically_known_true(value.shape[0] == 1)
            and statically_known_true(value.shape[1] == iota.length)
        ):
            return iota
        return None

    def _matches_reduction_shape(
        self, node: torch.fx.Node, index: torch.fx.Node
    ) -> bool:
        node_val = _tensor_val(node)
        index_val = _tensor_val(index)
        return (
            node_val is not None
            and index_val is not None
            and node_val.dim() == index_val.dim() == 2
            and statically_known_true(node_val.shape[1] == self.length)
            and statically_known_true(sym_eq(node_val.shape[0], index_val.shape[0]))
        )

    def _branch_is_compatible(
        self, where: torch.fx.Node, index: torch.fx.Node, branch: _Uniform
    ) -> bool:
        where_val = _tensor_val(where)
        index_val = _tensor_val(index)
        branch_val = _tensor_val(branch.expr)
        return (
            where_val is not None
            and index_val is not None
            and branch_val is not None
            and branch_val.device == where_val.device
            and len(branch_val.shape) <= len(index_val.shape)
            and all(
                statically_known_true(a == 1) or statically_known_true(sym_eq(a, b))
                for a, b in zip(
                    reversed(branch_val.shape), reversed(index_val.shape)
                )
            )
        )

    @staticmethod
    def _is_positive_zero(value: _Scalar | None) -> bool:
        return (
            isinstance(value, (int, float))
            and value == 0
            and math.copysign(1.0, value) > 0
        )

    def _dependencies(self, node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return ()
        if node.target is prims.iota.default or self._is_uniform_shape(node):
            return ()
        if node.target is aten.expand.default or node.target in _RESHAPE_OPS:
            source = node.args[0] if node.args else None
            return (source,) if isinstance(source, torch.fx.Node) else ()
        if node.target in (
            aten.eq.Tensor,
            aten.where.self,
            aten.mul.Tensor,
            prims.convert_element_type.default,
        ):
            return tuple(node.all_input_nodes)
        return ()

    def _value(self, value: Any) -> Any:
        return self.memo.get(value) if isinstance(value, torch.fx.Node) else value

    def _analyze_mul(self, node: torch.fx.Node) -> _OneHotRow | None:
        args = tuple(self._value(arg) for arg in node.args)
        rows = [value for value in args if isinstance(value, _OneHotRow)]
        if len(rows) != 1:
            return None
        row = rows[0]
        for original, value in zip(node.args, args):
            if value is row:
                continue
            if not isinstance(value, _Uniform) or value.expr is not original:
                return None
        node_val = _tensor_val(node)
        if (
            node_val is None
            or not node_val.dtype.is_floating_point
            or not self._matches_reduction_shape(node, row.index)
        ):
            return None
        return dataclasses.replace(row, steps=(*row.steps, node))

    def _analyze_convert(self, node: torch.fx.Node) -> _OneHotRow | None:
        source = self._value(node.args[0]) if node.args else None
        node_val = _tensor_val(node)
        if (
            not isinstance(source, _OneHotRow)
            or node_val is None
            or not node_val.dtype.is_floating_point
            or not self._matches_reduction_shape(node, source.index)
        ):
            return None
        source_node = source.tail
        source_val = _tensor_val(source_node)
        rounds_to_bf16 = (
            source_val is not None
            and source_val.dtype is torch.float32
            and node_val.dtype is torch.bfloat16
        )
        return dataclasses.replace(
            source,
            steps=(*source.steps, node),
            has_bf16_rounding=source.has_bf16_rounding or rounds_to_bf16,
        )

    def analyze(self, node: torch.fx.Node) -> _Value:
        """Compute a node's abstract value from its analyzed dependencies."""
        if node in self.memo:
            return self.memo[node]

        result: _Value = None
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            if self._is_uniform_shape(node):
                result = self._uniform(node)
        elif node.target is prims.iota.default:
            length = node.args[0] if node.args else None
            value = _tensor_val(node)
            if (
                isinstance(length, int)
                and node.kwargs.get("start", 0) == 0
                and node.kwargs.get("step", 1) == 1
                and value is not None
                and value.dtype is torch.int64
            ):
                result = _Iota(length)
        elif node.target is aten.expand.default:
            source = self._value(node.args[0])
            value = _tensor_val(node)
            if (
                isinstance(source, _Uniform)
                and value is not None
                and value.dim() == 2
                and statically_known_true(value.stride()[1] == 0)
            ):
                result = source
        elif node.target in _RESHAPE_OPS:
            source = self._value(node.args[0])
            if isinstance(source, _Iota):
                result = self._view_iota(node, source)
            elif isinstance(source, _Uniform) and self._is_uniform_shape(node):
                result = _Uniform(node, source.scalar)
        elif node.target is aten.eq.Tensor:
            lhs = self._value(node.args[0])
            rhs = self._value(node.args[1])
            if isinstance(lhs, _Iota) and isinstance(rhs, _Uniform):
                iota, index = lhs, rhs
            elif isinstance(rhs, _Iota) and isinstance(lhs, _Uniform):
                iota, index = rhs, lhs
            else:
                iota = index = None
            if (
                isinstance(iota, _Iota)
                and isinstance(index, _Uniform)
                and iota.length == self.length
            ):
                index_val = _tensor_val(index.expr)
                if index_val is not None and index_val.dtype is torch.int64:
                    result = _OneHotMask(index.expr, iota.length)
        elif node.target is aten.where.self:
            mask = self._value(node.args[0])
            hit = self._value(node.args[1])
            miss = self._value(node.args[2])
            node_val = _tensor_val(node)
            if (
                isinstance(mask, _OneHotMask)
                and isinstance(hit, _Uniform)
                and isinstance(miss, _Uniform)
                and node_val is not None
                and node_val.dtype.is_floating_point
                and self._is_positive_zero(miss.scalar)
                and self._matches_reduction_shape(node, mask.index)
                and self._branch_is_compatible(node, mask.index, hit)
                and self._branch_is_compatible(node, mask.index, miss)
            ):
                result = _OneHotRow(
                    mask.index,
                    mask.length,
                    node,
                    hit.expr,
                    miss.expr,
                    hit.scalar,
                    miss.scalar,
                )
        elif node.target is aten.mul.Tensor:
            result = self._analyze_mul(node)
        elif node.target is prims.convert_element_type.default:
            result = self._analyze_convert(node)

        if result is None and self._is_uniform_shape(node):
            result = self._uniform(node)
        self.memo[node] = result
        return result

    def analyze_subgraph(self, node: torch.fx.Node) -> _Value:
        order: list[torch.fx.Node] = []
        seen = OrderedSet[torch.fx.Node]()
        pending = [(node, False)]
        while pending:
            current, ready = pending.pop()
            if current in self.memo:
                continue
            if ready:
                order.append(current)
                continue
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > _MAX_ANALYSIS_NODES:
                return None
            pending.append((current, True))
            pending.extend(
                (dependency, False) for dependency in self._dependencies(current)
            )
        for current in order:
            self.analyze(current)
        return self.memo.get(node)


def _has_expanding_pointwise_consumer(reduction: torch.fx.Node) -> bool:
    reduction_val = _tensor_val(reduction)
    if reduction_val is None:
        return False
    for user in reduction.users:
        if user.op != "call_function" or not isinstance(
            user.target, torch._ops.OpOverload
        ):
            continue
        user_val = _tensor_val(user)
        if (
            torch.Tag.pointwise in user.target.tags
            and user_val is not None
            and statically_known_true(user_val.numel() > reduction_val.numel())
        ):
            return True
    return False


def _has_downstream_reduction(dense: torch.fx.Node, reduction: torch.fx.Node) -> bool:
    pending = [user for user in dense.users if user is not reduction]
    seen = OrderedSet[torch.fx.Node]()
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if len(seen) > _MAX_ANALYSIS_NODES:
            return True
        if user.op != "call_function" or not isinstance(
            user.target, torch._ops.OpOverload
        ):
            continue
        if torch.Tag.reduction in user.target.tags:
            return True
        if torch.Tag.pointwise in user.target.tags or is_view(user.target):
            pending.extend(user.users)
    return False


def _expand_branch(
    graph: torch.fx.Graph,
    branch: torch.fx.Node,
    row: _OneHotRow,
) -> torch.fx.Node:
    branch_val = _tensor_val(branch)
    root_val = _tensor_val(row.select)
    if branch_val is None or root_val is None:
        raise AssertionError("expected tensor metadata")
    if branch_val.dtype != root_val.dtype:
        branch = graph.call_function(
            prims.convert_element_type.default, (branch, root_val.dtype)
        )
    return _expand_to_index(graph, branch, row.index)


def _expand_to_index(
    graph: torch.fx.Graph, value: torch.fx.Node, index: torch.fx.Node
) -> torch.fx.Node:
    index_val = _tensor_val(index)
    if index_val is None:
        raise AssertionError("expected tensor metadata")
    shape = [
        size if isinstance(size, int) else graph.create_size_node(index, dim)
        for dim, size in enumerate(index_val.shape)
    ]
    return graph.call_function(aten.expand.default, (value, shape))


def _replay_branch(
    graph: torch.fx.Graph,
    row: _OneHotRow,
    branch: torch.fx.Node,
) -> torch.fx.Node:
    value = _project_first_mul(graph, row, branch)
    if value is None:
        value = _expand_branch(graph, branch, row)
        replacements = {row.select: value}
        steps = row.steps
    else:
        replacements = {row.select: value, row.steps[0]: value}
        steps = row.steps[1:]
    for step in steps:
        value = graph.node_copy(step, lambda node: replacements.get(node, node))
        replacements[step] = _expand_to_index(graph, value, row.index)
    return replacements[row.tail]


def _project_first_mul(
    graph: torch.fx.Graph,
    row: _OneHotRow,
    branch: torch.fx.Node,
) -> torch.fx.Node | None:
    if not row.steps:
        return None
    first = row.steps[0]
    if first.target is not aten.mul.Tensor or row.select not in first.args:
        return None
    other = first.args[1] if first.args[0] is row.select else first.args[0]
    first_val = _tensor_val(first)
    other_val = _tensor_val(other) if isinstance(other, torch.fx.Node) else None
    if (
        not isinstance(other, torch.fx.Node)
        or first_val is None
        or other_val is None
        or first_val.dtype != other_val.dtype
        or first_val.device != other_val.device
    ):
        return None

    if branch is row.hit and row.hit_scalar == -1:
        value = graph.call_function(aten.neg.default, (other,))
    elif branch is row.miss and row.miss_scalar == 0:
        # Only the zero-or-NaN property matters for unselected lanes.
        value = graph.call_function(aten.sub.Tensor, (other, other))
    else:
        return None
    return _expand_to_index(graph, value, row.index)


def _sum_args(node: torch.fx.Node) -> tuple[torch.fx.Node, int] | None:
    if node.target is not aten.sum.dim_IntList or len(node.args) < 3:
        return None
    dense, dims, keepdim = node.args[:3]
    if (
        not isinstance(dense, torch.fx.Node)
        or not isinstance(dims, (list, tuple))
        or len(dims) != 1
        or keepdim is not True
        or len(node.args) > 3
        or node.kwargs.get("dtype") is not None
    ):
        return None
    dense_val = _tensor_val(dense)
    if dense_val is None or dense_val.dim() == 0 or not isinstance(dims[0], int):
        return None
    return dense, dims[0] % dense_val.dim()


def eliminate_singleton_reductions(graph: torch.fx.Graph) -> int:
    """Replace profitable live one-hot row sums with compact expressions."""
    count = 0
    analysis_cache: dict[tuple[torch.fx.Node, int], _Value] = {}
    for reduction in list(graph.nodes):
        if reduction.op != "call_function":
            continue
        sum_args = _sum_args(reduction)
        if sum_args is None:
            continue
        dense, dim = sum_args
        dense_val = _tensor_val(dense)
        output_val = _tensor_val(reduction)
        if (
            dense_val is None
            or output_val is None
            or torch.version.hip is not None
            or dense_val.device.type != "cuda"
            or dense_val.dtype is not torch.float32
            or output_val.dtype is not torch.float32
            or dense_val.dim() != 2
            or dim != 1
            or not isinstance(dense_val.shape[dim], int)
            or dense_val.shape[dim] <= 1
            or len(dense.users) <= 1
            or config.force_shape_pad
        ):
            continue

        length = dense_val.shape[dim]
        key = (dense, length)
        if key not in analysis_cache:
            analysis_cache[key] = _RowAnalyzer(length).analyze_subgraph(dense)
        row = analysis_cache[key]
        if (
            not isinstance(row, _OneHotRow)
            or not row.has_bf16_rounding
            or len(row.steps) > _MAX_REPLAY_STEPS
            or length * dense_val.element_size() < _MIN_LIVE_REDUCTION_ROW_BYTES
            or not _has_expanding_pointwise_consumer(reduction)
            or _has_downstream_reduction(dense, reduction)
        ):
            continue

        index_val = _tensor_val(row.index)
        if (
            index_val is None
            or index_val.dtype is not torch.int64
            or index_val.device != dense_val.device
            or not statically_known_true(sym_eq(index_val.shape, output_val.shape))
        ):
            continue

        with graph.inserting_before(reduction):
            ge_zero = graph.call_function(aten.ge.Scalar, (row.index, 0))
            in_range = graph.call_function(
                aten.le.Scalar, (row.index, row.length - 1)
            )
            valid = graph.call_function(aten.bitwise_and.Tensor, (ge_zero, in_range))
            hit = _replay_branch(graph, row, row.hit)
            miss = _replay_branch(graph, row, row.miss)
            zero = graph.call_function(
                aten.full.default,
                ([], 0.0),
                {"dtype": dense_val.dtype, "device": dense_val.device},
            )
            # Match sum's signed-zero normalization and preserve NaNs produced
            # when an unselected zero is multiplied by a nonfinite value.
            normalized_hit = graph.call_function(aten.add.Tensor, (hit, zero))
            valid_hit = graph.call_function(
                aten.where.self, (valid, normalized_hit, zero)
            )
            miss_is_zero = graph.call_function(aten.eq.Scalar, (miss, 0))
            replacement = graph.call_function(
                aten.where.self, (miss_is_zero, valid_hit, miss)
            )

        reduction.replace_all_uses_with(replacement)
        graph.erase_node(reduction)
        counters["inductor"]["singleton_reduction_elimination"] += 1
        count += 1

    if count:
        graph.eliminate_dead_code()
    return count
