"""Eliminate reductions of values selected by at-most-one-hot masks.

The analysis tracks values along a reduction dimension through one-hot masks,
supported pointwise operations, and dtype conversions. It proves that off-index
values remain zero or NaN, then materializes only the selected value and guards
needed to preserve invalid-index and nonfinite behavior.
"""

import dataclasses
import enum
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..utils import is_view


aten = torch.ops.aten
prims = torch.ops.prims
_RESHAPE_OPS = (aten.reshape.default, aten.view.default, aten._unsafe_view.default)
_TOTAL_POINTWISE_OPS = (
    aten.abs.default,
    aten.add.Scalar,
    aten.add.Tensor,
    aten.clone.default,
    aten.mul.Scalar,
    aten.mul.Tensor,
    aten.neg.default,
    aten.sub.Scalar,
    aten.sub.Tensor,
    prims.convert_element_type.default,
)
_Scalar = int | float
# Bound analysis work across reductions with overlapping producer and consumer DAGs.
_MAX_ANALYSIS_NODES = 10_000
# Bound the semantic operations materialized in place of a reduction.
_MAX_MATERIALIZATION_OPS = 32
# Small live reductions can already use lower-overhead fused schedules.
_MIN_LIVE_REDUCTION_ROW_BYTES = 16 * 1024


class _ZeroKind(enum.Enum):
    UNKNOWN = 0
    EXACT = 1
    ZERO_OR_NAN = 2


@dataclasses.dataclass
class _Expr:
    target: torch.fx.node.Target
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclasses.dataclass
class _Shape:
    source: torch.fx.Node
    values: tuple[int | torch.SymInt, ...]


@dataclasses.dataclass
class _Unknown:
    pass


@dataclasses.dataclass
class _Uniform:
    expr: Any
    zero_kind: _ZeroKind = _ZeroKind.UNKNOWN
    scalar: _Scalar | None = None


@dataclasses.dataclass
class _Iota:
    length: int
    axis: int
    dtype: torch.dtype


@dataclasses.dataclass
class _SingletonMask:
    index: Any
    length: int


@dataclasses.dataclass
class _Singleton:
    index: Any
    length: int
    hit: Any
    miss: Any
    miss_zero_kind: _ZeroKind
    hit_scalar: _Scalar | None = None
    miss_scalar: _Scalar | None = None
    has_unmodeled_pointwise: bool = False
    has_low_precision_rounding: bool = False


_Value = _Unknown | _Uniform | _Iota | _SingletonMask | _Singleton | None


def _tensor_val(node: torch.fx.Node) -> torch.Tensor | None:
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def _is_float_tensor(node: torch.fx.Node) -> bool:
    val = _tensor_val(node)
    return val is not None and (val.dtype.is_floating_point or val.dtype.is_complex)


class _ReductionDimAnalyzer:
    def __init__(self, reduction: torch.fx.Node, dim: int, length: int) -> None:
        self.dim = dim
        self.length = length
        self.input_val = _tensor_val(reduction.args[0])
        self.memo: dict[torch.fx.Node, _Value] = {}
        self.nodes_visited = 0

    def _is_uniform_shape(self, node: torch.fx.Node) -> bool:
        val = _tensor_val(node)
        if val is None or self.input_val is None:
            return False
        rank_delta = self.input_val.dim() - val.dim()
        input_dim = self.dim - rank_delta
        return input_dim < 0 or (
            input_dim < val.dim()
            and statically_known_true(val.shape[input_dim] == 1)
        )

    @staticmethod
    def _constructor_scalar(node: torch.fx.Node) -> _Scalar | None:
        if node.target is aten.full.default and len(node.args) >= 2:
            value = node.args[1]
        elif node.target is aten.scalar_tensor.default and node.args:
            value = node.args[0]
        else:
            return None
        val = _tensor_val(node)
        if type(value) not in (int, float) or val is None:
            return None
        try:
            return torch.tensor(value, dtype=val.dtype, device="cpu").item()
        except (RuntimeError, TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _constructor_uniform(cls, node: torch.fx.Node) -> _Uniform:
        scalar = cls._constructor_scalar(node)
        zero_kind = _ZeroKind.EXACT if scalar == 0 else _ZeroKind.UNKNOWN
        return _Uniform(node, zero_kind, scalar)

    def _view_iota(self, node: torch.fx.Node, iota: _Iota) -> _Iota | None:
        val = _tensor_val(node)
        if val is None:
            return None
        axes = [
            i
            for i, size in enumerate(val.shape)
            if statically_known_true(size == iota.length)
        ]
        if len(axes) != 1 or any(
            not statically_known_true(size == 1)
            for i, size in enumerate(val.shape)
            if i != axes[0]
        ):
            return None
        return _Iota(iota.length, axes[0], iota.dtype)

    @staticmethod
    def _same_singleton(a: _Singleton, b: _Singleton) -> bool:
        return a.length == b.length and a.index is b.index

    def _matches_reduction_shape(
        self, node: torch.fx.Node, index: torch.fx.Node
    ) -> bool:
        node_val = _tensor_val(node)
        index_val = _tensor_val(index)
        return (
            node_val is not None
            and index_val is not None
            and node_val.dim() == index_val.dim()
            and self.dim < node_val.dim()
            and statically_known_true(node_val.shape[self.dim] == self.length)
            and all(
                dim == self.dim or statically_known_true(sym_eq(a, b))
                for dim, (a, b) in enumerate(zip(node_val.shape, index_val.shape))
            )
        )

    def _where_branch(
        self, node: torch.fx.Node, index: torch.fx.Node, branch: _Uniform
    ) -> Any | None:
        node_val = _tensor_val(node)
        index_val = _tensor_val(index)
        branch_val = _tensor_val(branch.expr)
        if (
            node_val is None
            or index_val is None
            or branch_val is None
            or branch_val.device != node_val.device
            or not self._matches_reduction_shape(node, index)
        ):
            return None
        if len(branch_val.shape) > len(index_val.shape):
            return None
        if not all(
            statically_known_true(a == 1) or statically_known_true(sym_eq(a, b))
            for a, b in zip(reversed(branch_val.shape), reversed(index_val.shape))
        ):
            return None
        expr = branch.expr
        if branch_val.dtype != node_val.dtype:
            expr = _Expr(
                prims.convert_element_type.default, (expr, node_val.dtype), {}
            )
        return _ReductionDimAnalyzer._expand_to_index(expr, index)

    @staticmethod
    def _expand_to_index(expr: Any, index: torch.fx.Node) -> Any | None:
        index_val = _tensor_val(index)
        if index_val is None:
            return None
        return _Expr(
            aten.expand.default, (expr, _Shape(index, tuple(index_val.shape))), {}
        )

    @staticmethod
    def _valid_index(iota: _Iota, index: _Uniform) -> bool:
        if not isinstance(index.expr, torch.fx.Node):
            return False
        val = _tensor_val(index.expr)
        return (
            val is not None
            and is_integer_dtype(val.dtype)
            and val.dtype is iota.dtype
            and torch.iinfo(val.dtype).max >= iota.length - 1
        )

    @staticmethod
    def _branch_arg(arg: Any, hit: bool) -> Any:
        def select(value: Any) -> Any:
            if isinstance(value, _Uniform):
                return value.expr
            if isinstance(value, _Singleton):
                return value.hit if hit else value.miss
            return value

        return pytree.tree_map(select, arg)

    def _analyze_arg(self, arg: Any) -> Any:
        def get_value(node: torch.fx.Node) -> Any:
            value = self.memo.get(node)
            return _Unknown() if value is None else value

        return torch.fx.map_arg(arg, get_value)

    @staticmethod
    def _abstract_leaves(arg: Any) -> list[_Value]:
        return [
            value
            for value in pytree.tree_leaves(arg)
            if isinstance(
                value, (_Unknown, _Uniform, _Iota, _SingletonMask, _Singleton)
            )
        ]

    @staticmethod
    def _infer_miss_zero_kind(
        node: torch.fx.Node, args: Any, singleton: _Singleton
    ) -> tuple[_ZeroKind, bool]:
        target = node.target
        if target in (aten.clone.default, aten.neg.default):
            return singleton.miss_zero_kind, True
        if target is prims.convert_element_type.default:
            if singleton.miss_zero_kind is _ZeroKind.EXACT:
                return _ZeroKind.EXACT, True
            if (
                singleton.miss_zero_kind is _ZeroKind.ZERO_OR_NAN
                and _is_float_tensor(node)
            ):
                return _ZeroKind.ZERO_OR_NAN, True
            return _ZeroKind.UNKNOWN, True
        if target in (aten.mul.Tensor, aten.mul.Scalar):
            kinds = [
                _ZeroKind.EXACT
                for value in args
                if type(value) in (bool, int, float, complex) and value == 0
            ]
            for value in _ReductionDimAnalyzer._abstract_leaves(args):
                if isinstance(value, _Uniform):
                    kinds.append(value.zero_kind)
                elif isinstance(value, _Singleton):
                    kinds.append(value.miss_zero_kind)
            if _ZeroKind.ZERO_OR_NAN in kinds:
                return _ZeroKind.ZERO_OR_NAN, True
            if _ZeroKind.EXACT in kinds:
                kind = (
                    _ZeroKind.ZERO_OR_NAN
                    if _is_float_tensor(node)
                    else _ZeroKind.EXACT
                )
                return kind, True
            return _ZeroKind.UNKNOWN, True
        return _ZeroKind.UNKNOWN, False

    @staticmethod
    def _rounds_to_low_precision(node: torch.fx.Node) -> bool:
        if node.target is not prims.convert_element_type.default:
            return False
        source = node.args[0] if node.args else None
        source_val = _tensor_val(source) if isinstance(source, torch.fx.Node) else None
        node_val = _tensor_val(node)
        return (
            source_val is not None
            and node_val is not None
            and source_val.dtype is torch.float32
            and node_val.dtype in (torch.bfloat16, torch.float16)
        )

    @staticmethod
    def _simplify_branch(
        node: torch.fx.Node,
        args: Any,
        singleton: _Singleton,
        hit: bool,
    ) -> Any | None:
        if node.target not in (aten.mul.Tensor, aten.mul.Scalar):
            return None
        if not isinstance(args, tuple) or len(args) != 2:
            return None

        lhs, rhs = args
        if lhs is singleton and isinstance(rhs, _Uniform):
            other = rhs.expr
        elif rhs is singleton and isinstance(lhs, _Uniform):
            other = lhs.expr
        else:
            return None

        scalar = singleton.hit_scalar if hit else singleton.miss_scalar
        node_val = _tensor_val(node)
        other_val = _tensor_val(other) if isinstance(other, torch.fx.Node) else None
        if (
            node_val is None
            or other_val is None
            or node_val.dtype != other_val.dtype
            or node_val.device != other_val.device
        ):
            return None
        if scalar == 1:
            result = other
        elif scalar == -1:
            result = _Expr(aten.neg.default, (other,), {})
        elif scalar == 0 and _is_float_tensor(node):
            result = _Expr(aten.sub.Tensor, (other, other), {})
        else:
            return None
        return result

    def _is_supported_pointwise(self, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target in _TOTAL_POINTWISE_OPS
            and torch.Tag.pointwise in node.target.tags
            and torch.Tag.out not in node.target.tags
            and torch.Tag.nondeterministic_seeded not in node.target.tags
            and torch.Tag.maybe_aliasing_or_mutating not in node.target.tags
            and not node.target._schema.is_mutable
            and all(ret.alias_info is None for ret in node.target._schema.returns)
            and _tensor_val(node) is not None
        )

    def _dependencies(self, node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return ()
        if node.target is prims.iota.default:
            return ()
        if node.target is aten.expand.default or node.target in _RESHAPE_OPS:
            source = node.args[0] if node.args else None
            return (source,) if isinstance(source, torch.fx.Node) else ()
        if self._is_uniform_shape(node):
            return ()
        if node.target in (
            aten.eq.Tensor,
            aten.eq.Scalar,
            aten.where.self,
        ):
            return tuple(node.all_input_nodes)
        return tuple(node.all_input_nodes) if self._is_supported_pointwise(node) else ()

    def analyze(self, node: torch.fx.Node) -> _Value:
        if node in self.memo:
            return self.memo[node]

        result: _Value = None
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            result = _Uniform(node) if self._is_uniform_shape(node) else None
        elif node.target is prims.iota.default:
            length = node.args[0] if node.args else None
            start = node.kwargs.get("start", 0)
            step = node.kwargs.get("step", 1)
            val = _tensor_val(node)
            if (
                isinstance(length, int)
                and start == 0
                and step == 1
                and val is not None
                and is_integer_dtype(val.dtype)
            ):
                result = _Iota(length, 0, val.dtype)
        elif node.target is aten.expand.default:
            source = self._analyze_arg(node.args[0])
            val = _tensor_val(node)
            if (
                isinstance(source, _Uniform)
                and val is not None
                and val.dim() > self.dim
                and statically_known_true(val.stride()[self.dim] == 0)
            ):
                result = source
        elif node.target in _RESHAPE_OPS:
            source = self._analyze_arg(node.args[0])
            if isinstance(source, _Iota):
                result = self._view_iota(node, source)
            elif isinstance(source, _Uniform) and self._is_uniform_shape(node):
                result = _Uniform(node, source.zero_kind, source.scalar)
        elif node.target in (aten.eq.Tensor, aten.eq.Scalar):
            lhs = self._analyze_arg(node.args[0])
            rhs = self._analyze_arg(node.args[1])
            if (
                isinstance(lhs, _Iota)
                and isinstance(rhs, _Uniform)
                and self._valid_index(lhs, rhs)
            ):
                if lhs.axis == self.dim and lhs.length == self.length:
                    result = _SingletonMask(rhs.expr, lhs.length)
            elif (
                isinstance(rhs, _Iota)
                and isinstance(lhs, _Uniform)
                and self._valid_index(rhs, lhs)
            ):
                if rhs.axis == self.dim and rhs.length == self.length:
                    result = _SingletonMask(lhs.expr, rhs.length)
        elif node.target is aten.where.self:
            mask = self._analyze_arg(node.args[0])
            hit = self._analyze_arg(node.args[1])
            miss = self._analyze_arg(node.args[2])
            if (
                isinstance(mask, _SingletonMask)
                and isinstance(hit, _Uniform)
                and isinstance(miss, _Uniform)
            ):
                hit_expr = self._where_branch(node, mask.index, hit)
                miss_expr = self._where_branch(node, mask.index, miss)
                if hit_expr is not None and miss_expr is not None:
                    result = _Singleton(
                        mask.index,
                        mask.length,
                        hit_expr,
                        miss_expr,
                        miss.zero_kind,
                        hit.scalar,
                        miss.scalar,
                    )
        elif self._is_uniform_shape(node):
            result = self._constructor_uniform(node)
        elif self._is_supported_pointwise(node):
            args = self._analyze_arg(node.args)
            kwargs = self._analyze_arg(node.kwargs)
            leaves = self._abstract_leaves((args, kwargs))
            singletons = [value for value in leaves if isinstance(value, _Singleton)]
            supported = all(
                isinstance(value, (_Uniform, _Singleton)) for value in leaves
            )
            if singletons and supported and all(
                self._same_singleton(singletons[0], value)
                for value in singletons[1:]
            ) and self._matches_reduction_shape(node, singletons[0].index):
                singleton = singletons[0]
                hit = self._simplify_branch(node, args, singleton, True)
                if hit is None:
                    hit = _Expr(
                        node.target,
                        self._branch_arg(args, True),
                        self._branch_arg(kwargs, True),
                    )
                miss = self._simplify_branch(node, args, singleton, False)
                if miss is None:
                    miss = _Expr(
                        node.target,
                        self._branch_arg(args, False),
                        self._branch_arg(kwargs, False),
                    )
                hit = self._expand_to_index(hit, singleton.index)
                miss = self._expand_to_index(miss, singleton.index)
                if hit is None or miss is None:
                    result = None
                else:
                    miss_kind, modeled = self._infer_miss_zero_kind(
                        node, args, singleton
                    )
                    result = _Singleton(
                        singleton.index,
                        singleton.length,
                        hit,
                        miss,
                        miss_kind,
                        has_unmodeled_pointwise=(
                            not modeled
                            or any(
                                value.has_unmodeled_pointwise
                                for value in singletons
                            )
                        ),
                        has_low_precision_rounding=(
                            self._rounds_to_low_precision(node)
                            or any(
                                value.has_low_precision_rounding
                                for value in singletons
                            )
                        ),
                    )

        if result is None and self._is_uniform_shape(node):
            result = self._constructor_uniform(node)
        self.memo[node] = result
        return result

    def analyze_subgraph(
        self, node: torch.fx.Node, max_nodes: int | None = None
    ) -> _Value:
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
            if max_nodes is not None and len(seen) > max_nodes:
                self.nodes_visited = len(seen)
                return None
            pending.append((current, True))
            pending.extend(
                (input_node, False) for input_node in self._dependencies(current)
            )
        for current in order:
            self.analyze(current)
        self.nodes_visited = len(seen)
        return self.memo.get(node)


def _has_downstream_reduction(
    dense: torch.fx.Node, reduction: torch.fx.Node, max_nodes: int
) -> tuple[bool | None, int]:
    pending = [user for user in dense.users if user is not reduction]
    seen = OrderedSet[torch.fx.Node]()
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if len(seen) > max_nodes:
            return None, len(seen)
        if user.op != "call_function" or not isinstance(
            user.target, torch._ops.OpOverload
        ):
            continue
        if torch.Tag.reduction in user.target.tags:
            return True, len(seen)
        if torch.Tag.pointwise in user.target.tags or is_view(user.target):
            pending.extend(user.users)
    return False, len(seen)


def _has_expanding_pointwise_consumer(
    reduction: torch.fx.Node, max_nodes: int
) -> tuple[bool | None, int]:
    reduction_val = _tensor_val(reduction)
    if reduction_val is None:
        return False, 0
    pending = list(reduction.users)
    seen = OrderedSet[torch.fx.Node]()
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if len(seen) > max_nodes:
            return None, len(seen)
        if user.op != "call_function" or not isinstance(
            user.target, torch._ops.OpOverload
        ):
            continue
        if torch.Tag.pointwise in user.target.tags:
            user_val = _tensor_val(user)
            if user_val is not None and statically_known_true(
                user_val.numel() > reduction_val.numel()
            ):
                return True, len(seen)
            pending.extend(user.users)
        elif is_view(user.target):
            pending.extend(user.users)
    return False, len(seen)


def _materialization_plan(value: Any, max_ops: int) -> list[_Expr] | None:
    roots = [item for item in pytree.tree_leaves(value) if isinstance(item, _Expr)]
    pending = [(root, False) for root in reversed(roots)]
    scheduled: set[int] = set()
    plan: list[_Expr] = []
    while pending:
        expr, ready = pending.pop()
        key = id(expr)
        if ready:
            plan.append(expr)
            continue
        if key in scheduled:
            continue
        scheduled.add(key)
        if len(scheduled) > max_ops:
            return None
        pending.append((expr, True))
        dependencies = [
            item
            for item in pytree.tree_leaves((expr.args, expr.kwargs))
            if isinstance(item, _Expr)
        ]
        pending.extend((dependency, False) for dependency in dependencies)
    return plan


def _materialize(graph: torch.fx.Graph, value: Any, plan: list[_Expr]) -> Any:
    memo: dict[Any, Any] = {}

    def replace(item: Any) -> Any:
        if isinstance(item, _Shape):
            key = ("shape", item.source)
            if key not in memo:
                memo[key] = [
                    dim
                    if isinstance(dim, int)
                    else graph.create_size_node(item.source, i)
                    for i, dim in enumerate(item.values)
                ]
            return memo[key]
        if isinstance(item, _Expr):
            return memo[id(item)]
        return item

    for expr in plan:
        key = id(expr)
        args = pytree.tree_map(replace, expr.args)
        kwargs = pytree.tree_map(replace, expr.kwargs)
        memo[key] = graph.call_function(expr.target, args, kwargs)
    return pytree.tree_map(replace, value)


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
    dim = dims[0] % dense_val.dim()
    return dense, dim


def eliminate_singleton_reductions(graph: torch.fx.Graph) -> int:
    count = 0
    analysis_nodes = 0
    analysis_cache: dict[tuple[torch.fx.Node, int, int], _Value] = {}
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
            or dense_val.device.type != "cuda"
            or not dense_val.dtype.is_floating_point
            or dense_val.dtype != output_val.dtype
            or not isinstance(dense_val.shape[dim], int)
            or dense_val.shape[dim] <= 1
        ):
            continue
        key = (dense, dim, dense_val.shape[dim])
        if key not in analysis_cache:
            remaining_nodes = _MAX_ANALYSIS_NODES - analysis_nodes
            if remaining_nodes <= 0:
                break
            analyzer = _ReductionDimAnalyzer(reduction, dim, dense_val.shape[dim])
            analysis_cache[key] = analyzer.analyze_subgraph(
                dense, max_nodes=remaining_nodes
            )
            analysis_nodes += analyzer.nodes_visited
        singleton = analysis_cache[key]
        if not isinstance(singleton, _Singleton) or singleton.miss_zero_kind not in (
            _ZeroKind.EXACT,
            _ZeroKind.ZERO_OR_NAN,
        ):
            continue
        materialized_values = (
            singleton.hit
            if singleton.miss_zero_kind is _ZeroKind.EXACT
            else (singleton.hit, singleton.miss)
        )
        materialization_plan = _materialization_plan(
            materialized_values, _MAX_MATERIALIZATION_OPS
        )
        if materialization_plan is None:
            continue
        materialization_size = len(materialization_plan)
        dense_is_live = any(user is not reduction for user in dense.users)
        # Limit live reuse to large low-precision rows with bounded materialization.
        # Full-width-only paths need scheduler-aware costing, and shape padding
        # can make retaining the original reduction cheaper through donation.
        row_bytes = singleton.length * dense_val.element_size()
        if dense_is_live and (
            torch.version.hip is not None
            or dense_val.dtype is not torch.float32
            or config.force_shape_pad
            or singleton.has_unmodeled_pointwise
            or not singleton.has_low_precision_rounding
            or row_bytes < _MIN_LIVE_REDUCTION_ROW_BYTES
            or materialization_size >= singleton.length
        ):
            continue
        if not isinstance(singleton.index, torch.fx.Node):
            continue
        index_val = _tensor_val(singleton.index)
        if (
            index_val is None
            or not is_integer_dtype(index_val.dtype)
            or index_val.device != dense_val.device
            or not statically_known_true(sym_eq(index_val.shape, output_val.shape))
        ):
            continue
        if dense_is_live:
            remaining_nodes = _MAX_ANALYSIS_NODES - analysis_nodes
            if remaining_nodes <= 0:
                break
            expanding, nodes_visited = _has_expanding_pointwise_consumer(
                reduction, remaining_nodes
            )
            analysis_nodes += nodes_visited
            if expanding is None:
                break
            if not expanding:
                continue

            remaining_nodes = _MAX_ANALYSIS_NODES - analysis_nodes
            if remaining_nodes <= 0:
                break
            downstream_reduction, nodes_visited = _has_downstream_reduction(
                dense, reduction, remaining_nodes
            )
            analysis_nodes += nodes_visited
            if downstream_reduction is None:
                break
            if downstream_reduction:
                continue

        with graph.inserting_before(reduction):
            ge_zero = graph.call_function(aten.ge.Scalar, (singleton.index, 0))
            in_range = graph.call_function(
                aten.le.Scalar, (singleton.index, singleton.length - 1)
            )
            valid = graph.call_function(aten.bitwise_and.Tensor, (ge_zero, in_range))
            materialized = _materialize(
                graph, materialized_values, materialization_plan
            )
            if singleton.miss_zero_kind is _ZeroKind.EXACT:
                hit = materialized
                miss = None
            else:
                hit, miss = materialized
            if not isinstance(hit, torch.fx.Node):
                raise AssertionError(f"expected FX node, got {type(hit)}")
            zero = graph.call_function(
                aten.full.default,
                ([], 0.0),
                {"dtype": dense_val.dtype, "device": dense_val.device},
            )
            normalized_hit = graph.call_function(aten.add.Tensor, (hit, zero))
            finite = graph.call_function(
                aten.where.self, (valid, normalized_hit, zero)
            )
            if singleton.miss_zero_kind is _ZeroKind.EXACT:
                replacement = finite
            else:
                if not isinstance(miss, torch.fx.Node):
                    raise AssertionError(f"expected FX node, got {type(miss)}")
                miss_is_zero = graph.call_function(aten.eq.Scalar, (miss, 0))
                replacement = graph.call_function(
                    aten.where.self, (miss_is_zero, finite, miss)
                )
        reduction.replace_all_uses_with(replacement)
        graph.erase_node(reduction)
        counters["inductor"]["singleton_reduction_elimination"] += 1
        count += 1
    if count:
        graph.eliminate_dead_code()
    return count
