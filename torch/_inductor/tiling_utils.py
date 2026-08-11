import dataclasses
import itertools
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload, TYPE_CHECKING, TypeVar, Union

import sympy

import torch
from torch._inductor import ir
from torch._inductor.dependencies import index_vars_no_squeeze, MemoryDep, ReadWrites
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .virtualized import V


T = TypeVar("T")
U = TypeVar("U")


Split = tuple[sympy.Expr, ...]
VarsAndRanges = tuple[list[sympy.Symbol], list[sympy.Expr]]


loop_tiling_log = torch._logging.getArtifactLogger(__name__, "loop_tiling")
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


if TYPE_CHECKING:
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        FusedSchedulerNode,
        SchedulerNode,
    )


def solve_for_zero(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
    if expr.is_constant():
        return None
    elif isinstance(expr, FloorDiv):
        return None

    if len(expr.free_symbols) != 1:
        raise AssertionError(
            f"expected exactly 1 free symbol, got {len(expr.free_symbols)}"
        )
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out[1]


def solve_for_tiling(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.

    Tiling an expression `x` by `y` means that the expression will now be indexed
    by both the original (x) and by (x * y). So we are looking for a
    multiplicative factor that will make ((x + 1) * y) - (x * y) == 1.

    To simplify things for sympy, we'll try just x * y == 1, check x(1) and x(0).
    """

    if len(expr.free_symbols) != 1:
        return None

    free_symbol = next(iter(expr.free_symbols))

    def _solve_simple_expr(expr: sympy.Expr) -> sympy.Expr | None:
        if expr.has(ModularIndexing) or expr.has(FloorDiv):
            # the div approximation could not eliminate all ModularIndexing /
            # FloorDiv nodes; we cannot solve this expression
            return None
        if len(expr.free_symbols) != 1:
            return None

        out = try_solve(sympy.Eq(expr, 1), free_symbol)
        if not out or not out[1].is_constant():
            return None
        return out[1]

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        return _solve_simple_expr(expr)

    required_values = []
    eq_1_expressions = []

    # very piecemeal solution if ModularIndexing or FloorDiv involved.
    # Look for terms we'll try to make 0, and then other terms we'll try to make 1.
    # Expand as needed.
    for arg in sympy.Add.make_args(expr):
        # Try to make mul terms 0
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO - only need one of these to be solvable to zero
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None:
                    continue

                if not out.is_constant():
                    raise AssertionError(f"expected constant, got {out}")
                seen = True
                required_values.append(out)

            if not seen:
                return None
        else:
            eq_1_expressions.append(arg)

    if not eq_1_expressions:
        return None

    eq_1_expr = sum(eq_1_expressions)

    def indexing_div_rep(
        x: sympy.Expr,
        y: sympy.Expr,
        z: sympy.Expr | None = None,
    ) -> sympy.Expr:
        return x / y

    # For the purposes of tiling/coalesced access, approximate ModularIndexing and FloorDiv
    # then check later. simultaneous=False rebuilds bottom up, collapsing nested
    # occurrences in one pass; the second sweep catches nodes that FloorDiv eval
    # reintroduces while rebuilding. Leftovers make _solve_simple_expr bail out.
    eq_1_expr_simplified = eq_1_expr
    for _ in range(2):
        # pyrefly: ignore [missing-attribute]
        eq_1_expr_simplified = eq_1_expr_simplified.replace(
            ModularIndexing, indexing_div_rep, simultaneous=False
        ).replace(FloorDiv, indexing_div_rep, simultaneous=False)

    out = _solve_simple_expr(eq_1_expr_simplified)

    # since we approximated FloorDiv/ModularIndexing, double check here
    if not out or sympy_subs(eq_1_expr, {free_symbol: out}) != 1:
        return None

    required_values.append(out)

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_broadcast_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> sympy.Expr | None:
    """
    Try to find the variable that this index is broadcast over.
    A broadcast pattern is one where consecutive values of a variable
    access the same memory location (e.g., x // 10).
    """
    # Approximate analysis by evaluating at 1 and 0
    variables: dict[sympy.Symbol, int] = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        if v not in index.free_symbols:
            continue

        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        # Broadcast means the value doesn't change when the variable increments
        if new_val == zero_index:
            return v
        variables[v] = 0

    return None


def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> sympy.Expr | None:
    """
    Try to find the symbol which coalesces this index
    """
    top_level_terms = sympy.Add.make_args(index)
    for v in var_ranges:
        if v in top_level_terms:
            return v

    # Approximate analysis by evaluating at 1 and 0
    variables: dict[sympy.Symbol, int] = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        if new_val - zero_index == 1:
            variables[v] = 2
            # in some more complex expressions, 0->1 will be coalesced,
            # but not 1->2
            if (sympy_subs(index, variables) - new_val) == 1:
                return v
        variables[v] = 0

    return None


def has_indirect_access(memory_expr: sympy.Expr) -> bool:
    """
    Check if this memory expression has any indirect indexing.
    """
    return any(symbol_is_type(s, SymT.INDIRECT) for s in memory_expr.free_symbols)


@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """
    Normalized reads and writes for nodes in the same FusedSchedulerNode.
    """

    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]
    node_var_mappings: dict[str, dict[sympy.Symbol, sympy.Expr]]


@dataclasses.dataclass(frozen=True)
class _FusedNodeView:
    nodes: Sequence["BaseSchedulerNode"]
    read_writes: ReadWrites
    group: Any

    def get_device(self) -> torch.device | None:
        return self.nodes[0].get_device()

    def get_nodes(self) -> Sequence["BaseSchedulerNode"]:
        return self.nodes

    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*(node.get_buffer_names() for node in self.nodes))

    def get_operation_names(self) -> OrderedSet[str]:
        return OrderedSet(node.get_name() for node in self.nodes)


def _supports_broadcast_work_node(
    node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
) -> bool:
    if torch.version.hip is not None:
        return False
    device = node.get_device()
    return (
        device is not None
        and device.type == "cuda"
        and not any(child.has_aliasing_or_mutation() for child in node.get_nodes())
    )


def _supports_broadcast_work_domain(
    index_vars: Sequence[sympy.Symbol] | OrderedSet[sympy.Symbol],
    reduce_vars: Sequence[sympy.Symbol] | OrderedSet[sympy.Symbol],
) -> bool:
    return not reduce_vars and len(index_vars) == 2


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[True],
) -> tuple[VarsAndRanges, VarsAndRanges] | None: ...


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[False] = False,
) -> tuple[VarsAndRanges, VarsAndRanges]: ...


def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: bool = False,
) -> tuple[VarsAndRanges, VarsAndRanges] | None:
    # nb: use statically_known_equals here to mimic scheduler.
    # TODO : store type of split/broadcast on fused node itself,
    # instead of re-deriving it.
    if n.is_reduction() or V.graph.sizevars.statically_known_equals(
        sympy_product(n._body.sizes[0]), pointwise_numel
    ):
        # pyrefly: ignore [bad-return]
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]

    if get_hint(sympy_product(n._body.sizes[0])) != get_hint(
        pointwise_numel * red_numel  # type: ignore[operator]
    ):
        raise AssertionError(
            "expected pointwise sizes to match pointwise_numel * red_numel"
        )
    i = len(n._body.sizes[0]) - 1
    prod = 1
    while i >= 0:
        prod *= n._body.sizes[0][i]
        if prod == red_numel:
            break
        i -= 1

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)  # type: ignore[return-value]

    if none_if_not_divisible:
        return None
    else:
        # pyrefly: ignore [bad-return]
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]


class NodeSplitGetter:
    """
    Finds a Pointwise, Reduction Split that compatible with all nodes in a SchedulerNode.
    """

    def __init__(
        self,
        node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
    ):
        self.node = node
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)
        self.red_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        self.reduction_split: Split = ()
        self.all_node_sizes: OrderedSet[tuple[Split, Split]] = OrderedSet()

        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            # if we can't split the pw ranges into a (pw, red) split,
            # don't add as a split option, but do make sure we check that this size
            # is splittable
            maybe_splits = get_pw_red_splits(
                n, self.pointwise_numel, self.red_numel, none_if_not_divisible=True
            )
            if maybe_splits is None:
                self.all_node_sizes.add(n._body.sizes)
                continue

            (_, n_pw_splits), (_, n_red_splits) = maybe_splits

            # fill in reduction size
            n_pw_splits, n_red_splits = (
                torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                    fused_group, (n_pw_splits, n_red_splits), self.red_numel
                )
            )

            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
            self.red_split_options[len(n_red_splits)].add(tuple(n_red_splits))

            if n_red_splits != ():
                self.reduction_split = (sympy_product(n_red_splits),)

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        self.seen_pw_splits: OrderedSet[Split] = OrderedSet()

    def get_node_splits(self) -> tuple[Split, Split]:
        """
        Get a compatible pointwise, reduction split of the node
        """

        if len(self.all_node_sizes) == 1:
            return next(iter(self.all_node_sizes))

        if len(self.pw_split_options) == 0:
            return ((self.pointwise_numel,), (self.red_numel,))

        max_pw_split = max(self.pw_split_options.keys())
        max_red_split = max(self.red_split_options.keys())

        def add_combined_split_options(
            split_options: dict[int, OrderedSet[Split]], curr_length: int
        ) -> None:
            for split in split_options[curr_length]:
                for i in range(len(split) - 1):
                    new_split = tuple(
                        split[0:i] + (sympy_product(split[i : i + 2]),) + split[i + 2 :]
                    )
                    split_options[len(new_split)].add(new_split)

        max_total_splits = max_pw_split + max_red_split
        for curr_iter, total_splits in enumerate(range(max_total_splits, 0, -1)):
            for pw_split_len in range(total_splits, 0, -1):
                for pw_split in self.pw_split_options[pw_split_len]:
                    for red_split in self.red_split_options[
                        total_splits - pw_split_len
                    ]:
                        if out := self.try_split(pw_split, red_split):
                            return out

            add_combined_split_options(self.pw_split_options, max_pw_split - curr_iter)
            add_combined_split_options(
                self.red_split_options, max_red_split - curr_iter
            )

        # if for whatever reason we couldn't split above, return default split
        return ((self.pointwise_numel,), (self.red_numel,))

    def try_split(self, pw: Split, red: Split) -> tuple[Split, Split] | None:
        """
        See if this split is compatible, and potentially returning a longer split
        than the input.
        """

        from torch._inductor.codegen.simd import CantSplit, SIMDKernel

        if pw in self.seen_pw_splits:
            return None
        self.seen_pw_splits.add(pw)

        for n_pw, n_red in self.all_node_sizes:
            try:
                groups = pw + red
                lengths = (n_pw, n_red)
                splits, getters = SIMDKernel._split_iteration_ranges(groups, lengths)
            except CantSplit:
                return None

            if len(getters) != 2:
                raise AssertionError(f"expected 2 getters, got {len(getters)}")
            pw_group_splits = splits[: len(pw)]
            # if we had to divide a variable into two to do this split,
            # then lets try the larger, induced split.
            # e.g. splitting (12, 2) into (2, 12) will split the first var into:
            # (2, 6) and produce an overall split of (2, 6, 2)
            flattened_pw_splits = tuple(itertools.chain.from_iterable(pw_group_splits))
            if flattened_pw_splits != pw:
                if out := self.try_split(flattened_pw_splits, red):
                    return out

        return pw, red


def apply_var_mapping(
    iter_vars: list[sympy.Symbol],
    red_vars: list[sympy.Symbol],
    norm_pw_vars: list[sympy.Symbol],
    norm_red_vars: list[sympy.Symbol],
    new_ranges: list[list[sympy.Expr]],
    return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Maps original variables to expressions using normalized variables."""

    # the output of split_iteration_range is a new_ranges, return_getters_groups
    # new_ranges is a flattened list of ranges corresponding to the new pw and red vars
    # for example, taking in pw vars of range (6, 6) to normalized range [36],
    # new_ranges would be [[6, 6]]
    # There is a return_getter callable for each input iter_var and red_vars.
    # if you flatten out all of the ranges, and create a variable for each index,
    # then applying the flattening vars to the callables in return_getters_groups
    # gives you the mapping from input vars -> flattened vars.
    # From there, we can compute the output, normalized variables.
    # For instance [6, 6] corresponding to flat vars v0, v1 will be
    # v0 + 6 * v1

    # Create flattened iteration variables
    num_vars = sum(len(s) for s in new_ranges)
    flat_vars = sympy.symbols(f"v_0:{num_vars}")
    count = 0

    if len(iter_vars) == 0 and len(red_vars) == 0:
        return {}

    if len(new_ranges) != len(norm_pw_vars + norm_red_vars):
        raise AssertionError(
            f"expected len(new_ranges) == len(norm_pw_vars + norm_red_vars), "
            f"got {len(new_ranges)} and {len(norm_pw_vars + norm_red_vars)}"
        )
    apply_groups = []
    for group in return_getters_groups:
        apply_groups.append([g(flat_vars) for g in group])

    iter_vars_to_flat_vars = {}
    for i, (group, var_group) in enumerate(
        zip(apply_groups, (iter_vars, red_vars), strict=True)
    ):
        # if the node has sizes (p0, 1) and the fused node is (p0, r0)
        # the reduction var gets filled in for split_iteration_range
        if len(group) != len(var_group):
            if i != 1:
                raise AssertionError(f"expected i == 1, got {i}")
            if len(var_group) != 0:
                raise AssertionError(
                    f"expected empty var_group, got len {len(var_group)}"
                )
            continue

        iter_vars_to_flat_vars.update({v: g for g, v in zip(group, var_group)})

    count = 0
    flat_vars_to_new_vars = {}
    for new_range, new_var in zip(
        new_ranges, norm_pw_vars + norm_red_vars, strict=True
    ):
        range_vars = []
        for _ in range(len(new_range)):
            range_vars.append(flat_vars[count])
            count += 1

        prod = 1
        for i in range(len(new_range) - 1, -1, -1):
            flat_vars_to_new_vars[range_vars[i]] = new_var * prod
            prod = new_range[i] * prod

    return {
        k: sympy_subs(v, flat_vars_to_new_vars)
        for k, v in iter_vars_to_flat_vars.items()
    }


def _group_accesses_by_index(
    accesses: dict[sympy.Expr, OrderedSet[str]],
    normalize: Callable[[sympy.Expr], sympy.Expr],
) -> dict[sympy.Expr, OrderedSet[str]]:
    result: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    for expr, buffer_names in accesses.items():
        result[normalize(expr)] |= buffer_names
    return dict(result)


def extract_normalized_read_writes(
    node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
) -> FusedNormalizedReadsWrites | None:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    node_var_mappings: dict[str, dict[sympy.Symbol, sympy.Expr]] = {}

    all_output_names = node.get_buffer_names()
    op_names = node.get_operation_names()
    outputs: OrderedSet[str] = OrderedSet()
    removed_buffers: OrderedSet[str] = OrderedSet()
    for buf_name in all_output_names:
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(buf_name, op_names):
            removed_buffers.add(buf_name)
        else:
            outputs.add(buf_name)

    inputs = OrderedSet(
        dep.name for dep in node.read_writes.reads if dep.name not in removed_buffers
    )

    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )
    retain_node_var_mappings = _supports_broadcast_work_domain(
        norm_pw_vars, norm_red_vars
    ) and _supports_broadcast_work_node(node)

    for n in list(node.get_nodes()):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body

        n_reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
        n_writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

        # TODO - will the names for all the inputs/outputs accurately
        # reflect mutation, or do I need to remap with mutation_real_name
        for inp in inputs:
            for expr in body.get_all_read_expr(inp):
                n_reads[expr].add(inp)

        for out in outputs:
            for expr in body.get_all_write_expr(out):
                n_writes[expr].add(out)

        mapping_only = not n_reads and not n_writes
        if mapping_only and not retain_node_var_mappings:
            continue

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(
            n, pointwise_numel, red_numel
        )

        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = (
            torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                groups, lengths, red_numel
            )
        )
        try:
            new_ranges, return_getters_groups = (
                torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(
                    groups, lengths
                )
            )
        except torch._inductor.codegen.simd.CantSplit as e:
            if mapping_only:
                # This body is needed only by the optional broadcast-work
                # analysis. Missing lineage makes that analysis fail closed,
                # but must not discard the existing read/write analysis.
                continue
            # occasionally with dynamic shapes, we will be unable to prove
            # divisibility
            if not (pointwise_numel.free_symbols or red_numel.free_symbols):
                raise AssertionError(
                    "expected dynamic shapes (free symbols) when split fails"
                ) from e
            return None

        var_map = apply_var_mapping(
            iter_vars,
            red_vars,
            norm_pw_vars,
            norm_red_vars,
            new_ranges,
            return_getters_groups,
        )
        if retain_node_var_mappings:
            node_var_mappings[n.get_name()] = var_map

        # We create Identity sympy.Functions to prevent expansion to int64,
        # unwrap for tiling analysis.
        def remove_identity(expr: sympy.Expr) -> sympy.Expr:
            return expr.replace(Identity, lambda x: x)

        for expr, buf_names in _group_accesses_by_index(
            n_reads,
            lambda expr: sympy_subs(remove_identity(expr), var_map),
        ).items():
            reads[expr] |= buf_names
        for expr, buf_names in _group_accesses_by_index(
            n_writes,
            lambda expr: sympy_subs(remove_identity(expr), var_map),
        ).items():
            writes[expr] |= buf_names

    reads = _group_accesses_by_index(
        reads,
        lambda expr: V.graph.sizevars.simplify_with_ranges(expr, ranges),
    )
    writes = _group_accesses_by_index(
        writes,
        lambda expr: V.graph.sizevars.simplify_with_ranges(expr, ranges),
    )

    fused_out = FusedNormalizedReadsWrites(
        norm_pw_vars,  # type: ignore[arg-type]
        norm_red_vars,  # type: ignore[arg-type]
        reads,
        writes,
        ranges,
        node_var_mappings,
    )
    loop_tiling_log.info("Normalized Fused reads: %s", fused_out)
    return fused_out


def get_score(
    addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int], buf_names: OrderedSet[str]
) -> int:
    """
    Score addr according to its approximate size.
    """
    # TODO - deduplicate with candidate_tilings
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v)
        # TODO - reason about indirect vars
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V

    return V.graph.sizevars.optimization_hint(sympy_product(var_sizes))


def try_get_buf_size(buf_name: str) -> int | None:
    buf = V.graph.try_get_buffer(buf_name)
    if not buf:
        return None
    return V.graph.sizevars.optimization_hint(sympy_product(buf.get_size()))


def get_hint(v: sympy.Expr | int) -> int:
    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.optimization_hint(v)


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """

    var: sympy.Symbol
    tiling_factor: int
    score: int


@dataclasses.dataclass(frozen=True)
class BroadcastWork:
    """Work saved by preserving one normalized pointwise split."""

    split_var: sympy.Symbol
    operation_score_bonus: int


@dataclasses.dataclass(frozen=True)
class _BroadcastValueDomain:
    """Iteration-symbol partition used by value-dependency analysis."""

    index_vars: OrderedSet[sympy.Symbol]
    preserved_vars: OrderedSet[sympy.Symbol]
    replay_vars: OrderedSet[sympy.Symbol]

    def is_preserved(self, dependencies: OrderedSet[sympy.Symbol]) -> bool:
        return bool(dependencies) and dependencies <= self.preserved_vars

    def is_replayed(self, dependencies: OrderedSet[sympy.Symbol]) -> bool:
        return bool(dependencies & self.replay_vars)


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    # Var -> Memory Score - not strictly the amount of memory
    # because we multiply writes x2
    # TODO: separate into dataclass that holds mem, dtype, is_write
    coalesced_by_var: dict[sympy.Expr, int]

    uncoalesced_addrs: dict[sympy.Expr, int]

    norm_read_writes: FusedNormalizedReadsWrites

    suggested_split: VarTiling | None = None
    broadcast_work: BroadcastWork | None = None


_BROADCAST_OP_COSTS = {
    op: cost
    for cost, ops in (
        (0, ("constant", "identity", "index_expr", "load", "store")),
        (
            1,
            (
                "abs",
                "add",
                "and_",
                "bitwise_and",
                "bitwise_not",
                "bitwise_or",
                "bitwise_xor",
                "eq",
                "ge",
                "gt",
                "le",
                "logical_and",
                "logical_not",
                "logical_or",
                "logical_xor",
                "lt",
                "maximum",
                "minimum",
                "mul",
                "ne",
                "neg",
                "or_",
                "relu",
                "sub",
                "to_dtype",
                "to_dtype_bitcast",
                "where",
                "xor",
            ),
        ),
        (
            4,
            (
                "div",
                "floordiv",
                "fmod",
                "mod",
                "reciprocal",
                "remainder",
                "rsqrt",
                "sqrt",
                "truediv",
            ),
        ),
        (
            8,
            (
                "acos",
                "acosh",
                "asin",
                "asinh",
                "atan",
                "atan2",
                "atanh",
                "cos",
                "cosh",
                "erf",
                "erfc",
                "erfinv",
                "exp",
                "exp2",
                "expm1",
                "lgamma",
                "log",
                "log1p",
                "log2",
                "sigmoid",
                "sin",
                "sinh",
                "tan",
                "tanh",
            ),
        ),
    )
    for op in ops
}
_MIN_BROADCAST_INNER_RANGE = 32
_BROADCAST_REPLAY_FACTOR = 8
_MIN_BROADCAST_WORK = 8
_MIN_SAVED_WORK_PER_BYTE = 2
_MAX_BROADCAST_LIVE_FULL_DOMAIN_VALUES = 5


def _broadcast_op_cost(node: torch.fx.Node) -> int | None:
    if node.op != "call_method":
        return None
    return _BROADCAST_OP_COSTS.get(str(node.target))


def _fx_expression_id(
    node: torch.fx.Node,
    expression_ids: dict[torch.fx.Node, int],
    interned_expressions: dict[tuple[Any, ...], int],
    normalized_index: sympy.Expr | None = None,
) -> int:
    def convert(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return ("node", expression_ids[arg])
        if isinstance(arg, tuple):
            return tuple(convert(item) for item in arg)
        if isinstance(arg, list):
            return ("list", *(convert(item) for item in arg))
        if isinstance(arg, dict):
            return tuple(
                (repr(key), convert(value))
                for key, value in sorted(arg.items(), key=lambda item: repr(item[0]))
            )
        return ("literal", type(arg).__qualname__, repr(arg))

    if normalized_index is not None:
        key = ("normalized_index", normalized_index)
    else:
        key = (node.op, str(node.target), convert(node.args), convert(node.kwargs))
    if key not in interned_expressions:
        interned_expressions[key] = len(interned_expressions)
    return interned_expressions[key]


def _is_dense_row_major_iteration(
    index: sympy.Expr,
    outer_var: sympy.Symbol,
    inner_var: sympy.Symbol,
    inner_range: sympy.Expr | int,
) -> bool:
    origin = sympy_subs(index, {outer_var: 0, inner_var: 0})
    inner_stride = sympy_subs(index, {outer_var: 0, inner_var: 1}) - origin
    outer_stride = sympy_subs(index, {outer_var: 1, inner_var: 0}) - origin
    return (
        V.graph.sizevars.statically_known_equals(inner_stride, 1)
        and V.graph.sizevars.statically_known_geq(outer_stride, inner_range)
        and V.graph.sizevars.statically_known_equals(
            index,
            origin + outer_var * outer_stride + inner_var,
        )
    )


def _loop_body_op_arg(
    node: torch.fx.Node,
    position: int,
    name: str,
) -> Any:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name)


def _normalized_get_index(
    node: Any,
    body: Any,
    var_map: dict[sympy.Symbol, sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
) -> sympy.Expr | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_module"
        or node.target != "get_index"
    ):
        return None
    index_name = _loop_body_op_arg(node, 0, "name")
    if not isinstance(index_name, str):
        return None
    index = body.indexing_exprs.get(index_name)
    if index is None:
        return None
    return V.graph.sizevars.simplify_with_ranges(
        sympy_subs(index, var_map),
        var_ranges,
    )


def _scheduled_peak_live_values(
    value_order: Sequence[int],
    direct_load_values: OrderedSet[int],
    value_inputs: dict[int, OrderedSet[int]],
    root_values: OrderedSet[int],
    reachable: OrderedSet[int],
) -> int:
    """Approximate pressure from live full-domain SSA values."""
    scheduled_values = [
        value_id
        for value_id in value_order
        if value_id in reachable and value_id in direct_load_values
    ]
    scheduled_values.extend(
        value_id
        for value_id in value_order
        if value_id in reachable and value_id not in direct_load_values
    )
    definitions = {
        value_id: position for position, value_id in enumerate(scheduled_values)
    }
    last_uses = definitions.copy()
    for value_id in scheduled_values:
        position = definitions[value_id]
        for input_id in value_inputs.get(value_id, ()):
            if input_id in reachable:
                last_uses[input_id] = max(last_uses[input_id], position)

    # Triton queues stores until after all compute. Consequently all externally
    # stored values are simultaneously live immediately before the first store.
    store_position = len(scheduled_values)
    for value_id in root_values:
        if value_id in reachable:
            last_uses[value_id] = max(last_uses[value_id], store_position)

    starts: dict[int, int] = defaultdict(int)
    expires: dict[int, int] = defaultdict(int)
    for value_id in reachable:
        if value_id not in definitions:
            continue
        starts[definitions[value_id]] += 1
        expires[last_uses[value_id]] += 1

    live = 0
    peak = 0
    for position in sorted(starts.keys() | expires.keys()):
        live += starts[position]
        peak = max(peak, live)
        live -= expires[position]
    return peak


def _analyze_broadcast_work(
    fused_node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
    normalized: FusedNormalizedReadsWrites,
) -> BroadcastWork | None:
    """
    Find pointwise work whose value domain omits a large inner iteration axis.

    This intentionally recognizes only a clean two-dimensional, row-major case.
    More general candidates need to account for layout conflicts, indirect
    indexing, and the cost of additional masks and block dimensions.
    """
    if not _supports_broadcast_work_node(fused_node):
        return None
    if not _supports_broadcast_work_domain(
        normalized.index_vars, normalized.reduce_vars
    ):
        return None

    outer_var, inner_var = normalized.index_vars
    outer_range = normalized.var_ranges[outer_var]
    inner_range = normalized.var_ranges[inner_var]
    if getattr(outer_range, "free_symbols", ()) or getattr(
        inner_range, "free_symbols", ()
    ):
        return None
    inner_hint = V.graph.sizevars.optimization_hint(inner_range, fallback=1)
    if inner_hint < _MIN_BROADCAST_INNER_RANGE:
        return None
    replay_factor = min(inner_hint, _BROADCAST_REPLAY_FACTOR)

    all_accesses = (*normalized.reads, *normalized.writes)
    if any(has_indirect_access(expr) for expr in all_accesses):
        return None

    expected_write = outer_var * inner_range + inner_var
    if not normalized.writes or any(
        not V.graph.sizevars.statically_known_equals(expr, expected_write)
        for expr in normalized.writes
    ):
        return None
    supported_broadcast_indices = (sympy.S.Zero, outer_var, inner_var)

    def is_supported_index(index: sympy.Expr) -> bool:
        return any(
            V.graph.sizevars.statically_known_equals(index, supported)
            for supported in supported_broadcast_indices
        ) or _is_dense_row_major_iteration(
            index,
            outer_var,
            inner_var,
            inner_range,
        )

    if any(not is_supported_index(expr) for expr in normalized.reads):
        return None

    total_numel = outer_range * inner_range
    output_names = OrderedSet[str]()
    for names in normalized.writes.values():
        output_names.update(names)
    if not output_names:
        return None
    for name in output_names:
        try:
            output_numel = V.graph.get_numel(name)
        except KeyError:
            return None
        if not V.graph.sizevars.statically_known_equals(output_numel, total_numel):
            return None
    total_numel_hint = V.graph.sizevars.optimization_hint(total_numel, fallback=0)
    if not total_numel_hint:
        return None

    scheduler_nodes = [
        node
        for node in fused_node.get_nodes()
        if isinstance(node, torch._inductor.scheduler.SchedulerNode)
    ]
    if not scheduler_nodes:
        return None
    if any(
        isinstance(dep, MemoryDep) and dep.mode is not None
        for node in scheduler_nodes
        for dep in node.read_writes.writes
    ):
        return None
    if any(
        node._body.indirect_vars or node._body.subblocks for node in scheduler_nodes
    ):
        return None

    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures

    try:
        memory = SIMDKernelFeatures(
            scheduler_nodes,
            total_numel,
        ).pointwise_memory_summary(
            (outer_range, inner_range),
        )
        external_memory_footprint_bytes = V.graph.sizevars.optimization_hint(
            memory.external_memory_bytes,
            fallback=0,
        )
    except (
        AssertionError,
        KeyError,
        NotImplementedError,
        RuntimeError,
        ValueError,
        torch._inductor.codegen.simd.CantSplit,
    ):
        return None
    if not external_memory_footprint_bytes:
        return None

    # These are external reads that vary with the outer tile but not the inner
    # tile. MemoryEstimator has already applied codegen-style read CSE and
    # removed kernel-local buffers.
    row_input_names = (
        memory.read_names_omitting_dim[1] - memory.read_names_omitting_dim[0]
    )
    if not row_input_names:
        return None

    domain = _BroadcastValueDomain(
        index_vars=OrderedSet((outer_var, inner_var)),
        preserved_vars=OrderedSet((outer_var,)),
        replay_vars=OrderedSet((inner_var,)),
    )
    work = 0
    full_domain_work = 0
    interned_expressions: dict[tuple[Any, ...], int] = {}
    expression_ids: dict[torch.fx.Node, int] = {}
    value_nodes: dict[int, torch.fx.Node] = {}
    value_dependencies: dict[int, OrderedSet[sympy.Symbol]] = {}
    value_inputs: dict[int, OrderedSet[int]] = {}
    value_order: list[int] = []
    direct_load_values = OrderedSet[int]()
    row_lineage = OrderedSet[int]()
    # This map is only for cross-node value lineage. Memory CSE and traffic
    # accounting above come from MemoryEstimator.
    stored_values: dict[str, int] = {}
    root_values = OrderedSet[int]()
    external_input_names = memory.external_read_names

    # Reductions are excluded above, so get_nodes() is the pointwise codegen
    # schedule. Replay its LoopBody SSA in that order. A fused store/load pair
    # aliases one value, matching codegen's store_cache behavior.
    for scheduler_node in scheduler_nodes:
        body = scheduler_node._body
        var_map = normalized.node_var_mappings.get(scheduler_node.get_name())
        if var_map is None:
            return None

        for node in body.root_block.graph.nodes:
            try:
                input_ids = OrderedSet(
                    expression_ids[input_node] for input_node in node.all_input_nodes
                )
            except KeyError:
                return None
            node_dependencies = OrderedSet[sympy.Symbol]()
            for input_id in input_ids:
                node_dependencies.update(value_dependencies.get(input_id, ()))

            normalized_index = _normalized_get_index(
                node,
                body,
                var_map,
                normalized.var_ranges,
            )
            if (
                node.op == "call_module"
                and node.target == "get_index"
                and normalized_index is None
            ):
                return None
            if normalized_index is not None:
                if not is_supported_index(normalized_index):
                    loop_tiling_log.info(
                        "Rejecting broadcast-work candidate: unsupported "
                        "normalized index %s",
                        normalized_index,
                    )
                    return None
                node_dependencies.update(
                    normalized_index.free_symbols & domain.index_vars
                )

            if node.op == "call_method" and node.target == "load":
                load_name = _loop_body_op_arg(node, 1, "name")
                if not isinstance(load_name, str):
                    return None
                if load_name in stored_values:
                    value_id = stored_values[load_name]
                    expression_ids[node] = value_id
                    continue
                if load_name not in external_input_names:
                    return None

            expression_id = _fx_expression_id(
                node,
                expression_ids,
                interned_expressions,
                normalized_index,
            )
            expression_ids[node] = expression_id

            if node.op == "call_method" and node.target == "store":
                store_name = _loop_body_op_arg(node, 1, "name")
                store_index = _normalized_get_index(
                    _loop_body_op_arg(node, 2, "index"),
                    body,
                    var_map,
                    normalized.var_ranges,
                )
                store_value = _loop_body_op_arg(node, 3, "value")
                store_mode = _loop_body_op_arg(node, 4, "mode")
                if (
                    not isinstance(store_name, str)
                    or store_index is None
                    or not isinstance(store_value, torch.fx.Node)
                    or store_mode is not None
                ):
                    return None
                store_value_id = expression_ids.get(store_value)
                if store_value_id is None:
                    return None
                previous_store = stored_values.setdefault(store_name, store_value_id)
                if previous_store != store_value_id:
                    return None
                if store_name not in output_names:
                    continue

                store_buffer = V.graph.try_get_buffer(store_name)
                if not isinstance(store_buffer, (ir.Buffer, ir.TensorBox)):
                    return None
                try:
                    store_layout = store_buffer.get_layout()
                except NotImplementedError:
                    return None
                store_numel = sympy_product(store_layout.size)
                store_offset = store_layout.offset
                store_origin = sympy_subs(
                    store_index,
                    {outer_var: 0, inner_var: 0},
                )
                if not (
                    V.graph.sizevars.statically_known_equals(store_numel, total_numel)
                    and V.graph.sizevars.statically_known_equals(
                        store_offset, store_origin
                    )
                    and _is_dense_row_major_iteration(
                        store_index,
                        outer_var,
                        inner_var,
                        inner_range,
                    )
                ):
                    return None
                root_values.add(store_value_id)
                continue

            if node.op == "output":
                continue

            if expression_id in value_nodes:
                continue

            value_nodes[expression_id] = node
            value_dependencies[expression_id] = node_dependencies
            value_inputs[expression_id] = input_ids
            value_order.append(expression_id)
            if node.op == "call_method" and node.target == "load":
                direct_load_values.add(expression_id)

            if row_lineage & input_ids:
                row_lineage.add(expression_id)
            elif (
                node.op == "call_method"
                and node.target == "load"
                and _loop_body_op_arg(node, 1, "name") in row_input_names
                and domain.is_preserved(node_dependencies)
            ):
                row_lineage.add(expression_id)

    reachable_values = OrderedSet[int]()
    pending = list(root_values)
    while pending:
        value_id = pending.pop()
        if value_id in reachable_values:
            continue
        reachable_values.add(value_id)
        pending.extend(value_inputs.get(value_id, ()))

    full_domain_values = OrderedSet(
        value_id
        for value_id in reachable_values
        if domain.is_replayed(value_dependencies.get(value_id, OrderedSet()))
    )
    peak_live_full_domain_values = _scheduled_peak_live_values(
        value_order,
        direct_load_values,
        value_inputs,
        root_values,
        full_domain_values,
    )
    loop_tiling_log.info(
        "Broadcast-work candidate: peak_live_full_domain_values=%s",
        peak_live_full_domain_values,
    )
    if peak_live_full_domain_values > _MAX_BROADCAST_LIVE_FULL_DOMAIN_VALUES:
        loop_tiling_log.info(
            "Rejecting broadcast-work candidate: peak_live_full_domain_values=%s",
            peak_live_full_domain_values,
        )
        return None

    for value_id in reachable_values:
        node = value_nodes.get(value_id)
        if node is None:
            continue
        node_dependencies = value_dependencies[value_id]
        if node.op == "call_method" and domain.is_replayed(node_dependencies):
            op_cost = _broadcast_op_cost(node)
            if op_cost is None:
                loop_tiling_log.info(
                    "Rejecting broadcast-work candidate: unknown full-domain op %s",
                    node.target,
                )
                return None
            full_domain_work += op_cost
        if value_id in row_lineage and domain.is_preserved(node_dependencies):
            op_cost = _broadcast_op_cost(node)
            if op_cost is None:
                loop_tiling_log.info(
                    "Rejecting broadcast-work candidate: unknown op %s",
                    node.target,
                )
                return None
            work += op_cost

    loop_tiling_log.info(
        "Broadcast-work candidate: split=%s work=%d full_domain_work=%d "
        "replay=%d numel=%d external_footprint_bytes=%d",
        outer_var,
        work,
        full_domain_work,
        replay_factor,
        total_numel_hint,
        external_memory_footprint_bytes,
    )
    if work < _MIN_BROADCAST_WORK:
        return None

    saved_work = work * (replay_factor - 1)
    one_lane_work = work + full_domain_work
    if saved_work < one_lane_work:
        loop_tiling_log.info(
            "Rejecting broadcast-work candidate: saved_work=%d full_domain_work=%d",
            saved_work,
            full_domain_work,
        )
        return None

    # The 1D kernel evaluates row-only expressions as lane vectors. A 2D
    # kernel broadcasts them across its inner tile. Require the savings over
    # eight lanes to cover one lane of the whole expression. The default and
    # first tuned 2D config use a 32-wide inner tile, so this remains a
    # conservative estimate of reuse.
    saved_work_numerator = saved_work * total_numel_hint
    required_work_numerator = (
        _MIN_SAVED_WORK_PER_BYTE * external_memory_footprint_bytes * replay_factor
    )
    if saved_work_numerator < required_work_numerator:
        loop_tiling_log.info(
            "Rejecting broadcast-work candidate: saved_work=%d required_work=%d",
            saved_work_numerator,
            required_work_numerator,
        )
        return None

    # The work/traffic model is an eligibility gate, not a conversion from
    # operations to bytes. Give the eligible split a small preference on the
    # same scale as the existing memory-coalescing scores.
    operation_score_bonus = max(1, external_memory_footprint_bytes // 20)
    loop_tiling_log.info(
        "Accepting broadcast-work candidate: operation_score_bonus=%d",
        operation_score_bonus,
    )
    return BroadcastWork(
        split_var=outer_var,
        operation_score_bonus=operation_score_bonus,
    )


def _analyze_memory_coalescing(
    fused_node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
) -> CoalesceVarAnalysis | None:
    """
    Implementation for BaseSchedulerNode.get_coalesce_analysis().
    Call that node method so loop-transform cache invalidation is honored.

    Find variables that coalesce the reads and writes and score the total size.

    If uncoalesced memory expressions are found, look for additionally tiling of variables
    which will coalesce memory accesses.

    For instance - for the following expression:

    (32*p0) // 2048

    Tiling p0 by 64 will make this expression coalesced.
    """

    norm_read_writes = extract_normalized_read_writes(fused_node)

    if norm_read_writes is None:
        return None

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges
    broadcast_work: BroadcastWork | None = None

    coalesced_by_var: dict[sympy.Symbol, int] = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = Counter()

    # Only check pointwise-only kernels
    index_vars = norm_read_writes.index_vars
    reduce_vars = norm_read_writes.reduce_vars
    innermost_var = (
        next(reversed(index_vars)) if index_vars and not reduce_vars else None
    )

    for is_read, (memory_expr, buf_names) in itertools.chain(
        ((True, item) for item in reads.items()),
        # pyrefly: ignore [bad-argument-type]
        ((False, item) for item in writes.items()),
    ):
        size = get_score(memory_expr, var_ranges, buf_names)
        if size == 0:
            continue

        # accesses with indirect expressions are never coalesced
        indirect_expr = has_indirect_access(memory_expr)

        if indirect_expr:
            maybe_coalesced_var = None
        else:
            maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
            # while broadcasting vars are not technically coalesced,
            # accesses at least stay in cache, so they provide most of the benefit.
            # treat the same for now.
            if maybe_coalesced_var is None:
                maybe_coalesced_var = find_broadcast_var(memory_expr, var_ranges)

        total_score = 0
        for buf_name in buf_names:
            if (buf := V.graph.try_get_buffer(buf_name)) and (
                buf_size := try_get_buf_size(buf_name)
            ):
                # constrain by buf size since we'll read at most that many elements
                # score could be more through either masking or by broadcasting (e.g. x // 16)
                total_score += min(buf_size, size) * buf.dtype.itemsize

        # coalesced writes more important
        total_score *= 1 if is_read else 2

        if maybe_coalesced_var:
            # Check if the coalescing is already achieved in 1D iteration.
            # Skip the innermost variable: it always varies across threads,
            # so its coalescing is always real.
            already_coalesced_1d = False
            if innermost_var is not None and maybe_coalesced_var != innermost_var:
                # Evaluate stride at two points (0->1 and 1->2) to catch
                # non-linear expressions that only look coalesced at the origin.
                subs = dict.fromkeys(var_ranges, 0)
                try:
                    val_0 = sympy_subs(memory_expr, subs)
                    subs[innermost_var] = 1
                    val_1 = sympy_subs(memory_expr, subs)
                    stride_01 = val_1 - val_0
                    if stride_01 in (0, 1):
                        subs[innermost_var] = 2
                        val_2 = sympy_subs(memory_expr, subs)
                        stride_12 = val_2 - val_1
                        if stride_12 in (0, 1):
                            already_coalesced_1d = True
                except (ZeroDivisionError, TypeError):
                    pass

            if not already_coalesced_1d:
                coalesced_by_var[maybe_coalesced_var] += total_score
            else:
                coalesced_by_var[innermost_var] += total_score
        else:
            uncoalesced_addrs[memory_expr] += total_score

    has_varying_uncoalesced_addr = any(
        memory_expr.free_symbols & var_ranges.keys()
        for memory_expr in uncoalesced_addrs
    )
    if not has_varying_uncoalesced_addr:
        broadcast_work = _analyze_broadcast_work(fused_node, norm_read_writes)
    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
            broadcast_work=broadcast_work,
        )

    # map from var -> tiling -> total_score
    tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        if has_indirect_access(uncoalesced_expr):
            continue

        expr_subs = dict.fromkeys(var_ranges.keys(), 0)
        for v in uncoalesced_expr.free_symbols & var_ranges.keys():
            # skip non iter/reduce var variables
            if v not in var_ranges:
                continue
            # skip small addrs
            if addr_score == 0:
                continue

            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0

            if len(single_var_expr.free_symbols) != 1:
                continue

            tiling_factor = solve_for_tiling(single_var_expr)

            if (
                tiling_factor is None
                or not tiling_factor.is_constant()
                or not tiling_factor.is_integer
            ):
                continue

            tiling_factor = int(tiling_factor)
            if not V.graph.sizevars.statically_known_lt(tiling_factor, var_ranges[v]):
                continue

            # TODO - if a var is in the middle, such as [n0, n1, n2]
            # n1 can be split beyond range

            MIN_TILING_BLOCK = 8
            if not all(
                V.graph.sizevars.statically_known_lt(MIN_TILING_BLOCK, block)
                for block in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            tiling_scores[v][tiling_factor] += addr_score

    if len(tiling_scores) == 0:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
            broadcast_work=broadcast_work,
        )

    best_tiling: tuple[sympy.Expr, int] | None = None
    best_tiling_score = 0

    for var, tiling_counter in tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            if tile_score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = tile_score

    if best_tiling is None:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
            broadcast_work=broadcast_work,
        )

    # TODO - for strictly pointwise fusions,
    # we can consider just swizzling the var if the var we are going to tile
    # does not coalesce a significant portion of global reads
    # TODO - could also prefer index var splits to reduction, better tested
    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        uncoalesced_addrs=uncoalesced_addrs,
        norm_read_writes=norm_read_writes,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], best_tiling_score),
        broadcast_work=broadcast_work,
    )


def analyze_memory_coalescing_for_nodes(
    nodes: Sequence["BaseSchedulerNode"],
) -> CoalesceVarAnalysis | None:
    if not nodes:
        return None

    from torch._inductor import scheduler

    node_types = (scheduler.FusedSchedulerNode, scheduler.SchedulerNode)
    if not all(isinstance(node, node_types) for node in nodes):
        return None

    if len(nodes) == 1:
        return nodes[0].get_coalesce_analysis()

    graph_scheduler = getattr(V.graph, "scheduler", None)
    if graph_scheduler is not None:
        fused_node = graph_scheduler.name_to_fused_node.get(nodes[0].get_first_name())
        if fused_node is not None:
            fused_nodes = list(fused_node.get_nodes())
            if len(fused_nodes) == len(nodes) and all(
                fused is node for fused, node in zip(fused_nodes, nodes, strict=True)
            ):
                return fused_node.get_coalesce_analysis()

    return _analyze_memory_coalescing(
        _FusedNodeView(
            nodes=nodes,
            read_writes=ReadWrites.merge_list([node.read_writes for node in nodes]),
            group=max(nodes, key=lambda node: int(node.is_reduction())).group,
        )
    )
