import dataclasses
import itertools
from collections import Counter, defaultdict
from typing import Optional, TYPE_CHECKING, Union, Sequence

import sympy

import torch
import heapq

from torch._inductor import config
from torch._inductor.dependencies import index_vars_no_squeeze
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .virtualized import V


if TYPE_CHECKING:
    Split = tuple[sympy.expr]
    from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode


def solve_for_zero(expr: sympy.Expr) -> Optional[tuple[sympy.Rel, sympy.Expr]]:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
    if expr.is_constant() and not expr == 0:
        return None
    elif isinstance(expr, FloorDiv):
        return None

    assert len(expr.free_symbols) <= 1
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out


def solve_for_tiling(expr: sympy.Expr) -> Optional[sympy.Expr]:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.
    """
    if len(expr.free_symbols) == 0:
        return None

    assert len(expr.free_symbols) == 1
    free_symbol = next(iter(expr.free_symbols))

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        expr_plus_one = sympy_subs(expr, {free_symbol: free_symbol + 1})

        diff = expr_plus_one - expr
        if diff.is_constant() and diff >= 0:
            # breakpoint()
            return diff

        out = try_solve(sympy.Eq(expr_plus_one - expr, 1), free_symbol)

        if not out or not out[1].is_constant():
            return None
        return out[1]

    required_values = []
    eq_1_expressions = []

    # very piecemeal solution if ModularIndexing or FloorDiv involved.
    # Expand as needed.
    for arg in sympy.Add.make_args(expr):
        # Try to make mul terms 0
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO - only need one of these to be solvable to zero
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None or not out[1].is_constant():
                    continue

                seen = True
                required_values.append(out[1])

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
        z: Optional[sympy.Expr] = None,
    ) -> sympy.Expr:
        return x / y

    is_non_differentiable = eq_1_expr.has(ModularIndexing) or eq_1_expr.has(
        ModularIndexing
    )
    # For the purposes of tiling/coalesced access, we can treat ModularIndexing and FloorDiv equivalently
    eq_1_expr = eq_1_expr.replace(ModularIndexing, indexing_div_rep).replace(
        FloorDiv, indexing_div_rep
    )
    out = try_solve(sympy.Eq(eq_1_expr, 1), free_symbol)
    if out is None or not out[1].is_constant():
        return None

    # since we approximated FloorDiv/ModularIndexing, double check here
    if (
        is_non_differentiable
        and not (sympy_subs(eq_1_expr, {free_symbol: out[1]})) == 1
    ):
        return None

    required_values.append(out[1])

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> Optional[sympy.Expr]:
    """
    Try to find the symbol which coalesces this index
    """
    # TODO - not sure what to do with indirect variable
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
    for v in var_ranges.keys():
        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            continue
        if new_val - zero_index == 1:
            return v
        variables[v] = 0

    return None


@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """
    Normalized reads and writes for nodes in the same FusedSchedulerNode.
    """

    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: OrderedSet[sympy.Expr]
    writes: OrderedSet[sympy.Expr]
    var_ranges: dict[sympy.Symbol, int]



def get_pw_red_splits(
    n: "SchedulerNode", pointwise_numel: sympy.Expr, red_numel: sympy.Expr
) -> tuple[
    tuple[list[sympy.Symbol], list[int]], tuple[list[sympy.Symbol], list[int]]
]:
    if n.is_reduction() or sympy_product(n._body.sizes[0]) == pointwise_numel:
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]

    assert sympy_product(n._body.sizes[0]) == pointwise_numel * red_numel  # type: ignore[operator]
    i = len(n._body.sizes[0]) - 1
    prod = 1
    while i >= 0:
        prod *= n._body.sizes[0][i]
        if prod == red_numel:
            break

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)  # type: ignore[return-value]

    # TODO - handle
    raise RuntimeError(
        f"Unhandled node: size: {n._body.sizes}, pw: {pointwise_numel}, red: {red_numel}"
    )


class NodeSplitGetter():

    def __init__(self, n: Union["FusedSchedulerNode", "SchedulerNode"],):
        self.node = n
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)
        self.red_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            # todo - check if same size.. take smaller 
            # p0: 128, p1: 384, p2: 196
            # vs
            # p0: 768, p1: 64, p2: 196
            # if not divisible.. flatten dims
            (_, n_pw_splits), (_, n_red_splits) = get_pw_red_splits(n, self.pointwise_numel, self.red_numel)
            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
            self.red_split_options[len(n_pw_splits)].add(tuple(n_red_splits))
        
        # pw, reduction splits
        self.seen_split_options: OrderedSet[tuple[Split, Split]] = OrderedSet()
        
        # this is a heap of pending splits to try, ordered based on length
        self.pending_splits: list[tuple[int, tuple[Split, Split]]] = []

        self.max_pw_splits = max(self.pw_split_options.keys())
        self.max_red_splits = max(self.red_split_options.keys())


    def get_node_splits(self):
        for diff in range(self.max_pw_splits + self.max_red_splits, 0, -1):
            self.add_pending_splits(diff)   

            while self.pending_splits:
                _, top_split = self.pending_splits[0]
                if self.try_split(top_split):
                    return top_split

                heapq.heappop(self.pending_splits)

        breakpoint()
        pass
        # TODO: return Default  

    def add_pending_splits(self, split_len: int) -> None:

        diff = split_len - (self.max_pw_splits + self.max_red_splits)
        for pw_diff in range(0, diff + 1):
            red_diff = (diff - pw_diff)

            pw_options = self.pw_split_options[self.max_pw_splits - pw_diff]
            red_options = self.red_split_options[self.max_red_splits - red_diff]

            pairs = itertools.product(  
                pw_options, red_options
            )
            for pw, red in pairs:  
                self.add_pending_split(pw, red)

    def add_pending_split(self, pw: Split, red: Split) -> None:
        if (pw, red) in self.seen_split_options:
            return

        heapq.heappush(self.pending_splits, (-len(pw) - len(red),(pw, red)))
        self.seen_split_options.add((pw, red))


    def try_split(self, pw_split: Split, red_split: Split):
        from torch._inductor.codegen.simd import SIMDKernel, CantSplit

        for n in self.node.get_nodes():
            try:
                (_, n_pw), (_, n_red) = get_pw_red_splits(n, self.pointwise_numel, self.red_numel)
                splits, getters = SIMDKernel._split_iteration_ranges(split, (n_pw, n_red))
            except CantSplit:
                return None

            split_groups = (
            (

                splits[:len(pw)],
                getters[0],
                True,
            ), (
                splits[len(pw):],
                getters[1], 
                False
            ))

            for group_split, group_getter, is_pointwise in split_groups:

                # If the number of returned splits is greater than the input, 
                # then we had to induce another variable.
                attempted_split = pw if is_pointwise else red
                num_split = sum(len(s) for s in group_split)
                if num_split > len(n_pw if is_pointwise else n_red):
                    out_vars = sympy.symbols(f"v_0:{num_split}")

                    var_to_split: dict[sympy.Symbol, int] = {}
                    var_i = 0
                    for i, s in enumerate(splits):
                        for _ in range(len(s)):
                            var_to_split[out_vars[var_i]] = i
                            var_i += 1
                    breakpoint()
                    
                    for i, getter in enumerate(group_getter):
                        expr_per_group = getter(out_vars)
                        for v in expr_per_group.free_symbols:
                            if var_to_split[v] != i:

                                dim_0, dim_1 = sorted([i, var_to_split[v]])
                                assert dim_0 == dim_1 - 1

                                # breakpoint()
                                out = (
                                    list(attempted_split[:dim_0]) +
                                    [sympy_product(attempted_split[dim_0:dim_0 + 2])] +
                                    list(attempted_split[dim_0 + 2:])
                                )
                                if is_pointwise:
                                    pw_split_options[len(out)].add(tuple(out))

                                return None

            
                    return None

        return pw, red



def get_node_splits(node):

    pointwise_numel = node.group[1][0]
    red_numel = node.group[1][1]

    # If there are fused nodes with one node having:
    # sizes = ([2048], [])
    # ranges: {p0: 2048}
    # and another node with
    # sizes: ([32, 64], [])
    # ranges: {p0: 32, p1: 64}
    # The p0 in the first node actually corresponds to
    # 64 * p0 + p1
    # So we find the node with the most number of splits, and
    # normalize the other nodes to use the same iter vars.

    pw_split_options: dict[int, OrderedSet[tuple[int]]] = defaultdict(OrderedSet)
    red_split_options: dict[int, OrderedSet[tuple[int]]] = defaultdict(OrderedSet)

    for n in reversed(node.get_nodes()):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        # todo - check if same size.. take smaller 
        # p0: 128, p1: 384, p2: 196
        # vs
        # p0: 768, p1: 64, p2: 196
        # if not divisible.. flatten dims
        (_, n_pw_splits), (_, n_red_splits) = get_pw_red_splits(n, pointwise_numel, red_numel)
        pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
        red_split_options[len(n_pw_splits)].add(tuple(n_red_splits))

    max_pw_splits = max(pw_split_options.keys())
    max_red_splits = max(red_split_options.keys())

    pw_splits: Optional[tuple[int]] = None
    red_splits: Optional[tuple[int]] = None



    seen_splits = OrderedSet()







def _extract_fused_node_meta(
    node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> FusedNormalizedReadsWrites:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()

    outputs = node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in node.read_writes.reads)

    pw_splits, red_splits = get_splits()
    # pw_splits = [128, 6, 64, 196]

    # def map_existing_vars_to_new_vars()

    # breakpoint()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )

    def apply_var_mapping(old_vars, new_vars, new_ranges, return_getters_groups):

        var_map = {}    

        num_vars = sum(len(s) for s in new_ranges)

        new_var_map = {}


        split_vars = sympy.symbols(f"v_0:{num_vars}")
        var_count = len(split_vars) - 1

        # ([p0, p1, p2], [[128, 6], [64], [196]])
        
        curr_count = 0
        new_var_map = {}
        for group, old_var in zip(new_ranges, old_vars):
            
            divis = None
            assert len(group) <= 2
            if len(group) == 2:
                new_var1 = split_vars[curr_count]
                new_var2 = split_vars[curr_count + 1]
                curr_count += 2
                # TODO _ think about
                new_var_map[new_var1] = (old_var * group[1])
                new_var_map[new_var2] = (old_var)
            else:
                new_var = split_vars[curr_count]
                curr_count += 1
                new_var_map[new_var] = old_var 

        out_exprs = [sympy_subs(g(split_vars), new_var_map) for g in return_getters_groups]

        var_map = {}

        var_map = defaultdict(list)

        for expr, new_var in zip(out_exprs, new_vars):
            repl_map = dict.fromkeys(expr.free_symbols, 0)
            for v in expr.free_symbols:
                repl_map[v] = new_var
                var_map[v].append(sympy_subs(expr, repl_map))
                repl_map[v] = 0

        var_map = {k: sum(v) for k, v in var_map.items()}
        return var_map

    for n in node.get_nodes():
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body
        n_reads: OrderedSet[sympy.Expr] = OrderedSet()
        n_writes: OrderedSet[sympy.Expr] = OrderedSet()
        for inp in inputs:
            n_reads |= body.get_all_read_expr(inp)
        for out in outputs:
            n_writes |= body.get_all_write_expr(out)

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(n, pointwise_numel, red_numel)
        # node_group = n._sizes[0] + n._sizes[1]

        new_ranges, return_getters_groups = torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(list(n_pw_splits) + list(n_red_splits), [pw_splits, red_splits])
        
        var_map = apply_var_mapping(iter_vars, norm_pw_vars, new_ranges, return_getters_groups[0])
        var_map.update(apply_var_mapping(red_vars, norm_red_vars, new_ranges[len(n_pw_splits):], return_getters_groups[1]))
        
        # iter_replacements = [g(norm_pw_vars) for g in return_getters_groups[0]]
        # red_replacements = [g(norm_pw_vars) for g in return_getters_groups[1]]
        
        # var_map = get_var_mapping(norm_pw_vars, iter_replacements, iter_vars)
        # var_map.update(get_var_mapping(norm_red_vars, red_replacements, red_vars))

        # breakpoint()
        # tmp_var_map = 

        n_reads_new = [sympy_subs(read, var_map) for read in n_reads]
        n_writes_new = [sympy_subs(read, var_map) for read in n_writes]

        reads |= n_reads_new
        writes |= n_writes_new

    breakpoint()
    return FusedNormalizedReadsWrites(
        norm_pw_vars,  # type: ignore[arg-type]
        norm_red_vars,  # type: ignore[arg-type]
        reads,
        writes,
        ranges,
    )


def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int:
    """
    Score addr according to its approximate size
    """

    # TODO - deduplicate with candidate_tilings
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v, None)
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V

    return V.graph.sizevars.atomically_apply_size_hint(
        sympy_product(var_sizes), fallback=config.unbacked_symint_fallback
    )


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """

    var: sympy.Symbol
    tiling_factor: int
    score: int


def get_hint(v: Union[sympy.Expr, int]) -> int:
    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.size_hint(v, fallback=config.unbacked_symint_fallback)


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]

    norm_read_writes: FusedNormalizedReadsWrites

    # Expression, split, score
    suggested_split: Optional[VarTiling] = None


def analyze_memory_coalescing(
    fused_node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> Optional[CoalesceVarAnalysis]:
    """
    Find variables that coalesce the reads and writes and score the total size.

    If uncoalesced memory expressions are found, look for additionally tiling of variables
    which will coalesce memory accesses.

    For instance - for the following expression:

    (32*p0) // 2048

    Tiling p0 by 64 will make this expression coalesced.
    """

    norm_read_writes = _extract_fused_node_meta(fused_node)

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges

    coalesced_by_var: dict[sympy.Symbol, int] = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = {}

    for memory_expr in itertools.chain(reads, writes):
        size = get_score(memory_expr, var_ranges)
        maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
        if maybe_coalesced_var:
            coalesced_by_var[maybe_coalesced_var] += size
        else:
            uncoalesced_addrs[memory_expr] = size

    breakpoint()
    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    # map from var -> tiling -> total_score
    potential_tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        expr_subs = dict.fromkeys(uncoalesced_expr.free_symbols, 0)
        for v in uncoalesced_expr.free_symbols:
            # skip non iter/reduce var variables
            if v not in var_ranges:
                continue
            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0
            if repr(single_var_expr) == "64*n1":
                breakpoint()
            tiling_factor = solve_for_tiling(single_var_expr)
            # breakpoint()
            if (
                tiling_factor is None
                or not tiling_factor.is_constant()
                or not tiling_factor.is_integer
            ):
                continue

            tiling_factor = int(tiling_factor)
            MIN_TILING_BLOCK = 4
            if any(
                (get_hint(b) < MIN_TILING_BLOCK)
                for b in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            potential_tiling_scores[v][tiling_factor] += addr_score

    best_tiling: Optional[tuple[sympy.Expr, int]] = None
    best_tiling_score = 0

    for var, tiling_counter in potential_tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            score = tile_score - coalesced_by_var[var]
            if score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = score

    if best_tiling is None:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    # TODO - for strictly pointwise fusions,
    # we can consider just swizzling the var if the var we are going to tile
    # does not coalesce a significant portion of global reads
    # TODO - could also prefer index var splits to reduction, better tested
    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        norm_read_writes=norm_read_writes,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], best_tiling_score),
    )
