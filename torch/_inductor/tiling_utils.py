import dataclasses
import itertools
from collections import Counter, defaultdict
from operator import is_
from re import A
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
Split = tuple[sympy.Expr]


if TYPE_CHECKING:
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

    def __init__(self, node: Union["FusedSchedulerNode", "SchedulerNode"],):
        from torch._inductor.codegen.simd import SIMDKernel, CantSplit
        self.node = node
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)
        self.red_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        self.pw_split_options[0].add(())
        self.red_split_options[0].add(())

        self.all_node_sizes: OrderedSet[tuple[Split, Split]] = OrderedSet()
        
        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            # todo - check if same size.. take smaller 
            # p0: 128, p1: 384, p2: 196
            # vs
            # p0: 768, p1: 64, p2: 196
            # if not divisible.. flatten dims
            (_, n_pw_splits), (_, n_red_splits) = get_pw_red_splits(n, self.pointwise_numel, self.red_numel)
                        
            # fill in reduction size
            n_pw_splits, n_red_splits = torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(fused_group, (n_pw_splits, n_red_splits), self.red_numel)
            
            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
            self.red_split_options[len(n_red_splits)].add(tuple(n_red_splits))

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        # pw, reduction splits
        self.seen_split_options: OrderedSet[tuple[Split, Split]] = OrderedSet()
        
        # this is a heap of pending splits to try, ordered based on length
        self.pending_splits: list[tuple[int, tuple[Split, Split]]] = []
        self.curr_attempted_split_len = -1

        self.max_pw_splits = max(self.pw_split_options.keys())
        self.max_red_splits = max(self.red_split_options.keys())


    def get_node_splits(self):
        # if len(self.all_node_sizes) == 1:
        #     return next(iter(self.all_node_sizes))

        for diff in range(self.max_pw_splits + self.max_red_splits, 0, -1):
            self.add_pending_splits(diff)   
            
            self.curr_attempted_split_len = diff
            while self.pending_splits:
                _, (pw_split, red_split) = heapq.heappop(self.pending_splits)

                if self.try_split(pw_split, red_split):
                    return pw_split, red_split
        
        # default
        return ((self.pointwise_numel), (self.red_numel))
        assert False
        # TODO: return Default  

    def add_pending_splits(self, split_len: int) -> None:
        """
        Add all combinations of pointwise and reduction splits that together
        have exactly the required number of splits (required_split_len).
        
        Args:
            required_split_len: Total number of splits required
        """
        # For each possible length of pointwise splits
        for pw_len in range(min(self.max_pw_splits, split_len) + 1):
            # Calculate the corresponding length needed for reduction splits
            red_len = split_len - pw_len
            
            # Skip invalid combinations
            if red_len > self.max_red_splits or red_len < 0:
                continue
                
            # Get all valid pointwise and reduction split options with these lengths
            pw_options = self.pw_split_options[pw_len]
            red_options = self.red_split_options[red_len]
            
            # Add all combinations as pending splits
            for pw_split, red_split in itertools.product(pw_options, red_options):
                self.add_pending_split(pw_split, red_split)

    def add_pending_split(self, pw: Split, red: Split) -> None:
        if (pw, red) in self.seen_split_options:
            return

        heapq.heappush(self.pending_splits, (-len(pw) - len(red),(pw, red)))
        self.seen_split_options.add((pw, red))


    def add_induced_split_option(self, split: Split, is_pointwise: bool):
        split_options = self.pw_split_options if is_pointwise else self.red_split_options
        split_options[len(split)].add(split)

        other_split_options = self.red_split_options if is_pointwise else self.pw_split_options
        split_len = len(split)
        curr_split = self.curr_attempted_split_len
        # we only need to iterate up to and including current split,
        # since subsequent split options will be iterated on.
        for i in range(split_len, curr_split - 1, -1):
            other_split = other_split_options[split_len - i]

            pairs = itertools.product(  
                *(([split], other_split) if is_pointwise else (other_split, [split]))
            )
            for pw, red in pairs: 
                self.add_pending_split(pw, red)
        

    def try_split(self, pw: Split, red: Split) -> bool:
        from torch._inductor.codegen.simd import SIMDKernel, CantSplit
        # breakpoint()
        return True
        for n_pw, n_red in self.all_node_sizes:
            try:
                groups = pw + red
                lengths = (n_pw, n_red)
                splits, getters = SIMDKernel._split_iteration_ranges(groups, lengths)
            except CantSplit:
                return False

            assert len(getters) == 2
            split_groups = (
                (splits[:len(pw)], getters[0],True), 
                (splits[len(pw):], getters[1],False),
            )
            breakpoint()

            all_vars = sympy.symbols(f"v_0:{sum(len(s) for s in splits)}")
            ret = True
            for group_split, group_getter, is_pointwise in split_groups:

                # If the number of returned splits is greater than the input, 
                # then we had to induce another variable.
                attempted_split = pw if is_pointwise else red
                num_split = sum(len(s) for s in group_split)
                # breakpoint()
                if (num_split > len(n_pw if is_pointwise else n_red)):

                    # add the flattened split
                    flattened_splits = tuple(itertools.chain.from_iterable(group_split))
                    self.add_induced_split_option(flattened_splits, is_pointwise)

                    var_to_split: dict[sympy.Symbol, int] = {}
                    var_i = 0 if is_pointwise else len(all_vars) - num_split
                    for i, s in enumerate(group_split):
                        for _ in range(len(s)):
                            var_to_split[all_vars[var_i]] = i
                            var_i += 1

                    exprs = [g(all_vars) for g in group_getter]
                    used_iter_vars = [
                        sorted([var_to_split[s] for s in expr.free_symbols]) for expr in exprs
                    ]

                    # assert no dim is combined across three different dims
                    combine_ranges = [var_list for var_list in used_iter_vars if len(var_list) > 1]
                    flattened_combined_idxs = [idx for range_list in combine_ranges for idx in range_list]
                    assert len(flattened_combined_idxs) == len(OrderedSet(flattened_combined_idxs))

                    split = list(attempted_split)
                    for idx_a, idx_b in combine_ranges:
                        combined_split = sympy_product(split[idx_a:idx_b + 1])
                        split[idx_a] = combined_split
                        split[idx_a + 1 : idx_b + 1] = [None for _ in range(idx_b - idx_a)]

                    split = tuple([a for a in split if a is not None])
                    self.add_induced_split_option(split, is_pointwise)

                    return False
                
        return True

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

def apply_unified_var_mapping(iter_vars, red_vars, norm_pw_vars, norm_red_vars, new_ranges, return_getters_groups):
    """
    Apply variable mapping for both pointwise and reduction variables using a unified approach.
    
    Args:
        iter_vars: Iteration variables for pointwise operations
        red_vars: Reduction variables
        norm_pw_vars: Normalized pointwise variables
        norm_red_vars: Normalized reduction variables
        new_ranges: Ranges for all variables (pointwise + reduction)
        return_getters_groups: List of lists of getter functions [pw_getters, red_getters]
        
    Returns:
        Combined variable mapping
    """
    # Create a flat list of all variables
    all_old_vars = list(iter_vars) + list(red_vars)
    all_new_vars = list(norm_pw_vars) + list(norm_red_vars)
    
    # Create symbolic variables for all splits
    num_vars = sum(len(s) for s in new_ranges)
    split_vars = sympy.symbols(f"v_0:{num_vars}")
    
    # Create mapping from split variables to original variable expressions
    var_idx = 0
    new_var_map = {}
    
    for group, old_var in zip(new_ranges, all_old_vars):
        if len(group) == 2:
            # Handle case where a variable is split into two dimensions
            new_var1 = split_vars[var_idx]
            new_var2 = split_vars[var_idx + 1]
            var_idx += 2
            
            new_var_map[new_var1] = (old_var * group[1])
            new_var_map[new_var2] = old_var
        else:
            # Handle case where variable is not split
            new_var = split_vars[var_idx]
            var_idx += 1
            new_var_map[new_var] = old_var
    
    # Create a unified list of getters with their corresponding output variables
    unified_getters = []
    for group_idx, (getters, output_vars) in enumerate(zip(return_getters_groups, [norm_pw_vars, norm_red_vars])):
        for getter_idx, getter in enumerate(getters):
            unified_getters.append((getter, output_vars[getter_idx], group_idx))
    
    # Process all getters in a single loop
    var_map = defaultdict(int)
    
    for getter, output_var, group_idx in unified_getters:
        # Apply substitution to get expression in terms of original variables
        expr = sympy_subs(getter(split_vars), new_var_map)
        
        # Map split variables to their contribution in the final expression
        for v in expr.free_symbols:
            # Create temporary replacement map to isolate this variable's contribution
            repl_map = {sym: 0 for sym in expr.free_symbols}
            repl_map[v] = output_var
            
            # Add this contribution to the variable mapping
            term = sympy_subs(expr, repl_map)
            var_map[v] += term
    
    return var_map


def _extract_fused_node_meta(
    node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> FusedNormalizedReadsWrites:

    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()

    outputs = node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in node.read_writes.reads)

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )
    node = node
    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]
    breakpoint()
    
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
       
        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(groups, lengths, red_numel)
        new_ranges, return_getters_groups = torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(groups, lengths)
        
        # Apply unified mapping
        var_map = apply_unified_var_mapping(
            iter_vars, red_vars, 
            norm_pw_vars, norm_red_vars, 
            new_ranges, return_getters_groups
        )
        var_map2 = apply_var_mapping(iter_vars, norm_pw_vars, new_ranges[:len(n_pw_splits)], return_getters_groups[0])
        
        for k, v in var_map2.items():
            assert var_map[k] == v
        # breakpoint()
        # var_map.update(apply_var_mapping(red_vars, norm_red_vars, new_ranges[len(n_pw_splits):], return_getters_groups[1]))

        breakpoint()
        n_reads_new = [sympy_subs(read, var_map) for read in n_reads]
        n_writes_new = [sympy_subs(read, var_map) for read in n_writes]

        reads |= n_reads_new
        writes |= n_writes_new

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

    # breakpoint()
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
            # if repr(single_var_expr) == "64*n1":
            #     breakpoint()
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
