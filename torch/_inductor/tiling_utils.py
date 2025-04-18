import dataclasses
import torch
import sympy
from collections import Counter, defaultdict
from typing import Optional, Union, Sequence, Any
from torch._inductor.utils import sympy_subs, sympy_product
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.solve import try_solve
from torch.utils._ordered_set import OrderedSet
from torch._inductor.ir import _IntLike
from torch._inductor.dependencies import MemoryDep
import itertools
from torch.utils._sympy.symbol import SymT, symbol_is_type



def replace_floor_div(expr):
    def indexing_div_rep(x, y):
        return x / y

    return expr.replace(
        FloorDiv, indexing_div_rep
    )

def replace_modular_indexing(expr):
    def mod_indexing_rep(x, y, z):
        return (x / y) % z

    return expr.replace(ModularIndexing, mod_indexing_rep)

def replace_symbols_for_solving(expr):
    return replace_modular_indexing(replace_floor_div(expr))

def solve_for_zero(expr) -> Optional[tuple[sympy.Rel, sympy.Expr]]:
    if expr.is_constant() and not expr == 0:
        return None
    elif isinstance(expr, ModularIndexing):
        return try_solve(sympy.Eq(expr.args[0] / expr.args[1], expr.args[2]), next(iter(expr.free_symbols)))
    elif isinstance(expr, FloorDiv):
        return None
    else:
        
        return try_solve(sympy.Eq(expr, 0), next(iter(expr.free_symbols)))

def solve_for_tiling(expr):

    potential_subsitutions = []
    required_substitutions = []
    relations = []
    possible_values = []

    exprs = expr.args if isinstance(expr, sympy.Add) else [expr]
    for arg in exprs:
        # try to make mul term 0 as it is likely not dense
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO: only need one of these to be true
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None:
                    continue
                
                seen = True
                relations.append(out[0])
                possible_values.append(out[1])

            if not seen:
                return None

            continue
        else:
            free_symbol = next(iter(arg.free_symbols))
            out = try_solve(sympy.Eq(replace_symbols_for_solving(arg), 1), free_symbol)
            if out is None:
                return None

            relations.append(out[0])
            possible_values.append(out[1])

    if len(set(possible_values)) == 1:
        return possible_values[0]
    
    return None



def is_coalesced(index: sympy.Expr, var_ranges: dict[sympy.Expr, int]) -> Optional[sympy.Expr]:

    top_level_terms = sympy.Add.make_args(index)
    # todo ignore indirect
    for v in var_ranges:
        if v in top_level_terms:
            return v

    # this is approximate. we could also take derivates and will later but 
    # that seems slower. i wonder what the above check would catch but this wouldnt..
    variables = {v: 0 for v in index.free_symbols}
    base_value = sympy_subs(index, variables)
    for v in var_ranges.keys():
        variables[v] = 1
        new_val = sympy_subs(index, variables)
        if new_val - base_value == 1:
            return v
        variables[v] = 0

    return None


def get_vars_and_var_ranges_v2(fused_node):
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()
    all_index_vars = OrderedSet()
    all_reduce_vars = OrderedSet()
    var_ranges = {}

    outputs = fused_node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in fused_node.read_writes.reads)

    for node in fused_node.get_nodes():
        body = node._body
        all_index_vars |= body.iter_vars
        all_reduce_vars |= body.reduce_vars
        var_ranges.update(body.var_ranges)

        for inp in inputs:
            reads |= (body.get_all_read_expr(inp))
        for out in outputs:
            writes |= (body.get_all_write_expr(out))

    return all_index_vars, all_reduce_vars, reads, writes, var_ranges
    





def get_vars_and_var_ranges(fused_node) -> tuple[OrderedSet[sympy.Symbol], OrderedSet[sympy.Symbol], OrderedSet[sympy.Expr], OrderedSet[sympy.Expr], dict[sympy.Symbol, _IntLike]]:
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()
    all_index_vars = OrderedSet()
    all_reduce_vars = OrderedSet()
    var_ranges = {}

    outputs = fused_node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in fused_node.read_writes.reads)

    # breakpoint()
    for node in fused_node.get_nodes():
        (index_size, reduce_size), body, (index_vars, reduce_vars) = node.node.get_default_sizes_body()
        all_index_vars |= OrderedSet(index_vars)
        all_reduce_vars |= OrderedSet(reduce_vars)

        assert len(index_size) == len(index_vars)
        assert len(reduce_size) == len(reduce_vars)

        for s, i in zip(itertools.chain(index_size, reduce_size), itertools.chain(index_vars, reduce_vars)):
            if ex_s := var_ranges.get(i):
                assert ex_s == s
            else:
                var_ranges[i] = s

        for inp in inputs:
            reads |= (body.get_all_read_expr(inp))
        for out in outputs:
            writes |= (body.get_all_write_expr(out))

    return all_index_vars, all_reduce_vars, reads, writes, var_ranges

def get_score(addr, var_ranges):
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v, None)
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V
    return V.graph.sizevars.size_hint(
        sympy_product(
            var_sizes
        )
    )



def tile_variables(fused_node) -> Optional[Union[sympy.Expr, int]]:
    bodies = [n._body for n in fused_node.get_nodes()]
    all_index_vars, all_reduce_vars, reads, writes, var_ranges = get_vars_and_var_ranges_v2(fused_node)
    return tile_variables_impl(fused_node, all_index_vars, all_reduce_vars, reads, writes, var_ranges)
    
@dataclasses.dataclass(frozen=True)
class TilingInfo:
    coalesced_by_var: dict[sympy.Expr, int]
    split_var: Optional[tuple[sympy.Expr, int]] = None


def tile_variables_impl(fused_node, all_index_vars: OrderedSet[sympy.Symbol], all_reduce_vars: OrderedSet[sympy.Symbol], reads: OrderedSet[sympy.Expr], writes: OrderedSet[sympy.Expr], var_ranges: dict[sympy.Symbol, _IntLike]):

    coalesced_by_var = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = {}

    # if all_index_vars & all_reduce_vars:

    total_size = 0
    for memory_expr in itertools.chain(reads, writes):
        # TODO - deduplicate with candidate_tilings
        size = get_score(memory_expr, var_ranges)
        total_size += size
        maybe_coalesced_var = is_coalesced(memory_expr, var_ranges)
        if maybe_coalesced_var:
            coalesced_by_var[maybe_coalesced_var] += size
        else:
            uncoalesced_addrs[memory_expr] = size

    if not uncoalesced_addrs:
        return TilingInfo(coalesced_by_var=coalesced_by_var)

    # If the last var is the coalescing var, it's okay to collaprse ranges
    # otherwise it's not !... todo not sure if thats right ??
    # Or is it, 
    # Or maybe just if there
    # map from var -> tiling -> total_score
    potential_tiling_scores = defaultdict(Counter)

    # breakpoint()
    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        expr_subs = {v: 0 for v in uncoalesced_expr.free_symbols}
        for v in uncoalesced_expr.free_symbols:
            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0
            tiling_factor = solve_for_tiling(single_var_expr)
            if tiling_factor is None or not tiling_factor.is_constant() or (tiling_factor >= (var_ranges[v] // 8)):
                continue
            potential_tiling_scores[v][tiling_factor] += addr_score


    best_tiling = None
    best_tiling_score = 0

    for var, tiling_counter in potential_tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            score = tile_score - coalesced_by_var[var] 
            if score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = score

    if not best_tiling:
        return TilingInfo(coalesced_by_var=coalesced_by_var)

    should_tile_var = best_tiling[0] in coalesced_by_var
    if not should_tile_var:
        return TilingInfo(coalesced_by_var=coalesced_by_var)
        

    size_ratio = best_tiling_score // (coalesced_by_var[best_tiling[0]])
    # TODO - tune / remove when 3d tiling is better supported
    # should_swizzle_var = size_ratio >= 16
    # breakpoint()
    # breakpoint()
    # if should_swizzle_var and False:
    #     bodies = [n._body for n in fused_node.get_nodes()]
    #     print("BEFORE")
    #     print([n._body for n in fused_node.get_nodes()])
    #     swizzle_nodes(fused_node, best_tiling[0], best_tiling[1])
    #     print("AFTER")
    #     print([n._body for n in fused_node.get_nodes()])
    return TilingInfo(coalesced_by_var=coalesced_by_var, split_var=(best_tiling[0], best_tiling[1], best_tiling_score))
    
    # breakpoint()
    return best_tiling[0], best_tiling[1], should_swizzle_var

def swizzle_nodes(fused_node, tiling_var: sympy.Expr, tiling_factor: int):
    for n in fused_node.get_nodes():
        out = apply_offset_transformation(n._body, tiling_var, tiling_factor)
        n._body = out
        n.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

# def apply_offset_transformation(body, var, offset_size):
#     """
#     Apply offset transformation where the targeted variable is expressed
#     as a composition of ModularIndexing and FloorDiv without changing
#     the number of iteration variables.
    
#     Args:
#         body: The original LoopBody
#         var: The variable to transform
#         offset_size: The offset factor
    
#     Returns:
#         New LoopBody with the transformation applied
#     """
#     # Get the current sizes and variables
#     iter_vars, reduce_vars = body.vars
#     iter_size, reduce_size = body.sizes

#     from . import dependencies
#     from . loop_body import LoopBody
    
#     # Find position of var in iter_vars
#     try:
#         var_idx = iter_vars.index(var)
#     except ValueError:
#         # If var not in iter_vars, return unchanged
#         return body
    
#     # Create new variables with the same structure
#     (new_iter_vars, new_reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
#         iter_size,
#         reduce_size,
#         prefix="p",
#     )
    
#     # Create substitution dictionary
#     substitution = {}
#     for i, old_var in enumerate(iter_vars):
#         if i == var_idx:
#             # Express the targeted variable as ModularIndexing + FloorDiv
#             target_var = new_iter_vars[i]
#             substitution[old_var] = ModularIndexing(target_var, 1, offset_size) + FloorDiv(target_var, offset_size) * offset_size
#         else:
#             substitution[old_var] = new_iter_vars[i]
    
#     # For reduction variables, keep the mapping straightforward
#     for i, old_var in enumerate(reduce_vars):
#         substitution[old_var] = new_reduce_vars[i]
    
#     # Create the reindexed variables for LoopBody constructor
#     reindexed_vars = []
#     for vars_list in [iter_vars, reduce_vars]:
#         reindexed_vars.append([substitution.get(v, v) for v in vars_list])
    
#     # Create a new LoopBody with the transformation
#     new_body = LoopBody(
#         body,
#         reindexed_vars,
#         var_ranges,
#         new_iter_vars,
#         new_reduce_vars,
#     )
    
#     return new_body

# def apply_offset_transformation(body, var, offset_size):
#     """
#     Apply offset transformation where the targeted variable is expressed
#     as a composition of ModularIndexing and FloorDiv without changing
#     the number of iteration variables.
    
#     Args:
#         body: The original LoopBody
#         var: The variable to transform
#         offset_size: The offset factor
    
#     Returns:
#         New LoopBody with the transformation applied
#     """
#     from . import dependencies
#     from . loop_body import LoopBody
    

#     # Get the current sizes and variables
#     old_body = body
#     iter_vars, reduce_vars = body.vars
#     iter_size, reduce_size = body.sizes
    
#     # Find position of var in iter_vars
#     try:
#         var_idx = iter_vars.index(var)
#     except ValueError:
#         # If var not in iter_vars, return unchanged
#         return body
    
#     # Create new variables with the same structure - first round with 't' prefix
#     (temp_iter_vars, temp_reduce_vars), temp_var_ranges = dependencies.index_vars_no_squeeze(
#         iter_size,
#         reduce_size,
#         prefix="t",  # Use 't' prefix for first round
#     )
    
#     # Define transformation function for the body
#     def transformed_body(*indices):
#         index = [*itertools.chain.from_iterable(indices)]
#         assert len(index) == len(iter_size) + len(reduce_size)
        
#         iter_idx = index[: len(iter_size)]
#         reduce_idx = index[len(iter_size):]
        
#         # Create modified indices with the transformation
#         modified_iter_idx = list(iter_idx)
#         # For the targeted variable, replace with the transformed expression
#         target_var = modified_iter_idx[var_idx]
#         modified_iter_idx[var_idx] = ModularIndexing(target_var, 1, offset_size) + offset_size * FloorDiv(target_var, offset_size)
        
#         # Call original body with modified indices
#         return old_body(modified_iter_idx, reduce_idx)
    
#     # Create intermediate LoopBody
#     intermediate_body = LoopBody(
#         transformed_body,
#         (temp_iter_vars, temp_reduce_vars),
#         temp_var_ranges,
#         temp_iter_vars,
#         temp_reduce_vars
#     )
    
#     # Second round of variable creation with 'p' prefix
#     (final_iter_vars, final_reduce_vars), final_var_ranges = dependencies.index_vars_no_squeeze(
#         iter_size,
#         reduce_size,
#         prefix="p",  # Use 'p' prefix for final variables
#     )
    
#     # Create final LoopBody
#     final_body = LoopBody(
#         intermediate_body,
#         (final_iter_vars, final_reduce_vars),
#         final_var_ranges,
#         final_iter_vars,
#         final_reduce_vars
#     )
#     return final_body


# def apply_offset_transformation(body, var, offset_size):
#     """
#     Apply offset transformation where the targeted variable is expressed
#     as a composition of ModularIndexing and FloorDiv to improve memory access patterns.
    
#     Args:
#         body: The original LoopBody
#         var: The variable to transform
#         offset_size: The offset factor
    
#     Returns:
#         New LoopBody with the transformation applied
#     """
#     # Get the current sizes and variables
#     old_body = body
#     iter_vars, reduce_vars = body.vars
#     iter_size, reduce_size = body.sizes

#     from . import dependencies
#     from . loop_body import LoopBody

#     # Find position of var in iter_vars
#     try:
#         var_idx = iter_vars.index(var)
#     except ValueError:
#         breakpoint()
#         # If var not in iter_vars, return unchanged
#         return body
    
#     # Create temporary variables with 't' prefix
#     (temp_iter_vars, temp_reduce_vars), temp_var_ranges = dependencies.index_vars_no_squeeze(
#         iter_size,
#         reduce_size,
#         prefix="t",
#     )
    
#     # Define the transformation function
#     def transformed_body(*indices: Sequence[sympy.Expr]) -> Any:
#         index = [*itertools.chain.from_iterable(indices)]
#         iter_idx = index[: len(iter_size)]
#         reduce_idx = index[len(iter_size):]
        
#         # Apply transformation to the targeted variable
#         modified_iter_idx = list(iter_idx)
#         target_var = modified_iter_idx[var_idx]
        
#         # Create the specific expression pattern:
#         # (offset_size * ModularIndexing(var, 1, max_block_size)) + FloorDiv(var, max_block_size)
#         max_block_size = 2048  # You might want to make this a parameter
#         modified_iter_idx[var_idx] = (offset_size * ModularIndexing(target_var, 1, max_block_size)) + FloorDiv(target_var, max_block_size)
        
#         return old_body(modified_iter_idx, reduce_idx)
    
#     # Create intermediate LoopBody
#     intermediate_body = LoopBody(
#         transformed_body,
#         (temp_iter_vars, temp_reduce_vars),
#         temp_var_ranges,
#         temp_iter_vars,
#         temp_reduce_vars
#     )
    
#     # Create final variables with 'p' prefix
#     (final_iter_vars, final_reduce_vars), final_var_ranges = dependencies.index_vars_no_squeeze(
#         iter_size,
#         reduce_size,
#         prefix="p",
#     )
    
#     # Create final LoopBody
#     final_body = LoopBody(
#         intermediate_body,
#         (final_iter_vars, final_reduce_vars),
#         final_var_ranges,
#         final_iter_vars,
#         final_reduce_vars
#     )
    
#     return final_body
def apply_offset_transformation(body, var, offset_size, max_block_size=2048):
    """
    Apply offset transformation where the targeted variable is expressed
    as a composition of ModularIndexing and FloorDiv to improve memory access patterns.
    
    Ensures both loads and stores are transformed consistently.
    
    Args:
        body: The original LoopBody
        var: The variable to transform
        offset_size: The offset factor
        max_block_size: The block size for ModularIndexing
    
    Returns:
        New LoopBody with the transformation applied
    """
    from . import dependencies
    from . loop_body import LoopBody

    # Get the current sizes and variables
    old_body = body
    iter_vars, reduce_vars = body.vars
    iter_size, reduce_size = body.sizes
    
    # Check if var is in iter_vars or reduce_vars
    var_in_iter = var in iter_vars
    var_in_reduce = var in reduce_vars
    
    if not var_in_iter and not var_in_reduce:
        # Variable not found in this body
        return body
    
    # Find position of var
    if var_in_iter:
        var_idx = iter_vars.index(var)
        var_is_iter = True
    else:
        var_idx = reduce_vars.index(var)
        var_is_iter = False
    
    # Create temporary variables
    (temp_iter_vars, temp_reduce_vars), temp_var_ranges = dependencies.index_vars_no_squeeze(
        iter_size,
        reduce_size,
        prefix="t",
    )
    
    # Define the transformation function that ensures consistent transformation
    # for both loads and stores
    def transformed_body(*indices: Sequence[sympy.Expr]) -> Any:
        index = [*itertools.chain.from_iterable(indices)]
        iter_idx = index[: len(iter_size)]
        reduce_idx = index[len(iter_size):]
        
        # Clone the indices
        modified_iter_idx = list(iter_idx)
        modified_reduce_idx = list(reduce_idx)
        
        # Apply the transformation
        if var_is_iter:
            target_var = modified_iter_idx[var_idx]
            # Use the consistent transformation pattern
            modified_iter_idx[var_idx] = (offset_size * ModularIndexing(target_var, 1, max_block_size)) + \
                                         FloorDiv(target_var, max_block_size)
        else:
            target_var = modified_reduce_idx[var_idx]
            modified_reduce_idx[var_idx] = (offset_size * ModularIndexing(target_var, 1, max_block_size)) + \
                                           FloorDiv(target_var, max_block_size)
        
        # Call the original body with consistently transformed indices
        return old_body(modified_iter_idx, modified_reduce_idx)
    
    # Create intermediate LoopBody
    intermediate_body = LoopBody(
        transformed_body,
        (temp_iter_vars, temp_reduce_vars),
        temp_var_ranges,
        temp_iter_vars,
        temp_reduce_vars
    )
    
    # Create final variables
    (final_iter_vars, final_reduce_vars), final_var_ranges = dependencies.index_vars_no_squeeze(
        iter_size,
        reduce_size,
        prefix="p",
    )
    
    # Create final LoopBody
    final_body = LoopBody(
        intermediate_body,
        (final_iter_vars, final_reduce_vars),
        final_var_ranges,
        final_iter_vars,
        final_reduce_vars
    )
    
    return final_body


# def apply_offset_transformation(body, var, offset_size):
    """
    Apply offset transformation where the targeted variable is expressed
    as a composition of ModularIndexing and FloorDiv to improve memory access patterns.
    
    Handles cases where var can be in either iter_vars or reduce_vars.
    
    Args:
        body: The original LoopBody
        var: The variable to transform
        offset_size: The offset factor
    
    Returns:
        New LoopBody with the transformation applied
    """
    # Get the current sizes and variables
    old_body = body
    iter_vars, reduce_vars = body.vars
    iter_size, reduce_size = body.sizes
    
    from . import dependencies
    from . loop_body import LoopBody


    # Check if var is in iter_vars or reduce_vars
    var_in_iter = var in iter_vars
    var_in_reduce = var in reduce_vars
    
    if not var_in_iter and not var_in_reduce:
        # Variable not found in this body
        return body
    
    # Find position of var in the appropriate vars list
    if var_in_iter:
        var_idx = iter_vars.index(var)
        var_is_iter = True
    else:
        var_idx = reduce_vars.index(var)
        var_is_iter = False
    
    # Create temporary variables with 't' prefix
    (temp_iter_vars, temp_reduce_vars), temp_var_ranges = dependencies.index_vars_no_squeeze(
        iter_size,
        reduce_size,
        prefix="t",
    )
    
    # Define the transformation function
    def transformed_body(*indices: Sequence[sympy.Expr]) -> Any:
        index = [*itertools.chain.from_iterable(indices)]
        iter_idx = index[: len(iter_size)]
        reduce_idx = index[len(iter_size):]
        
        # Apply transformation to the targeted variable
        modified_iter_idx = list(iter_idx)
        modified_reduce_idx = list(reduce_idx)
        
        max_block_size = 2048  # Parameter
        
        if var_is_iter:
            # Transform in iter_vars
            target_var = modified_iter_idx[var_idx]
            modified_iter_idx[var_idx] = (offset_size * ModularIndexing(target_var, 1, max_block_size)) + FloorDiv(target_var, max_block_size)
        else:
            # Transform in reduce_vars
            target_var = modified_reduce_idx[var_idx]
            modified_reduce_idx[var_idx] = (offset_size * ModularIndexing(target_var, 1, max_block_size)) + FloorDiv(target_var, max_block_size)
        
        return old_body(modified_iter_idx, modified_reduce_idx)
    
    # Create intermediate LoopBody
    intermediate_body = LoopBody(
        transformed_body,
        (temp_iter_vars, temp_reduce_vars),
        temp_var_ranges,
        temp_iter_vars,
        temp_reduce_vars
    )
    
    # Create final variables with 'p' prefix
    (final_iter_vars, final_reduce_vars), final_var_ranges = dependencies.index_vars_no_squeeze(
        iter_size,
        reduce_size,
        prefix="p",
    )
    
    # Create final LoopBody
    final_body = LoopBody(
        intermediate_body,
        (final_iter_vars, final_reduce_vars),
        final_var_ranges,
        final_iter_vars,
        final_reduce_vars
    )
    
    return final_body