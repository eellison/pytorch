# mypy: allow-untyped-defs
"""
Linear Reduction Algebraic Elimination Pass.

Eliminates dependent reduction passes by recognizing linear expressions.

Pattern:
    Pass 1: sum1 = sum(X, dims), sum2 = sum(X*Y, dims)    -- reads large tensor
    Pass 2: sum3 = sum(X * f(sum1, sum2), dims)            -- reads large tensor AGAIN

When f(sum1, sum2) is a per-channel scalar (broadcast over reduced dims), the
second pass can be algebraically eliminated:

    sum(X * scalar_per_channel, dims) = scalar_per_channel * sum(X, dims)

More generally:
    sum(a*X + b*Y + c, dims) = a*sum(X, dims) + b*sum(Y, dims) + c*N

where a, b, c are per-channel (broadcast over reduced dims) and N is the product
of the reduced dimension sizes.

For base tensors that don't already have a sibling sum, the pass CREATES one.
The scheduler will fuse these new sums with existing sibling reductions into
a single kernel pass, avoiding redundant data reads.

This is particularly effective for batch normalization backward, which computes:
    sum((grad_out - mean_grad - (x - mean) * var_term) * weight, [0, 2, 3])
where mean_grad and var_term are derived from earlier reductions over the same tensors.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass
class LinearTerm:
    """Represents a term: coefficient * base_tensor (to be summed).

    coefficient: a per-channel scalar (broadcast over reduce dims), or None for "1"
    base_tensor: the large tensor to sum, or None for a constant term (just coeff * N)
    negate: whether to negate the whole term
    """
    coefficient: object  # fx.Node, list[fx.Node], or None
    base_tensor: Optional[fx.Node]
    negate: bool = False


@dataclass
class LinearDecomposition:
    """A linear decomposition of an expression into terms suitable for reduction."""
    terms: list[LinearTerm] = field(default_factory=list)
    valid: bool = False


def _get_node_shape(node: fx.Node) -> Optional[list[int]]:
    """Get shape from node's fake tensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return list(val.shape)
    return None


def _is_broadcast_over_dims(node: fx.Node, full_shape: list[int], reduce_dims: list[int]) -> bool:
    """Check if node's value has size 1 in all reduce_dims (broadcast/per-channel)."""
    shape = _get_node_shape(node)
    if shape is None:
        return False

    ndim = len(full_shape)
    node_ndim = len(shape)

    if node_ndim < ndim:
        shape = [1] * (ndim - node_ndim) + shape
    elif node_ndim > ndim:
        return False

    for d in reduce_dims:
        dd = d if d >= 0 else ndim + d
        if dd >= len(shape):
            return False
        if shape[dd] != 1:
            return False

    return True


def _is_large_tensor(node: fx.Node, full_shape: list[int], reduce_dims: list[int]) -> bool:
    """Check if node is a full-sized tensor (not broadcast over reduce dims)."""
    shape = _get_node_shape(node)
    if shape is None:
        return False
    ndim = len(full_shape)
    node_ndim = len(shape)
    if node_ndim < ndim:
        shape = [1] * (ndim - node_ndim) + shape
    elif node_ndim != ndim:
        return False

    for d in reduce_dims:
        dd = d if d >= 0 else ndim + d
        if dd >= len(shape) or dd >= len(full_shape):
            return False
        if shape[dd] != full_shape[dd]:
            return False
    return True


def _find_existing_sum(tensor_node: fx.Node, reduce_dims: list[int], graph_nodes: set) -> Optional[fx.Node]:
    """Find an existing sum(tensor_node, reduce_dims) in the graph."""
    sorted_dims = sorted(reduce_dims)
    for user in tensor_node.users:
        if user not in graph_nodes:
            continue
        if user.op != "call_function":
            continue
        if user.target is not aten.sum.dim_IntList:
            continue
        if user.args[0] is not tensor_node:
            continue
        user_dims = user.args[1] if len(user.args) > 1 else None
        if user_dims is None:
            continue
        if sorted(user_dims) == sorted_dims:
            return user
    return None


def _trace_linear_expression(
    node: fx.Node,
    reduce_dims: list[int],
    full_shape: list[int],
    graph_nodes: set,
    depth: int = 0,
    max_depth: int = 15,
) -> Optional[LinearDecomposition]:
    """Decompose `node` as a linear combination of large tensors + constant terms.

    Each term has a per-channel coefficient (broadcast over reduce_dims) and a
    base tensor (or None for constant terms).
    """
    if depth > max_depth:
        return None

    if not isinstance(node, fx.Node):
        return None

    # Base case: node is itself a large tensor (leaf of the linear expression)
    # Check this before requiring call_function so placeholders and other node
    # types can serve as base tensors in decompositions.
    if node.op != "call_function":
        if _is_large_tensor(node, full_shape, reduce_dims):
            result = LinearDecomposition(valid=True)
            result.terms.append(LinearTerm(coefficient=None, base_tensor=node))
            return result
        return None

    target = node.target

    # Case: mul(A, B) where one side is broadcast (per-channel scalar)
    if target is aten.mul.Tensor:
        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            return None

        for scalar_side, tensor_side in [(lhs, rhs), (rhs, lhs)]:
            if _is_broadcast_over_dims(scalar_side, full_shape, reduce_dims):
                # tensor_side might be a deeper linear expression
                sub_decomp = _trace_linear_expression(
                    tensor_side, reduce_dims, full_shape, graph_nodes, depth + 1
                )
                if sub_decomp is not None and sub_decomp.valid:
                    return _scale_decomposition(sub_decomp, scalar_side)

                # Or tensor_side is a base large tensor
                if _is_large_tensor(tensor_side, full_shape, reduce_dims):
                    result = LinearDecomposition(valid=True)
                    result.terms.append(LinearTerm(
                        coefficient=scalar_side,
                        base_tensor=tensor_side,
                    ))
                    return result
        return None

    # Case: add(A, B)
    if target is aten.add.Tensor:
        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            return None

        lhs_bcast = _is_broadcast_over_dims(lhs, full_shape, reduce_dims)
        rhs_bcast = _is_broadcast_over_dims(rhs, full_shape, reduce_dims)

        if lhs_bcast and rhs_bcast:
            return None

        if lhs_bcast:
            rhs_decomp = _trace_linear_expression(
                rhs, reduce_dims, full_shape, graph_nodes, depth + 1
            )
            if rhs_decomp is not None and rhs_decomp.valid:
                rhs_decomp.terms.append(LinearTerm(coefficient=lhs, base_tensor=None))
                return rhs_decomp
            return None

        if rhs_bcast:
            lhs_decomp = _trace_linear_expression(
                lhs, reduce_dims, full_shape, graph_nodes, depth + 1
            )
            if lhs_decomp is not None and lhs_decomp.valid:
                lhs_decomp.terms.append(LinearTerm(coefficient=rhs, base_tensor=None))
                return lhs_decomp
            return None

        # Both large - decompose both
        lhs_decomp = _trace_linear_expression(
            lhs, reduce_dims, full_shape, graph_nodes, depth + 1
        )
        rhs_decomp = _trace_linear_expression(
            rhs, reduce_dims, full_shape, graph_nodes, depth + 1
        )
        if (lhs_decomp is not None and lhs_decomp.valid and
                rhs_decomp is not None and rhs_decomp.valid):
            lhs_decomp.terms.extend(rhs_decomp.terms)
            return lhs_decomp

        return None

    # Case: sub(A, B) = A + (-B)
    if target is aten.sub.Tensor:
        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            return None

        lhs_bcast = _is_broadcast_over_dims(lhs, full_shape, reduce_dims)
        rhs_bcast = _is_broadcast_over_dims(rhs, full_shape, reduce_dims)

        if lhs_bcast and rhs_bcast:
            return None

        if lhs_bcast:
            rhs_decomp = _trace_linear_expression(
                rhs, reduce_dims, full_shape, graph_nodes, depth + 1
            )
            if rhs_decomp is not None and rhs_decomp.valid:
                for term in rhs_decomp.terms:
                    term.negate = not term.negate
                rhs_decomp.terms.append(LinearTerm(coefficient=lhs, base_tensor=None))
                return rhs_decomp
            return None

        if rhs_bcast:
            lhs_decomp = _trace_linear_expression(
                lhs, reduce_dims, full_shape, graph_nodes, depth + 1
            )
            if lhs_decomp is not None and lhs_decomp.valid:
                lhs_decomp.terms.append(LinearTerm(coefficient=rhs, base_tensor=None, negate=True))
                return lhs_decomp
            return None

        # Both large
        lhs_decomp = _trace_linear_expression(
            lhs, reduce_dims, full_shape, graph_nodes, depth + 1
        )
        rhs_decomp = _trace_linear_expression(
            rhs, reduce_dims, full_shape, graph_nodes, depth + 1
        )
        if (lhs_decomp is not None and lhs_decomp.valid and
                rhs_decomp is not None and rhs_decomp.valid):
            for term in rhs_decomp.terms:
                term.negate = not term.negate
            lhs_decomp.terms.extend(rhs_decomp.terms)
            return lhs_decomp

        return None

    # Case: neg(A)
    if target is aten.neg.default:
        inner = node.args[0]
        if not isinstance(inner, fx.Node):
            return None
        inner_decomp = _trace_linear_expression(
            inner, reduce_dims, full_shape, graph_nodes, depth + 1
        )
        if inner_decomp is not None and inner_decomp.valid:
            for term in inner_decomp.terms:
                term.negate = not term.negate
            return inner_decomp
        return None

    # Base case: node is itself a large tensor (leaf of the linear expression)
    if _is_large_tensor(node, full_shape, reduce_dims):
        result = LinearDecomposition(valid=True)
        result.terms.append(LinearTerm(coefficient=None, base_tensor=node))
        return result

    return None


def _scale_decomposition(decomp: LinearDecomposition, scalar_node: fx.Node) -> LinearDecomposition:
    """Multiply all terms in a decomposition by scalar_node."""
    result = LinearDecomposition(valid=True)
    for term in decomp.terms:
        new_coeff = _multiply_coefficients(term.coefficient, scalar_node)
        result.terms.append(LinearTerm(
            coefficient=new_coeff,
            base_tensor=term.base_tensor,
            negate=term.negate,
        ))
    return result


def _multiply_coefficients(existing, new_scalar: fx.Node):
    """Combine coefficient representations.

    Coefficients can be:
    - None (meaning 1)
    - A single fx.Node
    - A list of fx.Nodes to multiply together
    """
    if existing is None:
        return new_scalar
    if isinstance(existing, list):
        return existing + [new_scalar]
    return [existing, new_scalar]


def _compute_reduce_numel(full_shape: list[int], reduce_dims: list[int]) -> int:
    """Product of sizes along reduced dimensions."""
    numel = 1
    ndim = len(full_shape)
    for d in reduce_dims:
        dd = d if d >= 0 else ndim + d
        numel *= full_shape[dd]
    return numel


def _squeeze_coeff_to_result_shape(
    graph: fx.Graph,
    coeff_node: fx.Node,
    reduce_dims: list[int],
    result_shape: list[int],
    result_dtype: torch.dtype,
    result_device: torch.device,
) -> fx.Node:
    """Squeeze a coefficient node (e.g., [1,128,1,1]) to match result shape (e.g., [128]).

    After reducing over dims [0,2,3], a per-channel coefficient [1,128,1,1] should
    become [128] to be compatible with the sum result.
    """
    coeff_shape = _get_node_shape(coeff_node)
    if coeff_shape is None:
        return coeff_node

    # If shapes already match, no squeezing needed
    if coeff_shape == result_shape:
        return coeff_node

    # Squeeze the reduce dims (they should all be size 1)
    # We squeeze all dims that are in reduce_dims
    ndim = len(coeff_shape)
    dims_to_squeeze = []
    for d in sorted(reduce_dims):
        dd = d if d >= 0 else ndim + d
        if dd < ndim and coeff_shape[dd] == 1:
            dims_to_squeeze.append(dd)

    if dims_to_squeeze:
        squeezed = graph.call_function(aten.squeeze.dims, (coeff_node, dims_to_squeeze))
        # Compute the squeezed shape
        new_shape = [s for i, s in enumerate(coeff_shape) if i not in dims_to_squeeze]
        if not new_shape:
            new_shape = []
        fake_val = coeff_node.meta["val"].squeeze(dim=dims_to_squeeze) if hasattr(coeff_node.meta.get("val", None), "squeeze") else None
        if fake_val is not None:
            squeezed.meta["val"] = fake_val
        else:
            squeezed.meta["val"] = torch.empty(new_shape, dtype=result_dtype, device=result_device)

        # May need further reshape if leading dims of 1 remain
        squeezed_shape = _get_node_shape(squeezed)
        if squeezed_shape != result_shape and squeezed_shape is not None:
            # Try reshape
            reshaped = graph.call_function(aten.reshape.default, (squeezed, result_shape))
            reshaped.meta["val"] = torch.empty(result_shape, dtype=result_dtype, device=result_device)
            return reshaped
        return squeezed

    # If no dims to squeeze, try reshape directly
    reshaped = graph.call_function(aten.reshape.default, (coeff_node, result_shape))
    reshaped.meta["val"] = torch.empty(result_shape, dtype=result_dtype, device=result_device)
    return reshaped


def _input_depends_on_reduction(node: fx.Node, reduce_dims: list[int], graph_nodes: set, max_depth: int = 12) -> bool:
    """Check if node's computation depends on a sum with matching reduce_dims.

    Ensures we only try to eliminate "second pass" sums whose inputs are derived
    from first-pass reduction outputs.
    """
    visited = set()
    stack = [(node, 0)]
    sorted_dims = sorted(reduce_dims)

    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        if current in visited:
            continue
        visited.add(current)

        if not isinstance(current, fx.Node):
            continue
        if current.op != "call_function":
            continue

        if current.target is aten.sum.dim_IntList:
            if len(current.args) >= 2:
                node_dims = current.args[1]
                if isinstance(node_dims, (list, tuple)) and sorted(node_dims) == sorted_dims:
                    return True

        if current.target in (
            aten.mul.Tensor, aten.mul.Scalar, aten.add.Tensor, aten.sub.Tensor,
            aten.neg.default, aten.div.Tensor, aten.div.Scalar,
            aten.unsqueeze.default, aten.squeeze.dims, aten.squeeze.default,
            aten.expand.default, aten.view.default, aten.reshape.default,
        ):
            for arg in current.args:
                if isinstance(arg, fx.Node):
                    stack.append((arg, depth + 1))

    return False


def _has_sibling_reduction(base_tensor: fx.Node, reduce_dims: list[int], target_sum: fx.Node, graph_nodes: set) -> bool:
    """Check if base_tensor participates in existing reductions over the same dims.

    A "sibling reduction" means there's already a sum over the same dims that reads
    this tensor (or a pointwise derivative of it), so a new sum would be fusable.
    """
    sorted_dims = sorted(reduce_dims)
    # Check direct users
    for user in base_tensor.users:
        if user not in graph_nodes:
            continue
        if user.op != "call_function":
            continue
        if user.target is aten.sum.dim_IntList and user is not target_sum:
            if len(user.args) >= 2 and sorted(user.args[1]) == sorted_dims:
                return True
    # Check one level of indirection
    for user in base_tensor.users:
        if user not in graph_nodes or user.op != "call_function":
            continue
        for user2 in user.users:
            if user2 not in graph_nodes or user2.op != "call_function":
                continue
            if user2.target is aten.sum.dim_IntList and user2 is not target_sum:
                if len(user2.args) >= 2 and sorted(user2.args[1]) == sorted_dims:
                    return True
    return False


def linear_reduction_elimination_pass(graph: fx.Graph) -> Optional[fx.Graph]:
    """
    Eliminate dependent reductions by algebraic decomposition of linear expressions.

    Finds sum(linear_expr(X, Y, ...), dims) where the linear expression is in
    terms of large tensors X, Y, and rewrites to algebraic combinations of
    per-tensor sums.
    """
    if not getattr(config, "linear_reduction_elimination", True):
        return None

    graph_nodes = set(graph.nodes)
    replacements_made = 0

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not aten.sum.dim_IntList:
            continue

        sum_node = node
        if len(sum_node.args) < 2:
            continue

        sum_input = sum_node.args[0]
        reduce_dims = sum_node.args[1]

        if not isinstance(sum_input, fx.Node):
            continue
        if not isinstance(reduce_dims, (list, tuple)):
            continue

        reduce_dims = list(reduce_dims)

        full_shape = _get_node_shape(sum_input)
        if full_shape is None:
            continue

        # Only process sums whose inputs depend on other sum outputs
        if not _input_depends_on_reduction(sum_input, reduce_dims, graph_nodes):
            continue

        # Try to decompose the sum input
        decomp = _trace_linear_expression(
            sum_input, reduce_dims, full_shape, graph_nodes
        )

        if decomp is None or not decomp.valid or len(decomp.terms) == 0:
            continue

        # Count how many base tensor terms need new sums created
        n_base_terms = sum(1 for t in decomp.terms if t.base_tensor is not None)
        n_existing = 0
        for term in decomp.terms:
            if term.base_tensor is None:
                continue
            existing = _find_existing_sum(term.base_tensor, reduce_dims, graph_nodes)
            if existing is not None and existing is not sum_node:
                n_existing += 1
                continue
            if _has_sibling_reduction(term.base_tensor, reduce_dims, sum_node, graph_nodes):
                n_existing += 1

        # Allow the rewrite when:
        # - At least one term has a sibling (original behavior), OR
        # - The number of new sums needed is small enough that creating them
        #   is worthwhile (the scheduler fuses same-shape reductions together,
        #   so the new sums won't be separate kernel passes in practice)
        n_new_sums_needed = n_base_terms - n_existing
        if n_existing == 0 and n_new_sums_needed > 3:
            # Too many new sums with no existing siblings to fuse with
            continue

        # Get result metadata
        sum_val = sum_node.meta.get("val")
        if sum_val is None:
            continue

        result_shape = list(sum_val.shape)
        result_dtype = sum_val.dtype
        result_device = sum_val.device
        reduce_numel = _compute_reduce_numel(full_shape, reduce_dims)

        # Emit the algebraic replacement
        with graph.inserting_before(sum_node):
            result_parts = []

            for term in decomp.terms:
                if term.base_tensor is not None:
                    # Get or create sum for this base tensor
                    base_sum = _find_existing_sum(term.base_tensor, reduce_dims, graph_nodes)
                    if base_sum is None:
                        base_sum = graph.call_function(
                            aten.sum.dim_IntList,
                            (term.base_tensor, reduce_dims),
                        )
                        base_sum.meta["val"] = torch.empty(
                            result_shape, dtype=result_dtype, device=result_device
                        )
                        graph_nodes.add(base_sum)

                    # Apply coefficient: coeff * sum(base_tensor, dims)
                    part = _emit_coeff_mul(
                        graph, term.coefficient, base_sum,
                        reduce_dims, result_shape, result_dtype, result_device
                    )
                else:
                    # Constant term: coeff * N
                    part = _emit_coeff_times_N(
                        graph, term.coefficient, reduce_numel,
                        reduce_dims, result_shape, result_dtype, result_device
                    )

                # Apply negation
                if term.negate:
                    part = graph.call_function(aten.neg.default, (part,))
                    part.meta["val"] = torch.empty(
                        result_shape, dtype=result_dtype, device=result_device
                    )

                result_parts.append(part)

            if len(result_parts) == 0:
                continue

            # Sum all parts
            replacement = result_parts[0]
            for part in result_parts[1:]:
                replacement = graph.call_function(aten.add.Tensor, (replacement, part))
                replacement.meta["val"] = torch.empty(
                    result_shape, dtype=result_dtype, device=result_device
                )

        sum_node.replace_all_uses_with(replacement)
        replacements_made += 1
        counters["inductor"]["linear_reduction_elimination"] += 1
        log.debug(
            "Eliminated dependent reduction: %s = sum(%s, %s) -> algebraic combination "
            "(%d terms, %d existing sums, %d created)",
            sum_node.name,
            sum_input.name,
            reduce_dims,
            len(decomp.terms),
            n_existing,
            n_new_sums_needed,
        )

    if replacements_made > 0:
        graph.eliminate_dead_code()
        graph.lint()
        log.info(
            "linear_reduction_elimination: eliminated %d dependent reduction(s)",
            replacements_made,
        )

    return graph if replacements_made > 0 else None


def _emit_coeff_mul(
    graph: fx.Graph,
    coeff,
    base_sum_node: fx.Node,
    reduce_dims: list[int],
    result_shape: list[int],
    result_dtype: torch.dtype,
    result_device: torch.device,
) -> fx.Node:
    """Emit: coeff * base_sum_node, with proper shape handling.

    Coefficients may have shapes like [1,128,1,1] that need squeezing to [128]
    to be compatible with the sum result shape.
    """
    if coeff is None:
        return base_sum_node

    def make_result_meta():
        return torch.empty(result_shape, dtype=result_dtype, device=result_device)

    if isinstance(coeff, list):
        # Chain of coefficients to multiply
        # First squeeze each to result shape, then multiply
        current = base_sum_node
        for c in coeff:
            squeezed_c = _squeeze_coeff_to_result_shape(
                graph, c, reduce_dims, result_shape, result_dtype, result_device
            )
            current = graph.call_function(aten.mul.Tensor, (current, squeezed_c))
            current.meta["val"] = make_result_meta()
        return current

    # Single coefficient
    squeezed = _squeeze_coeff_to_result_shape(
        graph, coeff, reduce_dims, result_shape, result_dtype, result_device
    )
    result = graph.call_function(aten.mul.Tensor, (squeezed, base_sum_node))
    result.meta["val"] = make_result_meta()
    return result


def _emit_coeff_times_N(
    graph: fx.Graph,
    coeff,
    N: int,
    reduce_dims: list[int],
    result_shape: list[int],
    result_dtype: torch.dtype,
    result_device: torch.device,
) -> fx.Node:
    """Emit: coeff * N for constant terms.

    sum(coeff_broadcast, dims) = coeff_squeezed * N
    """
    def make_result_meta():
        return torch.empty(result_shape, dtype=result_dtype, device=result_device)

    if coeff is None:
        # Pure constant 1 * N - create scalar tensor
        # This shouldn't happen in practice
        scalar_node = graph.call_function(
            aten.full.default,
            (result_shape, float(N)),
            {"dtype": result_dtype, "layout": torch.strided,
             "device": result_device, "pin_memory": False},
        )
        scalar_node.meta["val"] = make_result_meta()
        return scalar_node

    if isinstance(coeff, list):
        # Multiply coefficients together, then multiply by N
        # Start by squeezing first coeff and multiplying by N
        first = _squeeze_coeff_to_result_shape(
            graph, coeff[0], reduce_dims, result_shape, result_dtype, result_device
        )
        current = graph.call_function(aten.mul.Scalar, (first, float(N)))
        current.meta["val"] = make_result_meta()
        for c in coeff[1:]:
            squeezed_c = _squeeze_coeff_to_result_shape(
                graph, c, reduce_dims, result_shape, result_dtype, result_device
            )
            current = graph.call_function(aten.mul.Tensor, (current, squeezed_c))
            current.meta["val"] = make_result_meta()
        return current

    # Single coefficient * N
    squeezed = _squeeze_coeff_to_result_shape(
        graph, coeff, reduce_dims, result_shape, result_dtype, result_device
    )
    result = graph.call_function(aten.mul.Scalar, (squeezed, float(N)))
    result.meta["val"] = make_result_meta()
    return result
