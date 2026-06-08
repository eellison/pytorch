# mypy: allow-untyped-defs
"""
Expand-Sum Elision Pass: Simplifies sum(expand(x) / N, dims) -> sum(x, reduced_dims).

Pattern:
    y = expand(x, [B, C, H, W])        # x has shape [B, C, 1, 1]
    z = div(y, H*W)                     # or mul(y, 1/(H*W))
    result = sum(z, [0, 2, 3])          # sum over batch + spatial

When dim i is expanded (x.size(i)==1, expanded.size(i)==S_i) and then summed,
the expand+sum contributes a factor of S_i. If we also divide by the product of
those expanded sizes, the net contribution is 1 (the expand+div+sum is identity
for those dims).

Simplification:
    result = sum(x, [0, 2, 3])          # or sum(squeeze(x), non-trivial-dims)

In practice, the pass also handles chains where the div/mul scalar matches
the product of expanded sizes along the summed dims. This eliminates the
expensive expanded-spatial reduction (avg_pool2d_backward pattern).
"""

import logging
import math
from functools import reduce
from typing import Optional

import operator

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_node_shape(node: fx.Node) -> Optional[list[int]]:
    """Get static shape from node's fake tensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        shape = val.shape
        # Only return if all dims are static
        result = []
        for s in shape:
            if isinstance(s, int):
                result.append(s)
            else:
                return None
        return result
    return None


def _get_scalar_value(node_or_val) -> Optional[float]:
    """Try to extract a numeric scalar from an FX arg."""
    if isinstance(node_or_val, (int, float)):
        return float(node_or_val)
    if isinstance(node_or_val, fx.Node):
        val = node_or_val.meta.get("val")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _is_close(a: float, b: float, rtol: float = 1e-6) -> bool:
    """Check if two floats are approximately equal."""
    if b == 0:
        return abs(a) < rtol
    return abs(a - b) / max(abs(a), abs(b)) < rtol


def _trace_through_views(node: fx.Node) -> fx.Node:
    """Trace through view/reshape/permute/squeeze ops to find the source tensor.

    Returns the node closest to the computation (before any shape-only ops).
    We only trace through ops that don't change the data, just the shape/strides.
    """
    view_ops = {
        aten.view.default,
        aten.reshape.default,
        aten.permute.default,
        aten.squeeze.dim,
        aten.squeeze.dims,
        aten.squeeze.default,
        aten.unsqueeze.default,
        aten.expand.default,
        aten.as_strided.default,
    }
    current = node
    visited = set()
    while (
        isinstance(current, fx.Node)
        and current.op == "call_function"
        and current.target in view_ops
        and current not in visited
    ):
        visited.add(current)
        current = current.args[0]
    return current


def expand_sum_elision_pass(graph: fx.Graph) -> fx.Graph:
    """
    Simplify sum(expand(x, shape) / N, dims) -> sum(x, adjusted_dims) when:
    - The expanded dims (where x has size 1 but expanded shape doesn't) are
      all included in the sum dims
    - N equals the product of expanded sizes along those dims

    This eliminates the costly spatial expansion in avg_pool2d_backward patterns.
    """
    if not getattr(config, "expand_sum_elision", True):
        return graph

    num_elisions = 0

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not aten.sum.dim_IntList:
            continue

        sum_node = node
        sum_input = sum_node.args[0]
        if not isinstance(sum_input, fx.Node):
            continue
        sum_dims = sum_node.args[1] if len(sum_node.args) > 1 else None
        if not isinstance(sum_dims, (list, tuple)):
            continue
        sum_dims = list(sum_dims)

        # Check for keepdim
        keepdim = sum_node.args[2] if len(sum_node.args) > 2 else False

        # Try to match: sum(div/mul(expand(...), scalar), dims)
        # or: sum(expand(...), dims) (no div/mul)
        expand_node = None
        scalar_factor = None  # The divisor (for div) or 1/multiplier (for mul)

        if (sum_input.op == "call_function" and
                sum_input.target is aten.div.Scalar):
            # sum(div(expand_or_view, N), dims)
            divisor = _get_scalar_value(sum_input.args[1])
            if divisor is not None and divisor != 0:
                scalar_factor = divisor
                expand_candidate = sum_input.args[0]
                if isinstance(expand_candidate, fx.Node):
                    expand_node = expand_candidate
        elif (sum_input.op == "call_function" and
                sum_input.target is aten.div.Tensor):
            # sum(div(expand_or_view, tensor_scalar), dims)
            divisor_node = sum_input.args[1]
            if isinstance(divisor_node, fx.Node):
                divisor = _get_scalar_value(divisor_node)
                if divisor is not None and divisor != 0:
                    scalar_factor = divisor
                    expand_candidate = sum_input.args[0]
                    if isinstance(expand_candidate, fx.Node):
                        expand_node = expand_candidate
        elif (sum_input.op == "call_function" and
                sum_input.target is aten.mul.Scalar):
            # sum(mul(expand_or_view, 1/N), dims)
            multiplier = _get_scalar_value(sum_input.args[1])
            if multiplier is not None and multiplier != 0:
                scalar_factor = 1.0 / multiplier
                expand_candidate = sum_input.args[0]
                if isinstance(expand_candidate, fx.Node):
                    expand_node = expand_candidate
        elif (sum_input.op == "call_function" and
                sum_input.target is aten.mul.Tensor):
            # sum(mul(expand_or_view, scalar_tensor), dims) where scalar is 1/N
            for idx in [1, 0]:
                other_idx = 1 - idx
                mult_node = sum_input.args[idx]
                if isinstance(mult_node, fx.Node):
                    multiplier = _get_scalar_value(mult_node)
                    if multiplier is not None and multiplier != 0:
                        scalar_factor = 1.0 / multiplier
                        expand_candidate = sum_input.args[other_idx]
                        if isinstance(expand_candidate, fx.Node):
                            expand_node = expand_candidate
                        break
        else:
            # sum(expand(...), dims) without div/mul
            expand_node = sum_input
            scalar_factor = None  # Will check if expand sizes cancel with sum

        if expand_node is None:
            continue

        # Find the expand op (possibly through view/reshape)
        actual_expand = expand_node
        # Walk through views to find the expand
        view_chain = []
        visited = set()
        while (isinstance(actual_expand, fx.Node) and
               actual_expand.op == "call_function" and
               actual_expand not in visited):
            visited.add(actual_expand)
            if actual_expand.target is aten.expand.default:
                break
            elif actual_expand.target in (aten.view.default, aten.reshape.default):
                view_chain.append(actual_expand)
                actual_expand = actual_expand.args[0]
            else:
                actual_expand = None
                break

        if actual_expand is None:
            continue
        if not isinstance(actual_expand, fx.Node):
            continue
        if actual_expand.op != "call_function" or actual_expand.target is not aten.expand.default:
            continue

        # Extract expand info
        expand_input = actual_expand.args[0]
        if not isinstance(expand_input, fx.Node):
            continue

        input_shape = _get_node_shape(expand_input)
        expanded_shape = _get_node_shape(actual_expand)

        if input_shape is None or expanded_shape is None:
            continue
        if len(input_shape) != len(expanded_shape):
            continue

        # Find which dims were expanded (size 1 -> size > 1)
        ndim = len(input_shape)
        expanded_dims = []
        expanded_product = 1
        for d in range(ndim):
            if input_shape[d] == 1 and expanded_shape[d] > 1:
                expanded_dims.append(d)
                expanded_product *= expanded_shape[d]

        if not expanded_dims:
            continue

        # Normalize sum_dims
        norm_sum_dims = [d if d >= 0 else ndim + d for d in sum_dims]

        # But wait - if there were view/reshape ops between expand and div/sum,
        # the dims may have changed. For now, only handle the case where the
        # expand output feeds directly (or through the div) to the sum.
        # Check: the shape of sum_input should be expanded_shape (or same ndim)
        sum_input_shape = _get_node_shape(sum_input)
        if sum_input_shape is None:
            continue

        # The sum is over expanded_shape (or the shape after div, which is same)
        # Check that ALL expanded dims are included in sum_dims
        if not all(d in norm_sum_dims for d in expanded_dims):
            continue

        # Check that scalar_factor matches expanded_product (the expand+sum cancellation)
        if scalar_factor is not None:
            if not _is_close(scalar_factor, float(expanded_product)):
                continue
        else:
            # No div/mul - this means the expand contributes a factor.
            # We can only simplify if this factor is acceptable (caller wanted it).
            # Without div, sum(expand(x)) = x * expanded_product summed.
            # That's a different value, so we can't elide.
            continue

        # All conditions met. The expand+div+sum is identity for the expanded dims.
        # Replace: sum(div(expand(x, ...), N), dims) -> sum(x, remaining_dims)
        # where remaining_dims are the sum dims that aren't expanded dims,
        # adjusted for the input shape.

        # The remaining sum dims are those that aren't the expanded dims
        # (which cancel with div). These are dims where the input has actual data.
        remaining_sum_dims = [d for d in norm_sum_dims if d not in expanded_dims]

        # But we need to handle: if remaining dims have size 1 in input,
        # sum over them is trivial (just squeeze). Let's just emit the sum
        # over the input with the remaining dims.

        # Get metadata for the result
        sum_val = sum_node.meta.get("val")
        if sum_val is None:
            continue
        result_shape = list(sum_val.shape)
        result_dtype = sum_val.dtype
        result_device = sum_val.device

        with graph.inserting_before(sum_node):
            # Emit sum(expand_input, remaining_sum_dims)
            if remaining_sum_dims:
                new_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    (expand_input, remaining_sum_dims, keepdim),
                )
                new_sum_shape = list(input_shape)
                for d in remaining_sum_dims:
                    if keepdim:
                        new_sum_shape[d] = 1
                    else:
                        new_sum_shape[d] = 0  # placeholder for removal
                if not keepdim:
                    new_sum_shape = [s for i, s in enumerate(new_sum_shape)
                                    if i not in remaining_sum_dims]
            else:
                # All sum dims were expanded dims -> just squeeze the expanded dims
                # sum over empty dims is identity
                new_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    (expand_input, [], keepdim),
                )
                new_sum_shape = list(input_shape)

            # Set metadata
            new_sum.meta["val"] = torch.empty(
                result_shape, dtype=result_dtype, device=result_device
            )

            # The new sum might not match the expected result shape because
            # input has extra size-1 dims that get dropped by the original sum.
            # E.g., input [128, 640, 1, 1], sum over [0, 2, 3] -> [640]
            # but our new sum over remaining [0] with input [128, 640, 1, 1] -> [640, 1, 1]
            # So we need a reshape to match.
            new_sum_actual_shape = list(input_shape)
            if not keepdim:
                # Remove the summed dims
                for d in sorted(remaining_sum_dims, reverse=True):
                    new_sum_actual_shape.pop(d)
            else:
                for d in remaining_sum_dims:
                    new_sum_actual_shape[d] = 1

            if new_sum_actual_shape != result_shape:
                # Need a reshape
                replacement = graph.call_function(
                    aten.reshape.default,
                    (new_sum, result_shape),
                )
                replacement.meta["val"] = torch.empty(
                    result_shape, dtype=result_dtype, device=result_device
                )
            else:
                replacement = new_sum

        sum_node.replace_all_uses_with(replacement)
        num_elisions += 1
        counters["inductor"]["expand_sum_elisions"] += 1
        log.debug(
            "expand_sum_elision: replaced sum(div(expand(%s, %s), %s), %s) -> "
            "sum(%s, %s) + reshape",
            expand_input.name,
            expanded_shape,
            scalar_factor,
            sum_dims,
            expand_input.name,
            remaining_sum_dims,
        )

    if num_elisions > 0:
        graph.eliminate_dead_code()
        graph.lint()
        log.info(
            "expand_sum_elision: eliminated %d expand+div+sum patterns",
            num_elisions,
        )

    return graph
