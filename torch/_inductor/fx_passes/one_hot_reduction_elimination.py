# mypy: allow-untyped-defs
"""
One-Hot Reduction Elimination Pass.

Recognizes the pattern from cross-entropy backward where an iota-based one-hot
mask is constructed, multiplied by a scale, and then reduced over the vocabulary
dimension. This materializes a dense [batch, vocab] intermediate that is entirely
unnecessary.

THE PATTERN (from MobileBERT CE backward):
    iota(vocab_size) -> view([1, vocab_size])
    label -> unsqueeze -> expand([batch, vocab_size])
    eq(expanded_label, expanded_iota) -> one_hot mask [batch, vocab_size]
    where(one_hot, const_a, const_b) -> dense_values [batch, vocab_size]
    dense_values * scale -> scaled [batch, vocab_size]
    sum(scaled, dim=vocab_dim, keepdim=True) -> [batch, 1] result

THE ALGEBRAIC IDENTITY:
    sum(where(label == iota(V), a, b) * scale, dim=V) = a * scale + b * scale * (V - 1)

When b = 0 (the common CE backward case):
    sum(where(label == iota(V), a, 0) * scale, dim=V) = a * scale

This eliminates the entire [batch, vocab] intermediate, replacing the reduction
with a simple scalar multiply.
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims


def _get_node_shape(node: fx.Node) -> Optional[list[int]]:
    """Get static shape from node's fake tensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        shape = val.shape
        result = []
        for s in shape:
            if isinstance(s, int):
                result.append(s)
            else:
                return None
        return result
    return None


def _get_scalar_value(node_or_val) -> Optional[float]:
    """Try to extract a numeric scalar from an FX node or literal."""
    if isinstance(node_or_val, (int, float)):
        return float(node_or_val)
    if isinstance(node_or_val, fx.Node):
        val = node_or_val.meta.get("val")
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, torch.Tensor) and val.ndim == 0:
            # Scalar tensor - try to get val
            if val.numel() == 1:
                try:
                    return float(val.item())
                except Exception:
                    pass
            # Check if it came from scalar_tensor or full
            if node_or_val.op == "call_function":
                if node_or_val.target is aten.scalar_tensor.default:
                    return _get_scalar_value(node_or_val.args[0])
                if node_or_val.target is aten.full.default:
                    if len(node_or_val.args) >= 2:
                        shape = node_or_val.args[0]
                        if shape == [] or shape == ():
                            return _get_scalar_value(node_or_val.args[1])
    return None


def _is_iota_node(node: fx.Node) -> Optional[int]:
    """Check if a node is prims.iota and return its size, or None."""
    if (node.op == "call_function" and
            node.target is prims.iota.default):
        # prims.iota.default(size, start=0, step=1, ...)
        size = node.args[0] if node.args else None
        if isinstance(size, int):
            # Check start=0, step=1
            kwargs = dict(node.kwargs) if node.kwargs else {}
            start = kwargs.get("start", 0)
            step = kwargs.get("step", 1)
            if start == 0 and step == 1:
                return size
    return None


def _trace_to_iota(node: fx.Node) -> Optional[tuple[fx.Node, int]]:
    """Trace through view/expand to find an iota source.

    Returns (iota_node, vocab_size) or None.
    """
    view_ops = {
        aten.view.default,
        aten.reshape.default,
        aten.expand.default,
        aten.unsqueeze.default,
    }
    visited = set()
    current = node
    while (isinstance(current, fx.Node) and
           current.op == "call_function" and
           current not in visited):
        visited.add(current)
        iota_size = _is_iota_node(current)
        if iota_size is not None:
            return (current, iota_size)
        if current.target in view_ops:
            current = current.args[0]
        else:
            break
    return None


def _find_eq_with_iota(eq_node: fx.Node) -> Optional[tuple[fx.Node, fx.Node, int]]:
    """Check if eq_node is eq(expanded_label, iota_view) or eq(iota_view, expanded_label).

    Returns (label_source, iota_node, vocab_size) or None.
    """
    if (eq_node.op != "call_function" or
            eq_node.target not in (aten.eq.Tensor,)):
        return None

    lhs, rhs = eq_node.args[0], eq_node.args[1]
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return None

    # Try lhs as iota
    iota_info = _trace_to_iota(lhs)
    if iota_info is not None:
        return (rhs, iota_info[0], iota_info[1])

    # Try rhs as iota
    iota_info = _trace_to_iota(rhs)
    if iota_info is not None:
        return (lhs, iota_info[0], iota_info[1])

    return None


def _find_where_on_one_hot(where_node: fx.Node) -> Optional[tuple[fx.Node, float, float, int]]:
    """Check if where_node is where(one_hot_eq, const_a, const_b).

    Returns (eq_node, a_value, b_value, vocab_size) or None.
    The eq_node must be the result of eq(expanded_label, iota_view).
    """
    if (where_node.op != "call_function" or
            where_node.target is not aten.where.self):
        return None

    cond = where_node.args[0]
    if not isinstance(cond, fx.Node):
        return None

    # Get the scalar values for true/false branches
    true_val = _get_scalar_value(where_node.args[1])
    false_val = _get_scalar_value(where_node.args[2])
    if true_val is None or false_val is None:
        return None

    # Check if cond is an eq with iota
    eq_info = _find_eq_with_iota(cond)
    if eq_info is None:
        return None

    _, _, vocab_size = eq_info
    return (cond, true_val, false_val, vocab_size)


def _is_broadcast_mul(mul_node: fx.Node) -> Optional[tuple[fx.Node, fx.Node]]:
    """Check if mul_node is mul(big_tensor, small_scale).

    Returns (big_tensor_node, scale_node) or None.
    Big is the one whose shape has the vocab dim; scale is broadcast (has 1 in that dim).
    """
    if (mul_node.op != "call_function" or
            mul_node.target not in (aten.mul.Tensor, aten.mul.Scalar)):
        return None

    lhs, rhs = mul_node.args[0], mul_node.args[1]

    if mul_node.target is aten.mul.Scalar:
        # lhs is tensor, rhs is scalar
        if isinstance(lhs, fx.Node):
            return (lhs, rhs)
        return None

    # mul.Tensor - one arg should be broadcastable (smaller rank or has 1s)
    if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
        return None

    lhs_shape = _get_node_shape(lhs)
    rhs_shape = _get_node_shape(rhs)
    if lhs_shape is None or rhs_shape is None:
        return None

    # The "big" one has more elements
    lhs_numel = 1
    for s in lhs_shape:
        lhs_numel *= s
    rhs_numel = 1
    for s in rhs_shape:
        rhs_numel *= s

    if lhs_numel >= rhs_numel:
        return (lhs, rhs)
    else:
        return (rhs, lhs)


def one_hot_reduction_elimination_pass(graph: fx.Graph) -> fx.Graph:
    """
    Eliminate reductions over one-hot patterns from cross-entropy backward.

    Matches: sum(where(eq(label, iota), a, b) * scale, dim=vocab_dim, keepdim)
    Replaces with: a * scale + b * scale * (V - 1)
    When b == 0: just a * scale (broadcast to correct shape)
    """
    if not getattr(config, "one_hot_reduction_elimination", True):
        return graph

    num_eliminations = 0

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
        keepdim = sum_node.args[2] if len(sum_node.args) > 2 else False

        # Get sum input shape to normalize dims
        sum_input_shape = _get_node_shape(sum_input)
        if sum_input_shape is None:
            continue
        ndim = len(sum_input_shape)
        norm_sum_dims = [d if d >= 0 else ndim + d for d in sum_dims]

        # We look for two patterns:
        # Pattern A: sum(mul(where(eq(label, iota), a, b), scale), dim=vocab)
        # Pattern B: sum(where(eq(label, iota), a, b), dim=vocab) (no scale)

        where_node = None
        scale_node = None
        where_a = None
        where_b = None
        vocab_size = None
        vocab_dim = None

        # Also handle convert_element_type wrapping mul or where
        actual_sum_input = sum_input
        dtype_cast = None
        if (actual_sum_input.op == "call_function" and
                actual_sum_input.target is prims.convert_element_type.default):
            dtype_cast = actual_sum_input
            actual_sum_input = actual_sum_input.args[0]
            if not isinstance(actual_sum_input, fx.Node):
                continue

        # Try Pattern A: sum_input is mul(where(...), scale)
        mul_info = _is_broadcast_mul(actual_sum_input)
        if mul_info is not None:
            big_tensor, scale_arg = mul_info
            if isinstance(big_tensor, fx.Node):
                where_info = _find_where_on_one_hot(big_tensor)
                if where_info is not None:
                    _, where_a, where_b, vocab_size = where_info
                    where_node = big_tensor
                    scale_node = scale_arg

        # Try Pattern B: sum_input is where(...) directly
        if where_node is None:
            where_info = _find_where_on_one_hot(actual_sum_input)
            if where_info is not None:
                _, where_a, where_b, vocab_size = where_info
                where_node = actual_sum_input
                scale_node = None

        if where_node is None:
            continue

        # Find which dim is the vocab dim (should be in sum_dims and have size == vocab_size)
        found_vocab_dim = False
        for d in norm_sum_dims:
            if d < ndim and sum_input_shape[d] == vocab_size:
                vocab_dim = d
                found_vocab_dim = True
                break

        if not found_vocab_dim:
            continue

        # Verify the reduction is ONLY over the vocab dim
        # (if there are other dims being reduced, the identity is more complex)
        if len(norm_sum_dims) != 1 or norm_sum_dims[0] != vocab_dim:
            # Multi-dim reduction - the identity still holds if vocab_dim is one
            # of them, but we need to be careful. For now, only handle single-dim.
            continue

        # Compute the algebraic result:
        # sum(where(one_hot, a, b) * scale, dim=vocab) = a*scale + b*scale*(V-1)
        # When b=0: result = a * scale, broadcast along all non-vocab dims
        #
        # The result shape: same as sum_input_shape but with vocab_dim reduced
        # (kept as 1 if keepdim=True, removed otherwise)

        sum_val = sum_node.meta.get("val")
        if sum_val is None:
            continue
        result_shape = list(sum_val.shape)
        result_dtype = sum_val.dtype
        result_device = sum_val.device

        with graph.inserting_before(sum_node):
            if where_b == 0.0:
                # Simple case: result = a * scale per row
                # scale_node is the per-row scale tensor (e.g., [batch, 1])
                if scale_node is None:
                    # No scale: result is just `a` broadcast
                    # Create a full tensor of value `a`
                    new_result = graph.call_function(
                        aten.full.default,
                        (result_shape,),
                        {"fill_value": where_a, "dtype": result_dtype,
                         "layout": torch.strided, "device": result_device,
                         "pin_memory": False},
                    )
                    new_result.meta["val"] = torch.full(
                        result_shape, where_a, dtype=result_dtype, device=result_device
                    )
                elif isinstance(scale_node, fx.Node):
                    # Result = a * scale_node, with appropriate shape
                    # First create a * scale
                    if where_a == 1.0:
                        scaled = scale_node
                    elif where_a == -1.0:
                        scaled = graph.call_function(
                            aten.neg.default,
                            (scale_node,),
                        )
                        scale_shape = _get_node_shape(scale_node)
                        if scale_shape is not None:
                            scaled.meta["val"] = torch.empty(
                                scale_shape, dtype=result_dtype, device=result_device
                            )
                        else:
                            scaled.meta["val"] = torch.empty(
                                result_shape, dtype=result_dtype, device=result_device
                            )
                    else:
                        scaled = graph.call_function(
                            aten.mul.Scalar,
                            (scale_node, where_a),
                        )
                        scale_shape = _get_node_shape(scale_node)
                        if scale_shape is not None:
                            scaled.meta["val"] = torch.empty(
                                scale_shape, dtype=result_dtype, device=result_device
                            )
                        else:
                            scaled.meta["val"] = torch.empty(
                                result_shape, dtype=result_dtype, device=result_device
                            )

                    # Now reshape/broadcast to result_shape if needed
                    scaled_shape = _get_node_shape(scaled)
                    if scaled_shape is not None and scaled_shape != result_shape:
                        new_result = graph.call_function(
                            aten.expand.default,
                            (scaled, result_shape),
                        )
                        new_result.meta["val"] = torch.empty(
                            result_shape, dtype=result_dtype, device=result_device
                        )
                    else:
                        new_result = scaled
                        if not hasattr(new_result, 'meta') or 'val' not in new_result.meta:
                            new_result.meta["val"] = torch.empty(
                                result_shape, dtype=result_dtype, device=result_device
                            )
                else:
                    # scale_node is a scalar literal
                    fill_val = where_a * float(scale_node)
                    new_result = graph.call_function(
                        aten.full.default,
                        (result_shape,),
                        {"fill_value": fill_val, "dtype": result_dtype,
                         "layout": torch.strided, "device": result_device,
                         "pin_memory": False},
                    )
                    new_result.meta["val"] = torch.full(
                        result_shape, fill_val, dtype=result_dtype, device=result_device
                    )
            else:
                # General case: result = a*scale + b*scale*(V-1) per row
                # = scale * (a + b*(V-1))
                combined_const = where_a + where_b * (vocab_size - 1)
                if scale_node is None:
                    new_result = graph.call_function(
                        aten.full.default,
                        (result_shape,),
                        {"fill_value": combined_const, "dtype": result_dtype,
                         "layout": torch.strided, "device": result_device,
                         "pin_memory": False},
                    )
                    new_result.meta["val"] = torch.full(
                        result_shape, combined_const, dtype=result_dtype, device=result_device
                    )
                elif isinstance(scale_node, fx.Node):
                    scaled = graph.call_function(
                        aten.mul.Scalar,
                        (scale_node, combined_const),
                    )
                    scale_shape = _get_node_shape(scale_node)
                    if scale_shape is not None:
                        scaled.meta["val"] = torch.empty(
                            scale_shape, dtype=result_dtype, device=result_device
                        )
                    else:
                        scaled.meta["val"] = torch.empty(
                            result_shape, dtype=result_dtype, device=result_device
                        )

                    scaled_shape = _get_node_shape(scaled)
                    if scaled_shape is not None and scaled_shape != result_shape:
                        new_result = graph.call_function(
                            aten.expand.default,
                            (scaled, result_shape),
                        )
                        new_result.meta["val"] = torch.empty(
                            result_shape, dtype=result_dtype, device=result_device
                        )
                    else:
                        new_result = scaled
                else:
                    fill_val = combined_const * float(scale_node)
                    new_result = graph.call_function(
                        aten.full.default,
                        (result_shape,),
                        {"fill_value": fill_val, "dtype": result_dtype,
                         "layout": torch.strided, "device": result_device,
                         "pin_memory": False},
                    )
                    new_result.meta["val"] = torch.full(
                        result_shape, fill_val, dtype=result_dtype, device=result_device
                    )

            # Ensure final result has correct metadata
            if not hasattr(new_result, 'meta') or 'val' not in new_result.meta:
                new_result.meta["val"] = torch.empty(
                    result_shape, dtype=result_dtype, device=result_device
                )

        sum_node.replace_all_uses_with(new_result)
        num_eliminations += 1
        counters["inductor"]["one_hot_reduction_eliminations"] += 1
        log.debug(
            "one_hot_reduction_elimination: replaced sum(where(eq(label,iota(%d)),%s,%s)*scale, dim=%d) -> scalar",
            vocab_size, where_a, where_b, vocab_dim,
        )

    if num_eliminations > 0:
        graph.eliminate_dead_code()
        graph.lint()
        log.info(
            "one_hot_reduction_elimination: eliminated %d one-hot reduction patterns",
            num_eliminations,
        )

    return graph
