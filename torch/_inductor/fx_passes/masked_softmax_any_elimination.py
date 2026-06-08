# mypy: allow-untyped-defs
"""
Masked Softmax Any-Reduction Elimination Pass.

Eliminates a redundant any-reduction in the masked attention softmax pattern.

Pattern:
    mask: bool tensor with stride-zero along reduction dimension (constant per row)
    bias = where(mask, 0.0, -inf)
    scores = x + bias                    # [B, H, Q, K]
    eq = scores == -inf                  # True where masked
    not_eq = logical_not(eq)             # True where not masked
    any_valid = any(not_eq, dim=-1)      # reduction over K (the mask-constant dim)

Since mask is constant along K (stride=0 in that dimension), the result of
any(not_eq, dim=-1) is trivially equal to the mask value itself (True if
mask=True for any element along K, which means all elements since stride=0).

Simplification:
    any_valid = mask.any(dim=reduction_dim, keepdim=True)

Since mask has stride-zero along the reduction dim, mask.any() along that dim
is just the mask value at any single index (e.g., index 0) along that dim.
This eliminates a warp-shuffle reduction from the fused kernel.
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_node_val(node: fx.Node):
    """Get the FakeTensor value from node metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val
    return None


def _get_node_shape(node: fx.Node) -> Optional[list[int]]:
    """Get static shape from node's fake tensor metadata."""
    val = _get_node_val(node)
    if val is None:
        return None
    shape = val.shape
    result = []
    for s in shape:
        if isinstance(s, int):
            result.append(s)
        else:
            return None
    return result


def _get_node_stride(node: fx.Node) -> Optional[list[int]]:
    """Get stride from node's fake tensor metadata."""
    val = _get_node_val(node)
    if val is None:
        return None
    stride = val.stride()
    result = []
    for s in stride:
        if isinstance(s, int):
            result.append(s)
        else:
            return None
    return result


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize a potentially negative dimension."""
    if dim < 0:
        dim += ndim
    return dim


def _trace_to_where_mask(any_input_node: fx.Node, reduce_dim: int):
    """
    Trace backward from the any() input to find the masked-softmax pattern.

    Expected chain:
        mask (bool, stride-zero along reduce_dim)
        where(mask, 0.0, -inf) -> bias
        add(x, bias) -> scores
        eq(scores, -inf)
        logical_not(eq)  <-- this is any_input_node

    Returns the mask node if the pattern matches, None otherwise.
    """
    # Step 1: any_input should be logical_not(eq_node)
    if not (any_input_node.op == "call_function" and
            any_input_node.target is aten.logical_not.default):
        return None

    eq_node = any_input_node.args[0]
    if not isinstance(eq_node, fx.Node):
        return None

    # Step 2: eq_node should be eq(add_node, -inf) or eq.Scalar(add_node, -inf)
    if not (eq_node.op == "call_function" and
            eq_node.target in (aten.eq.Scalar, aten.eq.Tensor)):
        return None

    add_node = eq_node.args[0]
    eq_val = eq_node.args[1]

    # Check that the comparison value is -inf
    if isinstance(eq_val, fx.Node):
        # eq.Tensor case: check if the tensor is a scalar -inf
        eq_tensor_val = _get_node_val(eq_val)
        if eq_tensor_val is None or eq_tensor_val.numel() != 1:
            return None
        # Can't easily check the actual value of a fake tensor
        return None
    elif isinstance(eq_val, (int, float)):
        if eq_val != float("-inf"):
            return None
    else:
        return None

    if not isinstance(add_node, fx.Node):
        return None

    # Step 3: add_node should be add(view_node, where_node)
    if not (add_node.op == "call_function" and
            add_node.target in (aten.add.Tensor, aten.add.Scalar)):
        return None

    # One of the add inputs should be a where(mask, 0, -inf)
    where_node = None
    for arg in add_node.args[:2]:
        if not isinstance(arg, fx.Node):
            continue
        if arg.op == "call_function" and arg.target is aten.where.self:
            where_node = arg
            break

    if where_node is None:
        return None

    # Step 4: where_node should be where(mask, 0.0, -inf)
    if len(where_node.args) < 3:
        return None

    mask_node = where_node.args[0]
    true_val_node = where_node.args[1]
    false_val_node = where_node.args[2]

    if not isinstance(mask_node, fx.Node):
        return None

    # Check true_val is 0.0 and false_val is -inf
    # They might be scalar constants or full() tensors
    def _is_zero(node_or_val):
        if isinstance(node_or_val, (int, float)):
            return node_or_val == 0.0
        if isinstance(node_or_val, fx.Node) and node_or_val.op == "call_function":
            if node_or_val.target is aten.full.default:
                # full([], 0.0, ...)
                if len(node_or_val.args) >= 2:
                    return node_or_val.args[1] == 0.0
        return False

    def _is_neg_inf(node_or_val):
        if isinstance(node_or_val, (int, float)):
            return node_or_val == float("-inf")
        if isinstance(node_or_val, fx.Node) and node_or_val.op == "call_function":
            if node_or_val.target is aten.full.default:
                if len(node_or_val.args) >= 2:
                    return node_or_val.args[1] == float("-inf")
        return False

    if not (_is_zero(true_val_node) and _is_neg_inf(false_val_node)):
        return None

    # Step 5: Check that mask has stride-zero along the reduction dimension
    mask_stride = _get_node_stride(mask_node)
    mask_shape = _get_node_shape(mask_node)

    if mask_stride is None or mask_shape is None:
        return None

    # The mask might have fewer dims than the add result due to broadcasting.
    # We need to figure out which dim of the mask corresponds to reduce_dim.
    add_shape = _get_node_shape(add_node)
    if add_shape is None:
        return None

    ndim_add = len(add_shape)
    ndim_mask = len(mask_shape)

    # Normalize reduce_dim relative to add's ndim
    norm_reduce_dim = _normalize_dim(reduce_dim, ndim_add)

    # Map reduce_dim to mask's dimensions (accounting for broadcasting alignment)
    # Broadcasting aligns from the right
    dim_offset = ndim_add - ndim_mask
    mask_reduce_dim = norm_reduce_dim - dim_offset

    if mask_reduce_dim < 0:
        # The mask doesn't even have this dimension (it's broadcast from a scalar)
        # In that case, the mask is constant along reduce_dim - pattern matches!
        return mask_node

    if mask_stride[mask_reduce_dim] == 0:
        # Stride-zero along the reduction dimension - mask is constant per row
        return mask_node

    # Also check: if mask_shape[mask_reduce_dim] == 1, it's broadcast (size-1)
    if mask_shape[mask_reduce_dim] == 1:
        return mask_node

    return None


def masked_softmax_any_elimination_pass(graph: fx.Graph) -> Optional[fx.Graph]:
    """
    Eliminate redundant any() reductions in masked attention softmax patterns.

    Finds: any(logical_not(eq(x + where(mask, 0, -inf), -inf)), dim, keepdim)
    where mask has stride-zero along dim (constant per row).

    Replaces with: mask expanded/reshaped to the output shape of any().

    This saves one warp-shuffle reduction in the fused persistent kernel.
    """
    if not getattr(config, "masked_softmax_any_elimination", True):
        return None

    num_eliminations = 0

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not aten.any.dim:
            continue

        any_node = node
        if len(any_node.args) < 2:
            continue

        any_input = any_node.args[0]
        reduce_dim = any_node.args[1]
        keepdim = any_node.args[2] if len(any_node.args) > 2 else False

        if not isinstance(any_input, fx.Node):
            continue
        if not isinstance(reduce_dim, int):
            continue

        # Get the output shape of the any node
        any_val = _get_node_val(any_node)
        if any_val is None:
            continue
        any_shape = list(any_val.shape)

        # Try to trace back to the mask
        mask_node = _trace_to_where_mask(any_input, reduce_dim)
        if mask_node is None:
            continue

        # We found the pattern! Replace any(logical_not(eq(add(x, where(mask,0,-inf)), -inf)), dim)
        # with an equivalent expression that doesn't require a reduction.
        #
        # Since mask is constant along the reduction dim:
        #   any(not(x + where(mask, 0, -inf) == -inf), dim) = mask broadcasted to output shape
        #
        # More specifically: for each row (b, h, q):
        #   - If mask[b, ..., q, :] = True (all k, since stride-0): bias=0, scores=finite,
        #     not_eq=True for all k, any=True
        #   - If mask[b, ..., q, :] = False: bias=-inf, scores=-inf,
        #     not_eq=False for all k, any=False
        #
        # So any() = mask (with appropriate broadcasting).

        mask_val = _get_node_val(mask_node)
        if mask_val is None:
            continue
        mask_shape = list(mask_val.shape)

        # We need to produce a bool tensor with shape any_shape from the mask.
        # The mask.any(dim=reduce_dim, keepdim) is trivially equal to the mask
        # since mask is constant along that dim. But we need to handle shape
        # differences due to broadcasting (mask may have shape [B,1,Q,K] while
        # the any output has shape [B,H,Q,1]).

        with graph.inserting_before(any_node):
            # Strategy: slice the mask to get one element along the reduction dim,
            # then expand to the any output shape.
            #
            # The mask has stride-zero or size-1 along the reduction dim, so
            # any element along that dim gives the same value.

            ndim_mask = len(mask_shape)
            any_input_shape = _get_node_shape(any_input)
            if any_input_shape is None:
                continue
            ndim_input = len(any_input_shape)
            dim_offset = ndim_input - ndim_mask
            mask_reduce_dim = _normalize_dim(reduce_dim, ndim_input) - dim_offset

            # Select index 0 along the reduction dim to collapse it
            if mask_reduce_dim >= 0 and mask_reduce_dim < ndim_mask:
                if mask_shape[mask_reduce_dim] > 1:
                    # Need to slice: mask[:, :, :, 0:1] to get keepdim shape
                    # Use select + unsqueeze or narrow
                    sliced = graph.call_function(
                        aten.narrow.default,
                        (mask_node, mask_reduce_dim, 0, 1),
                    )
                    # Compute the shape for the sliced result
                    sliced_shape = list(mask_shape)
                    sliced_shape[mask_reduce_dim] = 1
                    sliced.meta["val"] = torch.empty(
                        sliced_shape, dtype=torch.bool,
                        device=mask_val.device,
                    )
                    current = sliced
                    current_shape = sliced_shape
                else:
                    # mask_shape[mask_reduce_dim] is already 1
                    current = mask_node
                    current_shape = list(mask_shape)
            elif mask_reduce_dim < 0:
                # Mask doesn't have this dim, it's implicitly broadcast
                # The mask value applies to all elements along reduce_dim
                current = mask_node
                current_shape = list(mask_shape)
            else:
                continue

            # Now we need to reshape/expand `current` to match any_shape.
            # The any output shape is the input shape with reduce_dim collapsed to 1 (if keepdim).
            # We need to handle the head dimension broadcasting (mask has H=1, output has H>1).

            # Build target shape for expand
            # First, unsqueeze to match ndim of output if needed
            target_ndim = len(any_shape)
            while len(current_shape) < target_ndim:
                current = graph.call_function(
                    aten.unsqueeze.default, (current, 0)
                )
                current_shape = [1] + current_shape
                current.meta["val"] = torch.empty(
                    current_shape, dtype=torch.bool,
                    device=mask_val.device,
                )

            # If keepdim=False, we need to squeeze the reduce dim out
            if not keepdim:
                # The any output doesn't have the reduced dim
                # But current still has it (as size 1)
                squeeze_dim = _normalize_dim(reduce_dim, target_ndim + 1)
                # Actually, if keepdim is False, any_shape has one fewer dim
                # We need to squeeze current to match
                if len(current_shape) > len(any_shape):
                    current = graph.call_function(
                        aten.squeeze.dim, (current, squeeze_dim)
                    )
                    current_shape = list(current_shape)
                    del current_shape[squeeze_dim]
                    current.meta["val"] = torch.empty(
                        current_shape, dtype=torch.bool,
                        device=mask_val.device,
                    )

            # Now expand to match any_shape exactly
            if current_shape != any_shape:
                expanded = graph.call_function(
                    aten.expand.default, (current, any_shape)
                )
                expanded.meta["val"] = torch.empty(
                    any_shape, dtype=torch.bool,
                    device=mask_val.device,
                )
                current = expanded

            # Replace any_node with current
            any_node.replace_all_uses_with(current)
            num_eliminations += 1
            counters["inductor"]["masked_softmax_any_elimination"] += 1
            log.debug(
                "Eliminated any() reduction: %s -> direct mask reference",
                any_node.name,
            )

    if num_eliminations > 0:
        graph.eliminate_dead_code()
        graph.lint()
        log.info(
            "masked_softmax_any_elimination: eliminated %d any-reduction(s)",
            num_eliminations,
        )
        return graph

    return None
