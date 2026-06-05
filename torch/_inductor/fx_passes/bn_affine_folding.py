# mypy: allow-untyped-defs
"""
Batch Normalization Inference Affine Folding Pass.

Folds the decomposed BN-inference graph pattern:
    sub(x, mean[C]) -> mul(result, rsqrt(var[C]+eps)) -> mul(result, weight[C]) -> add(result, bias[C])

Into a precomputed affine transform:
    scale[C] = weight[C] * rsqrt(var[C] + eps)
    shift[C] = bias[C] - mean[C] * scale[C]
    output = x * scale[C] + shift[C]

This reduces:
- 4 per-channel parameter loads (mean, var, weight, bias) to 2 (scale, shift)
- 5 ops per element (sub, rsqrt, mul, mul, add) to 2 ops (mul, add)

The transformation is numerically equivalent (same evaluation order for the
critical x*scale+shift computation).

Gated behind config.fold_bn_affine = True (default enabled).
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _is_op(node: fx.Node, target) -> bool:
    """Check if a node calls a specific aten op."""
    return node.op == "call_function" and node.target == target


def _get_shape(node: fx.Node) -> Optional[list[int]]:
    """Get shape from node's fake tensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return list(val.shape)
    return None


def _is_single_use(node: fx.Node) -> bool:
    """Check if a node has exactly one user."""
    return len(node.users) == 1


def _trace_unsqueeze_chain(node: fx.Node):
    """
    If node is unsqueeze(unsqueeze(x, -1), -1), return x.
    This matches the pattern where a [C] tensor is broadcast to [C, 1, 1].
    """
    if not _is_op(node, aten.unsqueeze.default):
        return None
    if node.args[1] != -1:
        return None
    inner = node.args[0]
    if not isinstance(inner, fx.Node):
        return None
    if not _is_op(inner, aten.unsqueeze.default):
        return None
    if inner.args[1] != -1:
        return None
    return inner.args[0]


def _match_bn_affine_chain(add_node: fx.Node):
    """
    Match the decomposed BN-inference affine chain ending at an add node.

    Pattern (reading backwards from add_node):
        add(mul(mul(sub(x, unsqueeze(unsqueeze(mean))), unsqueeze(unsqueeze(inv_std))), unsqueeze(unsqueeze(weight))), unsqueeze(unsqueeze(bias)))

    Where inv_std is one of:
        - rsqrt(add(var, eps))
        - mul(reciprocal(sqrt(add(var, eps))), 1)  [before canonicalization]
        - reciprocal(sqrt(add(var, eps)))  [partially canonicalized]

    Returns: (x, mean, var_or_inv_std, weight, bias, eps, all_intermediate_nodes) or None
    """
    if not _is_op(add_node, aten.add.Tensor):
        return None

    # add(scaled, unsqueeze(unsqueeze(bias)))
    mul_weight_node = add_node.args[0]
    bias_unsqueezed = add_node.args[1]

    if not isinstance(mul_weight_node, fx.Node) or not isinstance(bias_unsqueezed, fx.Node):
        return None

    # Trace bias: unsqueeze(unsqueeze(bias_param, -1), -1)
    bias = _trace_unsqueeze_chain(bias_unsqueezed)
    if bias is None or not isinstance(bias, fx.Node):
        return None

    # mul(normalized, unsqueeze(unsqueeze(weight)))
    if not _is_op(mul_weight_node, aten.mul.Tensor):
        return None

    mul_invstd_node = mul_weight_node.args[0]
    weight_unsqueezed = mul_weight_node.args[1]

    if not isinstance(mul_invstd_node, fx.Node) or not isinstance(weight_unsqueezed, fx.Node):
        return None

    # Trace weight
    weight = _trace_unsqueeze_chain(weight_unsqueezed)
    if weight is None or not isinstance(weight, fx.Node):
        return None

    # mul(sub_result, unsqueeze(unsqueeze(inv_std)))
    if not _is_op(mul_invstd_node, aten.mul.Tensor):
        return None

    sub_node = mul_invstd_node.args[0]
    invstd_unsqueezed = mul_invstd_node.args[1]

    if not isinstance(sub_node, fx.Node) or not isinstance(invstd_unsqueezed, fx.Node):
        return None

    # Trace inv_std computation
    invstd_1d = _trace_unsqueeze_chain(invstd_unsqueezed)
    if invstd_1d is None or not isinstance(invstd_1d, fx.Node):
        return None

    # inv_std could be:
    # Case 1: rsqrt(add(var, eps))  [after full canonicalization]
    # Case 2: mul(rsqrt(add(var, eps)), 1) [after rsqrt canon but mul-by-1 remains]
    # Case 3: mul(reciprocal(sqrt(add(var, eps))), 1) [before canonicalization]
    # Case 4: reciprocal(sqrt(add(var, eps)))
    var_node = None
    eps_val = None
    invstd_chain_nodes = []

    if _is_op(invstd_1d, aten.rsqrt.default):
        # Case 1: rsqrt(add(var, eps))
        add_var_eps = invstd_1d.args[0]
        invstd_chain_nodes.append(invstd_1d)
        if isinstance(add_var_eps, fx.Node) and _is_op(add_var_eps, aten.add.Tensor):
            var_node = add_var_eps.args[0]
            eps_val = add_var_eps.args[1]
            invstd_chain_nodes.append(add_var_eps)
    elif _is_op(invstd_1d, aten.mul.Tensor):
        # Case 2 or 3: mul(something, 1)
        inner = invstd_1d.args[0]
        mul_const = invstd_1d.args[1]
        if not isinstance(mul_const, (int, float)) or mul_const != 1:
            return None
        invstd_chain_nodes.append(invstd_1d)

        if isinstance(inner, fx.Node) and _is_op(inner, aten.rsqrt.default):
            # Case 2: mul(rsqrt(add(var, eps)), 1)
            add_var_eps = inner.args[0]
            invstd_chain_nodes.append(inner)
            if isinstance(add_var_eps, fx.Node) and _is_op(add_var_eps, aten.add.Tensor):
                var_node = add_var_eps.args[0]
                eps_val = add_var_eps.args[1]
                invstd_chain_nodes.append(add_var_eps)
        elif isinstance(inner, fx.Node) and _is_op(inner, aten.reciprocal.default):
            # Case 3: mul(reciprocal(sqrt(add(var, eps))), 1)
            sqrt_node = inner.args[0]
            invstd_chain_nodes.append(inner)
            if isinstance(sqrt_node, fx.Node) and _is_op(sqrt_node, aten.sqrt.default):
                add_var_eps = sqrt_node.args[0]
                invstd_chain_nodes.append(sqrt_node)
                if isinstance(add_var_eps, fx.Node) and _is_op(add_var_eps, aten.add.Tensor):
                    var_node = add_var_eps.args[0]
                    eps_val = add_var_eps.args[1]
                    invstd_chain_nodes.append(add_var_eps)
    elif _is_op(invstd_1d, aten.reciprocal.default):
        # Case 4: reciprocal(sqrt(add(var, eps)))
        sqrt_node = invstd_1d.args[0]
        invstd_chain_nodes.append(invstd_1d)
        if isinstance(sqrt_node, fx.Node) and _is_op(sqrt_node, aten.sqrt.default):
            add_var_eps = sqrt_node.args[0]
            invstd_chain_nodes.append(sqrt_node)
            if isinstance(add_var_eps, fx.Node) and _is_op(add_var_eps, aten.add.Tensor):
                var_node = add_var_eps.args[0]
                eps_val = add_var_eps.args[1]
                invstd_chain_nodes.append(add_var_eps)

    if var_node is None or eps_val is None:
        return None

    if not isinstance(var_node, fx.Node):
        return None

    # sub(x, unsqueeze(unsqueeze(mean)))
    if not _is_op(sub_node, aten.sub.Tensor):
        return None

    x = sub_node.args[0]
    mean_unsqueezed = sub_node.args[1]

    if not isinstance(x, fx.Node) or not isinstance(mean_unsqueezed, fx.Node):
        return None

    mean = _trace_unsqueeze_chain(mean_unsqueezed)
    if mean is None or not isinstance(mean, fx.Node):
        return None

    # Validate shapes: mean, var, weight, bias should be 1D [C]
    # x should be high-dimensional (at least 3D for broadcast benefit)
    x_shape = _get_shape(x)
    mean_shape = _get_shape(mean)
    var_shape = _get_shape(var_node)
    weight_shape = _get_shape(weight)
    bias_shape = _get_shape(bias)

    if x_shape is None or mean_shape is None:
        return None

    # x should be at least 3D (e.g., [N, C, H, W]) for this to be beneficial
    if len(x_shape) < 3:
        return None

    # Channel params should be 1D
    if mean_shape is not None and len(mean_shape) != 1:
        return None
    if var_shape is not None and len(_get_shape(var_node)) != 1:
        return None
    if weight_shape is not None and len(weight_shape) != 1:
        return None
    if bias_shape is not None and len(bias_shape) != 1:
        return None

    # Collect all intermediate nodes that will become dead after replacement
    intermediate_nodes = set()
    # The unsqueeze chains for mean, inv_std, weight, bias
    intermediate_nodes.add(mean_unsqueezed)
    intermediate_nodes.add(mean_unsqueezed.args[0])  # inner unsqueeze
    intermediate_nodes.add(invstd_unsqueezed)
    intermediate_nodes.add(invstd_unsqueezed.args[0])  # inner unsqueeze
    intermediate_nodes.add(weight_unsqueezed)
    intermediate_nodes.add(weight_unsqueezed.args[0])  # inner unsqueeze
    intermediate_nodes.add(bias_unsqueezed)
    intermediate_nodes.add(bias_unsqueezed.args[0])  # inner unsqueeze
    # The computation chain
    intermediate_nodes.add(sub_node)
    intermediate_nodes.add(mul_invstd_node)
    intermediate_nodes.add(mul_weight_node)
    intermediate_nodes.add(invstd_1d)
    for n in invstd_chain_nodes:
        intermediate_nodes.add(n)

    return {
        "x": x,
        "mean": mean,
        "var": var_node,
        "weight": weight,
        "bias": bias,
        "eps": eps_val,
        "intermediate_nodes": intermediate_nodes,
        "add_node": add_node,
    }


def _check_intermediates_single_use(match_info: dict) -> bool:
    """
    Check that all intermediate nodes in the BN chain are single-use.
    If any intermediate has other users, we can't safely eliminate them.
    """
    # The add_node (output) can have multiple users - that's fine
    for node in match_info["intermediate_nodes"]:
        if isinstance(node, fx.Node) and not _is_single_use(node):
            return False
    return True


def bn_affine_folding_pass(graph: torch.fx.Graph):
    """
    Main pass: find decomposed BN-inference patterns and fold them into
    precomputed scale/shift affine transforms.

    For each match:
        sub(x, mean) -> mul(_, inv_std) -> mul(_, weight) -> add(_, bias)
    becomes:
        scale = weight * rsqrt(var + eps)        [computed at graph level, 1D]
        shift = bias - mean * scale              [computed at graph level, 1D]
        output = x * unsqueeze(scale) + unsqueeze(shift)   [2 ops per element]
    """
    if not config.fold_bn_affine:
        return

    log.debug("bn_affine_folding_pass: scanning graph")

    replacements = 0

    # Find all add nodes that could be the end of a BN chain
    add_nodes = [
        n for n in graph.nodes
        if n.op == "call_function" and n.target == aten.add.Tensor
    ]

    nodes_to_erase = set()

    for add_node in add_nodes:
        if add_node in nodes_to_erase:
            continue

        match_info = _match_bn_affine_chain(add_node)
        if match_info is None:
            continue

        if not _check_intermediates_single_use(match_info):
            continue

        x = match_info["x"]
        mean = match_info["mean"]
        var_node = match_info["var"]
        weight = match_info["weight"]
        bias = match_info["bias"]
        eps = match_info["eps"]

        # Insert new nodes before the add_node
        with graph.inserting_before(add_node):
            # scale = weight * rsqrt(var + eps)   [1D, shape [C]]
            var_plus_eps = graph.call_function(aten.add.Tensor, args=(var_node, eps))
            inv_std = graph.call_function(aten.rsqrt.default, args=(var_plus_eps,))
            scale = graph.call_function(aten.mul.Tensor, args=(weight, inv_std))

            # shift = bias - mean * scale   [1D, shape [C]]
            mean_times_scale = graph.call_function(aten.mul.Tensor, args=(mean, scale))
            shift = graph.call_function(aten.sub.Tensor, args=(bias, mean_times_scale))

            # Broadcast scale and shift to match x's spatial dims
            # The original pattern used unsqueeze(-1) twice to go from [C] to [C,1,1]
            # We must match the same broadcast shape for correctness
            scale_bc = graph.call_function(aten.unsqueeze.default, args=(scale, -1))
            scale_bc = graph.call_function(aten.unsqueeze.default, args=(scale_bc, -1))

            shift_bc = graph.call_function(aten.unsqueeze.default, args=(shift, -1))
            shift_bc = graph.call_function(aten.unsqueeze.default, args=(shift_bc, -1))

            # output = x * scale + shift   [2 ops per element]
            x_scaled = graph.call_function(aten.mul.Tensor, args=(x, scale_bc))
            output = graph.call_function(aten.add.Tensor, args=(x_scaled, shift_bc))

        # Copy metadata from original add_node to new output
        if "val" in add_node.meta:
            output.meta["val"] = add_node.meta["val"]
        if "tensor_meta" in add_node.meta:
            output.meta["tensor_meta"] = add_node.meta["tensor_meta"]

        # Also propagate metadata to intermediate new nodes from their sources
        # (needed for shape inference in later passes)
        if "val" in var_node.meta:
            var_meta = var_node.meta["val"]
            if isinstance(var_meta, torch.Tensor):
                # var_plus_eps, inv_std, scale, shift all have same shape as var
                for n in [var_plus_eps, inv_std, scale, mean_times_scale, shift]:
                    n.meta["val"] = torch.empty_like(var_meta)

        # Replace all uses of the old add_node with the new output
        add_node.replace_all_uses_with(output)

        # Mark intermediate nodes for erasure
        nodes_to_erase.update(match_info["intermediate_nodes"])
        nodes_to_erase.add(add_node)

        replacements += 1
        log.debug("bn_affine_folding: folded BN chain ending at %s", add_node.name)

    # Erase dead nodes in reverse topological order
    if nodes_to_erase:
        for node in reversed(list(graph.nodes)):
            if node in nodes_to_erase and len(node.users) == 0:
                graph.erase_node(node)

    if replacements > 0:
        counters["inductor"]["bn_affine_folding"] += replacements
        log.debug("bn_affine_folding: folded %d BN-inference chains", replacements)
