# mypy: allow-untyped-defs
"""
Cat-Through-Reduction FX Pass.

Decomposes reductions over concatenated tensors into per-arm reductions:

    reduce(cat([a, b, ...], dim=D), reduce_dims)
        -> cat([reduce(a, reduce_dims), reduce(b, reduce_dims), ...], dim=adjusted_D)

This is valid when D is NOT in reduce_dims (the cat dimension is preserved
through the reduction).

WHY IT HELPS: Without this, Inductor materializes the full concatenated tensor
(e.g., 67MB for 6-branch Inception) to DRAM, then reads it back for the
reduction. With the decomposition, each branch reduces independently (staying
in registers or L2 cache) and only the small reduced results get concatenated.

Handles:
- aten.mean.dim, aten.sum.dim, aten.amax, aten.amin
- keepdim=True and keepdim=False
- Nested cats (recursively)
- Dim adjustment when keepdim=False (dims below the cat dim that are reduced
  cause the cat dim index to shift down)
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten

# Reduction ops we handle: maps target -> whether it has a keepdim arg
_REDUCTION_OPS = {
    aten.mean.dim: True,
    aten.sum.dim_IntList: True,
    aten.amax.default: True,
    aten.amin.default: True,
}


def _normalize_dims(dims: list[int], ndim: int) -> list[int]:
    """Normalize negative dims to positive and sort."""
    return sorted(d % ndim for d in dims)


def _get_cat_node(node: fx.Node) -> Optional[tuple[fx.Node, list[fx.Node], int]]:
    """If node is a cat op, return (cat_node, inputs_list, cat_dim).

    Handles both aten.cat.default(tensors, dim) forms.
    """
    if node.op != "call_function":
        return None
    if node.target != aten.cat.default:
        return None

    args = node.args
    if len(args) < 1:
        return None

    # First arg is the list of tensors
    tensors_arg = args[0]
    if not isinstance(tensors_arg, (list, tuple)):
        return None

    # Second arg is dim (default 0)
    cat_dim = args[1] if len(args) > 1 else 0
    if not isinstance(cat_dim, int):
        return None

    # All items in the list must be fx.Nodes
    tensor_nodes = list(tensors_arg)
    if not all(isinstance(t, fx.Node) for t in tensor_nodes):
        return None

    return node, tensor_nodes, cat_dim


def _get_tensor_ndim(node: fx.Node) -> Optional[int]:
    """Get the ndim from a node's fake tensor metadata."""
    if node.meta and "val" in node.meta:
        val = node.meta["val"]
        if hasattr(val, "ndim"):
            return val.ndim
    return None


def _compute_adjusted_cat_dim(cat_dim: int, reduce_dims: list[int], keepdim: bool) -> int:
    """Compute the cat dim in the output after reduction.

    When keepdim=True, dimension indices don't change.
    When keepdim=False, reduced dims below the cat dim shift it down.
    """
    if keepdim:
        return cat_dim
    # Count how many reduced dims are below the cat dim
    shift = sum(1 for d in reduce_dims if d < cat_dim)
    return cat_dim - shift


def _is_sole_user(cat_node: fx.Node) -> bool:
    """Check that the cat node is only used by the reduction (sole consumer).

    This ensures we don't pessimize cases where the cat result is also used
    elsewhere (in which case the cat must be materialized anyway).
    """
    return len(cat_node.users) == 1


def _try_decompose_reduction(node: fx.Node, graph: fx.Graph) -> bool:
    """Try to decompose a reduction over a cat into per-arm reductions.

    Returns True if the decomposition was applied.
    """
    if node.op != "call_function":
        return False
    if node.target not in _REDUCTION_OPS:
        return False

    args = node.args
    kwargs = node.kwargs

    # Get the input tensor and reduction dims
    if len(args) < 2:
        return False

    input_node = args[0]
    reduce_dims_arg = args[1]

    if not isinstance(input_node, fx.Node):
        return False
    if not isinstance(reduce_dims_arg, (list, tuple)):
        return False

    reduce_dims = list(reduce_dims_arg)

    # Get keepdim
    keepdim = False
    if len(args) > 2:
        keepdim = bool(args[2])
    elif "keepdim" in kwargs:
        keepdim = bool(kwargs["keepdim"])

    # Check if input is a cat
    cat_info = _get_cat_node(input_node)
    if cat_info is None:
        return False

    cat_node, cat_inputs, cat_dim = cat_info

    # Get ndim from the cat output
    ndim = _get_tensor_ndim(cat_node)
    if ndim is None:
        return False

    # Normalize cat_dim
    cat_dim_normalized = cat_dim % ndim

    # Normalize reduce_dims
    reduce_dims_normalized = _normalize_dims(reduce_dims, ndim)

    # Check the key condition: cat dim must NOT be in the reduce dims
    if cat_dim_normalized in reduce_dims_normalized:
        return False

    # Check that the cat is only used by this reduction (no other consumers).
    # If the cat has multiple users, materializing it is unavoidable, so the
    # decomposition wouldn't help and could hurt by creating more kernels.
    if not _is_sole_user(cat_node):
        return False

    # All checks pass - do the decomposition
    adjusted_cat_dim = _compute_adjusted_cat_dim(
        cat_dim_normalized, reduce_dims_normalized, keepdim
    )

    # Insert per-arm reductions before the current node
    with graph.inserting_before(node):
        reduced_arms = []
        for arm_node in cat_inputs:
            # Build the reduction call for this arm
            # Use the same args/kwargs pattern as the original reduction
            new_args: list = [arm_node, reduce_dims_arg]
            if len(args) > 2:
                new_args.append(args[2])  # keepdim positional arg

            new_node = graph.call_function(
                node.target,
                args=tuple(new_args),
                kwargs=dict(kwargs),
            )

            # Copy over metadata for the new reduction node
            # We need to compute the expected output shape
            arm_val = arm_node.meta.get("val", None) if arm_node.meta else None
            if arm_val is not None and hasattr(arm_val, "shape"):
                arm_shape = list(arm_val.shape)
                if keepdim:
                    out_shape = list(arm_shape)
                    for d in reduce_dims_normalized:
                        out_shape[d] = 1
                else:
                    out_shape = [
                        s for i, s in enumerate(arm_shape)
                        if i not in reduce_dims_normalized
                    ]
                new_node.meta["val"] = torch.empty(
                    out_shape,
                    dtype=arm_val.dtype,
                    device="meta",
                )
            reduced_arms.append(new_node)

        # Create the new cat over reduced arms
        new_cat = graph.call_function(
            aten.cat.default,
            args=(reduced_arms, adjusted_cat_dim),
        )

        # Copy metadata from the original reduction output
        if node.meta and "val" in node.meta:
            new_cat.meta["val"] = node.meta["val"]

    # Replace all uses of the original reduction with the new cat
    node.replace_all_uses_with(new_cat)

    # Remove the old reduction node and (now dead) cat node
    graph.erase_node(node)
    if len(cat_node.users) == 0:
        graph.erase_node(cat_node)

    counters["inductor"]["cat_through_reduction"] += 1
    log.debug(
        "cat_through_reduction: decomposed %s over %d-arm cat (dim=%d, reduce_dims=%s)",
        node.target,
        len(cat_inputs),
        cat_dim_normalized,
        reduce_dims_normalized,
    )
    return True


def cat_through_reduction_pass(graph: fx.Graph) -> Optional[fx.Graph]:
    """Main pass: find reductions over cats and decompose them.

    Returns the modified graph, or None if no changes were made.
    Iterates to fixed point to handle nested cats (e.g., cat(cat(a,b), c)).
    """
    if not config.cat_through_reduction:
        return None

    changed = False

    # Iterate to fixed point: decomposing an outer cat+reduce creates new
    # reduce nodes over inner cats which may themselves be decomposable.
    while True:
        iteration_changed = False
        nodes = list(graph.nodes)
        for node in reversed(nodes):
            if node.op != "call_function":
                continue
            if node.target not in _REDUCTION_OPS:
                continue
            # Node may have been erased by a previous iteration
            if node.graph is not graph:
                continue
            if _try_decompose_reduction(node, graph):
                iteration_changed = True
        if not iteration_changed:
            break
        changed = True

    if changed:
        graph.lint()
        return graph
    return None
