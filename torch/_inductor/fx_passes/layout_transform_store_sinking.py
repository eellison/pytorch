# mypy: allow-untyped-defs
"""
Layout-transform store sinking pass.

Detects the "channel shuffle" pattern where a producer's output is consumed
solely by a layout transform (view + permute + clone + view), and rewrites
the graph to eliminate the intermediate buffer by having the producer store
directly into the consumer's output layout.

Pattern detected:
    cat(inputs, dim=D) -> view(N, groups, C_per_group, ...) ->
        permute(0, 2, 1, ...) -> clone(contiguous) -> view(final_shape)

Rewritten to:
    [unsqueeze(inp, D+1) for inp in inputs] -> cat(unsqueezed, dim=D+1) ->
        view(final_shape)

This eliminates the clone kernel entirely since the cat on dim=D+1 produces
the same memory layout as the original permute+clone, and the cat fuses with
its upstream producers.
"""

import logging
from typing import Optional

import torch
from torch import fx
from torch._dynamo.utils import counters

log = logging.getLogger(__name__)

aten = torch.ops.aten


def _get_single_user(node: fx.Node) -> Optional[fx.Node]:
    """Return the single non-output user of a node, or None if != 1 such user."""
    users = [u for u in node.users if u.op != "output"]
    if len(users) == 1:
        return users[0]
    return None


def _is_view_op(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        aten.view.default,
        aten.reshape.default,
        aten._unsafe_view.default,
    )


def _is_permute_op(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target == aten.permute.default


def _is_clone_op(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target == aten.clone.default


def _is_cat_op(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target == aten.cat.default


def _get_val(node: fx.Node):
    """Get the fake tensor value from node metadata."""
    return node.meta.get("val", None)


def _static_dim(val, dim: int) -> Optional[int]:
    """Get a static integer size for a given dimension, or None."""
    if val is None:
        return None
    size = val.shape[dim]
    if isinstance(size, int):
        return size
    # Try to get the hint for symbolic sizes
    if hasattr(size, "node") and hasattr(size.node, "hint"):
        return size.node.hint
    return None


def _is_channel_shuffle_permutation(perm: list, ndim: int, swap_dim: int = 1) -> bool:
    """
    Check if permutation swaps dims swap_dim and swap_dim+1 and keeps
    everything else in place. The standard channel shuffle pattern is
    [0, 2, 1, 3, 4, ...] which swaps dims 1 and 2.
    """
    if len(perm) != ndim:
        return False
    if ndim < swap_dim + 2:
        return False
    expected = list(range(ndim))
    expected[swap_dim] = swap_dim + 1
    expected[swap_dim + 1] = swap_dim
    return perm == expected


def layout_transform_store_sinking_pass(graph: fx.Graph) -> Optional[int]:
    """
    Detect and rewrite channel-shuffle layout transform patterns.

    Returns the number of patterns replaced, or None if no changes.
    """
    count = 0

    for node in list(graph.nodes):
        if not _is_cat_op(node):
            continue

        # cat(inputs, dim=D)
        cat_node = node
        cat_inputs = cat_node.args[0]
        if not isinstance(cat_inputs, (list, tuple)) or len(cat_inputs) < 2:
            continue

        cat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0
        if not isinstance(cat_dim, int):
            continue

        # Check cat has a single user that is a view
        view1_node = _get_single_user(cat_node)
        if view1_node is None or not _is_view_op(view1_node):
            continue

        # Check view1 has a single user that is a permute
        permute_node = _get_single_user(view1_node)
        if permute_node is None or not _is_permute_op(permute_node):
            continue

        # Check permute has a single user that is a clone
        clone_node = _get_single_user(permute_node)
        if clone_node is None or not _is_clone_op(clone_node):
            continue

        # Verify clone uses contiguous memory format
        clone_kwargs = clone_node.kwargs
        memory_format = clone_kwargs.get("memory_format", None)
        if memory_format is not None and memory_format != torch.contiguous_format:
            continue

        # Check clone has a single user that is a view
        view2_node = _get_single_user(clone_node)
        if view2_node is None or not _is_view_op(view2_node):
            continue

        # Now validate the shapes and permutation
        cat_val = _get_val(cat_node)
        view1_val = _get_val(view1_node)
        permute_val = _get_val(permute_node)
        clone_val = _get_val(clone_node)

        if cat_val is None or view1_val is None or permute_val is None:
            continue

        # Check the view reshapes to (..., groups, C_per_group, ...)
        # For channel shuffle: [N, groups*C, H, W] -> [N, groups, C, H, W]
        cat_ndim = cat_val.ndim
        view1_ndim = view1_val.ndim

        if view1_ndim != cat_ndim + 1:
            continue

        # Get the permutation
        perm = permute_node.args[1]
        if not isinstance(perm, (list, tuple)):
            continue
        perm = list(perm)

        # Check it's the channel shuffle permutation (swap dims cat_dim and cat_dim+1)
        if not _is_channel_shuffle_permutation(perm, view1_ndim, swap_dim=cat_dim):
            continue

        # Validate: the view splits cat_dim into (groups, C_per_group)
        # After view, dim cat_dim has size 'groups' and dim cat_dim+1 has size 'C_per_group'
        groups = _static_dim(view1_val, cat_dim)  # groups (first of the split dims)
        c_per_group = _static_dim(view1_val, cat_dim + 1)  # C_per_group (second of split dims)

        if groups is None or c_per_group is None:
            continue

        # Validate that cat concatenated 'groups' tensors each of size C_per_group along cat_dim
        if len(cat_inputs) != groups:
            continue

        # Verify each input contributes C_per_group channels
        all_valid = True
        for inp in cat_inputs:
            if not isinstance(inp, fx.Node):
                all_valid = False
                break
            inp_val = _get_val(inp)
            if inp_val is None:
                all_valid = False
                break
            inp_size = _static_dim(inp_val, cat_dim)
            if inp_size != c_per_group:
                all_valid = False
                break
        if not all_valid:
            continue

        # We have a valid pattern! Rewrite:
        # cat([a, b], dim=D) -> view(N, 2, C, H, W) -> permute(0, 2, 1, 3, 4) -> clone -> view
        # becomes:
        # [unsqueeze(a, D+1), unsqueeze(b, D+1)] -> cat(dim=D+1) -> view(final_shape)

        log.info(
            "layout_transform_store_sinking: rewriting channel shuffle "
            "pattern at %s (groups=%d, C_per_group=%d)",
            cat_node.name,
            groups,
            c_per_group,
        )

        with graph.inserting_before(cat_node):
            # Create unsqueeze nodes for each input
            new_cat_inputs = []
            for inp in cat_inputs:
                unsqueeze_node = graph.call_function(
                    aten.unsqueeze.default, (inp, cat_dim + 1)
                )
                # Copy metadata
                inp_val = _get_val(inp)
                if inp_val is not None:
                    unsqueeze_node.meta["val"] = inp_val.unsqueeze(cat_dim + 1)
                new_cat_inputs.append(unsqueeze_node)

            # Create new cat on dim=D+1 (the groups dimension)
            new_cat_node = graph.call_function(
                aten.cat.default, (new_cat_inputs, cat_dim + 1)
            )
            # The result shape is [N, C_per_group, groups, H, W] - same as clone output
            if clone_val is not None:
                new_cat_node.meta["val"] = clone_val

            # Create view to final shape (same as view2's output shape)
            view2_shape = view2_node.args[1]
            new_view_node = graph.call_function(
                aten.view.default, (new_cat_node, view2_shape)
            )
            view2_val = _get_val(view2_node)
            if view2_val is not None:
                new_view_node.meta["val"] = view2_val

        # Replace all uses of view2_node with new_view_node
        view2_node.replace_all_uses_with(new_view_node)

        # Remove dead nodes in reverse order
        graph.erase_node(view2_node)
        graph.erase_node(clone_node)
        graph.erase_node(permute_node)
        graph.erase_node(view1_node)
        graph.erase_node(cat_node)

        count += 1
        counters["inductor"]["layout_transform_store_sinking"] += 1

    if count > 0:
        graph.lint()
        return count
    return None
