# mypy: allow-untyped-defs
"""
FX pass to pre-allocate outputs for torch.cond operations.

This pass transforms cond operations to pre-allocate output buffers before
the conditional execution, then have both branches write their results into
these pre-allocated buffers. This is required for CUDA graph conditional
nodes to work correctly, where both branches must write to the same output
memory locations.

The transformation works at the ATen level:

Before transformation:
    cond_result = cond(pred, true_fn, false_fn, operands)
    # true_fn/false_fn return new allocations

After transformation:
    out_buffers = [empty_strided(...), ...]  # pre-allocated
    cond(pred, true_fn_modified, false_fn_modified, operands + out_buffers)
    cond_result = out_buffers
    # true_fn_modified/false_fn_modified write into out_buffers via copy_

NOTE: In the future, we'd like to use out= variants of ops instead of copy_
when available (e.g., mm.out instead of mm + copy_). This requires adding
lowering support for out= ops first - see issue #138280.
"""

import operator
from typing import Sequence

import torch
from torch import fx
from torch.fx import GraphModule, Node


def transform_cond_to_preallocate_outputs(graph_or_gm) -> None:
    """
    Transform cond operations to pre-allocate outputs.

    This modifies the graph in-place.

    Can be called with either a Graph or a GraphModule.
    """
    # Handle both Graph and GraphModule inputs
    if isinstance(graph_or_gm, GraphModule):
        gm = graph_or_gm
        graph = gm.graph
    else:
        # It's a Graph - need to find the owning module
        graph = graph_or_gm
        gm = graph.owning_module
        if gm is None:
            # Can't do the transformation without access to the module
            return

    modified = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is not torch.ops.higher_order.cond:
            continue

        # Found a cond node
        # Args: (pred, true_graph, false_graph, operands)
        pred, true_graph_node, false_graph_node, operands = node.args

        # Get the subgraph modules
        if true_graph_node.op != "get_attr" or false_graph_node.op != "get_attr":
            continue

        true_graph: GraphModule = getattr(gm, true_graph_node.target)
        false_graph: GraphModule = getattr(gm, false_graph_node.target)

        # Get output metadata from the cond node
        # The node.meta should have 'val' with FakeTensors for outputs
        if "val" not in node.meta:
            continue

        output_vals = node.meta["val"]
        if not isinstance(output_vals, (list, tuple)):
            output_vals = [output_vals]

        # Check that all outputs are tensors we can pre-allocate
        if not all(isinstance(v, torch.Tensor) for v in output_vals):
            continue

        num_outputs = len(output_vals)
        num_original_operands = len(operands) if isinstance(operands, (list, tuple)) else 1

        # Insert allocation nodes before the cond
        with graph.inserting_before(node):
            alloc_nodes = []
            for i, out_val in enumerate(output_vals):
                # Create empty_strided call to allocate output buffer
                size = list(out_val.shape)
                stride = list(out_val.stride())
                dtype = out_val.dtype
                device = out_val.device

                alloc_node = graph.call_function(
                    torch.ops.aten.empty_strided.default,
                    args=(size, stride),
                    kwargs={"dtype": dtype, "device": device},
                )
                alloc_node.meta["val"] = out_val.clone()
                alloc_nodes.append(alloc_node)

        # Modify the true_graph to accept out_buffers and copy outputs into them
        _modify_subgraph_to_copy_outputs(true_graph, num_outputs, num_original_operands, output_vals)

        # Modify the false_graph to accept out_buffers and copy outputs into them
        _modify_subgraph_to_copy_outputs(false_graph, num_outputs, num_original_operands, output_vals)

        # Update the cond node's operands to include the pre-allocated buffers
        if isinstance(operands, (list, tuple)):
            new_operands = tuple(operands) + tuple(alloc_nodes)
        else:
            new_operands = (operands,) + tuple(alloc_nodes)

        # Create new cond node with updated operands
        with graph.inserting_before(node):
            new_cond = graph.call_function(
                torch.ops.higher_order.cond,
                args=(pred, true_graph_node, false_graph_node, new_operands),
            )
            # The new cond returns the out_buffers
            new_cond.meta = node.meta.copy()

        # The new cond returns the out_buffers (same as what was passed in)
        # We need to use getitem on the cond result to maintain the dependency
        # so DCE doesn't remove the cond. The subgraphs have copy_ which mutates
        # the buffers, so we need to ensure the cond actually executes.
        with graph.inserting_after(new_cond):
            result_nodes = []
            for i in range(num_outputs):
                getitem_node = graph.call_function(
                    operator.getitem,
                    args=(new_cond, i),
                )
                getitem_node.meta = alloc_nodes[i].meta.copy()
                result_nodes.append(getitem_node)

        # Replace uses of old cond outputs with the getitem results from new cond
        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                user.replace_all_uses_with(result_nodes[idx])
                graph.erase_node(user)

        # Remove old cond node if no users left
        node.replace_all_uses_with(new_cond)
        graph.erase_node(node)

        modified = True

    if modified:
        graph.lint()
        gm.recompile()


def _modify_subgraph_to_copy_outputs(
    subgraph: GraphModule, num_outputs: int, num_original_operands: int, output_vals
) -> None:
    """
    Modify a subgraph to:
    1. Accept additional out_buffer arguments (after the original operands)
    2. Copy its outputs into these buffers using aten.copy_
    3. Return the out_buffers
    """
    graph = subgraph.graph

    # Find existing placeholders and output
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    output_node = next(n for n in graph.nodes if n.op == "output")

    # Add new placeholder nodes for the out_buffers
    # Insert after existing placeholders
    last_placeholder = placeholders[-1] if placeholders else None

    out_buffer_nodes = []
    for i in range(num_outputs):
        with graph.inserting_after(last_placeholder):
            out_buf = graph.placeholder(f"out_buf_{i}")
            # Copy metadata from the corresponding output val
            if i < len(output_vals):
                out_buf.meta["val"] = output_vals[i].clone()
            out_buffer_nodes.append(out_buf)
            last_placeholder = out_buf

    # Get the current outputs
    current_outputs = output_node.args[0]
    if not isinstance(current_outputs, (list, tuple)):
        current_outputs = [current_outputs]

    # Insert copy_ operations before the output
    with graph.inserting_before(output_node):
        for i, (out_val, out_buf) in enumerate(zip(current_outputs, out_buffer_nodes)):
            if out_val is not None:  # Handle None outputs
                # Use aten.copy_ to copy out_val into out_buf
                copy_node = graph.call_function(
                    torch.ops.aten.copy_.default,
                    args=(out_buf, out_val),
                )
                # Copy metadata from out_val node if available
                if hasattr(out_val, 'meta') and 'val' in out_val.meta:
                    copy_node.meta["val"] = out_val.meta["val"]

    # Update output to return the out_buffers
    output_node.args = (tuple(out_buffer_nodes),)

    graph.lint()
    subgraph.recompile()

