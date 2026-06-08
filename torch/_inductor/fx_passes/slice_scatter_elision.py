# mypy: allow-untyped-defs
"""
Slice-Scatter Elision Pass: Eliminates redundant slice_scatter -> slice chains.

Pattern:
    result = slice_scatter(base, src, dim, start, end)
    consumer = slice(result, dim, start2, end2)
    where start2 >= start and end2 <= end and step == 1

When a slice extracts a region that is entirely within the region written by
slice_scatter, the base tensor is irrelevant -- we can read directly from src.

This is the common "stencil" pattern in scientific computing (pyhpc benchmarks):
    padded = full(0, [N+4, M+4, K])
    padded[1:-2, 2:-2, 1:] = compute(inputs)
    result = padded[1:-2, 2:-2, :]  # reads only the written region + zeros
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_static_value(val) -> Optional[int]:
    """Extract a static integer value from an FX arg."""
    if isinstance(val, int):
        return val
    if val is None:
        return None
    if hasattr(val, "meta") and "val" in val.meta:
        # scalar tensor -> try to get value
        t = val.meta["val"]
        if isinstance(t, (int, float)):
            return int(t)
    return None


def _normalize_slice_args(node: fx.Node, dim_size: int):
    """
    Normalize slice/slice_scatter start/end args to positive integers.
    Returns (start, end, step) as ints, or None if cannot be resolved statically.
    """
    target = node.target
    args = node.args
    kwargs = node.kwargs

    if target == aten.slice.Tensor:
        # slice.Tensor(self, dim=0, start=None, end=None, step=1)
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        start = args[2] if len(args) > 2 else kwargs.get("start", None)
        end = args[3] if len(args) > 3 else kwargs.get("end", None)
        step = args[4] if len(args) > 4 else kwargs.get("step", 1)
    elif target == aten.slice_scatter.default:
        # slice_scatter(self, src, dim=0, start=None, end=None, step=1)
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        start = args[3] if len(args) > 3 else kwargs.get("start", None)
        end = args[4] if len(args) > 4 else kwargs.get("end", None)
        step = args[5] if len(args) > 5 else kwargs.get("step", 1)
    else:
        return None

    dim = _get_static_value(dim)
    start = _get_static_value(start)
    end = _get_static_value(end)
    step = _get_static_value(step)

    if dim is None or step is None:
        return None

    # Normalize start
    if start is None:
        start = 0
    elif start < 0:
        start = max(0, dim_size + start)

    # Normalize end
    if end is None or end >= 2**62:
        end = dim_size
    elif end < 0:
        end = max(0, dim_size + end)
    end = min(end, dim_size)

    if step != 1:
        return None

    return (dim, start, end, step)


def _get_dim_size(node: fx.Node, dim: int) -> Optional[int]:
    """Get the size of a dimension from node metadata."""
    if "val" not in node.meta:
        return None
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor):
        return None
    if dim < 0:
        dim = len(val.shape) + dim
    if dim < 0 or dim >= len(val.shape):
        return None
    size = val.shape[dim]
    if isinstance(size, int):
        return size
    return None


def _is_full_constant(node: fx.Node) -> bool:
    """Check if a node is aten.full with a constant fill value."""
    return (
        node.op == "call_function"
        and node.target in (aten.full.default, torch.full)
    )


def slice_scatter_elision_pass(graph: fx.Graph) -> fx.Graph:
    """
    Eliminate slice(slice_scatter(base, src, dim, start, end), dim, start2, end2)
    when the slice reads entirely within the scatter-written region.

    Simplification:
        slice(slice_scatter(base, src, dim, a, b), dim, a2, b2) -> slice(src, dim, a2-a, b2-a)
        when a2 >= a and b2 <= b and step=1 for both

    Special case (most common in stencil codes):
        slice(slice_scatter(base, src, dim, a, b), dim, a, b) -> src
    """
    if not getattr(config, "slice_scatter_elision", True):
        return graph

    # Count targets for debugging
    _n_scatter = sum(1 for n in graph.nodes if n.op == "call_function" and n.target == aten.slice_scatter.default)
    _n_slice = sum(1 for n in graph.nodes if n.op == "call_function" and n.target == aten.slice.Tensor)
    log.debug("slice_scatter_elision: graph has %d slice_scatter, %d slice nodes", _n_scatter, _n_slice)

    num_elisions = 0
    nodes_to_erase = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target != aten.slice.Tensor:
            continue

        # This is a slice node. Check if its input is a slice_scatter.
        slice_input = node.args[0]
        if not isinstance(slice_input, fx.Node):
            continue
        if slice_input.op != "call_function":
            continue
        if slice_input.target != aten.slice_scatter.default:
            continue

        # Get scatter's base, src, dim, start, end
        scatter_node = slice_input
        scatter_base = scatter_node.args[0]
        scatter_src = scatter_node.args[1]

        # Get dim size from the scatter output
        scatter_dim_args = _normalize_slice_args(scatter_node, 0)  # need dim_size
        if scatter_dim_args is None:
            continue
        scatter_dim = scatter_dim_args[0]

        # Get the scatter output shape to normalize indices
        scatter_out_dim_size = _get_dim_size(scatter_node, scatter_dim)
        if scatter_out_dim_size is None:
            continue

        # Re-normalize with correct dim_size
        scatter_args = _normalize_slice_args(scatter_node, scatter_out_dim_size)
        if scatter_args is None:
            continue
        s_dim, s_start, s_end, s_step = scatter_args

        # Get the slice's dim and bounds
        slice_dim_size = scatter_out_dim_size  # slice reads from scatter output
        slice_args = _normalize_slice_args(node, slice_dim_size)
        if slice_args is None:
            continue
        sl_dim, sl_start, sl_end, sl_step = slice_args

        # Dimensions must match
        if s_dim != sl_dim:
            continue

        # The slice must read entirely within the scatter region
        if sl_start < s_start or sl_end > s_end:
            continue

        # The slice reads entirely from the scatter-written region.
        # We can replace: slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
        # with: slice(src, dim, a2 - a, b2 - a)

        new_start = sl_start - s_start
        new_end = sl_end - s_start

        # Get src's size in this dim to check if the new slice is the full extent
        src_dim_size = _get_dim_size(scatter_src, s_dim)
        if src_dim_size is None:
            continue

        # Verify dimensions: src should have size (s_end - s_start) in the scatter dim
        expected_src_dim = s_end - s_start
        if src_dim_size != expected_src_dim:
            # src might be smaller if scatter has step != 1, but we already checked step == 1
            continue

        # Check if this is an exact match (common case)
        if new_start == 0 and new_end == src_dim_size:
            # slice(slice_scatter(base, src, dim, a, b), dim, a, b) -> src
            # But we need to check that no other dimension slicing is happening
            # Actually this IS the right replacement -- the slice extracts exactly src
            node.replace_all_uses_with(scatter_src)
            nodes_to_erase.append(node)
            num_elisions += 1
            log.debug(
                "slice_scatter_elision: exact match - replaced %s with %s",
                node.name,
                scatter_src.name,
            )
        else:
            # Partial match: slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
            # -> slice(src, dim, a2-a, b2-a)
            with graph.inserting_before(node):
                new_slice = graph.call_function(
                    aten.slice.Tensor,
                    args=(scatter_src, s_dim, new_start, new_end, 1),
                )
                # Copy metadata
                if "val" in node.meta:
                    new_slice.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_slice.meta["tensor_meta"] = node.meta["tensor_meta"]

            node.replace_all_uses_with(new_slice)
            nodes_to_erase.append(node)
            num_elisions += 1
            log.debug(
                "slice_scatter_elision: partial - replaced %s with slice(%s, %d, %d, %d)",
                node.name,
                scatter_src.name,
                s_dim,
                new_start,
                new_end,
            )

    # Erase replaced nodes
    for node in reversed(nodes_to_erase):
        graph.erase_node(node)

    # Clean up dead slice_scatter nodes that have no remaining users
    graph.eliminate_dead_code()

    if num_elisions > 0:
        counters["inductor"]["slice_scatter_elisions"] += num_elisions
        log.info(
            "slice_scatter_elision: eliminated %d slice-of-scatter patterns",
            num_elisions,
        )
        graph.lint()

    # Second pass: handle partial overlap where read extends beyond the scatter
    # region (e.g., read [0:33] from scatter that wrote [1:33]).
    # Rewrite: slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
    # → slice_scatter(slice(base, dim, a2, b2), src_overlap, dim, a-a2, min(b,b2)-a2)
    # This creates a SMALL slice_scatter (output-sized) that Inductor lowers to a
    # fusible Pointwise node, avoiding the ConcatKernel overhead of the cat approach.
    if getattr(config, "slice_scatter_partial_elision", True):
        num_partial = _partial_overlap_elision(graph)
        if num_partial > 0:
            graph.lint()

    return graph


def _partial_overlap_elision(graph: fx.Graph) -> int:
    """Handle partial-overlap reads from slice_scatter outputs.

    Pattern: slice(slice_scatter(base, src, dim, a, b), dim, a2, b2)
    where the read range [a2, b2) partially overlaps with the write range [a, b).

    Rewrite to a small slice_scatter over only the read range:
        slice_scatter(slice(base, dim, a2, b2), src_overlap, dim, max(a,a2)-a2, min(b,b2)-a2)

    This produces a Pointwise IR node of size (b2-a2) instead of the full base size,
    which can fuse with downstream consumers (permute, clone, etc.) without
    materializing an intermediate ConcatKernel buffer.
    """
    num_elisions = 0
    nodes_to_erase = []

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != aten.slice.Tensor:
            continue

        slice_input = node.args[0]
        if not isinstance(slice_input, fx.Node):
            continue
        if slice_input.op != "call_function":
            continue
        if slice_input.target != aten.slice_scatter.default:
            continue

        scatter_node = slice_input
        scatter_base = scatter_node.args[0]
        scatter_src = scatter_node.args[1]

        # Get scatter dimension and bounds
        scatter_dim_args = _normalize_slice_args(scatter_node, 0)
        if scatter_dim_args is None:
            continue
        scatter_dim = scatter_dim_args[0]

        scatter_out_dim_size = _get_dim_size(scatter_node, scatter_dim)
        if scatter_out_dim_size is None:
            continue

        scatter_args = _normalize_slice_args(scatter_node, scatter_out_dim_size)
        if scatter_args is None:
            continue
        s_dim, s_start, s_end, s_step = scatter_args

        # Get the slice's dim and bounds
        slice_args = _normalize_slice_args(node, scatter_out_dim_size)
        if slice_args is None:
            continue
        sl_dim, sl_start, sl_end, sl_step = slice_args

        # Dimensions must match
        if s_dim != sl_dim:
            continue

        # Skip if already handled by the exact/subset match above
        if sl_start >= s_start and sl_end <= s_end:
            continue

        # Verify there IS overlap (not completely disjoint)
        if sl_end <= s_start or sl_start >= s_end:
            continue

        # Get src dim size to validate
        src_dim_size = _get_dim_size(scatter_src, s_dim)
        if src_dim_size is None:
            continue
        expected_src_dim = s_end - s_start
        if src_dim_size != expected_src_dim:
            continue

        # Get base dim size
        base_dim_size = _get_dim_size(scatter_base, s_dim)
        if base_dim_size is None:
            continue

        # Compute the overlap region in src coordinates
        overlap_start_in_src = max(0, sl_start - s_start)
        overlap_end_in_src = min(src_dim_size, sl_end - s_start)

        # Compute where src data lands in the output (read-range-relative coordinates)
        new_scatter_start = max(s_start, sl_start) - sl_start
        new_scatter_end = min(s_end, sl_end) - sl_start

        # Build the replacement: slice_scatter(slice(base, dim, a2, b2), src_overlap, dim, offset_start, offset_end)
        # IMPORTANT: Insert BEFORE the original scatter node, not before the consumer.
        # This ensures the new nodes are ordered before the scatter in the graph,
        # which allows the reinplace pass to still convert the original scatter+copy_
        # to an in-place mutation (reinplace checks for views used AFTER the scatter).
        with graph.inserting_before(scatter_node):
            # Create a view of base covering the read range [a2, b2)
            base_slice = graph.call_function(
                aten.slice.Tensor,
                args=(scatter_base, s_dim, sl_start, sl_end, 1),
            )
            if "val" in scatter_base.meta:
                base_val = scatter_base.meta["val"]
                base_slice.meta["val"] = base_val.narrow(s_dim, sl_start, sl_end - sl_start)

            # Get the src overlap portion
            if overlap_start_in_src == 0 and overlap_end_in_src == src_dim_size:
                src_overlap = scatter_src
            else:
                src_overlap = graph.call_function(
                    aten.slice.Tensor,
                    args=(scatter_src, s_dim, overlap_start_in_src, overlap_end_in_src, 1),
                )
                if "val" in scatter_src.meta:
                    src_val = scatter_src.meta["val"]
                    src_overlap.meta["val"] = src_val.narrow(
                        s_dim, overlap_start_in_src, overlap_end_in_src - overlap_start_in_src
                    )

            # Create the small slice_scatter over the read range
            replacement = graph.call_function(
                aten.slice_scatter.default,
                args=(base_slice, src_overlap, s_dim, new_scatter_start, new_scatter_end, 1),
            )
            if "val" in node.meta:
                replacement.meta["val"] = node.meta["val"]

        node.replace_all_uses_with(replacement)
        nodes_to_erase.append(node)
        num_elisions += 1
        log.info(
            "slice_scatter_elision: partial overlap - replaced %s with small "
            "slice_scatter (read [%d:%d] from scatter [%d:%d], dim=%d)",
            node.name,
            sl_start,
            sl_end,
            s_start,
            s_end,
            s_dim,
        )

    for node in reversed(nodes_to_erase):
        graph.erase_node(node)

    if num_elisions > 0:
        graph.eliminate_dead_code()
        counters["inductor"]["slice_scatter_partial_elisions"] = num_elisions
        log.info(
            "slice_scatter_elision: eliminated %d partial-overlap patterns",
            num_elisions,
        )

    return num_elisions
