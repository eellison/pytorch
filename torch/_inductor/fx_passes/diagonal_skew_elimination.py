# mypy: allow-untyped-defs
"""
Scatter Read Bypass Pass (diagonal_skew_elimination).

When a read (slice or select) accesses a region that was NOT modified by a
preceding scatter (slice_scatter or select_scatter), the read can bypass the
scatter and read directly from the scatter's base tensor.

Patterns handled:

1. select(slice_scatter(base, src, dim, start, end), dim, idx)
   where idx NOT in [start, end)
   -> select(base, dim, idx)

2. slice(select_scatter(base, src, dim, scatter_idx), dim, start, end)
   where scatter_idx NOT in [start, end)
   -> slice(base, dim, start, end)

3. select(select_scatter(base, src, dim, scatter_idx), dim, read_idx)
   where scatter_idx != read_idx
   -> select(base, dim, read_idx)

4. slice(slice_scatter(base, src, dim, s_start, s_end), dim, r_start, r_end)
   where [r_start, r_end) is entirely OUTSIDE [s_start, s_end)
   -> slice(base, dim, r_start, r_end)
   (Note: the existing slice_scatter_elision handles the case where the read
   is entirely INSIDE the scatter region. This handles the OUTSIDE case.)

Impact: In Longformer attention, the scatter chain has intermediate results
with multiple consumers. The "extra" consumers read from unwritten regions.
By bypassing the scatter, these reads go directly to the base, breaking the
multi-user dependency that forces materialization. This allows the scatter
chain to become a simple linear sequence that Inductor fuses into 1 kernel
instead of 4.

Target: 87 Longformer repros (largest single-model family, 2x+ gap).
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_dim_size(node: fx.Node, dim: int) -> Optional[int]:
    """Get the size of a specific dimension from node metadata."""
    val = node.meta.get("val")
    if val is None or not isinstance(val, torch.Tensor):
        return None
    shape = val.shape
    if dim < 0:
        dim = len(shape) + dim
    if dim < 0 or dim >= len(shape):
        return None
    size = shape[dim]
    if isinstance(size, int):
        return size
    return None


def _get_static_int(val) -> Optional[int]:
    """Extract a static integer from an FX node arg."""
    if isinstance(val, int):
        return val
    if val is None:
        return None
    return None


def _normalize_slice_bounds(start, end, dim_size: int):
    """Normalize slice start/end to positive integers within [0, dim_size]."""
    if start is None:
        start = 0
    elif start < 0:
        start = max(0, dim_size + start)
    start = min(start, dim_size)

    if end is None or (isinstance(end, int) and end >= 2**62):
        end = dim_size
    elif end < 0:
        end = max(0, dim_size + end)
    end = min(end, dim_size)

    return start, end


def _normalize_index(idx: int, dim_size: int) -> int:
    """Normalize a potentially negative index to positive."""
    if idx < 0:
        return idx + dim_size
    return idx


def _is_outside_slice_region(idx: int, start: int, end: int) -> bool:
    """Check if idx is outside the half-open interval [start, end)."""
    return idx < start or idx >= end


def _ranges_disjoint(r_start: int, r_end: int, s_start: int, s_end: int) -> bool:
    """Check if two half-open intervals are completely disjoint."""
    return r_end <= s_start or r_start >= s_end


def scatter_read_bypass_pass(graph: fx.Graph) -> fx.Graph:
    """
    Bypass scatter operations when reads access unwritten regions.

    This breaks multi-user dependencies in scatter chains, enabling fusion
    of the remaining linear chain into fewer kernels.
    """
    if not getattr(config, "diagonal_skew_elimination", True):
        return graph

    num_bypasses = 0
    nodes_to_erase = []

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # Pattern: select(scatter(...), dim, idx) or slice(scatter(...), dim, ...)
        if node.target == aten.select.int:
            result = _try_bypass_select(node)
            if result is not None:
                node.replace_all_uses_with(result)
                nodes_to_erase.append(node)
                num_bypasses += 1

        elif node.target == aten.slice.Tensor:
            result = _try_bypass_slice(node)
            if result is not None:
                node.replace_all_uses_with(result)
                nodes_to_erase.append(node)
                num_bypasses += 1

    # Clean up
    for node in reversed(nodes_to_erase):
        graph.erase_node(node)

    if num_bypasses > 0:
        graph.eliminate_dead_code()
        counters["inductor"]["scatter_read_bypass"] += num_bypasses
        log.info(
            "diagonal_skew_elimination: bypassed %d reads through scatter ops",
            num_bypasses,
        )
        graph.lint()

    return graph


def _try_bypass_select(node: fx.Node) -> Optional[fx.Node]:
    """
    Try to bypass a scatter when select reads from an unwritten region.

    Patterns:
    - select(slice_scatter(base, src, dim, start, end), dim, idx) where idx not in [start, end)
    - select(select_scatter(base, src, dim, scatter_idx), dim, idx) where idx != scatter_idx
    """
    # node is select.int(input, dim, index)
    if len(node.args) < 3:
        return None

    input_node = node.args[0]
    read_dim = _get_static_int(node.args[1])
    read_idx = _get_static_int(node.args[2])

    if read_dim is None or read_idx is None:
        return None
    if not isinstance(input_node, fx.Node):
        return None
    if input_node.op != "call_function":
        return None

    # Get dimension size for normalization
    dim_size = _get_dim_size(input_node, read_dim)
    if dim_size is None:
        return None

    read_idx_norm = _normalize_index(read_idx, dim_size)

    # Case 1: select from slice_scatter
    if input_node.target == aten.slice_scatter.default:
        # slice_scatter(base, src, dim, start, end, step)
        scatter_args = input_node.args
        if len(scatter_args) < 2:
            return None

        base_node = scatter_args[0]
        scatter_dim = _get_static_int(scatter_args[2]) if len(scatter_args) > 2 else 0
        scatter_start = _get_static_int(scatter_args[3]) if len(scatter_args) > 3 else None
        scatter_end = _get_static_int(scatter_args[4]) if len(scatter_args) > 4 else None
        scatter_step = _get_static_int(scatter_args[5]) if len(scatter_args) > 5 else 1

        if scatter_dim is None or scatter_step is None:
            return None
        if scatter_step != 1:
            return None  # Step != 1 complicates the region check
        if scatter_dim != read_dim:
            return None  # Different dimensions - can't determine bypass

        # Normalize scatter bounds
        s_dim_size = _get_dim_size(input_node, scatter_dim)
        if s_dim_size is None:
            return None
        s_start, s_end = _normalize_slice_bounds(scatter_start, scatter_end, s_dim_size)

        # Check: is read_idx outside the written region?
        if _is_outside_slice_region(read_idx_norm, s_start, s_end):
            # The select reads from an unwritten position -> bypass to base
            if not isinstance(base_node, fx.Node):
                return None

            log.debug(
                "scatter_read_bypass: select(slice_scatter(base, src, dim=%d, "
                "[%d,%d)), dim=%d, idx=%d) -> select(base, dim=%d, idx=%d) "
                "[idx outside written region]",
                scatter_dim, s_start, s_end, read_dim, read_idx_norm,
                read_dim, read_idx,
            )

            # Create: select(base, dim, idx) with same metadata
            graph = node.graph
            with graph.inserting_before(node):
                new_select = graph.call_function(
                    aten.select.int,
                    args=(base_node, read_dim, read_idx),
                )
                if "val" in node.meta:
                    new_select.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_select.meta["tensor_meta"] = node.meta["tensor_meta"]
            return new_select

    # Case 2: select from select_scatter
    elif input_node.target == aten.select_scatter.default:
        # select_scatter(base, src, dim, index)
        scatter_args = input_node.args
        if len(scatter_args) < 4:
            return None

        base_node = scatter_args[0]
        scatter_dim = _get_static_int(scatter_args[2])
        scatter_idx = _get_static_int(scatter_args[3])

        if scatter_dim is None or scatter_idx is None:
            return None
        if scatter_dim != read_dim:
            return None

        scatter_idx_norm = _normalize_index(scatter_idx, dim_size)

        # Check: different index?
        if read_idx_norm != scatter_idx_norm:
            if not isinstance(base_node, fx.Node):
                return None

            log.debug(
                "scatter_read_bypass: select(select_scatter(base, src, dim=%d, "
                "idx=%d), dim=%d, idx=%d) -> select(base, dim=%d, idx=%d) "
                "[different indices]",
                scatter_dim, scatter_idx_norm, read_dim, read_idx_norm,
                read_dim, read_idx,
            )

            graph = node.graph
            with graph.inserting_before(node):
                new_select = graph.call_function(
                    aten.select.int,
                    args=(base_node, read_dim, read_idx),
                )
                if "val" in node.meta:
                    new_select.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_select.meta["tensor_meta"] = node.meta["tensor_meta"]
            return new_select

    return None


def _try_bypass_slice(node: fx.Node) -> Optional[fx.Node]:
    """
    Try to bypass a scatter when slice reads from an unwritten region.

    Patterns:
    - slice(select_scatter(base, src, dim, scatter_idx), dim, start, end)
      where scatter_idx not in [start, end)
    - slice(slice_scatter(base, src, dim, s_start, s_end), dim, r_start, r_end)
      where [r_start, r_end) and [s_start, s_end) are disjoint
    """
    # node is slice.Tensor(input, dim, start, end, step)
    args = node.args
    if len(args) < 1:
        return None

    input_node = args[0]
    read_dim = _get_static_int(args[1]) if len(args) > 1 else 0
    read_start = _get_static_int(args[2]) if len(args) > 2 else None
    read_end = _get_static_int(args[3]) if len(args) > 3 else None
    read_step = _get_static_int(args[4]) if len(args) > 4 else 1

    if read_dim is None or read_step is None:
        return None
    if not isinstance(input_node, fx.Node):
        return None
    if input_node.op != "call_function":
        return None

    # Get dimension size for normalization
    dim_size = _get_dim_size(input_node, read_dim)
    if dim_size is None:
        return None

    r_start, r_end = _normalize_slice_bounds(read_start, read_end, dim_size)

    # Case 1: slice from select_scatter
    if input_node.target == aten.select_scatter.default:
        scatter_args = input_node.args
        if len(scatter_args) < 4:
            return None

        base_node = scatter_args[0]
        src_node = scatter_args[1]
        scatter_dim = _get_static_int(scatter_args[2])
        scatter_idx = _get_static_int(scatter_args[3])

        if scatter_dim is None or scatter_idx is None:
            return None
        if scatter_dim != read_dim:
            return None
        if not isinstance(base_node, fx.Node):
            return None

        scatter_idx_norm = _normalize_index(scatter_idx, dim_size)

        # Check: is scatter_idx outside the slice's read region?
        if _is_outside_slice_region(scatter_idx_norm, r_start, r_end):
            log.debug(
                "scatter_read_bypass: slice(select_scatter(base, src, dim=%d, "
                "idx=%d), dim=%d, [%d,%d)) -> slice(base, dim=%d, [%d,%d)) "
                "[scatter_idx outside read region]",
                scatter_dim, scatter_idx_norm, read_dim, r_start, r_end,
                read_dim, r_start, r_end,
            )

            graph = node.graph
            with graph.inserting_before(node):
                # Reconstruct the slice with same args but on base
                new_slice = graph.call_function(
                    aten.slice.Tensor,
                    args=(base_node,) + args[1:],
                )
                if "val" in node.meta:
                    new_slice.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_slice.meta["tensor_meta"] = node.meta["tensor_meta"]
            return new_slice

        # Case 1b: Push scatter through slice (scatter_idx IS in read region)
        # slice(select_scatter(base, src, dim, idx), dim, start, end)
        # = select_scatter(slice(base, dim, start, end), src, dim, idx - start)
        # This moves the dependency from scatter's output to scatter's base,
        # breaking the multi-user pattern.
        # Only apply when the scatter node has multiple users (otherwise no benefit)
        if (not _is_outside_slice_region(scatter_idx_norm, r_start, r_end)
                and len(input_node.users) > 1
                and read_step == 1
                and isinstance(src_node, fx.Node)):
            # Compute the new index within the sliced tensor
            new_scatter_idx = scatter_idx_norm - r_start

            log.debug(
                "scatter_read_bypass: slice(select_scatter(base, src, dim=%d, "
                "idx=%d), dim=%d, [%d,%d)) -> select_scatter(slice(base, ...), "
                "src, dim=%d, idx=%d) [push scatter through slice]",
                scatter_dim, scatter_idx_norm, read_dim, r_start, r_end,
                scatter_dim, new_scatter_idx,
            )

            graph = node.graph
            with graph.inserting_before(node):
                # First: slice the base directly
                sliced_base = graph.call_function(
                    aten.slice.Tensor,
                    args=(base_node,) + args[1:],
                )
                if "val" in node.meta:
                    sliced_base.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    sliced_base.meta["tensor_meta"] = node.meta["tensor_meta"]

                # Then: apply select_scatter with adjusted index
                new_scatter = graph.call_function(
                    aten.select_scatter.default,
                    args=(sliced_base, src_node, scatter_dim, new_scatter_idx),
                )
                if "val" in node.meta:
                    new_scatter.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_scatter.meta["tensor_meta"] = node.meta["tensor_meta"]

            return new_scatter

    # Case 2: slice from slice_scatter
    elif input_node.target == aten.slice_scatter.default:
        scatter_args = input_node.args
        if len(scatter_args) < 2:
            return None

        base_node = scatter_args[0]
        scatter_dim = _get_static_int(scatter_args[2]) if len(scatter_args) > 2 else 0
        scatter_start = _get_static_int(scatter_args[3]) if len(scatter_args) > 3 else None
        scatter_end = _get_static_int(scatter_args[4]) if len(scatter_args) > 4 else None
        scatter_step = _get_static_int(scatter_args[5]) if len(scatter_args) > 5 else 1

        if scatter_dim is None or scatter_step is None:
            return None
        if scatter_step != 1:
            return None
        if scatter_dim != read_dim:
            return None

        s_start, s_end = _normalize_slice_bounds(scatter_start, scatter_end, dim_size)

        # Check: are the regions completely disjoint?
        if _ranges_disjoint(r_start, r_end, s_start, s_end):
            if not isinstance(base_node, fx.Node):
                return None

            log.debug(
                "scatter_read_bypass: slice(slice_scatter(base, src, dim=%d, "
                "[%d,%d)), dim=%d, [%d,%d)) -> slice(base, dim=%d, [%d,%d)) "
                "[disjoint regions]",
                scatter_dim, s_start, s_end, read_dim, r_start, r_end,
                read_dim, r_start, r_end,
            )

            graph = node.graph
            with graph.inserting_before(node):
                new_slice = graph.call_function(
                    aten.slice.Tensor,
                    args=(base_node,) + args[1:],
                )
                if "val" in node.meta:
                    new_slice.meta["val"] = node.meta["val"]
                if "tensor_meta" in node.meta:
                    new_slice.meta["tensor_meta"] = node.meta["tensor_meta"]
            return new_slice

    return None


# Keep the old name for config compatibility
diagonal_skew_elimination_pass = scatter_read_bypass_pass
