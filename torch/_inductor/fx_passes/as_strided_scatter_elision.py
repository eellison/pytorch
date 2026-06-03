# mypy: allow-untyped-defs
"""
As-Strided-Scatter Elision Pass: Eliminates redundant full + as_strided_scatter + as_strided chains.

Pattern:
    buf = full([N], 0)
    result = as_strided_scatter(buf, src, size, stride, offset=0)
    output = as_strided(result, size2, stride2, offset=0)

When the scatter writes ALL elements of the buffer (prod(size) == N with contiguous
strides and offset=0), the full + scatter is redundant. The as_strided read can be
replaced with a view/reshape of src directly.

This pattern appears in batch-norm backward computations (timm_mobilevit, etc.)
where an intermediate is squeezed, scattered into a flat buffer, then strided back
into a higher-rank shape.
"""

import logging
import operator
from functools import reduce
from typing import List, Optional, Tuple

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_static_int(val) -> Optional[int]:
    """Extract a static integer from an FX arg or node meta."""
    if isinstance(val, int):
        return val
    if val is None:
        return None
    if hasattr(val, "meta") and "val" in val.meta:
        t = val.meta["val"]
        if isinstance(t, (int, float)):
            return int(t)
    return None


def _get_static_int_list(vals) -> Optional[List[int]]:
    """Extract a list of static integers from FX args."""
    result = []
    for v in vals:
        s = _get_static_int(v)
        if s is None:
            return None
        result.append(s)
    return result


def _is_contiguous_strides(size: List[int], stride: List[int]) -> bool:
    """Check if strides correspond to a contiguous (row-major) layout."""
    if len(size) != len(stride):
        return False
    if len(size) == 0:
        return True
    # Compute expected contiguous strides (row-major)
    expected = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        expected[i] = expected[i + 1] * size[i + 1]
    return stride == expected


def _is_full_zero(node: fx.Node) -> bool:
    """Check if a node is aten.full with fill value 0."""
    if node.op != "call_function":
        return False
    if node.target not in (aten.full.default, torch.full):
        return False
    # The fill_value is args[1]
    if len(node.args) < 2:
        return False
    fill_val = node.args[1]
    if isinstance(fill_val, (int, float)) and fill_val == 0:
        return True
    return False


def _get_full_numel(node: fx.Node) -> Optional[int]:
    """Get the total number of elements from a full() node's shape arg."""
    if len(node.args) < 1:
        return None
    shape_arg = node.args[0]
    if isinstance(shape_arg, (list, tuple)):
        sizes = _get_static_int_list(shape_arg)
        if sizes is None:
            return None
        return reduce(operator.mul, sizes, 1)
    return None


def _extract_as_strided_scatter_args(
    node: fx.Node,
) -> Optional[Tuple[fx.Node, fx.Node, List[int], List[int], int]]:
    """
    Extract (base, src, size, stride, offset) from an as_strided_scatter node.
    Returns None if args cannot be resolved statically.
    """
    if node.op != "call_function":
        return None
    if node.target != aten.as_strided_scatter.default:
        return None

    args = node.args
    # as_strided_scatter(self, src, size, stride, storage_offset=0)
    if len(args) < 4:
        return None

    base = args[0]
    src = args[1]
    size_arg = args[2]
    stride_arg = args[3]
    offset = args[4] if len(args) > 4 else node.kwargs.get("storage_offset", 0)

    if not isinstance(base, fx.Node) or not isinstance(src, fx.Node):
        return None

    size = _get_static_int_list(size_arg) if isinstance(size_arg, (list, tuple)) else None
    stride = _get_static_int_list(stride_arg) if isinstance(stride_arg, (list, tuple)) else None
    offset_val = _get_static_int(offset)

    if size is None or stride is None or offset_val is None:
        return None

    return (base, src, size, stride, offset_val)


def _extract_as_strided_args(
    node: fx.Node,
) -> Optional[Tuple[fx.Node, List[int], List[int], int]]:
    """
    Extract (input, size, stride, offset) from an as_strided node.
    Returns None if args cannot be resolved statically.
    """
    if node.op != "call_function":
        return None
    if node.target != aten.as_strided.default:
        return None

    args = node.args
    # as_strided(self, size, stride, storage_offset=0)
    if len(args) < 3:
        return None

    input_node = args[0]
    size_arg = args[1]
    stride_arg = args[2]
    offset = args[3] if len(args) > 3 else node.kwargs.get("storage_offset", 0)

    if not isinstance(input_node, fx.Node):
        return None

    size = _get_static_int_list(size_arg) if isinstance(size_arg, (list, tuple)) else None
    stride = _get_static_int_list(stride_arg) if isinstance(stride_arg, (list, tuple)) else None
    offset_val = _get_static_int(offset)

    if size is None or stride is None or offset_val is None:
        return None

    return (input_node, size, stride, offset_val)


def as_strided_scatter_elision_pass(graph: fx.Graph) -> fx.Graph:
    """
    Eliminate as_strided(as_strided_scatter(full(0, [N]), src, size, stride, 0), size2, stride2, 0)
    when the scatter writes ALL elements of the flat buffer (prod(size)==N, contiguous strides,
    offset=0).

    In this case the full+scatter is a no-op identity (just reshaping src into the buffer),
    and the subsequent as_strided is just another view. We replace with view(src, size2).
    """
    if not getattr(config, "as_strided_scatter_elision", True):
        return graph

    num_elisions = 0
    nodes_to_erase = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target != aten.as_strided.default:
            continue

        # This is an as_strided node. Check if its input is an as_strided_scatter.
        strided_args = _extract_as_strided_args(node)
        if strided_args is None:
            continue

        input_node, read_size, read_stride, read_offset = strided_args

        # The input must be an as_strided_scatter
        scatter_args = _extract_as_strided_scatter_args(input_node)
        if scatter_args is None:
            continue

        base, src, scatter_size, scatter_stride, scatter_offset = scatter_args

        # The base of the scatter must be full(0, [N])
        if not _is_full_zero(base):
            continue

        # The scatter must have offset 0
        if scatter_offset != 0:
            continue

        # The scatter must have contiguous strides (so it writes a dense block)
        if not _is_contiguous_strides(scatter_size, scatter_stride):
            continue

        # The scatter must write ALL elements of the buffer
        buf_numel = _get_full_numel(base)
        if buf_numel is None:
            continue

        scatter_numel = reduce(operator.mul, scatter_size, 1)
        if scatter_numel != buf_numel:
            continue

        # The read must also have offset 0
        if read_offset != 0:
            continue

        # The read must have contiguous strides for its shape
        if not _is_contiguous_strides(read_size, read_stride):
            continue

        # The read numel must not exceed scatter numel
        read_numel = reduce(operator.mul, read_size, 1)
        if read_numel > scatter_numel:
            continue

        # At this point:
        # - scatter writes ALL elements of buffer with contiguous layout from offset 0
        # - read reads a contiguous block from offset 0
        # - src has shape scatter_size with contiguous strides
        # So the buffer is just src laid out contiguously, and the read is just a view.
        # Replace with: view(src, read_size)

        with graph.inserting_before(node):
            new_view = graph.call_function(
                aten.view.default,
                args=(src, list(read_size)),
            )
            # Copy metadata from the original node
            if "val" in node.meta:
                new_view.meta["val"] = node.meta["val"]
            if "tensor_meta" in node.meta:
                new_view.meta["tensor_meta"] = node.meta["tensor_meta"]

        node.replace_all_uses_with(new_view)
        nodes_to_erase.append(node)
        num_elisions += 1
        log.debug(
            "as_strided_scatter_elision: replaced %s with view(%s, %s)",
            node.name,
            src.name,
            read_size,
        )

    # Erase replaced nodes
    for n in reversed(nodes_to_erase):
        graph.erase_node(n)

    # Clean up dead as_strided_scatter and full nodes
    graph.eliminate_dead_code()

    if num_elisions > 0:
        counters["inductor"]["as_strided_scatter_elisions"] += num_elisions
        log.info(
            "as_strided_scatter_elision: eliminated %d full+scatter+strided patterns",
            num_elisions,
        )
        graph.lint()

    return graph
