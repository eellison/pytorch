# mypy: allow-untyped-defs
"""
Structured Scatter Decode Pass: decodes index_put(accumulate=True) scatters
whose indices are an affine function of the source position (as_strided over
iota) into an explicit overlap-add of padded slices.

Pattern (Longformer sliding-window attention backward):

    iota(N) -> as_strided(size, stride, offset) -> [clone] -> view(flat)
    index_put(full([M], 0), [idx_flat], view(src, [-1]), accumulate=True)

The index tensor is as_strided over iota, so the destination address of
source element (i_0, ..., i_{D-1}) is the affine map

    dest(i) = offset + sum_d stride[d] * i_d

When exactly one dimension `b` makes this map non-injective (its stride is
smaller than the extent spanned by the inner dimensions - the "overlapping
window" dimension), the scatter is an overlap-add: fixing i_b = w, the
remaining map is injective and writes a strided slab of the output at a
per-window shift. The whole scatter therefore equals

    out.view(out_sz) = sum_w constant_pad_nd(src.select(b, w), pads(w))

which is pure pointwise (masked loads): no zero-memset of the scratch
buffer, no atomic scatter, no dense re-read - and unlike the gather-reduce
identity it is consumer-agnostic, so it applies even when the scatter result
feeds dense (non-reduction) outputs.

Controlled by: config.structured_scatter_decode (default True)
"""

import logging
import math
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# Maximum size of the overlapping (non-injective) dimension. The rewrite
# unrolls one padded slice per window, so each output element reads up to
# this many source positions in the fused kernel.
MAX_OVERLAP_UNROLL = 8


def _get_meta_val(node: fx.Node):
    if not isinstance(node, fx.Node):
        return None
    val = node.meta.get("val") if hasattr(node, "meta") else None
    if val is None or not hasattr(val, "shape"):
        return None
    return val


def _is_accumulate_index_put(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target not in (aten.index_put.default, aten.index_put_.default):
        return False
    if len(node.args) >= 4 and node.args[3] is True:
        return True
    return node.kwargs.get("accumulate", False) is True


def _is_zeros_init(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target == aten.full.default:
        if len(node.args) >= 2:
            return node.args[1] == 0 or node.args[1] == 0.0
    return node.target == aten.zeros.default


def _trace_affine_index(idx_node: fx.Node) -> Optional[tuple[list[int], list[int], int]]:
    """Trace an index tensor back to as_strided(iota(start=0, step=1), ...).

    Walks through value-preserving reshapes (view/clone/reshape) to find the
    affine index construction. Returns (size, stride, storage_offset) of the
    as_strided, i.e. dest(i) = offset + sum_d stride[d] * i_d for a source
    element at multi-index i in `size`-space, or None if the index is not of
    this form.
    """
    current = idx_node
    for _ in range(8):  # bounded walk
        if not isinstance(current, fx.Node) or current.op != "call_function":
            return None
        if current.target in (
            aten.view.default,
            aten.reshape.default,
            aten._unsafe_view.default,
            aten.clone.default,
        ):
            current = current.args[0]
            continue
        break
    if not isinstance(current, fx.Node) or current.op != "call_function":
        return None
    if current.target != aten.as_strided.default:
        return None

    base = current.args[0]
    size = current.args[1]
    stride = current.args[2]
    storage_offset = current.args[3] if len(current.args) > 3 else 0
    if not isinstance(base, fx.Node) or base.op != "call_function":
        return None

    # Base must be iota/arange with start=0, step=1 (identity: value == position)
    if base.target == prims.iota.default:
        start = base.kwargs.get("start", 0)
        step = base.kwargs.get("step", 1)
        if len(base.args) > 1:
            start = base.args[1]
        if len(base.args) > 2:
            step = base.args[2]
        if start != 0 or step != 1:
            return None
    elif base.target in (aten.arange.default, aten.arange.start_step):
        # arange(end) or arange(start, end, step)
        if base.target == aten.arange.start_step:
            if base.args[0] != 0:
                return None
            step = base.args[2] if len(base.args) > 2 else base.kwargs.get("step", 1)
            if step != 1:
                return None
    else:
        return None

    if not isinstance(size, (list, tuple)) or not isinstance(stride, (list, tuple)):
        return None
    if not all(isinstance(s, int) for s in size):
        return None
    if not all(isinstance(s, int) for s in stride):
        return None
    if not isinstance(storage_offset, int):
        return None
    return list(size), list(stride), storage_offset


def _decode_overlap_structure(
    size: list[int], stride: list[int], offset: int, out_numel: int
) -> Optional[dict]:
    """Decide whether the affine map factors as a single-overlap-dim slab write.

    Returns a dict with:
        - overlap_dim: the unique non-injective dimension (in original order)
        - out_view_shape: shape to view the flat output as (stride-desc order
          of the remaining dims)
        - per_window_pads: list (one per overlap index) of constant_pad_nd
          pad lists for the remaining-dims slice
    or None when the structure does not apply.
    """
    ndim = len(size)
    if ndim < 2 or len(stride) != ndim:
        return None
    if any(s <= 0 for s in size) or any(s <= 0 for s in stride):
        return None
    if offset < 0:
        return None

    # Bounds: every address must land inside the output buffer
    max_addr = offset + sum(st * (sz - 1) for st, sz in zip(stride, size))
    if max_addr >= out_numel:
        return None

    # Find the non-injective ("overlapping") dimensions: a dim overlaps when
    # its stride is smaller than the extent spanned by all strictly-smaller
    # strided dims.
    order = sorted(range(ndim), key=lambda d: stride[d], reverse=True)
    overlap_dims = []
    for pos, d in enumerate(order):
        inner_extent = sum(
            stride[j] * (size[j] - 1) for j in order[pos + 1:]
        )
        if stride[d] <= inner_extent:
            overlap_dims.append(d)
    if len(overlap_dims) != 1:
        return None
    b_dim = overlap_dims[0]
    num_windows = size[b_dim]
    if num_windows > MAX_OVERLAP_UNROLL:
        return None

    # The remaining dims must be strictly decreasing in stride in their
    # original order so the source slice (select on b_dim) aligns with the
    # output view without a permute.
    rem_dims = [d for d in range(ndim) if d != b_dim]
    rem_strides = [stride[d] for d in rem_dims]
    rem_sizes = [size[d] for d in rem_dims]
    if any(
        rem_strides[i] <= rem_strides[i + 1] for i in range(len(rem_strides) - 1)
    ):
        return None
    # Innermost remaining stride must be 1 and strides must nest by division
    # so the flat output factors into a dense view.
    if rem_strides[-1] != 1:
        return None
    out_view_shape = []
    for i in range(len(rem_strides) - 1):
        if rem_strides[i] % rem_strides[i + 1] != 0:
            return None
        out_view_shape.append(rem_strides[i] // rem_strides[i + 1])
    if out_numel % rem_strides[0] != 0:
        return None
    out_view_shape.insert(0, out_numel // rem_strides[0])
    # out_view_shape[d] is the extent of remaining dim d in the output view;
    # out_view_shape[d+1..] correspond to rem dims 1.. shifted: entry i of
    # out_view_shape pairs with rem dim i (sizes rem_sizes[i]).
    if len(out_view_shape) != len(rem_sizes):
        return None

    # For each window, decompose the slab offset into per-dim offsets within
    # the output view and check the slice fits.
    per_window_pads = []
    for w in range(num_windows):
        addr = offset + stride[b_dim] * w
        pads_lastdim_first: list[int] = []
        rem = addr
        offs = []
        for i in range(len(rem_dims)):
            st = rem_strides[i]
            off_i = rem // st
            rem -= off_i * st
            offs.append(off_i)
        if rem != 0:
            return None
        for i, off_i in enumerate(offs):
            if off_i + rem_sizes[i] > out_view_shape[i]:
                return None
        # constant_pad_nd pad list: (before_last, after_last, before_2ndlast, ...)
        for i in reversed(range(len(rem_dims))):
            before = offs[i]
            after = out_view_shape[i] - offs[i] - rem_sizes[i]
            pads_lastdim_first.extend([before, after])
        per_window_pads.append(pads_lastdim_first)

    return {
        "overlap_dim": b_dim,
        "num_windows": num_windows,
        "src_shape": list(size),
        "out_view_shape": out_view_shape,
        "per_window_pads": per_window_pads,
    }


def _find_structured_scatter_patterns(graph: fx.Graph) -> list[dict]:
    patterns = []
    for node in graph.nodes:
        if not _is_accumulate_index_put(node):
            continue
        # Only the functional form: replacing an in-place index_put_ would
        # drop the mutation of its base.
        if node.target != aten.index_put.default:
            continue
        base_node = node.args[0]
        indices = node.args[1]
        values_node = node.args[2]
        if not isinstance(base_node, fx.Node) or not isinstance(values_node, fx.Node):
            continue
        if not _is_zeros_init(base_node):
            continue
        if not isinstance(indices, (list, tuple)) or len(indices) != 1:
            continue
        idx_node = indices[0]
        if not isinstance(idx_node, fx.Node):
            continue

        base_val = _get_meta_val(base_node)
        values_val = _get_meta_val(values_node)
        if base_val is None or values_val is None:
            continue
        if base_val.ndim != 1:
            continue
        # accumulate=True is addition for numeric dtypes; bool accumulates
        # via logical-or, which the add-of-pads rewrite does not model.
        if values_val.dtype == torch.bool or base_val.dtype != values_val.dtype:
            continue
        out_numel = base_val.shape[0]
        if not isinstance(out_numel, int):
            continue

        affine = _trace_affine_index(idx_node)
        if affine is None:
            continue
        size, stride, storage_offset = affine

        # The flat values must align elementwise with the flattened index:
        # values numel == prod(size). (We re-view the flat values to `size`.)
        src_numel = values_val.numel()
        if not isinstance(src_numel, int) or src_numel != math.prod(size):
            continue
        if values_val.ndim != 1:
            continue

        structure = _decode_overlap_structure(
            size, stride, storage_offset, out_numel
        )
        if structure is None:
            continue

        # Only rewrite GPU graphs: the win is eliminating the zero-memset +
        # atomic scatter + dense re-read, which are GPU bandwidth costs.
        device = getattr(values_val, "device", None)
        if device is None or device.type != "cuda":
            continue

        patterns.append(
            {
                "index_put_node": node,
                "values_node": values_node,
                "out_numel": out_numel,
                "dtype": values_val.dtype,
                "device": device,
                **structure,
            }
        )
    return patterns


def _rewrite_structured_scatter(graph: fx.Graph, pattern: dict) -> bool:
    index_put_node = pattern["index_put_node"]
    values_node = pattern["values_node"]
    src_shape = pattern["src_shape"]
    b_dim = pattern["overlap_dim"]
    num_windows = pattern["num_windows"]
    out_view_shape = pattern["out_view_shape"]
    per_window_pads = pattern["per_window_pads"]
    out_numel = pattern["out_numel"]
    dtype = pattern["dtype"]

    slice_shape = [s for d, s in enumerate(src_shape) if d != b_dim]

    def _meta(node, shape):
        node.meta = {"val": torch.empty(shape, dtype=dtype, device="meta")}

    with graph.inserting_before(index_put_node):
        src_nd = graph.call_function(
            aten.view.default, args=(values_node, src_shape)
        )
        _meta(src_nd, src_shape)

        acc = None
        for w in range(num_windows):
            part = graph.call_function(
                aten.select.int, args=(src_nd, b_dim, w)
            )
            _meta(part, slice_shape)
            padded = graph.call_function(
                aten.constant_pad_nd.default,
                args=(part, per_window_pads[w], 0),
            )
            _meta(padded, out_view_shape)
            if acc is None:
                acc = padded
            else:
                acc = graph.call_function(
                    aten.add.Tensor, args=(acc, padded)
                )
                _meta(acc, out_view_shape)

        result = graph.call_function(
            aten.view.default, args=(acc, [out_numel])
        )
        if hasattr(index_put_node, "meta"):
            result.meta = dict(index_put_node.meta)

    index_put_node.replace_all_uses_with(result)

    log.info(
        "structured_scatter_decode: REWROTE affine index_put(accumulate=True) "
        "src_shape=%s overlap_dim=%d windows=%d -> overlap-add of padded "
        "slices into view %s (flat %d)",
        src_shape,
        b_dim,
        num_windows,
        out_view_shape,
        out_numel,
    )
    return True


def structured_scatter_decode_pass(graph: fx.Graph) -> fx.Graph:
    """Decode affine-index index_put(accumulate=True) scatters into overlap-adds.

    Controlled by: config.structured_scatter_decode (default True)
    """
    if not getattr(config, "structured_scatter_decode", True):
        return graph

    num_rewritten = 0
    patterns = _find_structured_scatter_patterns(graph)
    if patterns:
        log.info(
            "structured_scatter_decode: found %d structured scatter pattern(s)",
            len(patterns),
        )
        for pattern in patterns:
            if _rewrite_structured_scatter(graph, pattern):
                num_rewritten += 1
                counters["inductor"]["structured_scatter_decode_applied"] += 1

    if num_rewritten > 0:
        graph.eliminate_dead_code()
        graph.lint()

    return graph
