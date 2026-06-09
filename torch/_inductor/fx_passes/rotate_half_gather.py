# mypy: allow-untyped-defs
"""
Rotate-Half Gather Pass: rewrites RoPE-style half-rotation patterns into a
single-load gather with a register-only sign select.

THE PATTERN (RoPE rotate_half, as produced by vllm/HF traces):

  slice_scatter form:
      x1 = slice(src, dim, 0, H)         # first half
      x2 = slice(src, dim, H, D)         # second half
      zeros = full([..., D], 0)
      a = slice_scatter(zeros, [neg] x2, dim, 0, D-H)
      b = slice_scatter(zeros, [neg] x1, dim, D-H, D)
      out = add(a, b)

  cat form:
      out = cat([[neg] slice(src, dim, H, D), [neg] slice(src, dim, 0, H)], dim)

Inductor's slice_scatter / pointwise_cat lowerings turn each output element
into dual masked loads selected by `tl.where(index < bound, ...)`. Both
branches load from the SAME buffer at offsets differing only by +/-H, so the
two loads can be merged into one load at a computed index:

      out[..., j] = sign(j) * src[..., (j + H) % D]
      sign(j) = s_low if j < D - H else s_high

The `(j + H) % D` index is iota-derived, so Inductor's index propagation
folds it into direct ModularIndexing (a register computation, no indirect
load and no device assert), and the sign select becomes a register-only
`tl.where` between a value and its negation.

Gated by config.rotate_half_gather (default True).
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

_INT64_MAX = 9223372036854775807


def _static_int(val) -> Optional[int]:
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return None


def _node_val(node: fx.Node):
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val
    return None


def _static_dim_size(node: fx.Node, dim: int) -> Optional[int]:
    val = _node_val(node)
    if val is None:
        return None
    if dim >= val.ndim:
        return None
    size = val.shape[dim]
    return size if isinstance(size, int) else None


def _is_zero_full(node) -> bool:
    """Check node is aten.full.default filled with 0."""
    if not isinstance(node, fx.Node) or node.op != "call_function":
        return False
    if node.target is not aten.full.default:
        return False
    if len(node.args) < 2:
        return False
    fill = node.args[1]
    return isinstance(fill, (int, float)) and fill == 0


def _match_signed_slice(node) -> Optional[tuple[int, fx.Node, int, int, int]]:
    """Match [neg](slice(src, dim, start, end, step=1)).

    Returns (sign, src, dim, start, end) with start/end normalized to
    non-negative static ints, or None.
    """
    if not isinstance(node, fx.Node) or node.op != "call_function":
        return None
    sign = 1
    if node.target is aten.neg.default:
        sign = -1
        node = node.args[0]
        if not isinstance(node, fx.Node) or node.op != "call_function":
            return None
    if node.target is not aten.slice.Tensor:
        return None
    args = node.args
    src = args[0]
    if not isinstance(src, fx.Node):
        return None
    dim = _static_int(args[1] if len(args) > 1 else node.kwargs.get("dim", 0))
    start = args[2] if len(args) > 2 else node.kwargs.get("start", None)
    end = args[3] if len(args) > 3 else node.kwargs.get("end", None)
    step = args[4] if len(args) > 4 else node.kwargs.get("step", 1)
    if dim is None or step != 1:
        return None
    if dim < 0:
        val = _node_val(src)
        if val is None:
            return None
        dim += val.ndim
    dim_size = _static_dim_size(src, dim)
    if dim_size is None:
        return None
    start = 0 if start is None else _static_int(start)
    end = dim_size if end is None else _static_int(end)
    if start is None or end is None or start < 0:
        return None
    if end > dim_size:
        end = dim_size
    if not (0 <= start < end <= dim_size):
        return None
    return (sign, src, dim, start, end)


def _match_scatter_args(
    node: fx.Node,
) -> Optional[tuple[fx.Node, fx.Node, int, int, int]]:
    """Match slice_scatter(base, src, dim, start, end, step=1).

    Returns (base, src, dim, start, end) normalized, or None.
    """
    args = node.args
    if len(args) < 2:
        return None
    base, src = args[0], args[1]
    if not isinstance(base, fx.Node) or not isinstance(src, fx.Node):
        return None
    dim = _static_int(args[2] if len(args) > 2 else node.kwargs.get("dim", 0))
    start = args[3] if len(args) > 3 else node.kwargs.get("start", None)
    end = args[4] if len(args) > 4 else node.kwargs.get("end", None)
    step = args[5] if len(args) > 5 else node.kwargs.get("step", 1)
    if dim is None or step != 1:
        return None
    if dim < 0:
        val = _node_val(base)
        if val is None:
            return None
        dim += val.ndim
    dim_size = _static_dim_size(base, dim)
    if dim_size is None:
        return None
    start = 0 if start is None else _static_int(start)
    end = dim_size if end is None else _static_int(end)
    if start is None or end is None:
        return None
    if end > dim_size:
        end = dim_size
    if not (0 <= start < end <= dim_size):
        return None
    return (base, src, dim, start, end)


def _match_rotation(node: fx.Node) -> Optional[tuple[fx.Node, int, int, int, int, int]]:
    """Match a half-rotation rooted at `node` (an add of two slice_scatters
    into a zeros base, or a cat of two slices of the same tensor).

    Semantics matched:  out[..., j] = sign(j) * src[..., (j + H) % D]
    where sign(j) = sign_low for j < D - H else sign_high.

    Returns (src, dim, D, H, sign_low, sign_high) or None.
    """
    if node.op != "call_function":
        return None

    chunks = None  # (low_signed_slice, high_signed_slice, dim, D)
    if node.target is aten.add.Tensor:
        if len(node.args) != 2:
            return None
        a, b = node.args
        if not (
            isinstance(a, fx.Node)
            and isinstance(b, fx.Node)
            and a.op == "call_function"
            and b.op == "call_function"
            and a.target is aten.slice_scatter.default
            and b.target is aten.slice_scatter.default
        ):
            return None
        # Profitability: the scatters must exist solely to feed this add.
        if len(a.users) != 1 or len(b.users) != 1:
            return None
        ma = _match_scatter_args(a)
        mb = _match_scatter_args(b)
        if ma is None or mb is None:
            return None
        # Same zeros base, same dim.
        if ma[0] is not mb[0] or not _is_zero_full(ma[0]):
            return None
        if ma[2] != mb[2]:
            return None
        dim = ma[2]
        D = _static_dim_size(ma[0], dim)
        if D is None:
            return None
        # One scatter covers [0, K), the other [K, D).
        if ma[3] == 0 and mb[3] == ma[4] and mb[4] == D:
            low_scatter, high_scatter = ma, mb
        elif mb[3] == 0 and ma[3] == mb[4] and ma[4] == D:
            low_scatter, high_scatter = mb, ma
        else:
            return None
        low = _match_signed_slice(low_scatter[1])
        high = _match_signed_slice(high_scatter[1])
        if low is None or high is None:
            return None
        chunks = (low, high, dim, D)
    elif node.target is aten.cat.default:
        tensors = node.args[0]
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
            return None
        cat_dim = _static_int(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        )
        if cat_dim is None:
            return None
        out_val = _node_val(node)
        if out_val is None:
            return None
        if cat_dim < 0:
            cat_dim += out_val.ndim
        D = _static_dim_size(node, cat_dim)
        if D is None:
            return None
        low = _match_signed_slice(tensors[0])
        high = _match_signed_slice(tensors[1])
        if low is None or high is None:
            return None
        chunks = (low, high, cat_dim, D)
    else:
        return None

    low, high, dim, D = chunks
    sign_low, src_low, dim_low, start_low, end_low = low
    sign_high, src_high, dim_high, start_high, end_high = high
    # Both chunks must slice the same tensor along the rotation dim, and the
    # source must have the same dim size as the output (a pure rotation).
    if src_low is not src_high or dim_low != dim or dim_high != dim:
        return None
    if _static_dim_size(src_low, dim) != D:
        return None
    # low chunk (output [0, D-H)) reads src[H, D); high chunk reads src[0, H).
    H = start_low
    if H <= 0 or end_low != D:
        return None
    if start_high != 0 or end_high != H:
        return None
    # The rewrite produces a tensor with src's shape and dtype; require an
    # exact match with the matched node's output (slice_scatter can expand /
    # type-convert its src, in which case the rewrite would be wrong).
    src_val = _node_val(src_low)
    out_val = _node_val(node)
    if src_val is None or out_val is None:
        return None
    if [*src_val.shape] != [*out_val.shape] or src_val.dtype != out_val.dtype:
        return None
    return (src_low, dim, D, H, sign_low, sign_high)


def _build_gather(
    graph: fx.Graph,
    anchor: fx.Node,
    src: fx.Node,
    dim: int,
    D: int,
    H: int,
    sign_low: int,
    sign_high: int,
) -> Optional[fx.Node]:
    """Emit out[..., j] = sign(j) * src[..., (j + H) % D] before `anchor`."""
    src_val = _node_val(src)
    if src_val is None:
        return None
    fake_mode = getattr(src_val, "fake_mode", None)
    if fake_mode is None:
        return None
    device = src_val.device
    ndim = src_val.ndim

    def mk(target, args, kwargs=None, meta_fn=None):
        n = graph.call_function(target, tuple(args), kwargs or {})
        with fake_mode:
            fake_args = tuple(
                _node_val(a)
                if isinstance(a, fx.Node)
                else (
                    [_node_val(x) if isinstance(x, fx.Node) else x for x in a]
                    if isinstance(a, (list, tuple))
                    else a
                )
                for a in args
            )
            n.meta["val"] = target(*fake_args, **(kwargs or {}))
        return n

    with graph.inserting_before(anchor):
        # idx[j] = j + H        for j < D - H   (reads the high chunk)
        #          j - (D - H)  otherwise       (reads the low chunk)
        # The where on an iota stays pure index arithmetic (register-only);
        # deliberately NOT remainder((j + H), D): integer remainder lowers to
        # srem (an emulated division loop on GPU), which is much slower.
        base = mk(
            prims.iota.default,
            (D,),
            {
                "start": 0,
                "step": 1,
                "dtype": torch.int64,
                "device": device,
                "requires_grad": False,
            },
        )
        cond = mk(aten.lt.Scalar, (base, D - H))
        idx = mk(
            aten.where.self,
            (cond, mk(aten.add.Tensor, (base, H)), mk(aten.sub.Tensor, (base, D - H))),
        )
        index_args: list = [None] * ndim
        index_args[dim] = idx
        # _unsafe_index: the index is in [0, D) by construction; skip the
        # device assert so the load stays branch-free.
        gathered = mk(aten._unsafe_index.Tensor, (src, index_args))
        if sign_low == 1 and sign_high == 1:
            result = gathered
        elif sign_low == -1 and sign_high == -1:
            result = mk(aten.neg.default, (gathered,))
        else:
            # Multiply by a +/-1 vector computed from the iota (register-only;
            # folds into the kernel as index arithmetic). Deliberately NOT
            # where(cond, gathered, -gathered): that gives the gather node two
            # users, which triggers realize-on-reuse for indirect-indexing
            # producers and splits the gather into its own kernel.
            cond_f = mk(prims.convert_element_type.default, (cond, src_val.dtype))
            if sign_low == 1:
                # +1 where cond else -1:  2 * cond - 1
                sign = mk(aten.sub.Scalar, (mk(aten.mul.Scalar, (cond_f, 2)), 1))
            else:
                # -1 where cond else +1:  -2 * cond + 1
                sign = mk(aten.add.Scalar, (mk(aten.mul.Scalar, (cond_f, -2)), 1))
            trailing = ndim - 1 - dim
            if trailing > 0:
                sign = mk(aten.reshape.default, (sign, [D] + [1] * trailing))
            result = mk(aten.mul.Tensor, (gathered, sign))
    return result


def rotate_half_gather_pass(graph: fx.Graph) -> fx.Graph:
    """Rewrite RoPE half-rotation (slice_scatter/cat of two opposite halves of
    the same tensor) into a single-load modular gather with sign select."""
    num_rewrites = 0
    for node in list(graph.nodes):
        match = _match_rotation(node)
        if match is None:
            continue
        src, dim, D, H, sign_low, sign_high = match
        result = _build_gather(graph, node, src, dim, D, H, sign_low, sign_high)
        if result is None:
            continue
        node.replace_all_uses_with(result)
        num_rewrites += 1
        counters["inductor"]["rotate_half_gather"] += 1
        log.debug(
            "rotate_half_gather: rewrote %s (dim=%d, D=%d, H=%d, signs=(%d, %d))",
            node.name,
            dim,
            D,
            H,
            sign_low,
            sign_high,
        )

    if num_rewrites > 0:
        graph.eliminate_dead_code()
        graph.lint()
        log.info("rotate_half_gather: rewrote %d half-rotation patterns", num_rewrites)
    return graph
