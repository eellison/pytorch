# mypy: allow-untyped-defs
"""
Scatter-Reduce Fusion Pass: Eliminates scatter intermediates when followed by
full-dimension reductions.

Pattern:
    full(zeros) -> index_put(..., accumulate=True) x N -> [pointwise] -> sum(dim=all_scattered_dims)

When a scatter (index_put with accumulate=True) writes into a zero-initialized tensor,
and the result is subsequently reduced (summed) over ALL dimensions that the scatter
writes to, the scatter intermediate can be eliminated.

The key algebraic identity:
    sum(scatter_add(zeros, idx, values), dim=scattered_dims) == sum(values, dim=source_dims)

When a pointwise mask intervenes (e.g., ReLU gate from batch norm):
    sum(mask * scatter_add(zeros, idx, values), dim=all_dims)
    == sum(gather(mask, reverse_idx) * values, dim=source_dims)

This pass detects the pattern and rewrites it to avoid materializing the large
intermediate scatter buffer, eliminating expensive atomic operations.

Target patterns (from UNet backward):
- Bilinear upsample backward: 4x index_put(accumulate=True) -> add -> where(mask) -> sum
- MaxPool backward: scatter_add -> mask -> sum
"""

import logging
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)


# ============================================================================
# Custom op for masked scatter-reduce-gather fusion
# ============================================================================

_scatter_reduce_gather_lib = torch.library.Library(
    "scatter_reduce_fusion", "DEF"
)

_scatter_reduce_gather_lib.define(
    "masked_scatter_reduce(Tensor[] sources, Tensor[] row_indices, "
    "Tensor[] col_indices, Tensor mask, Tensor mask_fill_value, "
    "int[] output_shape) -> Tensor"
)

_scatter_reduce_gather_lib.define(
    "scatter_add_masked_reduce(Tensor src, Tensor index, Tensor mask, "
    "Tensor mask_fill_value, int scatter_dim, int num_channels, "
    "int output_spatial, int[] view_shape) -> Tensor"
)


def _masked_scatter_reduce_impl(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    """Implementation of fused scatter-reduce-gather.

    Computes: sum(where(mask, fill_value, scatter_add(zeros, indices, sources)), reduce_dims)

    This equals:
        gather_reduce_result + count_masked_per_channel * fill_value

    Where gather_reduce_result iterates over source positions and accumulates
    unmasked contributions per channel.
    """
    from torch._inductor.kernel.scatter_reduce_gather import (
        scatter_reduce_gather_fast,
    )
    # Compute the gather-reduce of unmasked scattered values
    gather_result = scatter_reduce_gather_fast(
        sources, row_indices, col_indices, mask, tuple(output_shape)
    )

    # Add the contribution from masked positions: count_true_per_channel * fill_value
    # mask shape: [B, C, H, W], reduction over [0, 2, 3] -> [C]
    fill_val = mask_fill_value.float()
    if fill_val.item() != 0.0:
        # Count True mask values per channel (summing over batch and spatial dims)
        count_true_per_channel = mask.float().sum(dim=[0, 2, 3])  # [C]
        gather_result = gather_result + count_true_per_channel * fill_val

    return gather_result


@torch.library.impl(
    _scatter_reduce_gather_lib, "masked_scatter_reduce", "CUDA"
)
def _masked_scatter_reduce_cuda(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    return _masked_scatter_reduce_impl(
        sources, row_indices, col_indices, mask, mask_fill_value, output_shape
    )


@torch.library.impl(
    _scatter_reduce_gather_lib, "masked_scatter_reduce", "CPU"
)
def _masked_scatter_reduce_cpu(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    return _masked_scatter_reduce_impl(
        sources, row_indices, col_indices, mask, mask_fill_value, output_shape
    )


@torch.library.impl(
    _scatter_reduce_gather_lib, "masked_scatter_reduce", "Meta"
)
def _masked_scatter_reduce_meta(
    sources: list[torch.Tensor],
    row_indices: list[torch.Tensor],
    col_indices: list[torch.Tensor],
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    output_shape: list[int],
) -> torch.Tensor:
    """Meta/FakeTensor implementation for shape inference."""
    # Result is [C] where C = output_shape[1] (channel dimension preserved)
    C = output_shape[1]
    return torch.empty(C, device=sources[0].device, dtype=sources[0].dtype)


# ============================================================================
# Custom op for scatter_add + where + sum (single scatter, maxpool pattern)
# ============================================================================


def _scatter_add_masked_reduce_impl(
    src: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    scatter_dim: int,
    num_channels: int,
    output_spatial: int,
    view_shape: list[int],
) -> torch.Tensor:
    """Implementation of fused scatter_add + where + sum.

    Computes:
        scattered = zeros.scatter_add(scatter_dim, index, src)
        viewed = scattered.view(view_shape)   # e.g. [B, C, H, W]
        masked = where(mask, fill_value, viewed)
        result = sum(masked, dim=[0, 2, 3])   # reduce all except channel

    Source-side approach: instead of materializing the full scattered intermediate,
    iterate over source elements per-channel, gather the mask at dest positions,
    mask the source, and accumulate.

    The key optimization vs the original: we avoid allocating [BC, S_out] and
    doing random atomic writes. Instead we do per-batch-channel sequential
    mask gathers from [H, W] which is much smaller.

    Args:
        src: [BC, S_src] source values (2D, flattened batch*channels x spatial)
        index: [BC, S_src] scatter indices (same shape as src)
        mask: [B, C, H, W] boolean mask (True = use fill_value)
        mask_fill_value: scalar tensor (the fill value for masked positions)
        scatter_dim: dimension for scatter (typically 1)
        num_channels: C
        output_spatial: H * W (output spatial size)
        view_shape: [B, C, H, W]
    """
    B, C, H, W = view_shape[0], view_shape[1], view_shape[2], view_shape[3]
    device = src.device
    dtype = src.dtype
    BC = src.shape[0]
    S_src = src.shape[1]

    # Reshape src and index from [BC, S_src] to [B, C, S_src]
    src_3d = src.view(B, C, S_src)
    index_3d = index.view(B, C, S_src)

    # mask is [B, C, H, W] -> view as [B, C, H*W] for spatial indexing
    mask_flat_spatial = mask.view(B, C, H * W)  # [B, C, S_out]

    # Gather mask at destination positions:
    # For each source element (b, c, s), its dest spatial position is index_3d[b, c, s]
    # We want mask_flat_spatial[b, c, index_3d[b, c, s]]
    # This is a gather along the last dimension
    gathered_mask = torch.gather(mask_flat_spatial, 2, index_3d)  # [B, C, S_src]

    # Zero out masked source contributions
    # mask=True means "use fill_value" (source contribution suppressed)
    unmasked_src = src_3d * (~gathered_mask).to(dtype)  # [B, C, S_src]

    # Sum over batch and source spatial dims -> [C]
    result = unmasked_src.sum(dim=[0, 2])  # [C]

    # Add fill_value contribution for masked positions
    fill_val = mask_fill_value.to(dtype)
    if fill_val.item() != 0.0:
        # Count True mask values per channel (over batch and spatial)
        count_masked_per_channel = mask.to(dtype).sum(dim=[0, 2, 3])  # [C]
        result = result + count_masked_per_channel * fill_val

    return result


@torch.library.impl(
    _scatter_reduce_gather_lib, "scatter_add_masked_reduce", "CUDA"
)
def _scatter_add_masked_reduce_cuda(
    src: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    scatter_dim: int,
    num_channels: int,
    output_spatial: int,
    view_shape: list[int],
) -> torch.Tensor:
    return _scatter_add_masked_reduce_impl(
        src, index, mask, mask_fill_value, scatter_dim,
        num_channels, output_spatial, view_shape
    )


@torch.library.impl(
    _scatter_reduce_gather_lib, "scatter_add_masked_reduce", "CPU"
)
def _scatter_add_masked_reduce_cpu(
    src: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    scatter_dim: int,
    num_channels: int,
    output_spatial: int,
    view_shape: list[int],
) -> torch.Tensor:
    return _scatter_add_masked_reduce_impl(
        src, index, mask, mask_fill_value, scatter_dim,
        num_channels, output_spatial, view_shape
    )


@torch.library.impl(
    _scatter_reduce_gather_lib, "scatter_add_masked_reduce", "Meta"
)
def _scatter_add_masked_reduce_meta(
    src: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    mask_fill_value: torch.Tensor,
    scatter_dim: int,
    num_channels: int,
    output_spatial: int,
    view_shape: list[int],
) -> torch.Tensor:
    """Meta/FakeTensor implementation for shape inference."""
    return torch.empty(num_channels, device=src.device, dtype=src.dtype)


log = logging.getLogger(__name__)
aten = torch.ops.aten


scatter_reduce_fusion_patterns = PatternMatcherPass(
    pass_name="scatter_reduce_fusion"
)


def scatter_reduce_fusion_pass(graph: fx.Graph) -> fx.Graph:
    """
    Detect scatter-then-reduce patterns and rewrite as gather-reduce.

    This eliminates the large intermediate tensor and expensive atomic operations
    when a scatter result is fully reduced over all scattered dimensions.

    Controlled by: config.scatter_reduce_fusion
    """
    if not getattr(config, "scatter_reduce_fusion", False):
        return graph

    num_rewritten = 0

    # Phase 1a: Detect scatter_add -> view -> where -> sum chains (MaxPool pattern)
    # Uses aten.index.Tensor (not aten.gather) to avoid unbacked symbol issues
    # in FakeTensor propagation, matching the approach used in the bilinear path.
    scatter_add_chains = _find_scatter_add_reduce_chains(graph)
    if scatter_add_chains:
        log.info(
            "scatter_reduce_fusion: found %d scatter_add-reduce chain(s)",
            len(scatter_add_chains),
        )
        for chain in scatter_add_chains:
            if _rewrite_scatter_add_reduce_chain(graph, chain):
                num_rewritten += 1
                counters["inductor"]["scatter_reduce_fusion_applied"] += 1

    # Phase 1b: Analysis pass - find index_put scatter-then-reduce chains (bilinear)
    matches = _find_scatter_reduce_chains(graph)
    if matches:
        log.info(
            "scatter_reduce_fusion: found %d scatter-reduce chain(s)",
            len(matches),
        )
        # Phase 2: Rewrite pass - replace scatter-reduce with gather-reduce
        for chain in matches:
            if _rewrite_scatter_reduce_chain(graph, chain):
                num_rewritten += 1
                counters["inductor"]["scatter_reduce_fusion_applied"] += 1

    # Phase 1c: Detect add(A, index_put(zeros, idx, val, accumulate=True))
    # and rewrite to index_put(A, idx, val, accumulate=True).
    # This eliminates the zeros initialization and the add kernel.
    # Pattern: embedding backward where scattered gradient is added to existing weight grad.
    scatter_add_into_chains = _find_scatter_add_into_patterns(graph)
    if scatter_add_into_chains:
        log.info(
            "scatter_reduce_fusion: found %d scatter-add-into pattern(s)",
            len(scatter_add_into_chains),
        )
        for chain_info in scatter_add_into_chains:
            if _rewrite_scatter_add_into(graph, chain_info):
                num_rewritten += 1
                counters["inductor"]["scatter_add_into_fusion_applied"] += 1

    if num_rewritten > 0:
        log.info(
            "scatter_reduce_fusion: rewrote %d chain(s)", num_rewritten
        )
        graph.eliminate_dead_code()
        graph.lint()

    return graph


class ScatterReduceChain:
    """Represents a detected scatter-then-reduce pattern."""

    def __init__(self):
        # The full/zeros node that initializes the scatter target
        self.zeros_node: Optional[fx.Node] = None
        # List of index_put nodes that scatter into the zeros buffer
        self.scatter_nodes: list[fx.Node] = []
        # Add nodes that combine multiple scatter results
        self.combine_adds: list[fx.Node] = []
        # The combined scatter result node (after all adds)
        self.combined_scatter: Optional[fx.Node] = None
        # Optional mask/where node
        self.mask_node: Optional[fx.Node] = None
        # The reduction node(s) that consume the scattered result
        self.reduction_nodes: list[fx.Node] = []
        # Reduction dimensions
        self.reduction_dims: Optional[list[int]] = None
        # Shape of the scatter output
        self.scatter_output_shape: Optional[list[int]] = None
        # Whether the pattern has an intervening mask
        self.has_mask: bool = False
        # All intermediate nodes that can be eliminated
        self.intermediate_nodes: list[fx.Node] = []


class ScatterAddReduceChain:
    """Represents a scatter_add -> [view] -> [add(skip, ...)] -> where -> sum pattern.

    This is the MaxPool backward pattern (VGG16, SqueezeNet):
        full_default[BC, S_out] = 0
        scatter_add(full_default, dim=1, index[BC, S_src], src[BC, S_src])
        view -> [B, C, H, W]
        where(mask[B, C, H, W], fill_value, viewed)
        sum(dim=[0, 2, 3]) -> [C]

    Extended for BN backward (ShuffleNet, UNet) where where_self has multiple users:
        - sum(where_self, [0,2,3]) -> rewritable
        - mul(where_self, other) -> sum([0,2,3]) -> rewritable
        - sub(where_self, ...) -> NOT rewritable (partial rewrite)

    Extended for UNet skip-connection pattern:
        scatter_add -> view -> add(skip, viewed) -> where -> sum
        (the add between scatter and where must be decomposed)
    """

    def __init__(self):
        self.zeros_node: Optional[fx.Node] = None
        self.scatter_add_node: Optional[fx.Node] = None
        self.scatter_dim: int = 1
        self.view_node: Optional[fx.Node] = None
        self.view_shape: Optional[list[int]] = None
        self.mask_node: Optional[fx.Node] = None  # the where node
        self.condition_node: Optional[fx.Node] = None  # the bool mask
        self.fill_value_node: Optional[fx.Node] = None  # scalar fill
        self.sum_node: Optional[fx.Node] = None
        self.reduction_dims: Optional[list[int]] = None
        self.src_node: Optional[fx.Node] = None
        self.index_node: Optional[fx.Node] = None
        # For multi-user where (partial rewrite):
        # List of (sum_node, optional_multiplier_node) pairs that can be rewritten
        self.rewrite_targets: list[tuple[fx.Node, Optional[fx.Node]]] = []
        # For skip-connection pattern (UNet):
        # The add node between view and where, and the skip tensor
        self.skip_add_node: Optional[fx.Node] = None
        self.skip_tensor_node: Optional[fx.Node] = None


def _get_tensor_meta(node: fx.Node) -> Optional[dict]:
    """Extract tensor metadata from a node."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None
    val = node.meta["val"]
    if not isinstance(val, torch.Tensor) and not hasattr(val, "shape"):
        # It's a FakeTensor
        pass
    if hasattr(val, "shape"):
        return {
            "shape": list(val.shape),
            "dtype": val.dtype,
            "numel": val.numel() if hasattr(val, "numel") else 0,
            "ndim": len(val.shape),
        }
    return None


def _is_zeros_init(node: fx.Node) -> bool:
    """Check if a node creates a zero-initialized tensor (full(..., 0))."""
    if node.op != "call_function":
        return False
    if node.target == aten.full.default:
        # full(shape, fill_value, ...)
        if len(node.args) >= 2:
            fill_value = node.args[1]
            return fill_value == 0 or fill_value == 0.0
    if node.target == aten.zeros.default:
        return True
    return False


def _is_accumulate_index_put(node: fx.Node) -> bool:
    """Check if a node is index_put with accumulate=True."""
    if node.op != "call_function":
        return False
    if node.target not in (aten.index_put.default, aten.index_put_.default):
        return False
    # Check accumulate argument (4th positional arg)
    if len(node.args) >= 4 and node.args[3] is True:
        return True
    # Check keyword argument
    if node.kwargs.get("accumulate", False) is True:
        return True
    return False


def _is_scatter_add(node: fx.Node) -> bool:
    """Check if a node is scatter_add (scatter_add(input, dim, index, src))."""
    if node.op != "call_function":
        return False
    return node.target in (aten.scatter_add.default, aten.scatter_add_.default)


def _get_scatter_add_info(node: fx.Node) -> Optional[dict]:
    """Extract scatter_add information.

    scatter_add(input, dim, index, src) where input is zeros.

    Returns dict with:
        - input_node: the target tensor (should be zeros)
        - dim: scatter dimension
        - index_node: the index tensor
        - src_node: the source tensor
    """
    if not _is_scatter_add(node):
        return None

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]
    src_node = node.args[3]

    if not isinstance(input_node, fx.Node):
        return None
    if not isinstance(dim, int):
        return None
    if not isinstance(index_node, fx.Node):
        return None
    if not isinstance(src_node, fx.Node):
        return None

    return {
        "input_node": input_node,
        "dim": dim,
        "index_node": index_node,
        "src_node": src_node,
    }


def _is_sum_reduction(node: fx.Node) -> Optional[list[int]]:
    """Check if node is sum reduction, return reduction dims or None."""
    if node.op != "call_function":
        return None
    if node.target == aten.sum.dim_IntList:
        if len(node.args) >= 2:
            dims = node.args[1]
            if isinstance(dims, (list, tuple)):
                return list(dims)
    return None


def _get_scatter_indices_info(node: fx.Node) -> Optional[dict]:
    """Extract scatter index information from an index_put node.

    Returns dict with:
        - indices: list of index tensors (or None for unindexed dims)
        - indexed_dims: list of dimensions that are actually indexed
        - values_node: the values being scattered
        - input_node: the target tensor
    """
    if not _is_accumulate_index_put(node):
        return None

    input_node = node.args[0]
    indices = node.args[1]  # list of index tensors or None
    values_node = node.args[2]

    if not isinstance(indices, (list, tuple)):
        return None

    indexed_dims = []
    for i, idx in enumerate(indices):
        if idx is not None:
            indexed_dims.append(i)

    return {
        "input_node": input_node,
        "indices": list(indices),
        "indexed_dims": indexed_dims,
        "values_node": values_node,
    }


def _reduction_covers_scatter_dims(
    reduction_dims: list[int], scatter_output_ndim: int, indexed_dims: list[int]
) -> bool:
    """Check if the reduction dimensions cover all the scattered (indexed) dimensions.

    For the scatter-reduce fusion to be valid, all dimensions that are indexed
    by the scatter must be reduced over. Non-indexed dimensions (those with None
    in the indices list) must also be reduced if they are spatial dims.

    In the UNet pattern, indices are [None, None, row_idx, col_idx] meaning
    dims 2,3 are scattered. The reduction is over [0, 2, 3] which covers both
    scattered dims. This is valid because dim 0 (batch) just broadcasts.
    """
    # Normalize negative dims
    normalized_reduction = [d % scatter_output_ndim for d in reduction_dims]

    # All indexed dimensions must be in the reduction set
    for dim in indexed_dims:
        if dim not in normalized_reduction:
            return False

    return True


def _is_view_node(node: fx.Node) -> bool:
    """Check if a node is a view/reshape operation."""
    if node.op != "call_function":
        return False
    return node.target in (
        aten.view.default,
        aten.reshape.default,
        aten._unsafe_view.default,
    )


def _find_scatter_add_reduce_chains(graph: fx.Graph) -> list[ScatterAddReduceChain]:
    """Find scatter_add -> view -> where -> sum patterns.

    The MaxPool backward pattern (VGG16, SqueezeNet):
        full[BC, S_out] = 0
        scatter_add(full, 1, index, src) -> [BC, S_out]
        view([B, C, H, W]) -> [B, C, H, W]
        where(mask, fill, viewed) -> [B, C, H, W]
        sum([0, 2, 3]) -> [C]

    Extended BN backward pattern (ShuffleNet, UNet):
        scatter_add -> view -> [optional add(skip, viewed)] -> where(mask, fill, ...)
        where has multiple users:
          - sum(where, [0,2,3]) -> rewritable
          - mul(where, other) -> sum([0,2,3]) -> rewritable
          - sub(where, ...) -> NOT rewritable (partial rewrite: scatter stays alive)
    """
    chains = []
    # Track which sum nodes have already been claimed by a chain to avoid
    # duplicates when multiple sums trace back to the same scatter
    claimed_sums = set()

    for node in graph.nodes:
        # Start from sum reduction nodes
        reduction_dims = _is_sum_reduction(node)
        if reduction_dims is None:
            continue
        if node in claimed_sums:
            continue

        chain = _trace_scatter_add_chain(node, reduction_dims)
        if chain is not None:
            # Mark all rewrite targets as claimed
            claimed_sums.add(node)
            for target_sum, _ in chain.rewrite_targets:
                claimed_sums.add(target_sum)
            chains.append(chain)

    return chains


def _trace_scatter_add_chain(
    sum_node: fx.Node, reduction_dims: list[int]
) -> Optional[ScatterAddReduceChain]:
    """Trace backwards from a sum node to find scatter_add -> view -> where -> sum.

    Handles multi-user where nodes (partial rewrite) and skip-connection adds.

    Returns a ScatterAddReduceChain if the pattern matches, None otherwise.
    """
    chain = ScatterAddReduceChain()
    chain.sum_node = sum_node
    chain.reduction_dims = reduction_dims

    # Validate reduction dims first - we only handle [0, 2, 3]
    normalized_dims = sorted([d % 4 for d in reduction_dims])
    if normalized_dims != [0, 2, 3]:
        return None

    # Get the input to the sum
    sum_input = sum_node.args[0]
    if not isinstance(sum_input, fx.Node):
        return None

    # Check if input is directly where, or mul(where, other) -> we trace through
    where_node = None
    if sum_input.op == "call_function" and sum_input.target == aten.where.self:
        where_node = sum_input
    elif sum_input.op == "call_function" and sum_input.target == aten.mul.Tensor:
        # sum(mul(where, other), dims) - check if either mul arg is a where
        for arg in sum_input.args:
            if isinstance(arg, fx.Node) and arg.op == "call_function" and arg.target == aten.where.self:
                where_node = arg
                break

    if where_node is None:
        return None

    # Extract where components: where(condition, fill_value, scattered_viewed)
    condition_node = where_node.args[0]
    fill_value_node = where_node.args[1]
    scattered_viewed_node = where_node.args[2]

    if not isinstance(condition_node, fx.Node):
        return None
    if not isinstance(fill_value_node, fx.Node):
        return None
    if not isinstance(scattered_viewed_node, fx.Node):
        return None

    chain.mask_node = where_node
    chain.condition_node = condition_node
    chain.fill_value_node = fill_value_node

    # Classify ALL users of where_self to determine rewrite targets
    # (multi-user partial rewrite support for BN backward pattern)
    where_users = list(where_node.users.keys())
    rewrite_targets: list[tuple[fx.Node, Optional[fx.Node]]] = []
    non_rewritable_users: list[fx.Node] = []

    for user in where_users:
        if user.op == "call_function":
            dims = _is_sum_reduction(user)
            if dims is not None:
                norm = sorted([d % 4 for d in dims])
                if norm == [0, 2, 3]:
                    rewrite_targets.append((user, None))
                    continue

            if user.target == aten.mul.Tensor:
                # mul(where_self, other) -> check if result goes to sum
                mul_users = list(user.users.keys())
                if len(mul_users) == 1:
                    mul_user = mul_users[0]
                    dims = _is_sum_reduction(mul_user)
                    if dims is not None:
                        norm = sorted([d % 4 for d in dims])
                        if norm == [0, 2, 3]:
                            # Identify the "other" multiplier
                            other_node = user.args[1] if user.args[0] == where_node else user.args[0]
                            if isinstance(other_node, fx.Node):
                                rewrite_targets.append((mul_user, other_node))
                                continue

        non_rewritable_users.append(user)

    if not rewrite_targets:
        return None

    # Partial rewrite profitability check:
    # If non-rewritable users exist, the scatter intermediate must still materialize.
    # In that case, the sum operations reading from `where_self` are cheap since the
    # buffer is already in cache. Adding gather operations would be strictly worse.
    # Only proceed if ALL users are rewritable (scatter can be fully eliminated).
    if non_rewritable_users:
        log.debug(
            "scatter_reduce_fusion: skipping scatter_add chain - where_self has "
            "%d non-rewritable user(s) (%s), scatter cannot be eliminated",
            len(non_rewritable_users),
            [u.name for u in non_rewritable_users[:3]],
        )
        return None

    chain.rewrite_targets = rewrite_targets

    # Trace backwards from where to find the scatter_add
    # Handle optional skip-connection add: scatter_add -> view -> add(skip, viewed)
    current = scattered_viewed_node

    # Check for add node (UNet skip-connection pattern):
    # where(mask, fill, add(skip, scatter_view))
    if (current.op == "call_function" and current.target == aten.add.Tensor):
        # One arg should trace back to scatter_add, the other is the skip tensor
        add_node = current
        arg0 = current.args[0]
        arg1 = current.args[1]

        # Try both orderings: add(skip, scatter_view) or add(scatter_view, skip)
        scatter_arg = None
        skip_arg = None
        for a, b in [(arg0, arg1), (arg1, arg0)]:
            if isinstance(a, fx.Node) and _is_view_node(a):
                # Check if view's input is scatter_add
                view_input = a.args[0]
                if isinstance(view_input, fx.Node) and _is_scatter_add(view_input):
                    scatter_arg = a
                    skip_arg = b
                    break
            elif isinstance(a, fx.Node) and _is_scatter_add(a):
                scatter_arg = a
                skip_arg = b
                break

        if scatter_arg is not None and isinstance(skip_arg, fx.Node):
            chain.skip_add_node = add_node
            chain.skip_tensor_node = skip_arg

            # Check that the add is only used by the where
            add_users = list(add_node.users.keys())
            if len(add_users) != 1 or add_users[0] != where_node:
                return None

            current = scatter_arg
        # else: not the skip pattern, fall through to check if it's view/scatter directly

    # Check for view node between scatter and where (or between scatter and add)
    if _is_view_node(current):
        chain.view_node = current
        view_meta = _get_tensor_meta(current)
        if view_meta is not None:
            chain.view_shape = view_meta["shape"]

        # Check the view is only used by the where or the skip-add
        view_users = list(current.users.keys())
        expected_view_user = chain.skip_add_node if chain.skip_add_node is not None else chain.mask_node
        if len(view_users) != 1 or view_users[0] != expected_view_user:
            return None

        current = current.args[0]
        if not isinstance(current, fx.Node):
            return None
    else:
        # No view - check metadata for shape
        view_meta = _get_tensor_meta(current)
        if view_meta is not None:
            chain.view_shape = view_meta["shape"]

    # Now current should be the scatter_add node
    if not _is_scatter_add(current):
        return None

    scatter_info = _get_scatter_add_info(current)
    if scatter_info is None:
        return None

    chain.scatter_add_node = current
    chain.scatter_dim = scatter_info["dim"]
    chain.src_node = scatter_info["src_node"]
    chain.index_node = scatter_info["index_node"]

    # Check scatter_add is only used by the view (or directly by where/add if no view)
    scatter_users = list(current.users.keys())
    if chain.view_node is not None:
        expected_user = chain.view_node
    elif chain.skip_add_node is not None:
        expected_user = chain.skip_add_node
    else:
        expected_user = chain.mask_node
    if len(scatter_users) != 1 or scatter_users[0] != expected_user:
        return None

    # Check the input to scatter_add is zeros
    input_node = scatter_info["input_node"]
    if not _is_zeros_init(input_node):
        return None
    chain.zeros_node = input_node

    # Validate: view_shape must be 4D
    if chain.view_shape is None or len(chain.view_shape) != 4:
        return None

    log.debug(
        "Found scatter_add-reduce chain: scatter_dim=%d, view_shape=%s, "
        "reduction_dims=%s, num_rewrite_targets=%d, has_skip=%s",
        chain.scatter_dim,
        chain.view_shape,
        reduction_dims,
        len(rewrite_targets),
        chain.skip_add_node is not None,
    )

    return chain


def _rewrite_scatter_add_reduce_chain(
    graph: fx.Graph, chain: ScatterAddReduceChain
) -> bool:
    """Rewrite scatter_add -> view -> [add(skip)] -> where -> sum using aten ops.

    Supports partial rewrites: when where_self has multiple users (BN backward),
    we rewrite only the sum-reduction users and leave the scatter alive for others.

    For skip-connection pattern (UNet): where(mask, fill, scatter_view + skip),
    we decompose sum(where(mask, fill, A+B), dims) into scatter and skip parts:
        result = scatter_contribution + skip_contribution + fill_contribution

    Original (simple):
        zeros[BC, S_out] -> scatter_add(zeros, 1, index, src) -> view[B,C,H,W]
        -> where(mask, fill, viewed) -> sum([0,2,3])

    Rewritten (simple):
        src_3d = view(src, [B, C, S_src])
        index_3d = view(index, [B, C, S_src])
        mask_flat = view(mask, [B, C, H*W])
        gathered_mask = index(mask_flat, [batch_idx, chan_idx, index_3d])
        unmasked_src = where(gathered_mask, 0, src_3d)
        result = sum(unmasked_src, [0, 2]) + fill * sum(mask, [0,2,3])

    Rewritten (with skip-connection):
        scatter_part = sum(where(gathered_mask, 0, src_3d), [0, 2])
        skip_part = sum(where(mask, 0, skip), [0, 2, 3])
        fill_part = fill * sum(mask, [0, 2, 3])
        result = scatter_part + skip_part + fill_part
    """
    if chain.scatter_add_node is None:
        return False
    if chain.src_node is None or chain.index_node is None:
        return False
    if chain.condition_node is None or chain.fill_value_node is None:
        return False
    if chain.view_shape is None:
        return False
    if not chain.rewrite_targets:
        return False

    B, C, H, W = chain.view_shape

    # Get src shape metadata to determine S_src
    src_meta = _get_tensor_meta(chain.src_node)
    if src_meta is None or len(src_meta["shape"]) != 2:
        return False
    S_src = src_meta["shape"][1]

    # Get index dtype for casting
    index_meta = _get_tensor_meta(chain.index_node)
    if index_meta is None:
        return False

    has_skip = chain.skip_add_node is not None

    # Process each rewrite target (sum node) independently
    num_rewritten = 0
    for target_sum_node, multiplier_node in chain.rewrite_targets:
        with graph.inserting_before(target_sum_node):
            # === Build the scatter contribution via gather-mask-reduce ===
            # src_3d = view(src, [B, C, S_src])
            src_3d = graph.call_function(
                aten.view.default,
                args=(chain.src_node, [B, C, S_src]),
            )
            if hasattr(chain.src_node, "meta") and "val" in chain.src_node.meta:
                val = chain.src_node.meta["val"]
                src_3d.meta = {"val": val.view(B, C, S_src) if hasattr(val, "view") else val}

            # index_3d = view(index, [B, C, S_src])
            index_3d = graph.call_function(
                aten.view.default,
                args=(chain.index_node, [B, C, S_src]),
            )
            if hasattr(chain.index_node, "meta") and "val" in chain.index_node.meta:
                val = chain.index_node.meta["val"]
                index_3d.meta = {"val": val.view(B, C, S_src) if hasattr(val, "view") else val}

            # mask_flat = view(mask, [B, C, H*W])
            mask_flat = graph.call_function(
                aten.view.default,
                args=(chain.condition_node, [B, C, H * W]),
            )
            if hasattr(chain.condition_node, "meta") and "val" in chain.condition_node.meta:
                val = chain.condition_node.meta["val"]
                mask_flat.meta = {"val": val.view(B, C, H * W) if hasattr(val, "view") else val}

            # batch_idx = arange(B).view(B,1,1).expand(B,C,S_src)
            batch_idx = graph.call_function(
                aten.arange.start_step,
                args=(0, B),
                kwargs={"dtype": torch.int64, "device": torch.device("cuda")},
            )
            batch_idx.meta = {"val": torch.arange(B, dtype=torch.int64, device="meta")}
            batch_idx = graph.call_function(
                aten.view.default,
                args=(batch_idx, [B, 1, 1]),
            )
            batch_idx.meta = {"val": torch.arange(B, dtype=torch.int64, device="meta").view(B, 1, 1)}
            batch_idx = graph.call_function(
                aten.expand.default,
                args=(batch_idx, [B, C, S_src]),
            )
            batch_idx.meta = {"val": torch.arange(B, dtype=torch.int64, device="meta").view(B, 1, 1).expand(B, C, S_src)}

            # chan_idx = arange(C).view(1,C,1).expand(B,C,S_src)
            chan_idx = graph.call_function(
                aten.arange.start_step,
                args=(0, C),
                kwargs={"dtype": torch.int64, "device": torch.device("cuda")},
            )
            chan_idx.meta = {"val": torch.arange(C, dtype=torch.int64, device="meta")}
            chan_idx = graph.call_function(
                aten.view.default,
                args=(chan_idx, [1, C, 1]),
            )
            chan_idx.meta = {"val": torch.arange(C, dtype=torch.int64, device="meta").view(1, C, 1)}
            chan_idx = graph.call_function(
                aten.expand.default,
                args=(chan_idx, [B, C, S_src]),
            )
            chan_idx.meta = {"val": torch.arange(C, dtype=torch.int64, device="meta").view(1, C, 1).expand(B, C, S_src)}

            # gathered_mask = aten.index.Tensor(mask_flat, [batch_idx, chan_idx, index_3d])
            gathered_mask = graph.call_function(
                aten.index.Tensor,
                args=(mask_flat, [batch_idx, chan_idx, index_3d]),
            )
            if hasattr(index_3d, "meta") and "val" in index_3d.meta:
                idx_fake = index_3d.meta["val"]
                gathered_mask.meta = {
                    "val": torch.empty(idx_fake.shape, dtype=torch.bool, device="meta")
                }

            # Create a zero scalar for the where
            zero_scalar = graph.call_function(
                aten.scalar_tensor.default,
                args=(0.0,),
                kwargs={"dtype": src_meta["dtype"],
                        "device": torch.device("cuda")},
            )
            if hasattr(src_3d, "meta") and "val" in src_3d.meta:
                zero_scalar.meta = {
                    "val": src_3d.meta["val"].new_zeros(())
                }

            # Apply multiplier to src if needed (for sum(where * other, dims))
            effective_src = src_3d
            if multiplier_node is not None:
                # Gather the multiplier at scatter destination positions
                # multiplier is in output space [B, C, H, W] -> flatten spatial
                mult_flat = graph.call_function(
                    aten.view.default,
                    args=(multiplier_node, [B, C, H * W]),
                )
                if hasattr(multiplier_node, "meta"):
                    mult_flat.meta = dict(multiplier_node.meta)

                # Gather multiplier at index positions
                gathered_mult = graph.call_function(
                    aten.index.Tensor,
                    args=(mult_flat, [batch_idx, chan_idx, index_3d]),
                )
                if hasattr(src_3d, "meta"):
                    gathered_mult.meta = dict(src_3d.meta)

                # effective_src = src_3d * gathered_mult
                effective_src = graph.call_function(
                    aten.mul.Tensor,
                    args=(src_3d, gathered_mult),
                )
                if hasattr(src_3d, "meta"):
                    effective_src.meta = dict(src_3d.meta)

            # unmasked_src = where(gathered_mask, 0.0, effective_src)
            unmasked_src = graph.call_function(
                aten.where.self,
                args=(gathered_mask, zero_scalar, effective_src),
            )
            if hasattr(src_3d, "meta"):
                unmasked_src.meta = dict(src_3d.meta)

            # scatter_contribution = sum(unmasked_src, [0, 2])  -> [C]
            scatter_contribution = graph.call_function(
                aten.sum.dim_IntList,
                args=(unmasked_src, [0, 2]),
            )
            if hasattr(target_sum_node, "meta"):
                scatter_contribution.meta = dict(target_sum_node.meta)

            accumulated = scatter_contribution

            # === Handle skip-connection (UNet pattern) ===
            # sum(where(mask, fill, scatter + skip), dims)
            # = sum_scatter_part + sum(where(mask, 0, skip), dims)
            #   + fill * sum(mask, dims)
            # For sum(where(mask, fill, scatter + skip) * mult, dims):
            # = scatter_part + sum(where(mask, 0, skip) * mult, dims)
            #   + fill * sum(mult * mask, dims)
            if has_skip:
                # zero_scalar_4d for where in 4D space
                zero_scalar_4d = graph.call_function(
                    aten.scalar_tensor.default,
                    args=(0.0,),
                    kwargs={"dtype": src_meta["dtype"],
                            "device": torch.device("cuda")},
                )
                zero_scalar_4d.meta = {"val": torch.tensor(0.0, dtype=src_meta["dtype"], device="meta")}

                # unmasked_skip = where(mask, 0, skip) -- zero out masked positions
                unmasked_skip = graph.call_function(
                    aten.where.self,
                    args=(chain.condition_node, zero_scalar_4d, chain.skip_tensor_node),
                )
                if hasattr(chain.skip_tensor_node, "meta"):
                    unmasked_skip.meta = dict(chain.skip_tensor_node.meta)

                if multiplier_node is not None:
                    # sum(where(mask, 0, skip) * mult, [0,2,3])
                    skip_times_mult = graph.call_function(
                        aten.mul.Tensor,
                        args=(unmasked_skip, multiplier_node),
                    )
                    if hasattr(chain.skip_tensor_node, "meta"):
                        skip_times_mult.meta = dict(chain.skip_tensor_node.meta)
                    skip_contribution = graph.call_function(
                        aten.sum.dim_IntList,
                        args=(skip_times_mult, [0, 2, 3]),
                    )
                else:
                    # sum(where(mask, 0, skip), [0,2,3])
                    skip_contribution = graph.call_function(
                        aten.sum.dim_IntList,
                        args=(unmasked_skip, [0, 2, 3]),
                    )
                if hasattr(target_sum_node, "meta"):
                    skip_contribution.meta = dict(target_sum_node.meta)

                accumulated = graph.call_function(
                    aten.add.Tensor,
                    args=(accumulated, skip_contribution),
                )
                if hasattr(target_sum_node, "meta"):
                    accumulated.meta = dict(target_sum_node.meta)

            # === Fill value contribution ===
            # fill * sum(mask, [0,2,3]) or fill * sum(mask * mult, [0,2,3])
            if multiplier_node is not None:
                # fill * sum(mult * mask, [0,2,3])
                fill_masked_mult = graph.call_function(
                    aten.mul.Tensor,
                    args=(multiplier_node, chain.condition_node),
                )
                if hasattr(chain.mask_node, "meta"):
                    fill_masked_mult.meta = dict(chain.mask_node.meta)
                fill_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    args=(fill_masked_mult, [0, 2, 3]),
                )
            else:
                # fill * sum(mask, [0,2,3])
                fill_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    args=(chain.condition_node, [0, 2, 3]),
                )
            if hasattr(target_sum_node, "meta"):
                fill_sum.meta = dict(target_sum_node.meta)

            fill_contribution = graph.call_function(
                aten.mul.Tensor,
                args=(chain.fill_value_node, fill_sum),
            )
            if hasattr(target_sum_node, "meta"):
                fill_contribution.meta = dict(target_sum_node.meta)

            final_result = graph.call_function(
                aten.add.Tensor,
                args=(accumulated, fill_contribution),
            )
            if hasattr(target_sum_node, "meta"):
                final_result.meta = dict(target_sum_node.meta)

            target_sum_node.replace_all_uses_with(final_result)
            num_rewritten += 1

    if num_rewritten > 0:
        log.info(
            "scatter_reduce_fusion: REWROTE scatter_add-reduce chain! "
            "scatter_dim=%d, view_shape=%s, %d/%d sum targets rewritten, "
            "has_skip=%s -> index-mask-reduce aten ops",
            chain.scatter_dim,
            chain.view_shape,
            num_rewritten,
            len(chain.rewrite_targets),
            has_skip,
        )
    return num_rewritten > 0


def _find_scatter_reduce_chains(graph: fx.Graph) -> list[ScatterReduceChain]:
    """Find all scatter-then-reduce patterns in the graph.

    Strategy: Start from sum reduction nodes and walk backwards to find
    scatter patterns that feed into them.
    """
    chains = []

    for node in graph.nodes:
        reduction_dims = _is_sum_reduction(node)
        if reduction_dims is None:
            continue

        # Walk backwards from the sum to find scatter patterns
        chain = _trace_back_to_scatter(node, reduction_dims)
        if chain is not None:
            chains.append(chain)

    return chains


def _trace_back_to_scatter(
    sum_node: fx.Node, reduction_dims: list[int]
) -> Optional[ScatterReduceChain]:
    """Trace backwards from a sum node to find a scatter-reduce pattern.

    Handles the common pattern:
        zeros -> index_put(accumulate=True) x N -> add x (N-1) -> [where/mask] -> sum

    Also handles:
        zeros -> index_put(accumulate=True) x N -> add x (N-1) -> [pointwise] -> sum
    """
    chain = ScatterReduceChain()
    chain.reduction_nodes.append(sum_node)
    chain.reduction_dims = reduction_dims

    # Get the input to the sum
    sum_input = sum_node.args[0]
    if not isinstance(sum_input, fx.Node):
        return None

    # Check if there's a where (mask) node between scatter and reduce
    current = sum_input

    # Handle where(condition, scalar_zero, scattered_value) pattern
    if (current.op == "call_function" and current.target == aten.where.self):
        # where.self(condition, if_true, if_false)
        # In our pattern: where(le_scalar, full_zero, scattered)
        # The scattered value is in position args[2] (if_false)
        chain.has_mask = True
        chain.mask_node = current
        chain.intermediate_nodes.append(current)
        # The scattered value is the third argument (if_false when condition=True means 0)
        current = current.args[2]
        if not isinstance(current, fx.Node):
            return None

    # Handle mul between scatter and reduce (e.g., mul_tensor_5 = where_self * sub_tensor_1)
    # This handles: sum(mul(where(..., scattered), other), dims)
    if (current.op == "call_function" and current.target == aten.mul.Tensor):
        # This is more complex - skip for now, handle the direct case
        # where scattered -> where -> sum
        pass

    # Now current should be the combined scatter result
    # Walk backwards through add nodes to find the scatter chain
    scatter_nodes = []
    visited = set()
    _collect_scatter_chain(current, scatter_nodes, chain.combine_adds, visited)

    if not scatter_nodes:
        return None

    # Validate: the chain must root in a zeros buffer
    # Two valid patterns:
    # 1. All scatters independently target the same zeros (parallel pattern with adds)
    # 2. Scatters are chained: index_put(index_put(...index_put(zeros)...)) (sequential)
    # In pattern 2, only the first scatter's input is zeros; the rest feed from previous.
    zeros_nodes = set()
    scatter_set = set(scatter_nodes)
    all_indexed_dims = None
    for snode in scatter_nodes:
        info = _get_scatter_indices_info(snode)
        if info is None:
            return None
        input_node = info["input_node"]
        if isinstance(input_node, fx.Node) and _is_zeros_init(input_node):
            zeros_nodes.add(input_node)
        elif isinstance(input_node, fx.Node) and input_node in scatter_set:
            # Chained scatter: this scatter's input is another scatter in the chain
            # This is valid - the chain roots at zeros through the first scatter
            pass
        else:
            # Unknown input - not a valid pattern
            return None

        indexed_dims = info["indexed_dims"]
        if all_indexed_dims is None:
            all_indexed_dims = indexed_dims
        elif indexed_dims != all_indexed_dims:
            # Different indexed dims across scatters - bail out
            return None

    if not zeros_nodes or all_indexed_dims is None:
        return None

    # Get scatter output shape
    scatter_meta = _get_tensor_meta(scatter_nodes[0])
    if scatter_meta is None:
        return None

    scatter_output_ndim = scatter_meta["ndim"]

    # Validate that reduction covers all scattered dimensions
    if not _reduction_covers_scatter_dims(
        reduction_dims, scatter_output_ndim, all_indexed_dims
    ):
        return None

    # Check that the combined scatter result is ONLY used by the reduction path
    # (and optionally the mask). If other nodes use it, we can't eliminate it.
    if not _check_single_use_chain(current, sum_node, chain.mask_node):
        return None

    chain.zeros_node = next(iter(zeros_nodes))
    chain.scatter_nodes = scatter_nodes
    chain.combined_scatter = current
    chain.scatter_output_shape = scatter_meta["shape"]
    chain.intermediate_nodes.extend(scatter_nodes)
    chain.intermediate_nodes.extend(chain.combine_adds)

    log.debug(
        "Found scatter-reduce chain: %d scatters, indexed_dims=%s, "
        "reduction_dims=%s, has_mask=%s, output_shape=%s",
        len(scatter_nodes),
        all_indexed_dims,
        reduction_dims,
        chain.has_mask,
        chain.scatter_output_shape,
    )

    return chain


def _collect_scatter_chain(
    node: fx.Node,
    scatter_nodes: list[fx.Node],
    add_nodes: list[fx.Node],
    visited: set,
):
    """Recursively collect scatter nodes through add chains or chained index_puts.

    Handles two patterns:
    1. add(index_put(...), add(index_put(...), index_put(...)))
       (parallel scatters combined with add)
    2. index_put(index_put(index_put(full, ...), ...), ...)
       (chained scatters where each feeds into the next as input)
    """
    if node in visited:
        return
    visited.add(node)

    if _is_accumulate_index_put(node):
        scatter_nodes.append(node)
        # Also recurse into the input of this index_put - it may be another
        # index_put in a chain (pattern: index_put(index_put(full, ...), ...))
        input_node = node.args[0]
        if isinstance(input_node, fx.Node):
            _collect_scatter_chain(input_node, scatter_nodes, add_nodes, visited)
        return

    if node.op == "call_function" and node.target == aten.add.Tensor:
        add_nodes.append(node)
        for arg in node.args:
            if isinstance(arg, fx.Node):
                _collect_scatter_chain(arg, scatter_nodes, add_nodes, visited)


def _check_single_use_chain(
    scatter_result: fx.Node,
    sum_node: fx.Node,
    mask_node: Optional[fx.Node],
) -> bool:
    """Check that the scatter result is only used in the reduction path.

    If the scatter result feeds multiple consumers, we cannot eliminate
    the materialization.
    """
    # For now, be conservative: check the scatter_result users
    users = list(scatter_result.users.keys())

    if mask_node is not None:
        # scatter_result should only be used by the mask node (and possibly the sum)
        allowed_users = {mask_node, sum_node}
        for user in users:
            if user not in allowed_users:
                # Check if user is a mul that feeds into a sum with same dims
                # (for the sum(where * sub) pattern)
                if user.op == "call_function" and user.target == aten.mul.Tensor:
                    # Check if this mul feeds a sum with same reduction dims
                    mul_users = list(user.users.keys())
                    for mul_user in mul_users:
                        dims = _is_sum_reduction(mul_user)
                        if dims is not None:
                            continue
                    # This is the mul(where_self, sub_tensor_1) -> sum pattern
                    # which is also part of the BN backward
                    continue
                return False
    else:
        # Without mask, scatter_result should only feed the sum
        allowed_users = {sum_node}
        for user in users:
            if user not in allowed_users:
                return False

    return True


def _rewrite_scatter_reduce_chain(
    graph: fx.Graph, chain: ScatterReduceChain
) -> bool:
    """Rewrite a scatter-reduce chain to eliminate the scatter intermediate.

    The rewrite strategy depends on whether a mask is present:

    WITHOUT mask:
        sum(scatter_add(zeros, idx, values), dims) = sum(values, source_dims)
        This is the trivial case - just sum the source values directly.

    WITH mask (where(condition, 0, scattered)):
        sum(where(cond, 0, scatter_add(zeros, idx, values)), dims)
        = sum(gather(cond, idx) * values, source_dims)  [if cond selects 0]
        OR more precisely for the UNet pattern:
        The mask is applied AFTER scatter, so we need to gather the mask
        into source space. But this requires knowing the reverse mapping.

    For the initial implementation, we handle the simpler NO-MASK case
    and emit a diagnostic for the masked case.
    """
    if chain.has_mask:
        # The masked case is more complex - the mask is in the output (scattered)
        # space, not the source space. To eliminate the scatter, we'd need to
        # either:
        # 1. Gather the mask into source space (requires reverse index mapping)
        # 2. Use a custom kernel that iterates output positions
        #
        # For now, log that we detected it and return False
        # A future version will handle this with a Triton template
        log.info(
            "scatter_reduce_fusion: detected masked scatter-reduce chain "
            "(%d scatters, shape=%s) but masked rewrite not yet implemented",
            len(chain.scatter_nodes),
            chain.scatter_output_shape,
        )
        # Still useful: for the UNet pattern, we CAN handle it because
        # the reduction is over ALL spatial + batch dims, and we can
        # reformulate as: for each output position, if mask allows,
        # sum the bilinear contributions from source.
        # This is the "gather-reduce" approach.
        return _rewrite_masked_scatter_reduce(graph, chain)

    # No mask case: sum(scatter(zeros, idx, src)) = sum(src)
    # This is straightforward - replace the sum with a sum over the source values
    return _rewrite_simple_scatter_reduce(graph, chain)


def _rewrite_simple_scatter_reduce(
    graph: fx.Graph, chain: ScatterReduceChain
) -> bool:
    """Rewrite: sum(scatter_add(zeros, idx, values), dims) -> sum(values, adjusted_dims)

    When scattering into zeros and then reducing over all scattered dimensions,
    the scatter is algebraically a no-op. We can sum the source values directly.
    """
    if len(chain.scatter_nodes) == 0:
        return False

    # For each reduction node, replace with a sum over the source values
    for sum_node in chain.reduction_nodes:
        # Collect all source value nodes from the scatter chain
        value_nodes = []
        for snode in chain.scatter_nodes:
            info = _get_scatter_indices_info(snode)
            if info is None:
                return False
            value_nodes.append(info["values_node"])

        if not value_nodes:
            return False

        # Get source tensor shape to compute correct reduction dims
        # The source values may have different shape than the scatter output
        first_value_meta = _get_tensor_meta(value_nodes[0])
        if first_value_meta is None:
            return False

        # Build the replacement: sum(value1 + value2 + ..., dims)
        # The reduction dims need to be adjusted for source shape
        # In the common case (same ndim), dims are the same
        source_ndim = first_value_meta["ndim"]
        scatter_ndim = len(chain.scatter_output_shape) if chain.scatter_output_shape else source_ndim

        # For index_put with indices [None, None, row_idx, col_idx]:
        # source shape = scatter output shape (broadcasting handles dims)
        # So reduction dims remain the same
        reduction_dims = chain.reduction_dims
        if source_ndim != scatter_ndim:
            # Need to adjust dims - for now bail out
            log.debug("scatter_reduce_fusion: source/scatter ndim mismatch, skipping")
            return False

        with graph.inserting_before(sum_node):
            # Sum all source values
            if len(value_nodes) == 1:
                combined_src = value_nodes[0]
            else:
                # Add all source values together
                combined_src = value_nodes[0]
                for vnode in value_nodes[1:]:
                    combined_src = graph.call_function(
                        aten.add.Tensor, args=(combined_src, vnode)
                    )
                    # Copy metadata from the first value
                    if hasattr(value_nodes[0], "meta"):
                        combined_src.meta = dict(value_nodes[0].meta)

            # Replace the sum node's input with the combined source
            new_sum = graph.call_function(
                aten.sum.dim_IntList, args=(combined_src, reduction_dims)
            )
            if hasattr(sum_node, "meta"):
                new_sum.meta = dict(sum_node.meta)

            sum_node.replace_all_uses_with(new_sum)

    return True


def _rewrite_masked_scatter_reduce(
    graph: fx.Graph, chain: ScatterReduceChain
) -> bool:
    """Rewrite masked scatter-reduce using output-centric iteration.

    Pattern:
        sum(where(mask, 0, scatter_add(zeros, [None,None,row_idx,col_idx], values)), dims=[0,2,3])

    The mask is computed from a separate skip-connection tensor (BN affine + ReLU),
    so it's defined in the scatter OUTPUT space, not source space.

    Rewrite strategy (gather-reduce approach):
        For each output position (b, c, h, w):
            if mask[b, c, h, w]:
                result[c] += scatter_output[b, c, h, w]

    But scatter_output[b,c,h,w] = sum of all source values that map to (h,w).
    For bilinear upsample backward with structured indices, each output position
    receives contributions from a fixed set of source positions.

    However, implementing this fully as an FX rewrite requires:
    1. Understanding the reverse mapping (output -> source positions)
    2. Emitting the gather + weighted sum + masked reduction

    For this prototype, we detect the pattern and emit a custom op call
    that will be lowered to an efficient Triton kernel.
    """
    if chain.mask_node is None:
        return False

    # Validate we have the expected structure
    if chain.mask_node.target != aten.where.self:
        return False

    # Extract mask components
    # where(condition, if_true_val, if_false_val)
    # In our case: where(le_scalar, full_zero, add_tensor_5)
    # condition = le_scalar (True where we want 0 -> relu gate is 0)
    # if_true = full (scalar 0)
    # if_false = scattered (the actual data when NOT masked)
    condition_node = chain.mask_node.args[0]
    zero_val_node = chain.mask_node.args[1]
    scattered_val_node = chain.mask_node.args[2]

    if not isinstance(condition_node, fx.Node):
        return False

    # For now, we log the detection but don't rewrite yet.
    # A full rewrite requires either:
    # (a) A custom Triton template registered as a fallback kernel
    # (b) Emitting the gather-reduce in FX ops (complex but portable)
    #
    # We'll implement option (b) for the special case where:
    # - The scatter indices are structured (bilinear 2x downsample)
    # - The reduction is over ALL non-channel dims
    # - The mask is a simple boolean tensor

    # Check if we can detect structured bilinear indices
    # The bilinear pattern uses indices of shape [H_out, 1] and [W_out]
    # where the values are floor(linspace(0, H_in-1, H_out))
    scatter_info_list = []
    for snode in chain.scatter_nodes:
        info = _get_scatter_indices_info(snode)
        if info is None:
            return False
        scatter_info_list.append(info)

    # Log detection for now - this is the valuable signal
    scatter_meta = _get_tensor_meta(chain.scatter_nodes[0])
    source_meta = _get_tensor_meta(scatter_info_list[0]["values_node"])

    log.info(
        "scatter_reduce_fusion: DETECTED masked bilinear scatter-reduce pattern! "
        "%d scatters, scatter_shape=%s, source_shape=%s, "
        "reduction_dims=%s. Rewrite via custom kernel needed.",
        len(chain.scatter_nodes),
        scatter_meta["shape"] if scatter_meta else "unknown",
        source_meta["shape"] if source_meta else "unknown",
        chain.reduction_dims,
    )

    # Attempt the gather-reduce rewrite for the specific UNet pattern
    return _emit_gather_reduce_for_bilinear(
        graph, chain, scatter_info_list, zero_val_node
    )


def _emit_gather_reduce_for_bilinear(
    graph: fx.Graph,
    chain: ScatterReduceChain,
    scatter_info_list: list[dict],
    mask_fill_value_node: Optional[fx.Node] = None,
) -> bool:
    """Emit a gather-reduce replacement for bilinear scatter-reduce.

    This replaces:
        scattered = sum_of_4_index_puts(zeros, [None,None,row_idx,col_idx], weighted_src)
        masked = where(condition, fill_val, scattered)
        result = sum(masked, dim=[0,2,3])
        [optional: result2 = sum(masked * other, dim=[0,2,3])]

    With standard aten ops that Inductor can fuse into efficient Triton kernels:
        For each scatter source i:
            gathered_mask_i = index(condition, [None, None, row_idx_i, col_idx_i])
            contrib_i = where(gathered_mask_i, 0, src_i).sum([0, 2, 3])
        result = sum(contrib_i) + fill_val * condition.sum([0, 2, 3])

    The key insight: index(mask, [None, None, row_idx, col_idx]) is the exact
    inverse of index_put(zeros, [None, None, row_idx, col_idx], src, accumulate=True).
    It gathers the mask at the destination positions for each source element.

    For sum(where_self * other, dims):
        sum(where(cond, fill, scatter) * other, dims)
        = fill * sum(other * cond, dims) + sum(scatter * other * ~cond, dims)
        = fill * sum(other * cond, dims) + sum_sources(src * gather(other) * gather(~cond))

    When ALL users of where_self are rewritten, the scatter chain becomes dead code
    and is eliminated by DCE, removing all expensive atomic scatter operations.

    PARTIAL REWRITE: When some users are NOT rewritable (e.g., sub in BN backward),
    we still rewrite the sum users we CAN handle. The scatter remains alive for the
    non-rewritten users, but the rewritten sums avoid redundant data passes through
    the large intermediate. This is still profitable when the sums dominate runtime.
    """
    if chain.mask_node is None:
        return False

    # Extract mask condition node (the boolean mask in output space)
    condition_node = chain.mask_node.args[0]
    if not isinstance(condition_node, fx.Node):
        return False

    # Analyze all users of the mask_node (where) to determine what we can rewrite.
    # We support PARTIAL rewrites: rewrite the sum users we CAN handle, leave the
    # scatter alive for other users (e.g., sub in BN backward). The scatter still
    # runs for non-rewritten users, but the sums that ARE rewritten save their
    # data passes through the large intermediate.
    #
    # Supported user patterns:
    #   1. sum(where_self, [0,2,3])          -> direct sum
    #   2. sum(where_self * other, [0,2,3])  -> multiply then sum
    sum_node = chain.reduction_nodes[0]
    mask_users = list(chain.mask_node.users.keys())
    log.debug(
        "scatter_reduce_fusion: where_self (%s) has %d users: %s",
        chain.mask_node.name,
        len(mask_users),
        [(u.name, u.target if u.op == "call_function" else u.op) for u in mask_users],
    )

    # Classify each user of where_self
    # Each entry: (sum_node, optional_multiplier_node)
    rewrite_targets: list[tuple[fx.Node, Optional[fx.Node]]] = []
    non_rewritable_users: list[fx.Node] = []

    for user in mask_users:
        if user.op == "call_function":
            dims = _is_sum_reduction(user)
            if dims is not None:
                # Direct sum: sum(where_self, dims)
                norm_dims = sorted([d % 4 for d in dims])
                if norm_dims == [0, 2, 3]:
                    rewrite_targets.append((user, None))
                    continue

            if user.target == aten.mul.Tensor:
                # Multiply: where_self * other -> check if result goes to sum
                mul_users = list(user.users.keys())
                if len(mul_users) == 1:
                    mul_user = mul_users[0]
                    dims = _is_sum_reduction(mul_user)
                    if dims is not None:
                        norm_dims = sorted([d % 4 for d in dims])
                        if norm_dims == [0, 2, 3]:
                            # Identify the "other" multiplier
                            # mul args are (where_self, other) or (other, where_self)
                            other_node = user.args[1] if user.args[0] == chain.mask_node else user.args[0]
                            if isinstance(other_node, fx.Node):
                                rewrite_targets.append((mul_user, other_node))
                                continue

        # User doesn't match a rewritable pattern - record but don't bail out.
        # The scatter will remain alive for these users (partial rewrite).
        non_rewritable_users.append(user)
        log.debug(
            "scatter_reduce_fusion: where_self has non-rewritable user: %s (%s) "
            "- will do partial rewrite leaving scatter alive",
            user.name,
            user.target if user.op == "call_function" else user.op,
        )

    if not rewrite_targets:
        return False

    # Validate all scatters use the expected [None, None, row_idx, col_idx] pattern
    # and extract the indices and source values
    source_nodes = []
    row_idx_nodes = []
    col_idx_nodes = []

    for info in scatter_info_list:
        indices = info["indices"]
        indexed_dims = info["indexed_dims"]

        # We expect indices like [None, None, row_idx, col_idx]
        # where indexed_dims are the spatial dims (2, 3 for 4D tensors)
        if len(indexed_dims) != 2:
            log.debug(
                "scatter_reduce_fusion: expected 2 indexed dims, got %d",
                len(indexed_dims),
            )
            return False

        # Extract the non-None index tensors
        row_dim = indexed_dims[0]
        col_dim = indexed_dims[1]
        row_idx = indices[row_dim]
        col_idx = indices[col_dim]

        if not isinstance(row_idx, fx.Node) or not isinstance(col_idx, fx.Node):
            return False

        source_nodes.append(info["values_node"])
        row_idx_nodes.append(row_idx)
        col_idx_nodes.append(col_idx)

    if not source_nodes:
        return False

    # Get output shape from scatter metadata
    output_shape = chain.scatter_output_shape
    if output_shape is None or len(output_shape) != 4:
        log.debug(
            "scatter_reduce_fusion: expected 4D scatter output, got shape=%s",
            output_shape,
        )
        return False

    # Validate that reduction dims are [0, 2, 3] (all except channel dim 1)
    reduction_dims = chain.reduction_dims
    if reduction_dims is None:
        return False
    normalized_dims = sorted([d % 4 for d in reduction_dims])
    if normalized_dims != [0, 2, 3]:
        log.debug(
            "scatter_reduce_fusion: expected reduction over [0,2,3], got %s",
            normalized_dims,
        )
        return False

    # Determine the fill value node (from where(cond, fill_val, scattered))
    if mask_fill_value_node is None or not isinstance(mask_fill_value_node, fx.Node):
        log.debug("scatter_reduce_fusion: mask_fill_value_node not available")
        return False

    # Get source dtype for casting bool mask
    src_meta = _get_tensor_meta(source_nodes[0])
    if src_meta is None:
        return False
    src_dtype = src_meta["dtype"]

    # Emit the gather-reduce rewrite for EACH sum target
    for target_sum_node, multiplier_node in rewrite_targets:
        with graph.inserting_before(target_sum_node):
            # Create a zero scalar for where masking
            zero_scalar = graph.call_function(
                aten.scalar_tensor.default,
                args=(0.0,),
                kwargs={"dtype": src_dtype, "device": torch.device("cuda")},
            )
            zero_scalar.meta = {
                "val": torch.tensor(0.0, dtype=src_dtype, device="meta")
            }

            # Accumulate contributions from each scatter source
            accumulated = None

            for src_node, row_idx, col_idx in zip(
                source_nodes, row_idx_nodes, col_idx_nodes
            ):
                # Gather the mask at source positions' destinations:
                # gathered_mask = condition[:, :, row_idx, col_idx]
                # aten.index.Tensor(mask, [None, None, row_idx, col_idx]) -> [B, C, H_src, W_src]
                gathered_mask = graph.call_function(
                    aten.index.Tensor,
                    args=(condition_node, [None, None, row_idx, col_idx]),
                )
                if hasattr(src_node, "meta") and "val" in src_node.meta:
                    src_val = src_node.meta["val"]
                    gathered_mask.meta = {
                        "val": torch.empty(
                            src_val.shape, dtype=torch.bool,
                            device=src_val.device if hasattr(src_val, "device") else "meta"
                        )
                    }
                else:
                    gathered_mask.meta = {}

                # Zero out masked contributions:
                # where(gathered_mask=True, 0, src) - condition True = masked out
                unmasked_src = graph.call_function(
                    aten.where.self,
                    args=(gathered_mask, zero_scalar, src_node),
                )
                if hasattr(src_node, "meta"):
                    unmasked_src.meta = dict(src_node.meta)

                # If there's a multiplier (sum(where * other, dims)):
                # We need to gather `other` at the same positions and multiply
                if multiplier_node is not None:
                    # Gather the multiplier at destination positions
                    gathered_mult = graph.call_function(
                        aten.index.Tensor,
                        args=(multiplier_node, [None, None, row_idx, col_idx]),
                    )
                    if hasattr(src_node, "meta"):
                        gathered_mult.meta = dict(src_node.meta)

                    # Multiply: unmasked_src * gathered_mult
                    unmasked_src = graph.call_function(
                        aten.mul.Tensor,
                        args=(unmasked_src, gathered_mult),
                    )
                    if hasattr(src_node, "meta"):
                        unmasked_src.meta = dict(src_node.meta)

                # Sum per channel: sum(unmasked_src, [0, 2, 3]) -> [C]
                channel_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    args=(unmasked_src, [0, 2, 3]),
                )
                if hasattr(target_sum_node, "meta"):
                    channel_sum.meta = dict(target_sum_node.meta)

                # Accumulate
                if accumulated is None:
                    accumulated = channel_sum
                else:
                    accumulated = graph.call_function(
                        aten.add.Tensor,
                        args=(accumulated, channel_sum),
                    )
                    if hasattr(target_sum_node, "meta"):
                        accumulated.meta = dict(target_sum_node.meta)

            # Add fill_value contribution:
            # For sum(where_self, dims): fill_val * sum(cond, dims)
            # For sum(where_self * other, dims): fill_val * sum(other * cond, dims)
            if multiplier_node is not None:
                # fill_val * sum(other * cond, [0,2,3])
                fill_masked_mult = graph.call_function(
                    aten.mul.Tensor,
                    args=(multiplier_node, condition_node),
                )
                if hasattr(chain.mask_node, "meta"):
                    fill_masked_mult.meta = dict(chain.mask_node.meta)

                fill_sum = graph.call_function(
                    aten.sum.dim_IntList,
                    args=(fill_masked_mult, [0, 2, 3]),
                )
                if hasattr(target_sum_node, "meta"):
                    fill_sum.meta = dict(target_sum_node.meta)

                fill_contribution = graph.call_function(
                    aten.mul.Tensor,
                    args=(mask_fill_value_node, fill_sum),
                )
            else:
                # fill_val * sum(cond, [0,2,3])
                mask_count = graph.call_function(
                    aten.sum.dim_IntList,
                    args=(condition_node, [0, 2, 3]),
                )
                if hasattr(target_sum_node, "meta"):
                    mask_count.meta = dict(target_sum_node.meta)

                fill_contribution = graph.call_function(
                    aten.mul.Tensor,
                    args=(mask_fill_value_node, mask_count),
                )
            if hasattr(target_sum_node, "meta"):
                fill_contribution.meta = dict(target_sum_node.meta)

            final_result = graph.call_function(
                aten.add.Tensor,
                args=(accumulated, fill_contribution),
            )
            if hasattr(target_sum_node, "meta"):
                final_result.meta = dict(target_sum_node.meta)

            target_sum_node.replace_all_uses_with(final_result)

    rewrite_kind = "FULL" if not non_rewritable_users else "PARTIAL"
    log.info(
        "scatter_reduce_fusion: %s REWRITE of bilinear scatter-reduce chain! "
        "%d scatters, %d sum targets rewritten, %d non-rewritable users remain "
        "-> gather-mask-reduce aten ops, output_shape=%s",
        rewrite_kind,
        len(chain.scatter_nodes),
        len(rewrite_targets),
        len(non_rewritable_users),
        output_shape,
    )
    return True


# ============================================================================
# Phase 1c: Scatter-add-into fusion (embedding backward pattern)
# ============================================================================
#
# Pattern:
#   full(shape, 0) -> index_put(full, [idx], values, accumulate=True) -> add(A, result)
#
# Algebraic identity:
#   add(A, index_put(zeros, [idx], val, accumulate=True)) == index_put(A, [idx], val, accumulate=True)
#
# This eliminates:
#   - The full(0) initialization of the scatter target buffer
#   - The add kernel that reads both A and the scatter result
# The rewritten form scatters directly into a copy of A.


def _find_scatter_add_into_patterns(graph: fx.Graph) -> list[dict]:
    """Find add(A, index_put(zeros, idx, val, accumulate=True)) patterns.

    Returns a list of dicts with keys:
        - add_node: the add node to replace
        - index_put_node: the index_put node
        - zeros_node: the full(0) node
        - other_node: the tensor A being added to
        - indices: the index list from index_put
        - values_node: the values being scattered
    """
    patterns = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target != aten.add.Tensor:
            continue

        # Try both orderings: add(A, index_put) or add(index_put, A)
        for i, j in [(0, 1), (1, 0)]:
            arg_ip = node.args[i]
            arg_other = node.args[j]

            if not isinstance(arg_ip, fx.Node) or not isinstance(arg_other, fx.Node):
                continue

            if not _is_accumulate_index_put(arg_ip):
                continue

            # Check that the index_put's input is a zeros tensor
            ip_input = arg_ip.args[0]
            if not isinstance(ip_input, fx.Node):
                continue
            if not _is_zeros_init(ip_input):
                continue

            # Check that the zeros tensor has the same shape as `other`
            zeros_meta = _get_tensor_meta(ip_input)
            other_meta = _get_tensor_meta(arg_other)
            if zeros_meta is None or other_meta is None:
                continue
            if zeros_meta["shape"] != other_meta["shape"]:
                continue

            # Check that index_put result is ONLY used by this add
            ip_users = list(arg_ip.users.keys())
            if len(ip_users) != 1 or ip_users[0] != node:
                continue

            # Check that zeros is ONLY used by the index_put
            zeros_users = list(ip_input.users.keys())
            if len(zeros_users) != 1 or zeros_users[0] != arg_ip:
                continue

            indices = arg_ip.args[1]
            values_node = arg_ip.args[2]

            patterns.append({
                "add_node": node,
                "index_put_node": arg_ip,
                "zeros_node": ip_input,
                "other_node": arg_other,
                "indices": indices,
                "values_node": values_node,
            })
            break  # Found pattern for this add node

    return patterns


def scatter_add_into_fusion_pass(graph: fx.Graph) -> fx.Graph:
    """Standalone pass for scatter-add-into optimization.

    Rewrites add(A, index_put(zeros, idx, val, accumulate=True))
    into index_put(A, idx, val, accumulate=True).

    Controlled by: config.scatter_add_into_fusion (default True)
    """
    if not getattr(config, "scatter_add_into_fusion", False):
        return graph

    num_rewritten = 0
    patterns = _find_scatter_add_into_patterns(graph)
    if patterns:
        log.info(
            "scatter_add_into_fusion: found %d scatter-add-into pattern(s)",
            len(patterns),
        )
        for chain_info in patterns:
            if _rewrite_scatter_add_into(graph, chain_info):
                num_rewritten += 1
                counters["inductor"]["scatter_add_into_fusion_applied"] += 1

    if num_rewritten > 0:
        graph.eliminate_dead_code()
        graph.lint()

    return graph


def _rewrite_scatter_add_into(graph: fx.Graph, chain_info: dict) -> bool:
    """Rewrite add(A, index_put(zeros, idx, val, True)) -> index_put(A, idx, val, True).

    This is safe because:
        index_put(A, idx, val, accumulate=True) = A.clone() + scatter_add(zeros, idx, val)
        = A + scatter_add(zeros, idx, val)  [when used as a new output, not in-place]
        = add(A, index_put(zeros, idx, val, accumulate=True))
    """
    add_node = chain_info["add_node"]
    index_put_node = chain_info["index_put_node"]
    other_node = chain_info["other_node"]
    indices = chain_info["indices"]
    values_node = chain_info["values_node"]

    with graph.inserting_before(add_node):
        # Create new index_put with A as the input instead of zeros
        new_index_put = graph.call_function(
            aten.index_put.default,
            args=(other_node, indices, values_node, True),
        )
        # Copy metadata from the add node (same output shape/dtype)
        if hasattr(add_node, "meta"):
            new_index_put.meta = dict(add_node.meta)

    add_node.replace_all_uses_with(new_index_put)

    log.info(
        "scatter_reduce_fusion: REWROTE scatter-add-into pattern! "
        "Eliminated full(0) init + add kernel for shape=%s",
        _get_tensor_meta(chain_info["zeros_node"])["shape"]
        if _get_tensor_meta(chain_info["zeros_node"]) else "unknown",
    )
    return True


# ============================================================================
# Pass registration and entry point
# ============================================================================

def register_scatter_reduce_fusion():
    """Register the scatter-reduce fusion patterns.

    Unlike pattern_matcher-based passes, this uses a direct graph traversal
    approach because the pattern is too complex for the pattern matcher DSL
    (variable number of scatter nodes, optional mask, etc.).
    """
    pass  # Registration is handled by scatter_reduce_fusion_pass directly


def get_scatter_reduce_stats(graph: fx.Graph) -> dict[str, Any]:
    """Analyze a graph for scatter-reduce patterns and return statistics.

    This is useful for benchmarking and debugging without actually rewriting.
    Returns information about detected patterns.
    """
    if not hasattr(config, "scatter_reduce_fusion_enabled"):
        pass  # config attr may not exist yet

    chains = _find_scatter_reduce_chains(graph)

    stats = {
        "num_chains": len(chains),
        "chains": [],
    }

    for chain in chains:
        chain_info = {
            "num_scatters": len(chain.scatter_nodes),
            "has_mask": chain.has_mask,
            "reduction_dims": chain.reduction_dims,
            "scatter_output_shape": chain.scatter_output_shape,
        }

        # Compute memory savings estimate
        if chain.scatter_output_shape:
            import math
            intermediate_elements = math.prod(chain.scatter_output_shape)
            # Each scatter writes atomically to the intermediate
            # Eliminating it saves: N_scatters * write + 1 read of intermediate
            chain_info["intermediate_elements"] = intermediate_elements
            chain_info["estimated_atomic_ops"] = intermediate_elements * len(chain.scatter_nodes)

        stats["chains"].append(chain_info)

    return stats
