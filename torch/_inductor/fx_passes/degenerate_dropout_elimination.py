# mypy: allow-untyped-defs
"""
Degenerate Dropout Elimination Pass.

Eliminates dropout patterns where the dropout mask is provably always True.

Pattern:
    seed = inductor_lookup_seed(seeds, idx)
    rand = inductor_random([shape], seed, 'rand')
    mask = gt.Scalar(rand, threshold)   # threshold < 1e-10 => mask always True
    masked = mul.Tensor(mask, value)     # masked == value since mask is all-True
    result = mul.Tensor(masked, scale)   # scale == 1.0 => result == value

When threshold is negligibly small (< 1e-10) and scale is 1.0, the entire
dropout chain is identity and can be replaced with just `value`.

This pattern appears in Longformer/DistilGPT2/FNet training where dropout
probability is set to effectively zero but the graph still carries the
RNG + mask + multiply operations, preventing fusion with downstream
LayerNorm templates.
"""

import logging
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# Threshold below which we consider dropout to be degenerate (always True).
# rand() produces values in [0, 1), so gt(rand, t) is True with probability 1-t.
# For t < 1e-10, the probability of a single element being dropped is < 1e-10,
# and for typical tensor sizes (millions of elements), the expected number of
# dropped elements is essentially zero.
DEGENERATE_THRESHOLD = 1e-10


def degenerate_dropout_elimination_pass(graph: fx.Graph) -> None:
    """Scan graph for degenerate dropout patterns and fold them to identity."""
    count = 0

    for node in list(graph.nodes):
        if _try_eliminate_degenerate_dropout(graph, node):
            count += 1

    if count:
        graph.eliminate_dead_code()
        counters["inductor"]["degenerate_dropout_eliminated"] += count
        log.debug("Eliminated %d degenerate dropout patterns", count)


def _try_eliminate_degenerate_dropout(graph: fx.Graph, node: fx.Node) -> bool:
    """Try to eliminate a degenerate dropout pattern rooted at a mul node.

    We look for:
        mul.Tensor(mul.Tensor(gt.Scalar(inductor_random(...), threshold), value), scale)
    where threshold < DEGENERATE_THRESHOLD and scale == 1.0.

    Also handles the simpler variant without the outer scale multiply:
        mul.Tensor(gt.Scalar(inductor_random(...), threshold), value)
    where threshold < DEGENERATE_THRESHOLD (implicit scale 1.0).
    """
    if node.op != "call_function":
        return False
    if node.target != aten.mul.Tensor:
        return False

    args = node.args
    if len(args) != 2:
        return False

    # Case 1: mul(mul(mask, value), 1.0) - outer scale multiply
    inner_mul, scale = _extract_scalar_mul(args)
    if inner_mul is not None and scale == 1.0:
        # Check the inner mul: mul.Tensor(mask, value)
        value = _extract_degenerate_mask_mul(inner_mul)
        if value is not None:
            # Verify shapes are compatible (the mask should broadcast to value's shape)
            if _shapes_compatible(node, value):
                node.replace_all_uses_with(value)
                return True

    # Case 2: mul(mask, value) directly (no outer scale)
    # Only match if this node itself is the degenerate mask mul
    value = _extract_degenerate_mask_mul(node)
    if value is not None:
        if _shapes_compatible(node, value):
            node.replace_all_uses_with(value)
            return True

    return False


def _shapes_compatible(original: fx.Node, replacement: fx.Node) -> bool:
    """Check that replacement has compatible shape/dtype metadata."""
    if not hasattr(original, 'meta') or not hasattr(replacement, 'meta'):
        return True  # No metadata to check, assume compatible

    orig_val = original.meta.get('val')
    repl_val = replacement.meta.get('val')

    if orig_val is None or repl_val is None:
        return True

    if not isinstance(orig_val, torch.Tensor) or not isinstance(repl_val, torch.Tensor):
        return True

    # Shapes must match (or be broadcastable to the same shape)
    if orig_val.shape != repl_val.shape:
        return False

    return True


def _extract_scalar_mul(args) -> tuple[Optional[fx.Node], Optional[float]]:
    """Extract (node, scalar) from a mul where one operand is a Python scalar."""
    left, right = args[0], args[1]

    if isinstance(right, (int, float)):
        if isinstance(left, fx.Node):
            return left, float(right)
    if isinstance(left, (int, float)):
        if isinstance(right, fx.Node):
            return right, float(left)

    return None, None


def _extract_degenerate_mask_mul(node: fx.Node) -> Optional[fx.Node]:
    """Check if node is mul.Tensor(degenerate_mask, value) and return value.

    A degenerate mask is: gt.Scalar(inductor_random(..., 'rand'), threshold)
    where threshold < DEGENERATE_THRESHOLD.
    """
    if node.op != "call_function" or node.target != aten.mul.Tensor:
        return None

    args = node.args
    if len(args) != 2:
        return None

    left, right = args[0], args[1]

    # Pattern: mul(mask, value) where mask is gt(rand, tiny_threshold)
    if isinstance(left, fx.Node) and _is_degenerate_gt_mask(left):
        if isinstance(right, fx.Node):
            return right

    if isinstance(right, fx.Node) and _is_degenerate_gt_mask(right):
        if isinstance(left, fx.Node):
            return left

    return None


def _is_degenerate_gt_mask(node: fx.Node) -> bool:
    """Check if node is gt.Scalar(inductor_random(...), threshold) with tiny threshold.

    Returns True if the pattern matches (indicating the mask is degenerate/always-True).
    """
    if not isinstance(node, fx.Node):
        return False

    if node.op != "call_function" or node.target != aten.gt.Scalar:
        return False

    args = node.args
    if len(args) != 2:
        return False

    rand_node, threshold = args[0], args[1]

    # Check threshold is negligibly small
    if not isinstance(threshold, (int, float)):
        return False
    if float(threshold) >= DEGENERATE_THRESHOLD:
        return False

    # Check the random source is inductor_random with 'rand' mode
    if not isinstance(rand_node, fx.Node):
        return False
    if rand_node.op != "call_function":
        return False
    if rand_node.target != prims.inductor_random.default:
        return False

    # Verify it's 'rand' mode (uniform [0,1))
    if len(rand_node.args) >= 3 and rand_node.args[2] != 'rand':
        return False

    return True
