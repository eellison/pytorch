# mypy: allow-untyped-defs
"""
Slice-output compaction pass.

When a graph output is `aten.slice.Tensor` of a single-use pointwise
intermediate, Inductor must realize the FULL base buffer and return a view into
it (graph outputs that are views realize their base; see
GraphLowering.run_node's output handling). The pointwise epilogue is then
computed and written for the entire base even though only the sliced region is
live. This pattern is common in DenseNet-style training backward graphs, where
each BN-backward returns only a channel slice of the full-shape gradient
(e.g. returns `slice(grad, 1, 224, 256)` of a [64, 256, 56, 56] tensor).

It also has a second-order fusion cost: if the pointwise base depends on a
reduction over the same inputs, the scheduler fuses the reduction with the
full-shape epilogue into a single kernel with two sequential loops over the
reduction range, re-reading every input twice (once for the reduction, once for
the epilogue) over the FULL base region.

This pass inserts `aten.clone` after such output slices and returns the clone
instead. The clone has a compact contiguous layout, so:
  - the full base buffer is never allocated or written;
  - the pointwise epilogue (inlined into the clone) is computed only for the
    sliced region;
  - the producer reduction stays in its own single-pass kernel.

The extra copy is free: the clone kernel IS the (now slice-restricted)
epilogue computation, not an additional pass.

Safety:
  - Only applied to output indices listed in
    `output_node.meta["user_visible_output_idxs"]`. Non-user-visible outputs
    (e.g. activations saved for backward) must keep their exact strides because
    the backward graph was traced and asserts against them.
  - Output stride metadata (`original_output_strides`) is updated to the
    clone's strides so GraphLowering's stride enforcement stays consistent.
    For user-visible view outputs Inductor only guarantees stride ORDER, and a
    contiguous tensor satisfies any downstream `view()`.
  - The base must be a single-use pointwise op (so no other consumer needs the
    full base) and the slice must be statically smaller than the base.
"""

import logging
from typing import Optional

import torch
from torch import fx
from torch._dynamo.utils import counters


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _is_pointwise_call(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    )


def _static_numel(val) -> Optional[int]:
    if not isinstance(val, torch.Tensor):
        return None
    numel = val.numel()
    # Require a static size; symbolic numel would need guards to compare.
    if isinstance(numel, int):
        return numel
    return None


def compact_slice_outputs_pass(graph: fx.Graph) -> None:
    """
    Replace user-visible graph outputs of the form `slice(pointwise_base, ...)`
    (single-use base, statically smaller slice) with `clone(slice(...))` so
    only the sliced region of the epilogue is materialized.
    """
    output_node = None
    for n in graph.find_nodes(op="output"):
        output_node = n
        break
    if output_node is None:
        return

    # Without this metadata we cannot tell user outputs from saved activations;
    # be conservative and do nothing.
    if "user_visible_output_idxs" not in output_node.meta:
        return
    user_visible_idxs = set(output_node.meta["user_visible_output_idxs"])

    outputs = output_node.args[0] if output_node.args else None
    if not isinstance(outputs, (list, tuple)):
        return

    # Map candidate slice node -> all output indices it occupies.
    positions: dict[fx.Node, list[int]] = {}
    for idx, out in enumerate(outputs):
        if isinstance(out, fx.Node) and out.target is aten.slice.Tensor:
            positions.setdefault(out, []).append(idx)

    original_output_strides = output_node.meta.get("original_output_strides")

    replaced = 0
    for slice_node, idxs in positions.items():
        # Every occurrence must be a user-visible output; the node must have no
        # other consumer (otherwise the base is materialized anyway and the
        # clone would be a pure extra copy).
        if not all(idx in user_visible_idxs for idx in idxs):
            continue
        if any(user is not output_node for user in slice_node.users):
            continue

        base = slice_node.args[0] if slice_node.args else None
        if not isinstance(base, fx.Node) or not _is_pointwise_call(base):
            continue
        if len(base.users) != 1:
            continue

        slice_val = slice_node.meta.get("val")
        base_val = base.meta.get("val")
        slice_numel = _static_numel(slice_val)
        base_numel = _static_numel(base_val)
        if slice_numel is None or base_numel is None:
            continue
        # Require the slice to be at most half the base so the saved epilogue
        # traffic clearly dominates any cost of changing the output layout.
        if slice_numel * 2 > base_numel:
            continue

        try:
            # FakeTensor clone; dispatches through the tensor's fake mode.
            clone_val = slice_val.clone()
        except Exception:
            continue

        with graph.inserting_after(slice_node):
            clone_node = graph.call_function(aten.clone.default, (slice_node,))
        clone_node.meta = {**slice_node.meta}
        clone_node.meta["val"] = clone_val
        # tensor_meta (if present) would carry the slice's strides; drop it so
        # nothing reads stale layout info for the clone.
        clone_node.meta.pop("tensor_meta", None)

        output_node.replace_input_with(slice_node, clone_node)

        if isinstance(original_output_strides, list):
            for idx in idxs:
                if idx < len(original_output_strides):
                    original_output_strides[idx] = tuple(clone_val.stride())

        replaced += 1
        log.debug(
            "compact_slice_outputs: cloned output slice %s of %s "
            "(numel %s -> %s)",
            slice_node,
            base,
            base_numel,
            slice_numel,
        )

    if replaced:
        graph.lint()
        counters["inductor"]["compact_slice_outputs"] += replaced
