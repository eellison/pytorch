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

Besides `aten.slice.Tensor`, `aten.select.int` outputs are handled the same
way (DeiT-distilled returns `x[:, 0]` / `x[:, 1]` of a full LayerNorm output;
Llama-style inference returns the last-token row of an RMSNorm epilogue).
A base may feed SEVERAL such output views (e.g. both the cls and dist token
selects of one LayerNorm result): in that case every consumer of the base must
be a compactable output view, their combined numel must still be statically
smaller than the base, and each view is cloned.

Safety:
  - Only applied to output indices listed in
    `output_node.meta["user_visible_output_idxs"]`. Non-user-visible outputs
    (e.g. activations saved for backward) must keep their exact strides because
    the backward graph was traced and asserts against them.
  - Output stride metadata (`original_output_strides`) is updated to the
    clone's strides so GraphLowering's stride enforcement stays consistent.
    For user-visible view outputs Inductor only guarantees stride ORDER, and a
    contiguous tensor satisfies any downstream `view()`.
  - The base must be a pointwise op whose ONLY consumers are the compacted
    output views (so no other consumer needs the full base) and the views
    combined must be statically smaller than the base.
  - The avoided write traffic must exceed `_MIN_SAVED_BYTES`: the clone splits
    the fused reduction+epilogue kernel in two, so for small bases the extra
    kernel launch costs more than the avoided full-base write.
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


# Output view ops we can compact by cloning. Each takes the base as its first
# argument and produces a statically-shaped sub-view of it.
_COMPACTABLE_VIEW_TARGETS = (aten.slice.Tensor, aten.select.int)

# Minimum write traffic (bytes) the compaction must save. Cloning splits the
# fused reduction+epilogue kernel in two, so below a few MB the extra kernel
# launch/wave costs more than the avoided full-base write. Measured on B200:
# 2MB saved -> ~1.5us REGRESSION (llama RMSNorm last-token select,
# mean_9c0fd9fb28b1); 12MB saved -> ~2.5us win (DenseNet BN-backward slice,
# sum_sum_98c4811f6ddf); 75MB saved -> ~10us win (DeiT dual token select,
# var_mean_c5067e6e3750).
_MIN_SAVED_BYTES = 4 * 1024 * 1024


def compact_slice_outputs_pass(graph: fx.Graph) -> None:
    """
    Replace user-visible graph outputs of the form `slice(pointwise_base, ...)`
    or `select(pointwise_base, ...)` with `clone(view(...))` so only the viewed
    region of the epilogue is materialized.

    A base may feed several such output views (all of its consumers must then
    be compactable output views); each view is cloned, and the combined view
    numel must be statically at most half the base numel.
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

    # Map candidate view node -> all output indices it occupies.
    positions: dict[fx.Node, list[int]] = {}
    for idx, out in enumerate(outputs):
        if isinstance(out, fx.Node) and out.target in _COMPACTABLE_VIEW_TARGETS:
            positions.setdefault(out, []).append(idx)

    # Group candidate views by their base node.
    by_base: dict[fx.Node, list[fx.Node]] = {}
    for view_node in positions:
        base = view_node.args[0] if view_node.args else None
        if isinstance(base, fx.Node):
            by_base.setdefault(base, []).append(view_node)

    original_output_strides = output_node.meta.get("original_output_strides")

    replaced = 0
    for base, view_nodes in by_base.items():
        if not _is_pointwise_call(base):
            continue
        # Every consumer of the base must be one of the candidate output views;
        # otherwise the full base is materialized anyway and the clones would
        # be pure extra copies.
        if any(user not in view_nodes for user in base.users):
            continue
        # Every occurrence of every view must be a user-visible output, and the
        # views must have no consumer besides the output node.
        ok = True
        for view_node in view_nodes:
            if not all(idx in user_visible_idxs for idx in positions[view_node]):
                ok = False
                break
            if any(user is not output_node for user in view_node.users):
                ok = False
                break
        if not ok:
            continue

        base_val = base.meta.get("val")
        base_numel = _static_numel(base_val)
        if base_numel is None:
            continue
        view_vals = [view_node.meta.get("val") for view_node in view_nodes]
        view_numels = [_static_numel(val) for val in view_vals]
        if any(numel is None for numel in view_numels):
            continue
        # Require the combined views to be at most half the base so the saved
        # epilogue traffic clearly dominates any cost of changing the output
        # layout. (Duplicate output positions share one clone, so each view
        # counts once.)
        total_view_numel = sum(view_numels)  # type: ignore[arg-type]
        if total_view_numel * 2 > base_numel:
            continue
        # Require the avoided write traffic to be large enough to pay for the
        # extra kernel the clone splits off (see _MIN_SAVED_BYTES).
        saved_bytes = (base_numel - total_view_numel) * base_val.element_size()
        if saved_bytes < _MIN_SAVED_BYTES:
            continue

        try:
            # FakeTensor clones; dispatch through the tensors' fake mode.
            clone_vals = [val.clone() for val in view_vals]
        except Exception:
            continue

        for view_node, clone_val in zip(view_nodes, clone_vals):
            with graph.inserting_after(view_node):
                clone_node = graph.call_function(
                    aten.clone.default, (view_node,)
                )
            clone_node.meta = {**view_node.meta}
            clone_node.meta["val"] = clone_val
            # tensor_meta (if present) would carry the view's strides; drop it
            # so nothing reads stale layout info for the clone.
            clone_node.meta.pop("tensor_meta", None)

            output_node.replace_input_with(view_node, clone_node)

            if isinstance(original_output_strides, list):
                for idx in positions[view_node]:
                    if idx < len(original_output_strides):
                        original_output_strides[idx] = tuple(clone_val.stride())

            replaced += 1
            log.debug(
                "compact_slice_outputs: cloned output view %s of %s "
                "(numel %s -> %s)",
                view_node,
                base,
                base_numel,
                _static_numel(clone_val),
            )

    if replaced:
        graph.lint()
        counters["inductor"]["compact_slice_outputs"] += replaced
