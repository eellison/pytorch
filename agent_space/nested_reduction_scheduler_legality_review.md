# Nested Reduction Scheduler Legality Review

## 1. Unrelated pointwise nodes in `_classify_grouped_pointwise_nodes`

The rejection is intentional. Nested lowering only has stages for pointwise
nodes that are semantically part of the grouped reduction pipeline:

- `LOCAL_REDUCTION_INPUT`: producer of the grouped reduction body
- `REDUCED`: consumer of the grouped reduced output
- `PARENT_FULL`: full parent-tile consumer of grouped-stage output

A pointwise node that is neither producer nor consumer has no modeled insertion
point in this pipeline. It may be benign for ordinary fusion, but nested codegen
would need to decide where to emit it and how its dependencies interact with the
staged grouped reduction. I split the combined `is_producer == is_consumer`
check into explicit cases and added comments.

## 2. `typing.cast` in `_pointwise_domain_is_compatible`

This should be fixed rather than documented away. The classifier now narrows
pointwise nodes to `SchedulerNode` before returning them. If a non-`SchedulerNode`
pointwise node appears, nested fusion rejects it at classification time. That
lets `_pointwise_domain_is_compatible` call `sn.get_ranges()` directly without
`typing.cast`.

## 3. Recomputing `NestedReduction.can_fuse` in backend `fuse`

This is real duplicate work, but I would not change it in this PR. `fuse()` is
a defensive construction boundary and mirrors the existing mix-order style:
before constructing the special fused node, it revalidates that the pair still
matches that special fused-node shape.

The cost is gated by `config.triton.nested_reduction` and by the dependent
reduction pair check. If profiling shows this matters when the feature is
enabled, the next step should be caching or returning a small analysis object
from `NestedReduction.can_fuse`, not weakening the construction-time guard.

## 4. `get_grouped_axis` and exact divisibility

The existing `FloorDiv(...) == iter_range` checks reject most inexact cases
implicitly, but the contract is clearer with an explicit divisibility guard.
I added a check after the grouped axis is known:

- group in `R`: require `outer_rnumel % group_size == 0`
- group in `X`: require `outer_numel % group_size == 0`

This keeps `get_grouped_axis` as shape classification and puts the legality
condition in `can_fuse`.

## 5. `_min_block_unprofitable_for_kernel` compile-time cost

This is worth watching, but I would not refactor it before enabling data. The
check is currently both a profitability guard and a codegen capability guard:
the grouped local reduction forces a minimum block on the split axis, and that
floor is only modeled for ordinary `x`/`r0_` kernels.

Because nested reduction is still gated and the earlier candidate checks are
cheap, this is acceptable for the staging PR. If compile-time profiles show it
is expensive, the better fix is to cache or move the tiling analysis into a
single nested analysis result.

## 6. `score_fusion_memory` early return

The early return is intentional. Exact dependency matches are the strongest
memory-sharing signal, and the buffer-overlap path is only a fallback heuristic
for same-buffer reads with different indexing. If exact dep scoring is nonzero,
adding buffer-overlap scoring can double-count the same fusion benefit.

So this is a behavioral cleanup, not a nested-specific correctness change.

## 7. Removed 4x asymmetry optimization

Agree with the review: always iterating the smaller dep set is simpler and
keeps the same intent. No further change needed.

## 8. `FusionScore.node_type_score` from bool to int

This is the right shape. The ordering is explicit:

- `-1`: nested dependent reductions, lower priority
- `0`: mixed-kind fusion
- `1`: ordinary same-kind fusion with memory sharing

The tuple comparison naturally gives `-1 < 0 < 1`.

## 9. Test coverage

For this scheduler-legality commit, unit-level coverage is appropriate because
codegen is introduced later in the stack. The tests cover grouped-axis
classification and the core symbolic geometry. End-to-end nested fusion/codegen
tests belong with the later lowering commits where the feature can actually run.

## Minor Nits

The separate `LOCAL_REDUCTION_INPUT` and `PARENT_FULL` rejects in
`FusedNestedReductions.can_fuse_with` are intentional documentation. They mark
two future relaxations with different implementation requirements: prologue
insertion before the grouped reduction, and full parent-tile consumer matching.

The duplicate `grouped_node.group` extraction in `FusedNestedReductions.__init__`
was unnecessary. I removed it by extracting `grouped_numel` and `grouped_rnumel`
once.
