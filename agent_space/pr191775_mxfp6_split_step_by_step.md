# MXFP6 4:3 layer: step-by-step review

Full path:
`/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_split_step_by_step.md`

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

The prerequisite is staged in this worktree. Plain `git diff` therefore shows
only the MXFP6 layer:

```bash
cd /data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt
git diff --stat
git diff
```

Use `git diff --cached` only when you want to revisit the prerequisite.

Steps marked **HIGH PRIORITY** contain the main new MXFP6 behavior.

## Independent review status

Each step below received a separate focused review. No architectural or
correctness blocker remains. The review found and fixed four local issues:

- generalized a stale `G=2` comment to the actual `group_size == factor` rule;
- pinned mixed-factor, reversed-group, and same-group read-before-write rejection;
- preserved value bounds while exposing split lanes;
- made preshuffled numerics and the ordinary 4:3 kernel-form test structural.

The final pass also removed an unreachable reverse-fusion retry, replaced the
parent-order tuple with a named result, and added an exact-vs-shifted scale
swizzle regression for the only new append rule.

## Step 1 - Establish the 4:3 workload

Read the packing helpers and first positive test:

- [_mxfp6_pack_four_to_three](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:118)
- [test_producer_consumer_mxfp6_four_to_three_pack](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1804)

```text
input lanes:   v0 v1 v2 v3
output bytes:  b0 b1 b2
rate:          4 inputs -> 3 outputs
```

The parent source is full resolution `[B, D]`; the sub-parent stage executes at
`[B, D/4]` and replays the packing body for output lanes 0, 1, and 2.

## Step 2 - Review supported rates

Read `SUB_PARENT_RATES` and `_sub_parent_epilogue_rate`:

- [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:578)
- [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:965)

Checkpoint: the output element-count equation is
`factor * node_numel == output_lanes * full_numel`. Only explicitly tested
rates are admitted.

Independent review: clean after correcting the shape-collision comment.

## Step 3 - Review ordered output grouping **HIGH PRIORITY**

Read:
[NestedReduction._group_sub_parent_epilogue_nodes](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:932).

All sub-parent nodes must share one input factor. Nodes are grouped by output
lane count without reordering stores. The planned order is therefore part of
the codegen contract.

Independent review: clean. A focused unit test now pins accepted grouping,
mixed-factor rejection, and reversed output-group rejection.

## Step 4 - Review the nested coordinate proof

Read:
[NestedReduction._nested_sub_parent_rate](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:987).

For `G=32` and rate `(4,3)`, the logical trailing grouped extent is
`G/4*3 = 24`. This need not be a literal source tensor dimension; it is the
planner's grouped coordinate frame.

Independent review: clean; both callers revalidate the factor and grouped
coordinate boundary.

## Step 5 - Review looped parent-chain placement **HIGH PRIORITY**

Read:
[NestedReduction._order_sub_parent_parent_nodes](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:785).

MXFP6 may consume a full-resolution value produced inside the fused kernel. A
looped reduction cannot retain that tile across reduction iterations, so its
producer chain is moved to the final parent pass and emitted with the epilogue.
The closure includes full-resolution producers and consumers, but stops at
external inputs and completed reduction outputs.

Checkpoint: nodes used by the reduction itself cannot be deferred, and leading
nodes cannot depend on deferred nodes.

The helper returns
[OrderedParentNodes](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:2213),
which names both results: the parent order and the index where its final
post-reduction loop begins.

Independent review: clean. Focused tests cover producer/consumer closure,
reduced-output boundaries, and a previously opened final pass.

## Step 6 - Review internal forwarding **HIGH PRIORITY**

Read:
[NestedReduction._sub_parent_internal_projections](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:1285).

There are two valid cases:

1. same output group: producer and consumer share one `_IterationSpace`, so
   order-preserving reshape/flatten equality is valid;
2. later output group: the consumer must preserve the source frame and may add
   only unused trailing output-lane axes.

Writers must be unique and precede their consumers. A transpose, regrouping, or
consumer-before-writer relation rejects the plan.

Independent review: clean after explicitly checking producer-before-consumer
order within an output group.

## Step 7 - Review the stage representation

Read:
[SubParentEpilogueStage](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:2156).

`output_groups` describes ordered `(output_lanes, nodes)` passes.
`required_post_reduction_index` identifies the first parent node that must run
after a reduction loop has completed. Earlier post-reduction nodes may already
have opened that final pass.

Read [Note [Sub-parent source layouts]](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:584)
before the three compatibility properties. The projection records are the
canonical plan; `source_layouts`, `broadcast_source_names`, and
`internal_dependency_names` are temporary name-based views for current
codegen. Their indexed-forwarding replacement is scoped in
[sub_parent_indexed_forwarding_followup.md](/data/users/eellison/pytorch/agent_space/sub_parent_indexed_forwarding_followup.md).

Independent review: clean; construction and codegen agree on both invariants.

## Step 8 - Review current codegen source adaptation

Read:
[_SubParentSourceLoadResolver](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2358).

This is still the name-based codegen compatibility layer. The prerequisite's
exact access records govern fusion legality, while current codegen resolves the
approved source by name/layout.

Independent review: clean. Missing in-kernel values still fail loudly, and the
name-keyed resolver remains documented follow-up work rather than a new
correctness dependency.

## Step 9 - Review the emission order

Read the output-group loop and final-pass parent scheduling:

- [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3417)
- [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3717)
- [_codegen_remapped_pointwise](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3657)

Checkpoint: persistent codegen can keep the parent tile live. Looped codegen
passes the planner's required post-reduction index through one call to
`generate_node_schedule`; that scheduler starts or reuses the final pass before
the sub-parent outputs consume the internal values.

Independent review: clean; state restoration, masks, CSE reuse, and final-loop
flushing were checked together.

## Step 10 - Review reshape and recursive split lowering

Read:

- [TritonKernel.emit_reshape](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/triton.py:6325)
- [TritonKernel.emit_split_via_reshape](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/triton.py:6407)

Factor 4 is materialized as recursive factor-2 splits. Bounds and derived mask
metadata must follow each resulting value.

Independent review: clean after explicitly preserving source bounds on each
split result.

## Step 11 - Review exact parent-stage append **HIGH PRIORITY**

Read the initial-formation guard in
[Scheduler._can_fuse](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:9858)
and its regression:
[test_producer_consumer_mxfp6_pack_scale_swizzle](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:2012).

The complete behavioral change is:

```diff
- elif not all(node in epilogue_nodes for node in node2.get_nodes()):
+ else:
+     existing_standalone_epilogue = isinstance(node1, FusedStagedReduction)
+     new_nodes_are_epilogue = all(
+         node in epilogue_nodes for node in node2.get_nodes()
+     )
+     if not existing_standalone_epilogue and not new_nodes_are_epilogue:
+         return False
```

Before, every fusion attempt required the newly appended `node2` to belong to
the derived epilogue. Afterward, that requirement remains for initial
formation, but an exact standalone `FusedStagedReduction` may append a node
classified elsewhere by the rebuilt complete plan.

`FusedNestedReductions` uses its dedicated append planner earlier in
`_can_fuse`, so the `isinstance` check here simply records that a staged
epilogue has already been formed.

This removes fusion-order dependence:

```text
(reduction + scale swizzle) + 4:3 pack  -> already worked
(reduction + 4:3 pack) + scale swizzle  -> rejected before, now replanned
```

Initial reduction-to-pointwise fusion still requires the pointwise node to be
the derived 4:3 stage. Once that exact `FusedStagedReduction` exists, a later
parent-shaped scale swizzle may append only after rebuilding the complete plan
and passing the same exact dependency proof. The unshifted case fuses to one
kernel; rolling the scale read by one group rejects fusion.

The earlier reverse reduction/pointwise retry is absent: scheduler nodes are
already topologically ordered, so that retry was unreachable outside an invalid
mock ordering.

## Step 12 - Review functional and rejection tests

MXFP6 tests start at:
[test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1804).

Review in this order:

1. ordinary 4:3 packing and exact packed bytes;
2. persistent and looped variants;
3. preshuffled packing, especially the unshifted one-kernel case;
4. shifted intermediate and non-trailing lane rejection;
5. internal full-resolution source and source-used-by-reduction rejection;
6. the factor-4 three-output RMSNorm epilogue, which also covers a
   batch-broadcast weight access;
7. dynamic batch coverage.

Independent review: the preshuffled input now varies across groups and tiles,
the shifted reference is required to differ, and packed bytes are compared
exactly.

## Step 13 - Review kernel-form checks

Read:
[test_mxfp6_four_to_three_pack_kernel_form](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:3498).

These checks pin one staged kernel and the expected split/load/store structure
for persistent and looped codegen. The ordinary 4:3 check now uses the same
structural helper as the internal-source case and requires exactly three
`tl.split` operations.

## Native MXFP6 validation

The production-shape validation is recorded in
[mxfp6_native_pack2_final_validation_20260826.md](/data/users/eellison/pytorch/agent_space/mxfp6_native_pack2_final_validation_20260826.md).
It uses `inline_asm_elementwise(..., pack=2)` with
`cvt.rn.satfinite.e2m3x2.f32`, exposes the two converted codes at full
resolution, and then uses the reviewed 4:3 packing stage. This needs no further
scheduler relation: conversion is elementwise and the exact projection proof
still governs packing and scale-swizzle fusion.

Across the four AITER comparison shapes it emitted one nested kernel and was
faster in every case. The 2048x3072 DCN plus scale-preshuffle case was exact,
one kernel, spill-free, and measured 9.014 us versus 24.166 us for the corrected
reference. These are local performance measurements, not checked-in test
expectations.
