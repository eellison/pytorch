# Explainer for 0659b54987fc: Support XBLOCK Nested Grouped Reductions

Commit: `0659b54987fc79f3171ba7a2fa0496813b75c606`

Title: `[inductor] Support XBLOCK nested grouped reductions`

## Summary

This commit extends nested reduction fusion from the original "grouped local
reduction splits the parent R tile" case to the complementary case where the
grouped local reduction splits the parent X tile.

The motivating pattern is a norm over a flattened batch/group axis followed by
a weighted reduction over the small group axis:

```python
x_flat = x.reshape(B * K, D)
x_normed = rmsnorm(x_flat).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)
```

Before this commit, the nested reduction machinery could fuse patterns like
layernorm followed by block amax over the R dimension, but it rejected the
X-split version above. The scheduler had an explicit `GroupedAxis.X` rejection,
and SIMD codegen assumed the grouped axis was always the R tree.

After this commit, nested reduction codegen can lower both shapes:

- Group-in-R: parent tile `[XBLOCK, RBLOCK]` becomes
  `[XBLOCK, RBLOCK / G, G]`, then reduces axis 2.
- Group-in-X: parent tile `[XBLOCK, RBLOCK]` becomes
  `[XBLOCK / G, G, RBLOCK]`, then reduces axis 1.

The feature is still gated by `triton.nested_reduction`, which this commit sets
to `False` by default. The tests explicitly patch it on.

## Existing Problem

Nested reduction fusion is intended for dependent reductions over the same
logical elements. The first reduction owns the parent tile and computes a value
such as RMSNorm or LayerNorm statistics. The second reduction reduces a small
local group over the same logical data.

The prior implementation only handled the case where the small local group is
inside the parent R axis. That matched block-scale patterns like:

```python
x_normed = layernorm(x)                         # outer reduction over D
out = x_normed.reshape(B, D // G, G).amax(-1)   # local reduction over G in R
```

It did not handle the small dimension in X case:

```python
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)     # local reduction over K in X
```

That second pattern has the same core property: both reductions traverse the
same logical elements. But the grouped reduction splits the parent X coordinate
instead of the parent R coordinate, so the old R-only layout was wrong.

## Scheduler Changes

The scheduler now admits `GroupedAxis.X` when it can prove the grouped
reduction is a legal nested reduction.

The safety checks are still conservative:

- The pair must be a dependent reduction pair.
- The outer and grouped reductions must cover the same total logical numel.
- The grouped stage must contain exactly one supported simple reduction.
- The group size must be a static power of two.
- The grouped parent extent must be divisible by the group size.
- The min-block guard must allow the forced `min_xblock` or `min_rblock`.
- Pointwise nodes must classify into supported nested domains.
- Normal fusion legality still checks dependencies before admitting follow-up
  consumers.

The main new scheduler work is axis classification:

- Direct shape checks still detect obvious group-in-R and group-in-X forms.
- For ambiguous equal-size cases, the scheduler falls back to loop-body load
  indexing. It compares the coefficient of the grouped reduction variable with
  coefficients in the outer reduction load index:
  - If it matches the outer reduction variable, the grouped axis is R.
  - If it matches an outer iteration variable, the grouped axis is X.
  - If evidence conflicts or is ambiguous, fusion is rejected.

This keeps the X support local to provable cases rather than relaxing dependency
matching globally.

## Codegen Changes

The central refactor is `_GroupedReductionLayout`. It now carries
`local_reduction_in_r`, which decides which parent tree is grouped and which
tree passes through unchanged.

For group-in-R:

- `group_tree = r_tree`
- `passthrough_tree = x_tree`
- `parent_axis = 1`
- reshape shape is `[XBLOCK, nested_R0_REDUCED_BLOCK, G]`
- reduction axis is 2
- output shape is `[XBLOCK, nested_R0_REDUCED_BLOCK]`
- codegen sets `kernel.min_rblock = G`

For group-in-X:

- `group_tree = x_tree`
- `passthrough_tree = r_tree`
- `parent_axis = 0`
- reshape shape is `[nested_X_REDUCED_BLOCK, G, RBLOCK]`
- reduction axis is 1
- output shape is `[nested_X_REDUCED_BLOCK, RBLOCK]`
- codegen sets `kernel.min_xblock = G`

The commit also updates index remapping:

- Grouped body iter vars are remapped differently depending on whether the
  local group is in X or R.
- Reduced-output families rewrite the grouped tree, not always the R tree.
- Parent-full broadcasts now support both orientations:
  - R case: `[X, groups] -> [X, groups, G] -> [X, R]`
  - X case: `[groups, R] -> [groups, G, R] -> [X, R]`
- Full-resolution pointwise consumers choose the source iteration space that
  matches their actual ranges. This matters for collapsed or degenerate shapes,
  especially `B=1`, where the natural full-resolution view can look like the
  local-reduction input domain.

## Tests Added Or Updated

Existing small-dim-in-X tests now expect nested fusion instead of fallback:

- RMSNorm weighted sum/max over `K`
- LayerNorm weighted sum over `K`
- `B=1` variants
- dynamic shape variants where `K` is static
- BF16 epilogue and reduced-output pointwise epilogue cases

New coverage checks:

- Full-resolution epilogue after an X-split nested reduction.
- Rejection when a full-resolution X epilogue depends on an intermediate node
  that cannot be fused yet.
- Bad X group sizes, including non-power-of-two and too-large groups.
- Rejection of 3D X-axis grouped reductions, which need explicit higher-rank
  mapping before they are safe to support.
- Scheduler unit coverage for loop-body axis disambiguation.
- Kernel-form checks that X-split nested reductions carry `min_xblock` and emit
  the expected `nested_X_*` reshape structure.

## How To Review

A useful review order:

1. `torch/_inductor/scheduler.py`

   Confirm that `GroupedAxis.X` is admitted only after the same legality gates
   as the R case. Pay particular attention to the loop-body fallback. It should
   return `None` on ambiguous or conflicting evidence.

2. `torch/_inductor/codegen/simd.py`

   Review `_GroupedReductionLayout` as the core abstraction. The key question is
   whether every place that used to assume "group tree means R tree" now asks
   the layout for the grouped tree, passthrough tree, parent axis, reshape
   shape, output shape, and broadcast orientation.

3. `test/inductor/test_nested_reduction.py`

   Check that the tests cover both successful X-split fusion and rejection
   cases. The important positive cases are weighted RMSNorm reduce over `K`,
   `B=1`, full-resolution epilogues, dynamic shapes, and kernel-form checks.

4. `test/inductor/test_inductor_scheduler.py`

   Check the coefficient-based axis classification unit test. This is the guard
   for equal-size/ambiguous axis cases.

## Non-Goals And Remaining Limits

This commit does not try to support every possible grouped reduction shape.
It intentionally keeps several limits:

- `triton.nested_reduction` is disabled by default in this commit.
- The grouped reduction size must still be static.
- The group size must be a power of two and within existing min-block caps.
- Only one grouped reduction is supported in the grouped stage.
- Higher-rank X-grouped patterns such as `[B, H, K, D].sum(dim=2)` still fall
  back.
- Tuple/stateful reductions such as argmax and Welford remain unsupported in
  the nested grouped stage.
- C++ wrapper support remains disabled for nested reduction.

## Mental Model

The easiest way to read the patch is:

The scheduler proves which parent axis owns the local group. Codegen then
builds one layout object from that answer and asks it how to reshape, reduce,
broadcast, and remap pointwise consumers.

That means the X support is not a separate codegen path. It is the same nested
pipeline with the grouped and passthrough trees swapped.
