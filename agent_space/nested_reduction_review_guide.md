# Nested Reduction Review Guide

This guide is for reviewing the current landable nested-reduction commit:

```text
ea02af97780 [inductor] Fuse dependent cross-axis reductions
```

The two commits above it are follow-ups and should be reviewed separately:

```text
727270d529a [inductor] Fuse NVFP4 nested-reduction packing
35112510c77 [inductor] Checkpoint follow-up fusion work
```

## What This PR Does

The core feature fuses two dependent reductions that operate over the same
logical data but at different resolutions.

The canonical pattern is:

1. Compute a row-wise reduction, such as RMSNorm or LayerNorm statistics.
2. Use that result to produce a full-resolution normalized tensor.
3. Immediately consume the normalized tensor in a second small grouped
   reduction, such as FP8 block amax.

Without nested reduction, Inductor normally materializes the normalized tensor
or the first reduction output and launches another kernel for the grouped
reduction. With this change, the first reduction, grouped reduction, and
supported pointwise prologue/epilogue work can run in one Triton kernel.

The important design split is:

- The scheduler decides whether the fusion is legal and useful.
- SIMD/Triton codegen decides how to materialize the different iteration
  resolutions inside one kernel.

## Mental Model

Think of the kernel as having a parent 2D tile:

```text
[XBLOCK, RBLOCK]
```

The first reduction runs in that parent tile. The second grouped reduction then
views one parent dimension as:

```text
[num_groups, group_size]
```

For the common layernorm to block-amax pattern:

```text
x: [B, D]
G = 128

parent tile:
  [B, D]

grouped view:
  [B, D / G, G]

grouped output:
  [B, D / G]
```

For `D = 4096` and `G = 128`, the grouped output has `4096 / 128 = 32`
values per row.

The key codegen trick is that the grouped output needs a different logical
iteration range, but it still lives inside the same physical kernel launch and
same parent loop. That is what the derived iteration family represents.

For example, in a looped parent reduction:

```text
parent r offset:       r0_offset
parent r index:        r0_offset + tl.arange(0, R0_BLOCK)

derived group offset:  r0_offset // 128
derived group index:   r0_offset // 128 + tl.arange(0, R0_BLOCK // 128)
```

A fresh independent `IterationRangesRoot` would not know that its offset is
derived from the parent loop variable. The derived root is a lens over the
same physical loop.

## Review Order

### 1. Behavior Tests First

Start with:

```text
test/inductor/test_nested_reduction.py
```

This file explains the user-visible contract better than the implementation.
Review these groups first:

- Pattern 1: small dim in X, such as RMSNorm over `D` followed by reduce over
  `K`.
- Pattern 2: small dim in R, such as LayerNorm/RMSNorm followed by block amax.
- Producer-consumer cases where node2 reads node1 output instead of only a
  shared input.
- Dynamic shape cases.
- `B=1` cases.
- Non-persistent forced cases.
- Rejection cases for unsupported or unprofitable patterns.

Then read:

```text
test/inductor/test_nested_reduction_internals.py
```

This checks generated kernel structure:

- one launched nested kernel where expected
- no unnecessary intermediate loads/stores
- `min_xblock` and `min_rblock` metadata
- derived masks
- persistent versus non-persistent kernel form
- expected Triton source snippets

Use the internals tests as the map for what codegen invariants must be true.

### 2. Scheduler Legality

Read these in `torch/_inductor/scheduler.py`:

```text
NestedReduction
NestedReduction._get_grouped_reduction_info
NestedReduction.can_fuse
NestedReduction.is_group_size_in_r
FusedNestedReductions
FusedNestedReductions.can_fuse_with
Scheduler.score_fusion_memory nested hook
BaseScheduling.fuse nested hook
```

Questions to answer while reviewing:

- Does node1 have to be a reduction?
- Does node2 depend on node1?
- Do node1 and node2 traverse the same total element count?
- Is node2 exactly one grouped reduction?
- Is the grouped reduction type simple and single-value?
- Is `group_size` static, power-of-two, and small enough?
- Is the outer reduction large enough to be worth fusing?
- Are downstream pointwise nodes only admitted when their resolution is known?
- Are extra unmet dependencies rejected rather than creating a cycle?

This part should be reviewed as "when do we fuse?" independent of the Triton
lowering.

### 3. Iteration Range Plumbing

Read these in `torch/_inductor/codegen/simd.py`:

```text
DerivedIterationRangesRoot
SIMDKernel.use_range_trees
_DerivedIterationFamily
```

The important invariant:

```text
Only one iteration family is active at a time.
```

The parent family is used for the outer reduction. The reduced-output family is
temporarily activated for stores and pointwise epilogues at grouped-output
resolution. Full-resolution epilogues reuse the parent trees but may lazily
materialize broadcasted grouped values.

What to check:

- Derived headers are emitted lazily and only once.
- Looped derived headers are emitted inside the loop body when they depend on
  the current reduction offset.
- Stores through a derived family remap the index before delegating to the
  normal kernel store.
- `use_range_trees` restores the previous active trees after the scoped region.

### 4. Grouped Layout And Handlers

Read these in `torch/_inductor/codegen/simd.py`:

```text
_GroupReductionLayout
_GroupedReductionOpsHandler
_PointwiseRemapHandler
```

`_GroupReductionLayout` is the geometry object. It answers:

- Which parent tree is split by `group_size`?
- Which tree passes through unchanged?
- Which axis is reduced after reshape?
- What is the grouped output shape?
- How do reduced values broadcast back to full resolution?

`_GroupedReductionOpsHandler` intercepts the grouped reduction body. Its core
operation is:

```text
reshape full-resolution tile -> reduce over group axis -> reduced-output value
```

`_PointwiseRemapHandler` runs pointwise bodies in a remapped iteration space.
It is used for reduced-output and full-resolution pointwise prologue/epilogue
nodes. The review point here is that it delegates normal loads/stores to the
existing kernel machinery after remapping indices, instead of inventing a
parallel indexing system.

### 5. Main Codegen Flow

Read `SIMDScheduling.codegen_nested_reduction` in
`torch/_inductor/codegen/simd.py`.

The flow is:

1. Split node1 into the outer reduction schedule plus any full-resolution
   epilogues that need to run later.
2. Generate node2's normal schedule.
3. Classify node2 pointwise nodes by resolution.
4. Build the regular parent kernel from node1's schedule.
5. Set `min_xblock` or `min_rblock` so a tile contains a full group.
6. Internalize removable node1 and node2 temporary buffers.
7. Emit node1.
8. Build `_GroupReductionLayout`.
9. Construct remapped grouped iteration variables.
10. Emit node1 full-resolution pointwise epilogues.
11. Emit node2 grouped reduction through `_GroupedReductionOpsHandler`.
12. Emit reduced-output and full-resolution node2 pointwise nodes.
13. Clear the normal post-loop grouped reduction path because this grouped
    reduction stores inside the nested loop.
14. Generate the kernel, launch it, and free buffers after hooks.

Review questions:

- Does every internalized buffer remain available in `kernel.cse.store_cache`
  until its consumers run?
- Are buffers removed only after scheduler lifetime checks?
- Does the code use normal node schedules where possible?
- Does remapping reuse `_split_iteration_ranges` instead of manual shape hacks?
- Are full-resolution epilogues excluded for the small-dim-in-X cases that are
  only size-compatible but not dependency-order compatible?

### 6. Triton And Autotune Metadata

Skim these files after the main codegen makes sense:

```text
torch/_inductor/codegen/triton.py
torch/_inductor/runtime/triton_heuristics.py
torch/_inductor/runtime/coordinate_descent_tuner.py
torch/_inductor/config.py
torch/_inductor/metrics.py
```

The nested codegen needs three pieces of metadata:

- `min_xblock`: when the small grouped dimension is in X.
- `min_rblock`: when the small grouped dimension is in R.
- `max_xblock`: to avoid creating an excessively large `XBLOCK * RBLOCK` tile.

Review questions:

- Does the first heuristic choice satisfy the min/max constraints?
- Does coordinate descent preserve those constraints?
- Are the defaults `None`, so non-nested kernels behave the same?
- Is the nested-reduction metric incremented only for the nested path?

## Key Invariants

Scheduler invariants:

- node1 is a reduction.
- node2 depends on node1.
- node2 contains exactly one grouped reduction.
- grouped reduction type is single-value, not Welford or arg reduction.
- group size is static and exactly known.
- total traversed elements match.
- grouped output and pointwise node resolutions are compatible.
- extra unmet dependencies do not get pulled into the nested kernel.

Codegen invariants:

- Parent range trees are restored after derived range use.
- Derived reduced-output offsets are based on parent loop offsets.
- Grouped values are reshaped and reduced exactly once.
- Internalized buffers are either in `store_cache` or not removed.
- Full-resolution loads lazily broadcast reduced grouped values when needed.
- Buffer freeing happens after intermediate hooks on the normal path.

Test invariants:

- `B=1` still fuses for supported patterns.
- Dynamic shapes fuse for the supported dynamic patterns.
- Non-persistent and persistent lowering both work.
- Masks are exercised by non-full tiles.
- Unsupported reductions reject instead of reaching codegen.
- Small outer reductions reject.

## Concrete Examples

### Small Dim In R

LayerNorm plus block amax:

```text
x: [B, D]
G: 128

node1:
  reduce D
  output shape [B, 1] stats

node2:
  view normalized x as [B, D / G, G]
  reduce G
  output [B, D / G]
```

For `B = 4`, `D = 4096`, `G = 128`:

```text
parent R numel: 4096
derived R numel: 4096 / 128 = 32

parent r index:
  r0_offset + tl.arange(0, R0_BLOCK)

derived reduced_r index:
  r0_offset // 128 + tl.arange(0, R0_BLOCK // 128)
```

The grouped reduction reshapes:

```text
[XBLOCK, R0_BLOCK] -> [XBLOCK, R0_BLOCK / 128, 128]
```

Then reduces axis `2`.

### Small Dim In X

RMSNorm weighted sum:

```text
x: [B, K, D]

node1:
  reshape [B * K, D]
  reduce D for RMSNorm stats

node2:
  consume normalized [B, K, D]
  reduce K
  output [B, D]
```

Here `group_size = K` splits X, not R. The nested kernel sets `min_xblock = K`
so the X tile includes a whole K group.

## Suggested Reviewer Checklist

- Run or trust the two main tests:
  - `TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py`
  - `TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction_internals.py`
- In scheduler, verify all rejection paths happen before codegen.
- In codegen, verify derived range activation is scoped.
- In codegen, verify internal buffer removal is guarded.
- In heuristics, verify min/max block metadata is optional for normal kernels.
- In tests, verify each scheduler/codegen invariant has at least one positive
  or negative case.

