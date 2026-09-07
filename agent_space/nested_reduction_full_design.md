# Nested Reduction Full Design

This note describes the current nested-reduction design in the
`nested_reduction_single_commit_wip` branch in:

- `/data/users/eellison/pytorch_nested_reduction_single_commit_wip`

It is intended to be read before reviewing the implementation. The hard
part of this feature is not Triton syntax. The hard part is that one
kernel executes several logically different iteration spaces in a fixed
order while keeping the legality and buffer-ownership rules coherent.


## Goal

Fuse a dependent two-stage reduction into one Triton kernel when:

1. `node1` performs an outer reduction over a large shared input.
2. `node2` performs a grouped reduction over data already resident in
   `node1`'s tile.
3. Optional pointwise consumers can run after the grouped reduction at:
   - grouped output resolution
   - full outer-tile resolution
   - half resolution of the grouped axis

Canonical examples:

- `rms_norm -> grouped amax`
- `rms_norm -> grouped fp8 quantization`
- `rms_norm -> grouped amax -> NVFP4 pass-3 packing`

The main win is that the shared large input is read once and reused
through later phases in registers when legal.


## High-Level Kernel Shape

A fused nested-reduction kernel runs in this order:

1. Outer reduction
2. Group reduction
3. Grouped-output epilogues
4. Full-resolution epilogues
5. Half-resolution consumers
6. Final kernel body / wrapper finalization

The key point is that these are not separate kernels. They are separate
iteration families executed inside one kernel.


## Supported Structural Cases

There are two ways the grouped reduction can be embedded in the outer
tile.

### `small_dim_in_r`

`group_size` lives in the outer reduction's reduction axis.

Conceptually:

- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK, RBLOCK / G, G]`
- grouped reduction axis: the last axis

This is the common quantization case.

### `small_dim_in_x`

`group_size` lives in the outer reduction's non-reduction axis.

Conceptually:

- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK / G, G, RBLOCK]`
- grouped reduction axis: the middle axis

This is the cross-axis case.

The scheduler computes this classification once and stores it on the
fused node. Codegen reads that result rather than re-deriving it.


## Core Legality Invariants

The fused kernel is only valid when all of the following hold:

1. The grouped reduction is a legal reinterpretation of one axis of the
   outer-reduction tile.
2. `group_size` is statically known.
3. The grouped output iteration space is representable as a derived
   range-tree family.
4. Full-resolution epilogues are only accepted when the grouped output
   can be lifted back to the original outer-tile resolution.
5. Half-resolution consumers are only accepted when the consumer's lane
   selection is invariant modulo the split factor.
6. The early buffer-ownership pass and the late codegen pass must agree
   on which half-resolution consumers are fused.

The current half-resolution path is intentionally narrow. It is built
for the even/odd style access pattern used by the NVFP4 pass-3 kernel.


## Main Abstraction: Derived Iteration Families

The implementation is organized around one abstraction:

- `_DerivedIterationFamily`

A family defines the active iteration space for one consumer stage. It
contains:

- `kind`
- `range_trees`
- `index_subs`
- `remapped_values`
- `flat_index_expr`
- `flat_index_derived_tree`

There are three current family kinds:

1. `reduced_output`
2. `full_resolution`
3. `half_resolution`

The family owns:

- temporary activation of its range-tree family
- remapped loads
- remapped stores
- header emission for derived trees

This keeps the pointwise handler thin. The handler is mostly a shim that
delegates load/store policy to the family.


## Why Families Matter

The important design choice is that later phases are modeled as
"activate a different iteration family and reuse normal indexing /
load / store machinery", not as handwritten masks and indices.

Without families, nested reduction would need to manually:

- construct reduced masks
- construct reduced indices
- emit custom reduced loads
- emit custom reduced stores
- special-case full-resolution and half-resolution consumers

With families:

- codegen temporarily swaps to a derived range-tree family
- normal `kernel.load()` and `kernel.store()` stay in use
- pointwise consumer code does not need its own Triton load/store path

This is the main architectural cleanup compared to the earlier design.


## Main Data Object: Group Reduction Layout

`_GroupReductionLayout` is the structural description of the grouped
reduction. It centralizes:

- which tree is grouped
- which tree is the "other" tree
- whether the grouped axis is in `x` or `r`
- grouped reshape shape
- grouped reduction axis
- grouped output shape
- broadcast shapes for lifted values
- child shapes for half-resolution consumers
- full-resolution and half-resolution flat-index construction
- parent-tile-shape predicates for half-resolution capture/split

This object is the source of truth for how the grouped phase and later
consumer phases interpret the tile.


## Current Phase Breakdown

### 1. Outer Reduction

`codegen_node_schedule_with_kernel()` emits the normal outer-reduction
body for `node1`.

Before this happens, node1 outputs that are only used internally by the
fused node are marked in two places:

- `kernel.inline_reduction_buffers`
- `V.graph.removed_buffers`

After schedule codegen, `kernel.remove_buffer(name)` is called for those
buffers so the kernel argument list matches the wrapper-level removed
buffer decision.

That split is necessary because `kernel.args.output_buffers[name]` does
not exist until the schedule has actually emitted the output.

### 2. Group Reduction

The grouped reduction stage uses:

- `_GroupReductionLayout`
- `layout.make_reduced_output_family(...)`
- `_GroupedReductionOpsHandler`

The grouped reduction handler:

- reads node1 outputs from `node1_cse_vars`
- reshapes the full-resolution tile into the grouped layout
- reduces over the grouped axis
- stores through the reduced-output family

The grouped stage is the only stage that performs the reduction itself.

### 3. Grouped-Output Epilogues

Grouped-output epilogues use the reduced-output family. Their logical
iteration space is the grouped output tile, so the family uses derived
range trees and `index_subs`.

These epilogues run through `_PointwiseRemapHandler` with the
reduced-output family active for the whole stage.

### 4. Full-Resolution Epilogues

Full-resolution epilogues use:

- `layout.make_full_resolution_family(...)`

This family:

- reuses the outer trees
- defines `flat_index_expr`
- pre-populates `remapped_values` for lifted grouped outputs

If a grouped value must be lifted to the full outer tile, the family
precomputes the broadcasted value once and pointwise consumers read it
through `remapped_values`.

### 5. Half-Resolution Consumers

Half-resolution consumers use:

- `layout.make_half_resolution_family(factor=2)`

This family:

- uses one derived tree and one outer tree
- defines a half-resolution flat index
- records the derived tree explicitly for split registration

Half-resolution consumer loads are satisfied from three sources:

1. `kernel.inline_reduction_buffers`
2. `kernel.cse.store_cache`
3. `captured_load_values`

The family still owns remapped load/store execution; the layout owns how
split and broadcast values are registered into the family.


## Captured Load Values

The NVFP4 pass-3 path needs more than just grouped outputs. It also
needs already-loaded full-resolution values so it can split them into
even and odd lanes in registers instead of reloading from memory.

The kernel therefore has an optional, tightly-scoped capture window:

- `kernel.captured_load_values`
- `kernel.captured_load_range_trees`

This window is enabled only while codegen is interpreting the grouped
reduction body.

The Triton `load()` path opportunistically records the first shaped load
for each input name, but only when:

- capture is active
- the active range-tree family matches the captured outer family

The half-resolution stage can then reuse those captured full-resolution
loads.

This is load-only on purpose. Stores already flow through
`kernel.cse.store_cache`, so there is no need for a parallel store
capture path.


## Half-Resolution Split and Broadcast Rules

Half-resolution registration is layout-owned.

There are three cases:

1. Split
   - for parent-tile-shaped full-resolution values
   - emits `tl.split(tl.reshape(...))`
   - produces lane-indexed tuples in `family.remapped_values`

2. Broadcast
   - for grouped values that need to be expanded to half resolution
   - emits reshape + broadcast + reshape

3. Passthrough
   - for scalar or shape-less values

The split path now has an explicit shape precondition:

- parent dimension must match the parent tile
- otherwise it errors loudly instead of silently splitting an unexpected
  shape

This is important because the captured-load optimization depends on the
captured value having the parent-tile shape the half-resolution consumer
expects.


## Early and Late Half-Resolution Discovery

Half-resolution consumers are discovered twice.

### Early pass

This happens before codegen and is used to decide buffer ownership:

- whether node1 outputs can stay inline
- whether they can be removed as external kernel outputs

### Late pass

This happens after the grouped stage has been emitted and can therefore
see actual kernel state:

- `inline_reduction_buffers`
- `store_cache`
- final captured-load values

Both passes share the same BFS implementation.

The design now includes a hard divergence check. If the early pass and
the late pass disagree on which half-resolution consumers are fused, the
kernel errors instead of silently proceeding with inconsistent buffer
ownership.


## `B=1` / Singleton-Extent Canonicalization

The `B=1` NVFP4 extra-load issue came from logically equivalent indices
surviving in different singleton-`x` forms, so CSE never unified them.

The fix is in generic `SIMDKernel.prepare_indexing()`:

- if an active range tree has statically known `numel == 1`
- its iteration variables are substituted with `0`

This is mathematically correct and canonicalizes degenerate iteration
spaces so equivalent singleton indices become identical before Triton
load/store indexing is formed.

This is broader than nested reduction, but the nested-reduction branch
now has both nested-reduction tests and loop-ordering tests covering the
change.


## Tests and Structural Invariants

`test/inductor/test_nested_reduction.py` now checks both:

- numerical correctness
- kernel form

The current structural checks assert, for representative patterns:

- no dead intermediate scratch allocation
- split present when expected
- broadcast present when expected
- graph inputs loaded exactly once
- expected number of output stores

This is important because the key regressions in this work were
structural, not numerical. The `B=1` NVFP4 bug was "same answer, wrong
kernel shape" until inspected directly.


## Triton vs SIMD Responsibilities

The current split is:

- `simd.py`
  - legality
  - layout
  - iteration families
  - consumer staging
  - buffer ownership decisions
- `triton.py`
  - backend emission
  - `load` / `store`
  - split / broadcast helpers
  - capture of shaped loads during the explicit capture window

The important cleanup relative to the older design is that `simd.py`
does not hand-emit nested-reduction `tl.load` / `tl.store` anymore.


## What This Design Solves

This design directly supports:

- dependent cross-axis grouped reductions
- grouped-output epilogue fusion
- full-resolution epilogue fusion
- half-resolution pass-3 consumer fusion
- `B=1` cases that still fit the current grouped / split model

It also makes future scheduler-owned family construction more plausible,
because the family object is now the right shape to move up a level.


## What This Design Does Not Solve

This design does not attempt to solve:

- arbitrary affine masked subregion families
- heterogeneous sibling pointwise fusion like `Q` and `K` RoPE in one
  kernel
- general combo-kernel reuse of this machinery
- scheduler-owned family construction
- a fully backend-neutral family abstraction

Those are follow-on projects.


## Design Summary

The current nested-reduction design is:

- one outer kernel
- one grouped reduction layout
- one generic family abstraction for later consumer stages
- one pointwise remap handler that delegates load/store to the family
- one explicit load-capture optimization for the half-resolution path

The important shift is from "nested reduction has bespoke pass-specific
codegen" to "nested reduction runs several iteration families inside one
kernel, each using standard load/store/indexing once activated."

That is the current design center.
