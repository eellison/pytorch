# Nested Reduction Design Overview

This note is meant to be read before diving into the implementation.
The code is not hard because of Triton syntax; it is hard because one
kernel executes several logically different iteration spaces in a
specific order. The important thing to understand is the iteration-space
model and the legality conditions.

## Goal

Fuse a dependent two-stage reduction into one Triton kernel when:

1. `node1` performs an outer reduction over a large shared input.
2. `node2` performs a grouped reduction over data that is already
   resident in `node1`'s tile.
3. Optional pointwise consumers can run after the grouped reduction at:
   - reduced resolution
   - full resolution
   - half resolution

Canonical examples:

- `layernorm -> grouped amax`
- `rmsnorm -> grouped quantization prep`

The main win is that the shared large input is read once and reused
through later phases in registers when legal.

## Kernel Phases

A fused nested-reduction kernel runs in this order:

1. Outer reduction
   - Execute `node1`.
   - Keep `node1` outputs in registers / CSE when possible.

2. Group reduction
   - Reinterpret the outer-reduction tile as grouped structure.
   - Reduce over `group_size`.
   - Produce one value per group.

3. Optional consumers
   - Reduced-resolution epilogues consume one value per group.
   - Full-resolution epilogues consume values lifted back to the
     original tile resolution.
   - Half-resolution consumers consume values at `parent_extent / 2`,
     typically even/odd split patterns.

The core question is not "can these ops all be emitted in one kernel?"
It is "can each later phase be expressed as a legal derived iteration
space of the original tile?"

## Axis Classification

There are two supported ways the grouped reduction can be embedded in
the outer-reduction tile.

### `small_dim_in_r`

`group_size` lives in the outer reduction's reduction axis.

Conceptually:

- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK, RBLOCK / G, G]`
- reduction axis: the last axis

This is the common `layernorm -> per-group amax` / quantization shape.

### `small_dim_in_x`

`group_size` lives in the outer reduction's non-reduction axis.

Conceptually:

- outer tile: `[XBLOCK, RBLOCK]`
- grouped view: `[XBLOCK / G, G, RBLOCK]`
- reduction axis: the middle axis

This is the cross-axis case where the smaller grouped dimension is
embedded in `X`, not `R`.

The scheduler should compute this classification once and codegen should
reuse it. Re-deriving it independently is what caused earlier
scheduler/codegen mismatches.

## Legality Invariants

The fusion is only legal when all of the following hold:

1. `node2` can be expressed as a grouped reinterpretation of one axis of
   `node1`'s tile.
2. `group_size` is statically known.
3. The grouped reduction's output iteration space is well-defined as a
   derived range tree.
4. Full-resolution epilogues are only accepted when the lifted value
   truly corresponds to the original full tile.
5. Half-resolution consumers are only accepted when split-lane selection
   is provably invariant modulo the split factor.

For the current half-resolution path, that last rule effectively means
"even/odd style access patterns only."

## Why Derived Iteration Spaces Matter

The implementation becomes much easier to reason about once the reduced
iteration space is treated as an actual range-tree family rather than as
handwritten masks and indices.

Without derived range trees, codegen has to manually:

- build reduced indices
- build reduced masks
- emit custom reduced loads/stores
- special-case the alternate iteration space in multiple places

With derived range trees, the kernel temporarily swaps to a different
range-tree family and the normal indexing/load/store flow still works.

That is the key architectural idea behind the cleanup work: model the
later phases as "same kernel, different active iteration space" instead
of as bespoke codegen.

## Main Abstractions

These names refer to the current single-commit WIP implementation in:

- `/data/users/eellison/pytorch_nested_reduction_single_commit_wip/torch/_inductor/codegen/simd.py`

### `DerivedIterationRangesRoot`

A temporary root whose geometry is derived from a parent tree.

Used for the grouped output space, where the grouped axis has:

- smaller logical `numel`
- smaller effective block size
- smaller block offset

This is what lets reduced-resolution codegen use the standard indexing
machinery.

### `_ReducedOutputSpace`

A thin wrapper around the derived range-tree family for the
group-reduction output tile.

It owns:

- the temporary range trees
- the substitution from original body iter vars into reduced-space vars

It does not own the grouped reduction logic itself.

### `_GroupReductionLayout`

The shared structural description of the grouped reduction.

This is the most important data object. It centralizes:

- which tree is the grouped one
- which tree is the "other" one
- whether the grouped axis is `X` or `R`
- reshape shape
- reduce axis
- output shape
- broadcast shapes
- child shapes for half-resolution consumers
- flat-index reconstruction helpers

If the code feels dense, this is usually the first place to read.

### `_GroupedReductionOpsHandler`

Owns the grouped reduction phase.

Responsibilities:

- intercept loads so the grouped reduction can read `node1` values from
  registers / CSE
- reshape the full-resolution tile into the grouped layout
- reduce over the grouped axis
- store under the grouped reduction's own buffer name

This handler is specific to the reduction phase itself.

### `_PointwiseRemapHandler`

Runs pure pointwise bodies after the grouped reduction.

It is shared by:

- reduced-resolution epilogues
- full-resolution epilogues
- half-resolution consumers

Its job is simple: satisfy loads either from a precomputed value map or
through a remapped iteration space, then let the pointwise body run.

### `_HalfResolutionContext`

Specialized support for half-resolution consumers.

It registers three kinds of values:

- split full-resolution values
- broadcast reduced-resolution values
- passthrough scalar values

This is the most specialized part of the design. It exists because the
half-resolution path is not just "use a smaller range tree"; it also
needs register remapping for split and broadcast behavior.

## How To Read The Code

Recommended order:

1. `codegen_nested_reduction`
   - high-level phase ordering
2. `_GroupReductionLayout`
   - structural model
3. `DerivedIterationRangesRoot` and `_ReducedOutputSpace`
   - how reduced iteration spaces become first-class
4. `_GroupedReductionOpsHandler`
   - the grouped reduction itself
5. `_PointwiseRemapHandler`
   - how pointwise consumers run in remapped spaces
6. `_HalfResolutionContext`
   - specialized half-resolution value remapping

Reading `_HalfResolutionContext` first tends to make the feature look
more ad hoc than it really is.

## What Is Generic vs Feature-Specific

Generic infrastructure:

- derived range trees
- temporary range-tree swapping
- remapped pointwise execution over alternate iteration spaces

Feature-specific logic:

- grouped reduction layout rules
- legality of `small_dim_in_r` vs `small_dim_in_x`
- full-resolution lifting
- half-resolution split/broadcast behavior
- flattened `B=1` handling

This distinction matters when deciding how to restack the work.

## Non-Goals

This design does not claim to solve every mismatched-iteration-space
fusion problem.

In particular, it does not by itself solve:

- heterogeneous sibling pointwise fusion with different logical outputs
- the MLA `Q RoPE + K RoPE` mismatch case
- arbitrary masked affine subregions outside the current grouped/split
  model

Those are related future directions, but they are not what nested
reduction should claim today.

## Review Checklist

When reviewing the implementation, the useful questions are:

1. Is the grouped reduction really a legal reinterpretation of one axis
   of the outer tile?
2. Is the axis classification computed once and reused consistently?
3. Are later consumers running in a clearly defined derived iteration
   space?
4. Are full-resolution and half-resolution paths guarded by explicit
   legality checks rather than wishful codegen?
5. Is the code using standard indexing/load/store paths wherever
   possible, instead of hand-emitting alternate-coordinate logic?

If those answers are good, the rest of the code becomes much easier to
evaluate.
