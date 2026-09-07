# Sub-Parent Epilogue Split Plan

Base: `96ac987dbfb [inductor] Fuse NVFP4 nested-reduction packing`

The current dirty tree is represented by the patch series in
`agent_space/sub_parent_split_patches/series`. Apply those patches in order.

## 1. Half-Output Leaf Fix

Patch: `0001-half-output-leaf-fix.patch`

This is the correctness fix for the confirmed factor-2 half-output
read-before-write bug in the existing NVFP4 half-resolution epilogue path.

Contents:

- minimal factor-2 half-epilogue candidate extraction
- schedule-time leaf guard in `SIMDScheduling.can_fuse`
- plan-time `outputs_unread` guard
- sibling parent-source guard
- `FusedNestedReductions.can_fuse_with` output-unread protection
- fullres-reader regression tests for producer-consumer, standalone, and NVFP4

Both guards are required. The schedule-time guard prevents the bad fusion from
forming; the plan-time guard fails closed if a bad group reaches codegen.

Do not add a blanket `other.has_aliasing_or_mutation()` guard to
`FusedNestedReductions.can_fuse_with`. That rejects unrelated reduced/full
append fusions before the normal mutation-aware legality checks run.

## 2. Interleaved Sub-Parent Core

Patch: `0002-sub-parent-interleaved-looped-core.patch`

This turns the factor-2 half-resolution model into the factor/layout data model
without adding chunk/SwiGLU yet. It keeps only `INTERLEAVED`, so the old NVFP4
path remains the specialization.

Contents:

- `SubParentSourceLayout.INTERLEAVED`
- `HalfResolutionEpiloguePlan.sub_parent_factor`
- `HalfResolutionEpiloguePlan.source_layouts`
- `_sub_parent_epilogue_*` source and factor helpers
- `materialize_value_at_sub_parent_resolution`
- looped/non-persistent interleaved support
- kernel-form updates for looped vs persistent signatures

The factor scan is intentionally preserved instead of using a quotient; it is
more robust for symbolic-size proof.

## 3. Contiguous Chunk/SwiGLU

Patch: `0003-contiguous-chunk-swiglu.patch`

This adds the chunk/SwiGLU layout and the profitability guard that keeps it from
firing on the measured net-negative large-D/small-B shape.

Contents:

- `SubParentSourceLayout.CONTIGUOUS`
- contiguous source-dep proof
- `_ContiguousSubParentRemappedValue`
- contiguous branch in `materialize_value_at_sub_parent_resolution`
- `TritonKernel.emit_split_via_reshape_permute`
- contiguous lane selection in `_resolve_remapped_value`
- chunk2/chunk4/chunk8/chunk16 numeric and kernel-form tests
- unprofitable large-D/small-B fallback test

Current guard:

- fuse all contiguous cases with `rnumel <= 4096`
- for larger reductions, require `numel >= 128`

This reflects the measured split: chunk/SwiGLU was a clear win at small/medium
D and became break-even or negative for large D with small batch. The threshold
is deliberately simple and should be easy to tune in review.

## Verification

- The patch series applies cleanly to a fresh detached worktree at
  `96ac987dbfb`.
- The applied series matches the current dirty tree exactly for the four
  touched tracked files.
- `git diff --check` is clean.
- `python -m py_compile` succeeds on the four touched files.
- `python test/inductor/test_nested_reduction.py -v`: 246 passed.
- `python test/inductor/test_inductor_scheduler.py -v`: 32 passed, 2 skipped.
