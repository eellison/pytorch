# Historical #191975 guide - cleanup folded into owner commits

#191975 is no longer a commit in the local stack. Its final tree is unchanged:
the tile-shape cleanup moved to #190594, required materialization state moved to
#190595, and source-dependency lookup cleanup moved to #191775. The text below
documents the former standalone diff.

Small source-only cleanup. The larger representation, lane-matching, split
emitter, backend-hook, and fusion-decision changes from the old version were
folded into the commits that introduce those mechanisms.

## What changes

### Remove redundant tile-shape predicates

`_GroupedReductionLayout.parent_dim()` already accepts only a rank-2 tile or a
rank-1 tile with a trivial passthrough axis and returns the grouped dimension.
The deleted `_is_axis_tile_shaped` wrappers repeated those checks. Codegen now
compares `parent_dim` directly with the parent or child block.

### Make materialization state explicit

`must_materialize_names` is required by `_codegen_sub_parent_pointwise` and
`_SubParentPointwiseRemapHandler`. Both nested and standalone callers already
pass it, so removing the silent empty default changes no behavior. The distinct
store-cache-miss and unusable-layout assertions remain intact.

### Simplify source lookup

`_sub_parent_epilogue_source_deps` builds `full_resolution_source_deps`
directly from the reduction reads or internal full-resolution writes. The
fused/removed-buffer cases are unchanged; the rewrite only removes an
intermediate variable and the misleading `reduction_deps` name.

## Deliberately unchanged

- no plan cache or cross-phase retained plan
- no fusion-decision or leaf-legality change
- no planner-side source-chain reorder or rejection
- no multi-output CONTIGUOUS restriction
- no Triton emitter or test change

The verified full-resolution MXFP6 fork therefore continues to form one staged
kernel in looped and persistent modes. The original #191975 commit message and
PR description describe the old, much larger diff; the PR description needs to
be updated before submission.

## Verification

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py
  360 passed, 5 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_inductor_scheduler.py
  76 passed, 4 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py -k sub_parent
  80 passed, 1 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py -k mxfp6
  20 passed, 1 skipped
```

`python -m py_compile` and `git diff --check` passed. `lintrunner -a`
reported no file lint or applied patch; Pyrefly alone failed while fetching
`spmd-types` through the network tunnel.
