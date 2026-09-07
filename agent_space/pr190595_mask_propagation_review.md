# PR #190595 local mask-propagation review

## Review target

- Worktree: `/data/users/eellison/pytorch/agent_space/pr190594_ci_fix`
- HEAD: `b55adf52d4dae589a004fe31e313f310e168db4f`
- Exact uncommitted diff: `agent_space/pr190595_mask_propagation_review.patch`
- Patch SHA256: `4dbc7540a7f0e32524396c792c419a60323f65e6ae697fef75a512ac362135b4`
- Nothing was amended or submitted.

The patch contains the already-reviewed local #190595 cleanup plus the new
derived-mask fix. The mask-specific source changes are in
`_DerivedIterationFamily.mask_vars_for_shape`,
`_DerivedIterationFamily.set_value_masks`, and the three calls from
`materialize_value_at_sub_parent_resolution`.

## Contract

Values receive `mask_vars` when they are materialized in the sub-parent domain,
matching ordinary Triton loads and CSE operations. Parent splits, group
broadcasts, and already-child-width values receive the active derived family's
shape-compatible masks. There is no planner rejection and no repair in
`indirect_indexing`.

For a factor-2 R projection:

```text
[X, R]     -> [X, R/2] gets xmask + half2_r0_index_mask
[X, R/G]   -> [X, R/2] gets xmask + half2_r0_index_mask
[X, R/2]   -> [X, R/2] gets xmask + half2_r0_index_mask
```

Masks that would widen a broadcast-invariant value are omitted.

## Verification

- New indirect-index tests: 6 passed. These cover standalone split, nested
  group broadcast, and nested `G=2` direct-width forwarding in persistent and
  looped modes.
- Generated persistent and looped kernels put
  `half2_r0_index_mask & xmask` on both `tl.device_assert` and the indirect
  `tl.load`.
- NVFP4 inline-asm kernel form: 4 passed.
- Dynamic sub-parent tests: 4 passed.
- Existing mismatched masked-source tests: 2 passed.
- Scheduler suite: 60 passed, 6 skipped.
- Full nested-reduction file: 342 passed, 2 skipped, plus the two known
  `kernel_num_gb` accounting failures reproduced on the clean baseline.
- `with-proxy spin quicklint`, `py_compile`, and `git diff --check` pass.

The deterministic reproductions and generated fixed kernels are under
`agent_space/rev195_cg/`.
