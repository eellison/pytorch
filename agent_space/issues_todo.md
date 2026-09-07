# Nested Reduction: Remaining Review Notes

These are architecture and design observations — not bugs. They do not
block landing but should be visible during review.

## Code Structure Notes

### `_GroupReductionLayout` carries a lot of semantic weight
- Owns: grouped axis semantics, reshape geometry, reduction axis, family
  construction, broadcast lifting, flat-index reconstruction.
- A future simplification would move grouped/block-local semantics upward
  into scheduler/IR.

### Epilogue family classification is rediscovered in codegen
- `_codegen_group_reduction_epilogue()` partitions epilogues by comparing
  `numel` with the reduced-output `numel`. Same distinction already exists
  in `FusedNestedReductions.can_fuse_with()`.

### Stage ordering is procedural
- `codegen_nested_reduction()` manually sequences flush → grouped → epilogue
  stages via `codegen_body()` calls. The persistent and non-persistent paths
  diverge through control flow, not through an explicit stage plan.

### Tiling constraints are feature-local kernel state
- `nested_reduction_min_rblock`, `min_xblock`, `max_xblock` live as ad hoc
  fields on the kernel rather than on a scheduler plan.

### `FusedNestedReductions` doesn't carry the full block-local reduction spec
- Stores `small_dim_in_r` but not `group_size`, reduction output name, or
  consumer resolution classification. Codegen reconstructs those pieces.

## Test Notes

### Dynamic pattern1 doesn't fuse (skipped)
- `group_size` is not statically known with `dynamic=True`, so
  `NestedReduction.can_fuse()` rejects the pair. Test is skipped.

### Redundant graph-input loads in multi-iter-var patterns
- Only affects pattern1 (weighted norm + sum, 2 iter vars). The core
  RMSNorm + amax/FP8 pattern (1 iter var) has no redundant load.
- Root cause: the weight load `w[B, K]` forces the X-tree to decompose
  `x0` into `x2 (x%K)` and `x3 (x//K)`. Subsequent input loads use
  these sub-entries instead of flat `x0`, defeating CSE.

## Deliberate Design Decisions

### Scheduler requires exact static `rnumel2`
- `NestedReduction.can_fuse()` rejects unless
  `statically_known_equals(rnumel2, sympy.Integer(rnumel2_hint))`.
- Reasonable for current patterns. A deliberate capability restriction.

### Full-resolution epilogues limited to `small_dim_in_r`
- Deliberate capability boundary for the current core branch.

## Fresh Pass: 2026-04-30

Validation run on `/data/users/eellison/pytorch`, branch
`nested_reduction_v3_test`:

```
python test/inductor/test_nested_reduction_internals.py -v
python test/inductor/test_nested_reduction.py -v
python agent_space/capture_nested_reduction_core.py
```

Results:
- Internals kernel-form tests: 39 passed, 15 skipped.
- Numeric/fusion tests: 156 passed, 39 skipped.
- Captured representative producer-consumer AMAX, full-resolution epilogue,
  and dynamic pattern2 kernels in persistent and forced non-persistent modes.

### Captured core kernels look structurally correct
- No captured core kernel reloads an intermediate output buffer.
- Non-persistent kernels emit two sequential `tl.range` loops: one for the
  outer reduction and one for the grouped stage.
- Dynamic pattern2 emits loop-local derived reduced headers inside the second
  loop, which is the important `is_loop=parent.is_loop` behavior.
- Full-resolution epilogue kernels emit both the reduced scale store and the
  full-resolution output store in the same kernel.

### Captured-kernel extraction includes wrapper text after the last kernel
File:
- `test/inductor/test_nested_reduction_internals.py`

The regex over `combined_code` stops at the next `@triton_heuristics` or EOF.
For the final kernel, EOF includes the wrapper module tail. Current checks are
mostly safe because they count `tl.load` / `tl.store`, which the wrapper tail
does not contain, but future FileCheck assertions could accidentally match
wrapper text.

Suggested cleanup:
- Extract kernels from individual `source_codes` entries, or stop the regex at
  the generated string terminator / `async_compile` boundary.

### Full-resolution epilogue still has redundant input reloads
Files:
- `agent_space/captured_nested_reduction_core_round2/fullres_persistent_kernel.py`
- `agent_space/captured_nested_reduction_core_round2/fullres_nonpersistent_kernel.py`

Fresh capture:
- Persistent fullres: actual input loads = 4, metadata `num_load` = 3.
- Non-persistent fullres: actual input loads = 5, metadata `num_load` = 3.

The persistent path now reuses the original `x` tile for the full-resolution
output, but it still reloads `weight` because grouped-stage indexing uses the
decomposed grouped index and full-resolution indexing uses the parent index.
The non-persistent path also reloads `x` inside loop 2 for the full-resolution
stage.

This is not a correctness blocker. It is a perf/codegen-normalization follow-up:
canonicalize equivalent grouped/parent index expressions, or teach the family
to reuse compatible graph-input loads across grouped/full-resolution stages.

### `num_load` metadata no longer means textual `tl.load` count for fullres
The internals tests intentionally pass `num_load=3` while textual load counts
are 4 or 5 for full-resolution kernels. This is consistent with current
bookkeeping, but reviewers should know `num_load` is counting standard handler
loads, not every remapped/family load emitted into the kernel.

If autotune or diagnostics ever start relying more directly on `num_load`,
nested-reduction family loads should be routed through the same accounting path
or counted explicitly.

### Grouped-reduction store path still mirrors generic store bookkeeping
File:
- `torch/_inductor/codegen/simd.py`

`_GroupedReductionOpsHandler.store_reduction()` manually updates
`store_cache`, mutation aliases, and `num_store`, then routes the physical store
through the reduced-output family. This is now correct, but it remains less
idiomatic than the normal `KernelHandler.store_reduction()` path.

Longer-term normalization would be a generic "store through alternate range
trees" path so remapped stores can preserve normal store bookkeeping and
removal behavior without manual mirroring.

### `CSE.invalidate(...store_cache.values())` is a nested-reduction stage boundary
File:
- `torch/_inductor/codegen/simd.py`

In non-persistent mode, `codegen_nested_reduction()` invalidates expression CSE
while preserving values currently in `store_cache`. This models "start loop 2;
reload loop-local inputs; keep post-loop reduction values."

The behavior is correct for current kernels, but the API is not self-describing.
A small generic helper such as "invalidate expression cache but preserve
store-backed values" would make this look like normal phase management rather
than a nested-reduction-specific CSE trick.

### The new shared test utility must be committed or folded back
File:
- `test/inductor/nested_reduction_test_utils.py`

Current `git status` shows this helper as untracked, while both nested-reduction
test files import `choices_context` from it. Before finalizing the stack, either
add this file to the commit or inline the helper back into the test files.
