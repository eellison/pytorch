# Nested Reduction: Cleanup Tracking

## Done in this PR

- [x] `_launch_kernel_and_cleanup()` extracted
- [x] `GPU_TYPE` in internals tests
- [x] `meta_num_load` rename in test helper
- [x] Kernel extraction regex stops at `async_compile.wait`
- [x] Fusion-rejection tests (non-pow-2, too-large group_size)
- [x] `from_kernel` simplified -- assert 2 trees, index directly
- [x] Delete unused `_DerivedIterationFamily.load()`
- [x] `NoNestedReductionTest` patches to `False` explicitly
- [x] Name `MAX_TILE_ELEMENTS = 1048576`
- [x] Comment on `make_full_resolution_family` -- register-tile broadcast
- [x] Dynamic test varying both B and D at runtime
- [x] max_xblock vs min_xblock conflict -- clamped
- [x] Config comment: "TOPK" -> "FP8 block size"
- [x] check_fusion uses assertEqual(1) not >=1
- [x] Unicode arrows/dashes -> ASCII
- [x] score_fusion_memory overload stubs include allow_nested_reduction
- [x] Shared pointwise epilogue body emitter (`_emit_pointwise_epilogue`)
- [x] Enforce nested min/max block constraints in coordinate descent and
  dynamic R-block scaling
- [x] Keep nested XBLOCK max compatible with Triton's hard XBLOCK limit in
  coordinate descent.
- [x] Reorganize supported dynamic-shape tests so `dynamic` is an explicit
  parameter and dynamic cases run multiple sizes
- [x] Clarify and test the dynamic-shape contract:
  `group_size` is static, the split dimension must be divisible by it, and the
  resulting group count may be symbolic. Pattern1 keeps the `K` axis static and
  varies `B/D`; pattern2 can vary `B/D` with a literal group size.
- [x] Reuse the generic name-based fusion score for nested reductions.
  The old nested-specific byte scorer mostly encoded ordering: fuse ordinary
  sibling reductions before nested producer-consumer pairs. That policy now
  lives as a small nested-late priority in `FusionScore`, while the actual
  shared-data score uses the same name-based path as template/user Triton
  epilogues. See `agent_space/nested_reduction_generic_fusion_irregularities.md`.
- [x] Move the full-resolution epilogue reindex fallback into generic vertical
  fusion.
  If `can_fuse_vertical()` fails, the scheduler now tries the existing
  reduction/pointwise reindexer and verifies vertical legality before keeping
  the mutation. This removes the nested-specific reindex retry from
  `FusedNestedReductions.can_fuse_with()`.
- [x] Collapse reduced/full nested pointwise emission into
  `_codegen_remapped_pointwise()`.
  Reduced-output and full-resolution pointwise stages differ in their
  iteration family, source coordinate groups, and load resolver; the actual
  body emission now shares one helper.
- [x] Inline one-use broadcast shape helpers in `_GroupReductionLayout`.
  The remaining manual part is only the two legal axis orderings for lifting a
  grouped value back to the parent tile before calling the shared Triton
  `emit_broadcast_via_reshape()` helper.
- [x] Remove dead `min_xblock` threading through tiled/3D reduction config
  helpers.
  Nested codegen rejects `y`/`z` tiled reductions before kernel creation, so
  only the 1D reduction/persistent paths need the nested block constraints in
  this PR.

## First follow-up (after landing)

- [ ] `NestedReductionPlan` on `FusedNestedReductions`
- [ ] `can_fuse()` returns analysis object
- [ ] Epilogue resolution classification as named helper or on plan
- [ ] FP8 unfused-vs-fused compiled comparison test
- [ ] Additional rejection tests: mismatched totals, no shared reads
- [ ] Make B=1 / single-group support first-class in the plan/layout.
  For small-dim-in-x with batch size 1, `numel1 == group_size`, so the
  grouped axis has exactly one group (`groups == 1`). That is not
  `group_size == 1`. Keep this path explicit so future cleanups do not
  accidentally break guaranteed fusion for batch-1 workloads.
- [ ] Treat consumer resolution as first-class.
  The concrete landable first step is a small `NestedConsumerResolution`
  enum/helper that classifies reduced/full today without changing generated
  code. Later, NVFP4 adds `HALF` to the same helper. See
  `agent_space/nested_reduction_resolution_plan.md` and
  `agent_space/nested_reduction_pointwise_resolution_handoff.md`.
- [x] Collapse reduced/full epilogue emission into one pointwise-stage emitter.
  This is the larger simplification that still supports NVFP4 cleanly:
  reduced, full, and later half-resolution consumers differ in family
  construction and value materialization, not in how their pointwise bodies are
  emitted.
- [ ] Fold loop-local full-resolution prologue handling into the plan.
  The current PR has a targeted fix that defers full-resolution outer epilogues
  into the grouped pass when they feed nested work. A future
  `NestedReductionPlan` should make that role/resolution/lifetime
  classification explicit. See
  `agent_space/nested_reduction_loop_local_prologue_issue.md`.
- [x] Remove the nested pointwise flat-index shortcut.
  Full-resolution pointwise work should be emitted from the grouped
  reduction's logical coordinates, not from a flattened parent `[X, R]` tile.
  The flat path was wrong for small-dim-in-x when the body order is
  `[B, D, K]` but the physical tile is `[B*K, D]`. The current PR now maps
  both full-resolution and reduced-output pointwise bodies from explicit
  source groups/source values. See
  `agent_space/nested_reduction_fullres_prologue_order_issue.md`.
- [ ] Future: reuse dep-aware loop mapping before enabling small-dim-in-x
  full-resolution epilogues.
  After the coordinate-mapping cleanup, it is tempting to lift the remaining
  small-dim-in-x full-res epilogue guard. A quick experiment fused
  `return x_normed + s[:, None, :]` but produced wrong numerics:
  `SIMDKernel.is_compatible([B, D, K], [B, K, D])` can pass by factorizing
  `D`, even though semantically this is not a legal coordinate mapping. The
  future fix should reuse normal fusion's dependency-aware loop matching or
  reindexing logic, not only size splitting.

## Current PR-compatible simplifications

- [x] Guard combo kernels from unwrapping nested reductions.
  `ComboKernel` currently flattens fused nodes with `get_nodes()` and then
  emits a normal node schedule. That bypasses `codegen_nested_reduction()` and
  hit the original iteration-space mismatch for two independent RMSNorm -> FP8
  quants. The current PR now filters `FusedNestedReductions` from combo
  candidates, matching `FusedMixOrderReductions`, and adds a regression test.

- [ ] Optionally extract epilogue partitioning into a tiny helper.
  This is lower value than the shared emitter, but it gives the reduced/full
  split one name and leaves room for a future `HALF` bucket without changing
  call structure.

- [x] Add a focused positive test for reduced-output masks.
  Most current positive cases use power-of-two group counts such as
  `4096 / 16`, `4096 / 128`, or `8192 / 16`, so
  `reduced_r0_index_mask = reduced_r0_index < nested_R0_REDUCED_NUMEL` is
  emitted but often not forced false. A case like `B=4, D=384, G=128` should
  still fuse and gives `D / G == 3`, while the Triton block rounds the parent
  R tile up. Without the reduced mask, the extra group lane can overwrite the
  next row's first group.

## Fresh output-code findings

- [ ] Decide what to do about B=1 with large outer reductions that get split.
  Probing `B=1, K=16, D=16384` did not take nested fusion: fusion logs show the
  outer mean split into two reduction nodes (`[8192]` and `[2]`), and the
  nested candidate is rejected with "intermediate nodes between node1 & node2".
  `D=8192` still fuses, and the `D=16384` case fuses again if
  `split_reductions=False`, so this is specifically split-reduction
  interaction rather than grouped codegen. This should not be framed as a
  general split-reduction policy. Nested reduction only accepts a small exact
  grouped reduction today (`MAX_SMALL_REDUCTION == 128`, power of 2), so large
  staged reductions such as `256k -> 4k -> 1` are outside this path: a `4k`
  second-stage reduction would be rejected instead of treated as nested. Any
  future fix should stay limited to recognized nested-reduction patterns where
  the second reduction is the small local/grouped reduction.

- [ ] Consider a small generated-code cleanup for B=1 stores.
  B=1 pattern1 emits a correct fused store with
  `tl.broadcast_to(r0_1, [nested_X_REDUCED_BLOCK, R0_BLOCK])` where
  `nested_X_REDUCED_BLOCK == 1`. If store indexing can preserve the existing
  `[1, R]` shape, avoiding this no-op-looking broadcast would make the B=1 path
  look less special.

## Noted but not actionable now

- vars_and_sizes prefix filtering -- theoretical, not a practical bug
- can_fuse_with bypass of generic scheduler checks -- documented, intentional
- persistent config path threading -- constraints flow correctly today
- default-on config -- product decision, not a code cleanup
- FP8 on XPU -- needs XPU CI visibility
- Generated-code locality for derived headers. The old explicit early
  `ensure_headers()` call is gone, but `ensure_headers()` still emits grouped
  constants, reduced index, and mask as one bundle. The grouped reduction needs
  the constants before `tl.reshape`; the reduced index/mask are only useful at
  stores. Splitting this into shape-constant emission and index/mask emission
  would make generated code prettier, but adds backend API/state churn without
  changing behavior.
- Reduced-output tile flattening. Stores and reduced-resolution epilogues could
  technically run over a flat `[XBLOCK * nested_R0_REDUCED_BLOCK]` tile instead
  of a 2D `[XBLOCK, nested_R0_REDUCED_BLOCK]` tile. The grouped reduction still
  needs the 3D `[XBLOCK, nested_R0_REDUCED_BLOCK, GROUP_SIZE]` view, so this
  would only simplify the reduced-output consumer/store shape. Unclear perf
  impact; keep the 2D shape for now because it matches the logical output and
  keeps x/r masks separate.

## Skipped (not this PR)

- Template/matmul reduction epilogues. A prototype can fuse the narrow
  `matmul -> grouped amax` case by reshaping the accumulator tile and storing a
  reduced tile, including ragged `M` masks. It is not enough for the real block
  quant pattern because the next reduced-resolution scale node and the
  full-resolution quant consumer need a template-side resolution/range resolver.
  Keep the existing template + reduction scheduler guard for this PR. See
  `agent_space/template_reduction_epilogue_findings.md`.
- Shared test helper file -- reviewer asked to inline
- Tiling constraints object -- 3 fields don't need a dataclass
- Symbolic shapes instead of strings -- significant refactor
- Lazy value sources -- too abstract now
- Grouped-lane full-res family -- major range-tree change
- Extract module -- do after plan cleanup
- Broadcast helper reuse -- manual sequence is necessary
