# Review: correctness follow-up (crashes A/B) and padded-scatter support

Date: 2026-09-02. Both diffs uncommitted, reviewed before amend/publish.

This is an AI-assisted local review document. It is not intended to be pasted
into GitHub without human review and the disclosure required by `AI_POLICY.md`.

## 1. Correctness follow-up (`pr191974_simplified_wt`, on 61b333901fa)

Diff: scheduler.py +25 (two changes), test_nested_reduction.py +45 (two tests).

### Verdict

APPROVE the crash-A change as is. APPROVE the crash-B change as an interim
step, with one condition (broaden the predicate) and one disclosure (it also
drops the persistent one-kernel case). Both regression tests are well built.

### Crash A: `merge_loops` skips `FusedNestedReductions`

Verified on this tree: `mark_dynamic` batch -> nested=1, kernels=1, correct on
the recompile; automatic dynamic -> two compiles, both correct; static
unchanged. This is the fix I validated earlier by monkeypatch, now in place.
Skipping only `FusedNestedReductions` (not the standalone
`FusedStagedReduction`) is fine: the standalone relation proofs are
merge-invariant (verified in the previous pass), so nothing else needs the
freeze. Side benefit: the last post-fusion body mutation for nested members is
gone, so codegen's re-plan is a pure recomputation of the fusion-time plan.

### Crash B: decline required-live sources written by reduction ancestors

Cures the reported crash: the residual pattern now compiles at every D
(nested kernel = prologue + rms + block amax; pack split into its own
pointwise kernel, confirmed by kernel numels x=B*D/2, r0=1).

Two issues.

1. Declines for all policies. At D=512 (persistent) the same graph was one
   kernel before this change and is two kernels after (measured:
   `prologue_out_norealize 512` went from nested=1 kernels=1 to kernels=2).
   That is the opposite of the stated intent to keep the persistent case. It
   is an acceptable interim cost only if the reload fix follows promptly, and
   the commit message should say the persistent case is temporarily split.

2. The predicate is too narrow. "Writer is an ancestor of the outer reduction"
   covers prologues on the reduction's path, but the set that dies at loop
   close is every non-displaced outer node: anything emitted in the parent
   stage. A parent-stage pointwise that consumes the prologue but does not
   feed the reduction (probe: `s = realize(sigmoid(h))`, read by the pack at
   lane positions) is neither displaced nor an ancestor. Planner-level
   instrumentation (`tmp_sibling_instrument.py`) shows the outer group held
   that node (role SIBLING) and `_plan_nested_sub_parent_stage` accepted the
   plan in 6 of 6 calls. The end-to-end probe did not crash only because the
   pack had already fused with the block amax through the standalone staged
   path, and a separate guard refused to nest an existing staged reduction
   (fusion log: "sub-parent epilogue planning failed" for outer+pack, i.e. the
   standalone planner correctly rejects the leading source; then "staged
   reduction plan would be lost" for outer+FusedStagedReduction). That is a
   fusion-order race, not a check. Fix: use the displacement predicate the
   classifier already uses: decline when the writer is an outer node, not a
   reduction, whose ancestors contain no outer-reduction name (i.e. it is not
   LOCAL_REDUCTION_INPUT). Same cost, strict superset, and it matches the
   standalone path's leading-vs-final rule. Pin it with a planner-level unit
   test (mock style, like `test_nested_reduction_rejects_ambiguous_pointwise_
   domain`) since end-to-end reachability depends on fusion order.

### Tests

- `test_dynamic_materialized_parent_output`: mark -> 1 kernel, automatic -> 2
  kernels, numerics asserted on both calls, both persistent configs. Good.
- `test_nested_sub_parent_rejects_parent_prologue_source`: D chosen per config
  (4096 looped / 512 persistent), output and realized prologue kinds,
  nested-vs-unnested oracle, `expected_kernels=2`. Good. Note it also pins the
  persistent split described above; when the reload fix lands, the persistent
  expectation should flip back to 1.

### Verification (this tree)

Nested 403 ran = 402 green + the known openssl AOTI casualty; scheduler 144 OK;
both crash repros cured; standalone F1 path unaffected.

## 2. Padded-scatter support (`pr191974_padded_wt`, on 61b333901fa)

Diff: ir.py +31/-6, lowering.py +36/-3, scheduler.py +9, new
`test/inductor/test_padded_scatter.py` (10 tests).

### What it does

`index_put` with a single boolean mask and a scalar value on a realized
destination now lowers to an `ir.Scatter` with a `store_mask`: a predicated
store that never reads the destination, instead of `where(mask, value, self)`
which re-reads the whole output. Gated by
`BackendFeature.MASKED_SCATTER_WITH_INDEX` (exists; Triton declares it;
precedent in decomposition.py:1206). Falls back to the old path for
accumulate, unrealized destinations, and masks or values that read the
destination. `Scatter.store_output` wraps the store in `ops.masked`; Triton's
`indexing()` folds `_load_mask` into store masks (triton.py:4217, 4323), so the
store is predicated. The scheduler refuses consumer fusion into a group that
contains a predicated scatter, because `CSEProxy.store` populates
`store_cache` unconditionally (common.py:3052-3053, including mutation
aliases) and a forwarded value would be wrong wherever the mask is false.

### Verdict

APPROVE. Sound as written; two hardening suggestions and two hygiene notes.

Verified:
- Control flow: staged and nested plans in `can_fuse` fall through to the
  common legality branch, so the predicated-store rule is reached for nested
  consumer fusions too (my ordering concern from the first read is resolved).
- `MutationLayoutSHOULDREMOVE.__init__` calls `mark_buffer_mutated`, so prior
  readers are realized before the mutation, equivalent to `realize_into`. View
  destinations index through the target's real layout.
- Tests: 10/10 pass. Nested suite on this tree: 411 ran = 410 green + the
  openssl casualty. `test_torchinductor` index_put / masked_fill / scatter
  slices: 99 ran, 0 GPU errors or failures (GPUTests: 46 pass, 3 skips); the 48 errors are all the openssl
  `_get_file_checksum` casualty on cpp paths (the one CUDA-named failure,
  `test_ctr_not_moved_to_cuda_when_used_in_index_put`, compiles a CPU
  `cpp_fused_lift_fresh` kernel and fails identically on the baseline tree).

Suggestions:
1. Defense in depth: make `CSEProxy.store` skip `_update_store_cache` when the
   kernel has an active load mask. "A predicated store never populates the
   store cache" is the precise invariant; it protects any fusion path that does
   not pass through this exact `can_fuse` branch, and it would even let
   consumer fusion stay legal (the consumer would physically reload). Keep the
   scheduler rule or retire it afterwards.
2. `register_users_of` runs only from `run_node` on the FX result (`self`), so
   the scatter's reads of mask and value are not recorded in `name_to_users`.
   Correctness holds through scheduler dependencies (the mutated-after-store
   tests cover it), but lowering heuristics that consult users will not see
   the scatter; consider registering explicitly.
3. `Loops.get_read_names()` for the bare `Scatter` traces the store function
   for `ComputedBuffer`, so the mask read is visible to the scheduler. Note
   only.
4. This worktree does not include the correctness follow-up; rebase before
   publishing.

## Scratch (git-ignored, `agent_space/pr191974_failure_investigation/`)

`tmp_sibling_source_check.py`, `tmp_sibling_instrument.py`,
`tmp_sibling_instrument2.py`, `tmp_run_with_nms_stub.py` (torchvision::nms
stub for running `test_torchinductor` under the overlay), plus the earlier
crash repros.

## Addendum (same day): widened guard implemented and verified

Implemented in `pr191974_simplified_wt` (uncommitted): the crash-B predicate
now declines when a required-live source is written by an outer node that is
neither a reduction nor `LOCAL_REDUCTION_INPUT`, i.e. any node emitted inside
the parent loop. Pins: `test_nested_sub_parent_rejects_parent_stage_live_source`
(planner-level, three writer roles) and
`test_nested_sub_parent_rejects_parent_stage_sibling_source` (end-to-end, both
persistent configs). The real sibling plan now returns None (previously
accepted 6/6). Nested suite 413 = 412 green + openssl env casualty; scheduler
150 OK; lint clean; battery rerun of all 11 formerly-loud cases across
P/L/LF: 31 PASS, 2 PASS* (bf16 cast placement; emulate siblings byte-equal).

## Landing checks (candidate = simplified worktree with the full follow-up)

- Kernel byte-identity: 30/30 normalized kernel hashes identical to the
  published head across every `_capture_*_sources` helper in the nested test
  module, both persistent settings (`tmp_kernel_hash_corpus.py`).
- Perf A/B (`bench_nested_freeze.py --mode current --coordinate-descent`,
  external CUDA graphs): block quant 2.180 -> 2.181 us, swizzle 2.325 ->
  2.344 us; kernel and nested counts unchanged; X-parent shapes unfused on
  both trees.
- Amended commit message draft: `PR191974_COMMIT_MESSAGE_v2.md`.
