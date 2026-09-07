# PR #190594 step-by-step reading checklist

Full path:
`/data/users/eellison/pytorch/agent_space/pr190594_step_by_step_checklist.md`

This is a check-off companion to
[`/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md).
It follows execution order rather than file order.

## Target and reading rule

- GitHub PR: [pytorch/pytorch#190594](https://github.com/pytorch/pytorch/pull/190594)
- Current local base commit: `b3c4e7cb830`
- The checklist follows the current working tree, including the pending
  simplification on top of that commit. Open the files directly for this
  walkthrough. Use `git show b3c4e7cb830:<path>` only to compare against the
  last committed version.

Open the current implementation with:

```bash
less torch/_inductor/scheduler.py
less torch/_inductor/codegen/simd.py
less torch/_inductor/codegen/triton.py
```

Do not check a step merely because you read the function. Check it when you can
answer the question under **Checkpoint**.

## One example for the entire walkthrough

Keep this example beside the code:

```text
B = 32
D = 1024
G = 16

numel       = B * (D / G) = 2048
rnumel      = G           = 16
full_numel  = 2048 * 16   = 32768
child_numel = full_numel / 2 = 16384

reduction read: base + r
lane-0 read:    base + 2*c
lane-1 read:    base + 2*c + 1
```

The reduction scheduler group is `(2048, 16)`. The packing epilogue group is
`(16384, 1)`. Ordinary fusion rejects that intentional numel mismatch; the
sub-parent plan proves why it is safe.

## The complete path

```text
fusion request
  -> SIMDScheduling admission gates                 simd.py, first visit
  -> dependency and lane proof                      scheduler.py
  -> FusedStagedReduction identity
  -> Scheduler.merge_loops
  -> staged dispatch and plan rebuild               simd.py, second visit
  -> parent emission and CSE-liveness boundary      simd.py
  -> live source: reshape + tl.split                 triton.py
     expired source: derived-domain reload           simd.py
  -> sub-parent pointwise stores                     simd.py
```

## Phase 0: see the graph before reading the machinery

- [x] **Step 1 - Read one positive test.**

  Exact PR location: `test/inductor/test_nested_reduction.py:864`,
  `test_standalone_sub_parent_epilogue`.

  Local file:
  [`/data/users/eellison/pytorch/test/inductor/test_nested_reduction.py`](/data/users/eellison/pytorch/test/inductor/test_nested_reduction.py)

  Follow the view into groups of 16, the amax reduction, the two `[..., 0]` and
  `[..., 1]` lane reads, and the half-resolution outputs.

  **Checkpoint:** Why is the packing output `full_numel / 2`, rather than
  reduced resolution or full resolution?

- [x] **Step 2 - Read the scheduler-dependency example in the main guide.**

  Read `## The problem` and `## What "lane k" means` in
  [`pr190594_interleaved_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md).

  **Checkpoint:** Point at the three `MemoryDep` indices and identify the
  reduction read, lane 0, and lane 1 without referring to graph shape.

## Phase 1: first visit to SIMD - may the backend attempt this fusion?

- [x] **Step 3 - Enter through `SIMDScheduling.can_fuse`.**

  Exact PR file: `torch/_inductor/codegen/simd.py`. Find `can_fuse`, then follow
  its call to `_sub_parent_epilogue_decision` at line 2513.

  Local file:
  [`/data/users/eellison/pytorch/torch/_inductor/codegen/simd.py`](/data/users/eellison/pytorch/torch/_inductor/codegen/simd.py)

  This is why `simd.py` is read before the scheduler planner: it decides whether
  the backend has a special answer or should continue through ordinary fusion.

  **Checkpoint:** Explain the three results: a valid staged plan (`FUSE`), a
  specialized shape with no legal plan (`REJECT`), and a pair ordinary fusion
  may handle (`DEFER`). Why must a pre-existing `FusedStagedReduction` never
  fall through to ordinary fusion if its rebuilt plan is lost?

- [x] **Step 4 - Read `_sub_parent_epilogue_plan` in `simd.py`.**

  Exact line: `simd.py:2560`.

  This is the backend wrapper, not the semantic planner. Read its checks for:

  - the feature flag;
  - `supports_sub_parent_epilogue`;
  - the semantic scheduler plan;
  - two-dimensional tiling.

  **Checkpoint:** Which failures mean "this backend cannot emit it" rather than
  "the memory-dependency relationship is illegal"?

- [x] **Step 5 - Read `_sub_parent_tiling_is_2d`.**

  Exact line: `simd.py:2586`.

  The derived child tree comes from the parent's R tree. A y/z tiling has no
  representation in this implementation.

  **Checkpoint:** Why must this be decided before fusion rather than asserted
  for the first time during codegen?

## Phase 1: scheduler - prove the fusion is semantically legal

- [x] **Step 6 - Read `NestedReduction.sub_parent_epilogue_plan`.**

  Exact line: `scheduler.py:611`.

  Local file:
  [`/data/users/eellison/pytorch/torch/_inductor/scheduler.py`](/data/users/eellison/pytorch/torch/_inductor/scheduler.py)

  Read every `return None` in order. This method assembles the proof; the next
  steps unpack its individual claims.

  **Checkpoint:** List the facts that must be established before it returns a
  `StagedReductionPlan`.

- [x] **Step 7 - Read `_sub_parent_epilogue_candidate_nodes`.**

  Exact line: `scheduler.py:684`.

  Apply the example: `2 * 16384 == 2048 * 16`. Then follow how every other
  member must fit the reduction, reduced-output, or full-parent domain.

  **Checkpoint:** Why must classification reject a sibling that fits none of
  the four supported execution domains?

- [x] **Step 8 - Read `try_get_sub_parent_extent_subs` and
  `interleaved_sub_parent_lane`.**

  Exact lines: `scheduler.py:740` and `scheduler.py:757`.

  First prove the parent extent is divisible by two. Then normalize symbolic
  extents and recover `Mod(index, 2)` as a static lane.

  **Checkpoint:** Why must `2*c + d` be rejected when `d` cannot be proved to
  be the constant 0 or 1?

- [ ] **Step 9 - Read `_try_get_sub_parent_source_layouts`.**

  Exact line: `scheduler.py:797`.

  This is the main source proof. `normalized_source_read_indices` calls
  `MemoryDep.normalize_with_ranges` to express parent reads over explicit
  `(X, R)` coordinates and epilogue reads over `(X, R / 2)`. The dependency
  method reuses `SIMDKernel.map_kernel_groups_to_node_sizes` for split/merged
  native dimensions. Internal and external parent sources follow this same
  proof; unrelated epilogue inputs remain ordinary loads.

  **Checkpoint:** Why do raw `MemoryDep` variables not identify corresponding
  parent and child coordinates, and what ordering contract lets
  `normalize_with_ranges` establish that correspondence?

- [ ] **Step 10 - Read the core index proof.**

  Continue within `_try_get_sub_parent_source_layouts` at `scheduler.py:863`.
  Require one parent index for the source, recover the exact physical lane, then
  substitute `parent_r = 2 * child_r + lane` into the parent index and require
  static equality with every child index.

  **Checkpoint:** Show why `base + 2*c + 1` passes for lane 1 but
  an odd-offset source cannot be treated as the same resident lane projection.

- [ ] **Step 11 - Read the ambiguity and leaf checks.**

  Read:

  1. the one-parent-index requirement inside the source proof at
     `scheduler.py:871`;
  2. `_sub_parent_epilogue_outputs_unread`.

  The leaf check applies to readers outside the sub-parent stage. A consumer at
  the same derived resolution is classified as another epilogue node and runs
  in stage order. A reduced- or full-resolution reader would require a later
  stage that is not represented yet and therefore rejects the plan.

  There is deliberately no blanket source-lifetime check. Parent-stage
  sources use the same normalized index proof, then codegen decides whether a
  value is still CSE-live or must be loaded again.

  **Checkpoint:** Why does more than one normalized parent index make resident
  lane selection ambiguous, and why can a non-epilogue reader of an epilogue
  output not run before the staged output exists?

## Phase boundary: preserve identity, not the old plan

- [ ] **Step 12 - Read the plan and identity types.**

  Exact locations:

  - `NestedReductionStage` at `scheduler.py:1437`;
  - `SubParentEpilogueStage` at `scheduler.py:1448`;
  - `StagedReductionPlan` at `scheduler.py:1457`;
  - `FusedStagedReduction` at `scheduler.py:3457`.

  **Checkpoint:** Which fields describe topology and emission details that must
  be rebuilt after loop merging, and which object merely tells later
  scheduler/codegen paths that generic SIMD emission is forbidden?

- [ ] **Step 13 - Read the `merge_loops` boundary.**

  Read `Note [Sub-parent reduction epilogues]` near the top of
  `scheduler.py`, then inspect `Scheduler.merge_loops` and
  `LoopBody.merge_loops`.

  Local files:

  - [`scheduler.py`](/data/users/eellison/pytorch/torch/_inductor/scheduler.py)
  - [`loop_body.py`](/data/users/eellison/pytorch/torch/_inductor/loop_body.py)

  **Checkpoint:** What can change after fusion, and why does that make a saved
  fusion-time plan unsafe even though the scheduler group stays the same?

## Phase 2: second visit to SIMD - rebuild and emit

- [ ] **Step 14 - Read staged dispatch.**

  Follow:

  ```text
  Scheduler._codegen
    -> backend.codegen_staged_reduction
    -> SIMDScheduling.codegen_staged_reduction
  ```

  Exact SIMD line: `simd.py:2991`.

  **Checkpoint:** Why is failure to rebuild the final plan an assertion rather
  than permission to fall back to a second kernel?

- [ ] **Step 15 - Read the final plan rebuild.**

  From `codegen_staged_reduction`, follow `_find_sub_parent_epilogue_plan` back
  at `simd.py:2614` through the SIMD backend gates and scheduler legality
  planner.

  **Checkpoint:** Which inputs now reflect the post-`merge_loops`,
  removed-operation state?

- [ ] **Step 16 - Read `_codegen_reduction_with_sub_parent_epilogue`.**

  Exact line: `simd.py:3446`.

  First read only its outline:

  1. construct the parent reduction schedule;
  2. include epilogue nodes in index-width analysis;
  3. create the kernel and force the approved 2D tiling;
  4. construct the sub-parent family;
  5. record planned source values while emitting the parent stage;
  6. flush the parent stage, establishing which values remain CSE-live;
  7. emit the epilogue through the shared remapped-pointwise path.

  **Checkpoint:** Which schedule chooses tiling, and which larger schedule must
  be considered for address-width safety?

- [ ] **Step 17 - Read the derived coordinate construction.**

  Read in this order:

  1. `_GroupedReductionLayout.make_sub_parent_family` at `simd.py:1897`;
  2. `_GroupedReductionLayout.sub_parent_iteration_values` at `simd.py:1918`.

  Apply the example: `(B, groups, local) = (32, 64, 16)` becomes
  `(32, 64, 8)` for each factor-2 output lane.

  **Checkpoint:** Which axis shrinks, and why does the X axis remain unchanged?

- [ ] **Step 18 - Read parent-source recording and resolution.**

  Read:

  1. `_SubParentSourceLoadResolver.load` at `simd.py:2234`;
  2. `_SubParentSourceLoadResolver.resolve_load` at `simd.py:2241`;
  3. `_GroupedReductionLayout.materialize_value_at_sub_parent_resolution` at
     `simd.py:1978`.

  The wrapper records the value returned by the normal parent load or store
  cache, but does not split it immediately. After parent codegen is flushed,
  `resolve_load` asks whether that exact value is still present in CSE, store,
  or reduction caches. A live parent-resolution value is reshaped and split;
  an expired value falls through to an ordinary derived-domain load.

  Explicitly masked parent loads are not recorded, and explicitly masked child
  loads bypass forwarding, so their distinct masks and fill values remain on
  the normal load path.

  **Checkpoint:** Why is liveness checked after the parent flush rather than
  equating persistent with reusable and looped with expired? Give one looped,
  loop-invariant value that can still be forwarded.

- [ ] **Step 19 - Read the looped source path.**

  Return to `_codegen_reduction_with_sub_parent_epilogue`. It now flushes the
  parent stage unconditionally. In a looped reduction, `TritonKernel.codegen_body`
  invalidates loop-local CSE and store-cache values. The resolver therefore
  declines them and the epilogue reloads at the derived index. For an internal
  source, the ordinary invalidated-store path keeps the buffer materialized and
  emits the existing read-after-write barrier.

  **Checkpoint:** Why can the looped form remain one kernel even though it does
  not retain the complete source tile in registers?

- [ ] **Step 20 - Read `_DerivedIterationFamily.resolve_load` and
  `_codegen_remapped_pointwise`.**

  Exact lines: `simd.py:1552` and `simd.py:3390`.

  `_codegen_sub_parent_pointwise` no longer exists. Standalone sub-parent and
  existing nested pointwise stages share `_codegen_remapped_pointwise` and
  `_PointwiseRemapHandler`. The handler first checks family-owned forwarded
  values, then the optional source resolver, then performs an ordinary remapped
  load. Activating the family swaps the active range trees so an ordinary
  pointwise body runs at child resolution.

  **Checkpoint:** What does the pointwise loop body know about the fact that it
  is executing in a derived domain?

- [ ] **Step 21 - Read `TritonKernel.emit_split_via_reshape`.**

  Exact isolated line: `triton.py:6165`.

  Local file:
  [`/data/users/eellison/pytorch/torch/_inductor/codegen/triton.py`](/data/users/eellison/pytorch/torch/_inductor/codegen/triton.py)

  This is the final lowering detail: reshape `[..., n]` into
  `[..., n/2, 2]`, then use `tl.split` on the trailing lane axis.

  **Checkpoint:** Why is `tl.split` an implementation of an already-proved
  schedule rather than the feature's semantic abstraction?

## Tests to close the loop

- [ ] **Step 22 - Read the typed-identity test.**

  Exact location: `test_nested_reduction.py:996`,
  `test_standalone_sub_parent_has_staged_identity`.

  **Checkpoint:** Which generic scheduler/codegen routes must not receive this
  fused node?

- [ ] **Step 23 - Read two rejections and one allowed sibling.**

  Read:

  - `test_standalone_sub_parent_rejects_ambiguous_source_load` at line 1272;
  - `test_standalone_sub_parent_rejects_output_fullres_reader` at line 1083;
  - `test_standalone_sub_parent_allows_reduced_sibling_source`.

  **Checkpoint:** Connect each rejection to its exact planner predicate, then
  explain why the sibling's unplanned source is an ordinary reload rather than
  a resident-lane lifetime violation.

- [ ] **Step 24 - Read the kernel-form tests.**

  Read:

  - `test_standalone_nvfp4_inline_asm_kernel_form` at line 2199;
  - `test_standalone_sub_parent_epilogue_kernel_form` at line 2215;
  - `test_uses_indexing_schedule` in `test_indexing.py` at line 878.

  **Checkpoint:** Identify the assertions for one kernel, `tl.split`, expected
  load behavior, derived stores, and epilogue-aware index-width selection.

## Final check-off

- [ ] I can explain why ordinary fusion rejects `(2048, 16)` with `(16384, 1)`.
- [ ] I can derive lane 0 and lane 1 from the `MemoryDep` index expressions.
- [ ] I can separate SIMD backend admission from scheduler semantic legality.
- [ ] I can explain why only `FusedStagedReduction` crosses `merge_loops`.
- [ ] I can describe the persistent source path without looking at the code.
- [ ] I can describe the looped source path without looking at the code.
- [ ] I can explain why CSE liveness, rather than the persistent/looped label,
  decides whether a parent value is split or reloaded.
- [ ] I can explain why `tl.split` is code generation, not the scheduling proof.
- [ ] I can explain why internal and external parent sources use the same index
  proof but different persistent/looped value paths.
