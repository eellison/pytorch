# PR #190595 step-by-step reading checklist

Full path:
`/data/users/eellison/pytorch/agent_space/pr190595_step_by_step_checklist.md`

This is a check-off companion to
[pr190595_nested_interleaved.md](/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md)
(the guide) and
[pr190595_review_walkthrough.md](/data/users/eellison/pytorch/agent_space/pr190595_review_walkthrough.md)
(file-by-file review questions). Like the #190594 checklist, it follows
execution order rather than file order.

## Target and reading rule

- GitHub PR: [pytorch/pytorch#190595](https://github.com/pytorch/pytorch/pull/190595)
- Local commit: `359b11d5aa1`; parent `86038d6e1cd` (#190594, landed).
- Current isolated review diff: run
  `git diff 86038d6e1cd` inside the worktree below so the uncommitted
  review/removal changes are included.
- All file links below point into the single working review tree:
  `/data/users/eellison/pytorch/agent_space/pr190594_ci_fix`. It is stopped at
  `359b11d5aa1` with the current #190595 review/removal edits left uncommitted. Do NOT
  read the main checkout for this review; it is on a different commit.

Open the implementation with:

- [scheduler.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py)
- [simd.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py)
- [triton.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/triton.py)
- [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py)
- [test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_inductor_scheduler.py)

Current isolated verification is 296 passed / 1 skipped in the nested test
file and 60 passed / 6 skipped in the scheduler test file; diff-check, syntax,
and quicklint are clean.

This checklist assumes #190594 has been reviewed. It does not re-derive the
lane proof, `FusedStagedReduction` identity, the derived iteration family, or
the basic source-load resolver -- only how this PR attaches them to the
existing nested-reduction pipeline. The current worktree uses the direct eager
broadcast path; the abandoned projection experiments are not part of this
review or the planned two-PR stack.

Do not check a step merely because you read the function. Check it when you
can answer the question under **Checkpoint**.

Steps marked **HIGH PRIORITY** contain the main new correctness or design
surface in #190595; the remaining steps are mostly context, inherited
machinery, or verification.

## One example for the entire walkthrough

```text
B = 32,  D = 1024,  G = 16

outer reduction (rms-norm mean):  scheduler group (32, 1024)
grouped amax:                     scheduler group (2048, 16)   # (B*D/G, G)

grouped coordinates (outer, groups, local) = (32, 64, 16):

  REDUCED                (32, 64)       numel  2048   # the scale
  LOCAL_REDUCTION_INPUT  (32, 64, 16)   numel 32768   # feeds the grouped body
  PARENT_FULL            (32, 64, 16)   numel 32768   # normalized y
  SUB_PARENT             (32, 64, 8)    numel 16384   # pair epilogue (factor 2)

G = 2 hazard: SUB_PARENT and REDUCED numels coincide (both B*D/2).
```

And the eager-broadcast example, for phase 2:

```text
amax[32,64] -> /6 -> clamp -> to(fp8)        # one logical scale, inlined twice
   -> swizzled scale output    (REDUCED domain, group width)
   -> .float() -> reciprocal -> packing
                               (SUB_PARENT domain, lane width)

#190595 eagerly broadcasts the named group value before replaying the packing
consumer. That is the complete behavior being reviewed here.
```

## The complete path

```text
dependent reduction pair
  -> is_candidate / plan                     scheduler.py, pair admission
  -> PointwiseDomainContext.create           the four domains
  -> _classify_nested_pointwise_nodes        one domain per member
  -> _plan_nested_sub_parent_stage           sources: split | broadcast | ordinary
  -> FusedNestedReductions identity
  -> append fusion (can_fuse_with -> ordinary gates -> fuse_with)
  -> Scheduler.merge_loops
  -> codegen_staged_reduction -> plan_from_topology
  -> _codegen_nested_reduction               outer -> grouped -> reduced/full -> SUB_PARENT
  -> eager group-to-lane materialization      reshape+broadcast before epilogue replay
  -> emit_broadcast_via_reshape              triton.py
```

## Phase 0: see the graph before reading the machinery

- [ ] **Step 1 - Read one positive test.**

  [test_nested_reduction.py:804](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:804),
  `test_producer_consumer_rmsnorm_interleaved_pair_epilogue`.

  Follow the outer rms-norm reduction over D, the grouped amax over G, the
  scale math, and the two interleaved pair reads.

  **Checkpoint:** The two reductions have scheduler groups `(32, 1024)` and
  `(2048, 16)`. Why can generic SIMD scheduling never place both in one
  kernel, and what does #190594's machinery contribute that this PR builds on?

- [ ] **Step 2 - Read the domain table in the guide.**

  Read `## What this commit adds` in
  [pr190595_nested_interleaved.md](/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md).

  **Checkpoint:** For the example, write the grouped-coordinate shape and
  numel of all four domains without looking at the table above.

## Phase 1: pair admission and domain classification (scheduler.py)

- [ ] **Step 3 - Read `_is_enabled_for` and `is_candidate`.**

  [scheduler.py:559](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:559)
  and
  [scheduler.py:572](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:572).

  **PR-local focus:** `_is_enabled_for` at lines 562-568 changes by removing
  the C++ wrapper exclusion and documenting strict-reduction support. The
  `is_candidate` body is inherited context.

  **Checkpoint:** Why are strict reductions excluded here (what property of a
  planned strict `R0_BLOCK` breaks the grouped layout), and why must the cheap
  candidate filter run before any planning?

- [ ] **Step 4 - Skim the inherited `NestedReduction.plan` admission funnel.**

  [scheduler.py:1357](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1357).
  **PR-local focus:** context only. Apart from removing the stale C++ wrapper
  TODO at the entry, #190595 does not change this admission funnel.

  This method was introduced as a plan-producing refactor in #190594; its
  reduction-pair checks originated in the pre-stack `NestedReduction.can_fuse`.
  #190595 does not add the dependency, extent, group-size, grouped-axis, or
  profitability checks here. Skim them to establish the approved topology,
  then follow the final call into `plan_from_topology`, where this PR adds the
  sub-parent domain and stage.

  **Checkpoint:** Which stable topology facts does `plan` pass across the
  phase boundary (the two candidate groups, reduction node, group size, and
  grouped axis), and which mutable domains are deliberately left for
  `plan_from_topology` to rebuild?

- [ ] **Step 5 - Read `PointwiseDomainContext.create`.**

  [scheduler.py:614](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:614)
  (class) and
  [scheduler.py:623](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:623).

  **PR-local focus:** lines 586-652 add `SUB_PARENT`, its fixed factor, and
  the factory that preserves grouped coordinates.

  The sub-parent domain keeps grouped coordinates `(outer, groups, local/2)`
  rather than flattening to `(parent_numel, parent_rnumel/2)`.

  **Checkpoint:** Construct a reshape whose numel matches SUB_PARENT but whose
  elements cross a group boundary, and show which representation rejects it.

- [ ] **Step 6 - Read `_classify_nested_pointwise_nodes`.** **HIGH PRIORITY**

  Start with the inherited wrapper at
  [scheduler.py:1042](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1042),
  then review `_classify_grouped_pointwise_nodes` at
  [scheduler.py:1081](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1081).

  **PR-local focus:** lines 1095-1165 add source-aware SUB_PARENT
  classification and the G=2 disambiguation. The outer-node wrapper is mostly
  inherited context.

  Follow the priority among the four domains, then the G=2 tiebreak: a node
  compatible with both REDUCED and SUB_PARENT is SUB_PARENT only if it reads
  the grouped reduction's *source*.

  **Checkpoint:** The source-name set is built from
  `grouped_reduction.read_writes.reads`. Why would `used_buffer_names()` make
  the tiebreak vacuous? (Hint: what else does that set contain, and who reads
  it?)

- [ ] **Step 7 - Read `_plan_nested_sub_parent_stage`.** **HIGH PRIORITY**

  [scheduler.py:1208](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1208).

  **PR-local focus:** lines 1208-1291 are the new nested sub-parent stage
  planner.

  Separate every sub-parent read into its three categories:

  1. parent-resolution sources -- #190594's lane proof, including values
     *written by pointwise nodes inside the group*;
  2. `broadcast_source_names` -- group-constant values (the scale) broadcast
     into the lane domain;
  3. unrelated inputs -- ordinary derived-index loads.

  **Checkpoint:** In the example, classify `y` (normalized input), the scale,
  and an external lane-shaped tensor `z`. Then: why must dependency names stay
  node-local (`read_writes`) instead of applying the scheduler-global
  `mutation_renames` map -- what goes wrong if a source is in-place mutated
  *later* in the graph?

## Phase 1b: append fusion must not bypass the scheduler

- [ ] **Step 8 - Read `FusedNestedReductions.can_fuse_with` and the scheduler
  re-entry.** **HIGH PRIORITY**

  [scheduler.py:3737](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3737),
  then `_can_fuse_nested_reduction_append` at
  [scheduler.py:8681](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:8681).

  **PR-local focus:** scheduler.py:3737-3804 adds append discovery and full-plan
  validation; 7592-7599 normalizes reverse discovery; 8681-8722 re-enters
  normal fusion legality; 8757-8770 performs typed dispatch after placement checks.

  The helper re-enters ordinary `_can_fuse` producer-first, so device, stream,
  mempool, `no_fuse_buffer_names`, resource, and vertical legality checks all
  still run. The dependency relaxation covers proven reads from *both* the
  grouped and outer stages.

  **Checkpoint:** Why must reverse discovery be normalized producer-first
  before stream/mempool bookkeeping, and what would output-order dependence
  look like as a user-visible symptom? (The regression test is
  `test_sub_parent_fusion_is_independent_of_nested_append_order`,
  [test_nested_reduction.py:841](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:841).)

- [ ] **Step 9 - Read `_plan_append` and `fuse_with`.**

  [scheduler.py:3779](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3779)
  and
  [scheduler.py:3795](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3795).

  **PR-local focus:** lines 3737-3804 are new or materially changed in this PR.

  Both `can_fuse_with` and `fuse_with` call the same full `_plan_append`
  predicate. This preserves the straightforward #190594 flow while ensuring
  fusion approval and construction validate the same prospective topology.
  The internally created grouped node inherits its mempool explicitly because
  this path bypasses `Scheduler.fuse_two_nodes`.

  **Checkpoint:** Name the scheduler bookkeeping `fuse_two_nodes` normally
  performs, and say for each item why this path either replicates it or does
  not need it. (Unit test: `test_nested_reduction_fuse_with_propagates_mempool`,
  [test_inductor_scheduler.py:172](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_inductor_scheduler.py:172).)

## Phase boundary: topology crosses, domains rebuild

- [ ] **Step 10 - Read the types.**

  `NestedReductionStage`
  [scheduler.py:1660](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1660),
  `SubParentEpilogueStage`
  [scheduler.py:1671](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1671),
  `StagedReductionPlan`
  [scheduler.py:1682](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1682),
  `FusedNestedReductions`
  [scheduler.py:3706](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3706).

  **PR-local focus:** `SubParentEpilogueStage.broadcast_source_names` at 1678,
  support for a plan containing both nested and sub-parent stages at
  1691-1695, and the stable nested-topology fields retained by
  `FusedNestedReductions` at 3713-3735. There is no mutable append-approval
  state; `_plan_append` rebuilds the prospective plan when needed. The base
  staged types are inherited from #190594.

  **Checkpoint:** Which fields are stable topology (safe to carry across
  `merge_loops`) and which are mutable emission detail that must be rebuilt?

- [ ] **Step 11 - Read `plan_from_topology`.** **HIGH PRIORITY**

  [scheduler.py:1457](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1457).
  Contrast with `plan` (step 4).

  **PR-local focus:** lines 1476-1536 create the domain context, plan the
  optional sub-parent stage, and remove those nodes from the ordinary grouped
  stage.

  **Checkpoint:** Why is re-running full nested admission at codegen time
  *incorrect* (not merely wasteful)? What can grouped-axis discovery no longer
  recover after `merge_loops`?

## Phase 2: codegen (simd.py, triton.py)

- [ ] **Step 12 - Read staged dispatch and the nested branch.**

  `codegen_staged_reduction`
  [simd.py:3089](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:3089)
  -> `_codegen_nested_reduction`
  [simd.py:3116](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:3116).

  **PR-local focus:** simd.py:3089-3437 changes staged dispatch and extends
  nested emission through the sub-parent stage. Read the unchanged outer
  reduction mechanics only as context.

  Emission order: outer reduction, grouped reduction, reduced/full pointwise,
  SUB_PARENT epilogue -- all in one kernel context; kernel-local buffer
  removal happens only after every stage emits.

  **Checkpoint:** Which schedule owns tiling and the grid, and which larger
  schedule must index-width analysis see? What breaks if buffer removal runs
  between stages?

- [ ] **Step 13 - Read the grouped stage's reshape-and-reduce.**

  `_GroupedReductionOpsHandler`
  [simd.py:2178](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2178)
  and the named constants at `_grouped_axis_named_constants`
  [simd.py:1759](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:1759).

  **PR-local focus:** named-constant emission at 1759-1775 and the grouped
  handler changes at 2178-2237. The underlying reshape-and-reduce mechanism
  predates this PR.

  In the generated kernel this is
  `tl.reshape(v, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])`
  followed by a reduce over axis 2 -- a register re-view, no data movement.

  **Checkpoint:** Why is the same line correct in both the persistent kernel
  (whole row resident) and each iteration of a looped kernel, and what
  guarantees a group never straddles a loop iteration?

- [ ] **Step 14 - Read source materialization for the nested stage.**

  `_SubParentSourceLoadResolver`
  [simd.py:2333](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2333)
  (now with `broadcast_source_names`), `make_sub_parent_family`
  [simd.py:1945](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:1945),
  and `_codegen_remapped_pointwise`
  [simd.py:3570](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:3570).

  **PR-local focus:** changes around 1945-2207 add grouped broadcasts;
  2333-2412 extends the resolver; 3570-3604 threads the new behavior through
  the shared remapped emitter.

  On the first sub-parent load, the resolver materializes the requested source
  if its producer value is still live. Parent-resolution values become lane
  tuples. Group/reduced values are reshaped and broadcast to lane width.
  Unrelated inputs remain ordinary derived-index loads.

  **Checkpoint:** For the example's epilogue, which value is register-split,
  which is eagerly broadcast, and which would be an ordinary derived-index
  load? What single condition (not the persistent/looped label) decides
  whether a recorded value is forwarded at all?

- [ ] **Step 15 - Confirm the eager-broadcast boundary.**

  This is a short boundary check, not another subsystem to review. Read only:

  1. `_SubParentSourceLoadResolver.materialize`
     ([simd.py:2369](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2369)):
     it finds the still-live source value, derives `must_forward` from
     `kernel.store_buffer_names`, and delegates materialization;
  2. `materialize_value_at_sub_parent_resolution`
     ([simd.py:2041](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2041)):
     a parent-width value is split, while a reduced/group-width value takes
     `_broadcast_value_to_axis_resolution`;
  3. `_PointwiseRemapHandler.load`
     ([simd.py:2304](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2304)):
     the first actual epilogue load asks the resolver for the value.
  4. `_DerivedIterationFamily.set_value_masks`
     ([simd.py:1617](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:1617)):
     split, broadcast, and already-child-width values receive the target
     derived-domain masks when they are materialized.

  Stop there. Materialization is on demand, but a group value is still
  broadcast eagerly to lane width when first used.

  **Checkpoint:** Why must a name in `kernel.store_buffer_names` forward or
  fail, while an external input may safely fall back to a normal load? Why is
  the derived tail mask attached here rather than repaired at an eventual
  indirect-indexing consumer?

- [ ] **Step 16 - Read the broadcast emitter and named-constant dedup.**

  `emit_broadcast_via_reshape`
  [triton.py:6286](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/triton.py:6286);
  `TritonKernel._codegen_named_constant` and where `_named_constant_defs`
  splices into the kernel prologue (search both names in
  [triton.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/triton.py)).

  **PR-local focus:** triton.py:6286-6314 adds the broadcast emitter changes;
  7927-7960 implements function-scope named-constant deduplication.

  **Checkpoint:** The reduced-output and sub-parent families both request
  grouped-axis constants. Why must definitions live at kernel-function scope,
  and what happens on a conflicting redefinition?

- [ ] **Step 17 - Read the dynamic-shape paths.**

  Symbolic group handling in `_grouped_axis_named_constants`
  ([simd.py:1791](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:1791))
  and the divisibility fallback in `make_sub_parent_family`
  ([simd.py:1945](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:1945)).

  **PR-local focus:** the symbolic named-constant path at 1791-1807 and the
  sub-parent family changes beginning at 1945.

  **Checkpoint:** What proof admits a dynamic reduction extent (the dynamic
  test runs `D=4096` and `D=4608` through one graph), and why is
  power-of-two *not* required on this interleaved path?

## Tests to close the loop

- [ ] **Step 18 - The classification and parent-schedule tests.**

  `test_multiple_parent_reductions_block_amax`
  ([test_nested_reduction.py:271](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:271))
  returns both parent-reduction results and the downstream block-local result.
  Its one-kernel assertion verifies that nested codegen preserves the complete
  parent schedule rather than selecting a single RMSNorm-like reduction.

  `test_producer_consumer_rmsnorm_interleaved_pair_epilogue`
  ([test_nested_reduction.py:804](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:804);
  its G=2 parametrization makes misclassification fail *numerically*) and
  `test_nested_reduction_reduced_only_consumer_group_size_two`
  ([test_nested_reduction.py:825](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:825);
  a reduced-only consumer at G=2 must not be dragged into the sub-parent
  stage). Domain-context unit test:
  `test_nested_reduction_sub_parent_domain_preserves_group_axis`
  ([test_inductor_scheduler.py:396](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_inductor_scheduler.py:396)).

  **PR-local focus:** all four named tests are added by #190595.

  **Checkpoint:** Connect each to the exact predicate from steps 5-6.

- [ ] **Step 19 - The eager materialization tests.**

  Read `test_nvfp4_inline_asm_kernel_form`
  ([test_nested_reduction.py:2772](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:2772)),
  `test_mxfp4_inline_asm_kernel_form`
  ([test_nested_reduction.py:2794](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:2794)),
  and `test_rmsnorm_block_scale_swizzle_kernel_form`
  ([test_nested_reduction.py:2757](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:2757)).

  **PR-local focus:** these tests pin the fused nested kernel, eager
  broadcast/split operations, load/store counts, and the independent blocked
  scale-layout store.

  **Checkpoint:** Which checks prove that eager group values are broadcast and
  parent values are split without an extra kernel?

- [ ] **Step 20 - The gate and source-category tests.**

  `test_sub_parent_append_respects_fusion_gate`
  ([test_nested_reduction.py:898](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:898)),
  `test_dynamic_sub_parent_epilogue`
  ([test_nested_reduction.py:860](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:860)),
  `test_producer_consumer_sub_parent_intermediate`
  ([test_nested_reduction.py:1147](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:1147))
  and its inlined-parent-full and independent-source siblings, and the
  `rejects_*` block (shifted, conflicting, mutated, grouped-axis-X, non-leaf).

  **PR-local focus:** these tests are part of the new block beginning near
  test_nested_reduction.py:792.

  **Checkpoint:** For each rejection, name the planner predicate from step 7
  that declines it, and confirm the fallback is two correct kernels.

- [ ] **Step 21 - The kernel-form tests.**

  `test_nvfp4_inline_asm_kernel_form`
  ([test_nested_reduction.py:2772](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py:2772))
  and the MXFP4 sibling: exact load/store counts, `tl.split` counts,
  `min_rblock`, packing instructions, in both forced-persistent and
  forced-looped classes.

  **PR-local focus:** the nested NVFP4/MXFP4 kernel-form tests and their exact
  load/store assertions are added or updated by #190595.

  **Checkpoint:** Which assertion would fire if `triton.multi_kernel`
  silently replaced the staged persistent form with its looped twin?

## Final check-off

- [ ] I can name the four pointwise domains and derive their grouped-coordinate
  shapes for `(B, D, G) = (32, 1024, 16)`.
- [ ] I can explain the G=2 ambiguity and why the tiebreak keys on the grouped
  reduction's *reads*.
- [ ] I can list the three source categories and their distinct codegen paths.
- [ ] I can explain why append fusion re-enters ordinary `_can_fuse` and what
  producer-first normalization protects.
- [ ] I can explain why topology crosses `merge_loops` but domains are rebuilt
  through `plan_from_topology`.
- [ ] I can point at the grouped reshape in a generated kernel and say why it
  is register-only.
- [ ] I can explain why #190595 eagerly broadcasts group values before the
  sub-parent body and why this is correct despite possible duplicated work.
- [ ] I can explain why CSE liveness, not the persistent/looped label, decides
  forwarding -- same rule as #190594, one level up.

## Later-stack boundary

#191775 replaces the stored factor-2 `sub_parent_domain` with per-candidate
rates (`_nested_sub_parent_rate`), widens interleaved factors to 4, and adds
the `(4, 3)` output rate. Review this commit at its factor-2 INTERLEAVED
boundary. The current landing scope stops after #191775.
