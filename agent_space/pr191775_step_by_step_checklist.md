# PR #191775 step-by-step reading checklist

> Historical monolithic checklist. The current uncommitted review proposal
> splits exact staged-dependency legality from the MXFP6 extension. Use
> [pr191775_split_step_by_step.md](/data/users/eellison/pytorch/agent_space/pr191775_split_step_by_step.md)
> for the active review order.

Full path:
`/data/users/eellison/pytorch/agent_space/pr191775_step_by_step_checklist.md`

This is the check-off companion to the overall guide:
[pr191775_mxfp6_staged_packing.md](/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_staged_packing.md).
It follows the planner-to-codegen flow rather than file order.

## Target and reading rule

- GitHub PR: [pytorch/pytorch#191775](https://github.com/pytorch/pytorch/pull/191775)
- Submitted base orig: `70d9ccee025` (#190595)
- Current ghstack orig commit: `b225e3b078b`
- Review worktree:
  `/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt`
- Run `git diff` in that worktree to inspect the local dependency-plan redesign.
- Run `git diff HEAD^` to inspect the complete submitted PR plus local changes.
- Exact saved diff:
  [pr191775_exact_dependency_plan.patch](/data/users/eellison/pytorch/agent_space/pr191775_exact_dependency_plan.patch)

Open the implementation with:

- [scheduler.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py)
- [simd.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py)
- [triton.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/triton.py)
- [common.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/common.py)
- [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py)
- [test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_inductor_scheduler.py)

Steps marked **HIGH PRIORITY** contain the main new correctness surface.

## One example for the walkthrough

```text
four logical 6-bit values:       v0 v1 v2 v3
three packed output bytes:       b0 b1 b2
input/output rate:               4:3

parent source tile:              [B, D]
sub-parent child coordinates:    [B, D/4]
emission passes:                 output lane 0, 1, 2

For each child coordinate, split the parent tile into four register lanes,
then replay the pointwise pack body three times to produce b0, b1, and b2.
```

# Phase 0: establish the workload

- [ ] **Step 1 - Read the MXFP6 helpers and positive test.**

  Start at
  [`_mxfp6_pack_four_to_three`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:118)
  and
  [`test_producer_consumer_mxfp6_four_to_three_pack`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:1764).

  **Checkpoint:** Identify the four source lanes, three output bytes, scale
  reduction, and final packed shape. Why is this not representable by
  #190595's fixed two-input/one-output rate?

- [ ] **Step 2 - Read the exact reference test.**

  [`test_producer_consumer_mxfp6_four_to_three_pack_exact`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:1811).

  **Checkpoint:** Which part of the test is exact integer packing, and why is
  this stronger than relying only on floating tolerances?

# Phase 1: enter through scheduler fusion

- [ ] **Step 3a - Enter the nested-append path.** **HIGH PRIORITY**

  Start at
  [`Scheduler._can_fuse_impl`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9402).
  When `node1` is already a `FusedNestedReductions`, it delegates to
  [`FusedNestedReductions.can_fuse_with`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:4024).

  **Inherited:** The special `FusedNestedReductions` append branch and the
  requirement that the new node depend on the grouped stage come from the
  earlier nested/sub-parent work. There is no new #191775 mechanism to review
  in this entry check.

  **Checkpoint:** Why is this an append into the existing grouped stage rather
  than construction of a new nested-reduction pair?

- [ ] **Step 3b - Build the prospective topology plan.** **HIGH PRIORITY**

  Read
  [`FusedNestedReductions._plan_append`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:4070),
  then follow it into
  [`NestedReduction.plan_from_topology`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1682)
  and
  [`_plan_nested_sub_parent_stage`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1416).

  `_plan_append` first creates the prospective grouped node, then asks the
  normal staged planner whether the complete topology is emit-able.

  **Inherited:** `_plan_append`, `plan_from_topology`, and rebuilding a plan
  from the known grouped axis and group size were introduced by #190594/#190595.
  `plan_from_topology` exists because fusion and `merge_loops` can rewrite the
  ranges after the topology was originally discovered.

  **New in submitted #191775:** The plan can describe the factor-4,
  three-output-lane MXFP6 stage. Review
  [`_sub_parent_epilogue_rate`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:892),
  [`_nested_sub_parent_rate`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:920),
  and
  [`_group_sub_parent_epilogue_nodes`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:870).

  **New in the local redesign:** `_plan_append` returns the complete
  `StagedReductionPlan`, not only its nested stage, so domain classification and
  exceptional dependency proof come from the same plan. The changed return and
  call sites are in
  [`_plan_append`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:4070)
  and
  [`can_fuse_with`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:4024).

  **Checkpoint:** What does this plan prove that ordinary producer/consumer
  dependency matching cannot prove by itself?

- [ ] **Step 3c - Classify only the nodes being appended.**

  Return to
  [`FusedNestedReductions.can_fuse_with`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:4024).
  It reads the appended nodes' domains from the complete plan, rejects
  `LOCAL_REDUCTION_INPUT`, and chooses whether legality must consider only the
  grouped stage or the whole existing nested node.

  **Inherited:** The supported-domain checks and the distinction between the
  grouped stage and whole nested producer already existed.

  **New in the local redesign:** `can_fuse_with` no longer separately calls the
  pointwise classifier before planning. It reads the classifications from the
  prospective plan built in Step 3b.

  **Checkpoint:** Why does a `SUB_PARENT` consumer use the whole nested node as
  its producer, while another grouped-stage consumer can use `node2`?

- [ ] **Step 3d - Prove every exceptional dependency.** **HIGH PRIORITY**

  Read
  [`Scheduler._can_fuse_nested_reduction_append`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9357)
  and
  [`Scheduler._plan_fusion_dependency_matches`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9158).

  Strict `MemoryDep` matching is tried first. Each remaining read of a producer
  output must pass the inherited reshape/broadcast proof or match an exact
  source/consumer relation from the staged plan. One uncovered read rejects the
  candidate.

  **Inherited:** The conservative normalized reshape/broadcast proof, producer
  density/injectivity checks, score bridge, and ordinary vertical legality came
  from the landed scheduler-equivalence work (#183432).

  **New in submitted #191775:** MXFP6 lane reads were admitted using names whose
  indices the staged plan had proved.

  **New in the local redesign:** The name permission is removed. The staged
  plan retains `ProjectedSourceAccess` records, and fusion derives exact
  `MemoryDepMatch(write, read)` pairs. All unmatched producer-output reads must
  be covered, and one producer write contributes to the score only once.

  **Checkpoint:** Why is an exact `(write, read)` relation safer than allowing
  every read with the same buffer name? Why do `StarDep` and `WeakDep` stay on
  ordinary vertical legality?

- [ ] **Step 3e - Replan instead of caching approval.** **HIGH PRIORITY**

  After legality succeeds, `can_fuse_with` calls `_plan_append` again. If
  [`Scheduler.fuse`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:11931)
  applies the fusion, `FusedNestedReductions.fuse_with` rebuilds it once more.
  Codegen later rebuilds the post-`merge_loops` plan independently.

  **Inherited:** `fuse_with` and codegen already rebuilt the plan rather than
  carrying a mutable approval object across phases.

  **New in the local redesign:** `can_fuse_with` also replans after ordinary
  legality, and nonempty exact dependency matches prevent later optional loop
  expansion/reordering/inversion from invalidating the proof.

  **Checkpoint:** Why are exact dependency records valid only for the current
  loop state? Why is rebuilding safer than carrying an approval object across
  scheduler phases?

# Phase 2: inspect the plan built by that path

- [ ] **Step 4 - Read `_sub_parent_epilogue_rate`.** **HIGH PRIORITY**

  [`scheduler.py:890`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:890).

  The helper first recognizes ordinary power-of-two `factor:1` shapes, then
  derives a rational shape ratio and accepts the one additional `(4, 3)` case.

  **Checkpoint:** For `node_numel = 3 * full_numel / 4`, show why the reduced
  rational is `(factor, output_lanes) = (4, 3)`. Why should an arbitrary
  rational rate still be rejected here?

- [ ] **Step 5 - Read `_nested_sub_parent_rate`.** **HIGH PRIORITY**

  [`scheduler.py:920`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:920).

  This replaces #190595's stored factor-2 sub-parent domain with a rate derived
  per candidate. It also reconstructs the exact grouped-coordinate shape.

  **Checkpoint:** Why must the grouped axis be R, why must `group_size` divide
  `factor`, and how does `expected_groups` reject a same-numel reshape that
  crosses group boundaries?

- [ ] **Step 6 - Read `_group_sub_parent_epilogue_nodes`.**

  [`scheduler.py:870`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:870).

  **Checkpoint:** Why must all candidates share one input factor, and why must
  output-lane groups already appear in strictly increasing order rather than
  being reordered by the planner?

- [ ] **Step 7 - Read the plan representation.**

  [`ProjectedSourceAccess`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1904)
  and
  [`SubParentEpilogueStage`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1927).

  **Checkpoint:** Explain the distinction among `factor`, `output_groups`,
  `source_projections`, `internal_dependency_names`, and
  `broadcast_source_names`.

# Phase 3: prove internal-source scheduling

- [ ] **Step 8 - Read `_sub_parent_internal_dependencies`.**

  [`scheduler.py:1193`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1193).

  **Checkpoint:** Why does an internal epilogue edge require exactly one
  normalized `MemoryDep` write? What ambiguity would multiple writers create?

- [ ] **Step 9 - Read `_order_sub_parent_parent_nodes`.** **HIGH PRIORITY**

  [`scheduler.py:746`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:746).

  This moves the full-resolution producer chain to the final reduction pass in
  a looped kernel. It declines if the chain contains a reduction, has an
  incompatible domain, or feeds a node that must remain earlier.

  **Checkpoint:** Construct the final order as `leading_nodes + deferred_nodes`.
  Why must `deferred_start` be strictly inside the sequence, and why would
  emitting the source before the final loop lose it from CSE?

- [ ] **Step 10 - Read the nested stage builder.**

  [`_plan_nested_sub_parent_stage`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1418).

  **Checkpoint:** Where are the rate, internal dependencies, output-read
  restriction, parent lane sources, and group-constant broadcast sources each
  proved? Which parts are inherited from #190595 versus generalized here?

# Phase 4: inspect the fusion-legality details

- [ ] **Step 11 - Read the exact producer/read relation.** **HIGH PRIORITY**

  [`ProjectedSourceAccess`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1904),
  [`MemoryDepMatch`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1918),
  [`_plan_fusion_dependency_matches`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9159),
  and
  [`_nested_fusion_dependency_matches`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9209).

  **Checkpoint:** Which residual reads use the staged plan's exact lane proof?
  Which use the inherited normalized reshape/broadcast proof? Why must every
  residual producer-output `MemoryDep` be covered, while `StarDep` and
  `WeakDep` remain on the ordinary path? Why are synchronization-requiring and
  non-injective writes excluded?

- [ ] **Step 12 - Read speculative consumer reindexing and rollback.**

  [`_reindex_sub_parent_consumer`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9257).

  **Checkpoint:** What exact producer/read/write triple permits the reindex?
  Where is loop state restored if the rebuilt plan rejects the candidate?

# Phase 5: follow the approved plan into codegen

- [ ] **Step 13 - Read the staged-codegen entry and generalized layout.** **HIGH PRIORITY**

  Begin at
  [`SIMDScheduling.codegen_staged_reduction`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:3185),
  which rebuilds the approved topology plan and dispatches a
  `FusedNestedReductions` node to `_codegen_nested_reduction`.

  [`make_sub_parent_family`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:1955),
  [`sub_parent_iteration_values`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:1991), and
  [`materialize_value_at_sub_parent_resolution`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:2059).

  **Checkpoint:** Follow `factor=4`, `output_lanes=3`, and each
  `output_lane=0/1/2`. Show how one child coordinate maps to the correct packed
  output coordinate and how a parent `[B,D]` value becomes four
  `[B,D/4]` register values.

- [ ] **Step 14 - Review the visible rebase integration fix.**

  [`set_value_masks`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:1620).

  The rebase integration delta wraps mask derivation in `ensure_active`. MXFP6 may
  materialize an internal source before entering the derived epilogue, so the
  target family's mask trees must be active while their names and shapes are
  derived.

  **Checkpoint:** Why is this a property of the mask helper rather than every
  caller? Why is the nested activation a no-op when the family is already
  active?

- [ ] **Step 15 - Read recursive split emission.**

  [`_emit_recursive_split`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/triton.py:6367)
  and
  [`emit_split_via_reshape`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/triton.py:6407).

  **Checkpoint:** Draw the binary split tree that turns four interleaved lanes
  into `v0, v1, v2, v3`. Why is float8 bitcast to uint8 before splitting and
  bitcast back afterward?

- [ ] **Step 16 - Read standalone emission.** **HIGH PRIORITY**

  [`_codegen_reduction_with_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:3682).

  **Checkpoint:** Follow the order: construct the parent schedule, create one
  kernel, capture source values, optionally delay the body flush for internal
  sources, materialize required internal values, emit three output-lane
  passes, then flush. Why may external values reload while internal values
  must forward or fail loudly?

- [ ] **Step 17 - Read nested emission.**

  [`simd.py:3388`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:3388).

  **Checkpoint:** Which #190595 stages remain unchanged? Where does #191775
  add the loop over `output_groups` and `output_lane`, and how are internal
  epilogue stores forwarded between those passes?

# Phase 6: verify failure modes and generated form

- [ ] **Step 18 - Read the rejection tests.**

  Start at
  [`test_producer_consumer_mxfp6_rejects_shifted_intermediate`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:1936)
  and continue through the source-before-reduction cases.

  **Checkpoint:** For each rejection, identify whether the failed proof is
  rate/domain compatibility, exact index equivalence, unique internal access,
  or safe looped ordering.

- [ ] **Step 19 - Read the kernel-form tests.**

  [`test_mxfp6_four_to_three_pack_kernel_form`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:3385)
  and
  [`test_mxfp6_internal_source_kernel_form`](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:3398).

  **Checkpoint:** Verify one staged kernel, three split operations, the
  expected store count, and the persistent-versus-looped input-load count.

## Current verification

```text
test_nested_reduction.py -k mxfp6: 30 passed, 3 skipped
test_nested_reduction.py -k indirect_index_mask: 6 passed
full rebased-stack test_nested_reduction.py: 441 passed, 12 skipped,
  2 known kernel_num_gb narrow-runner failures
with-proxy spin quicklint: clean
py_compile and git diff --check: clean
```

The reviewed state is amended and submitted to #191775.
