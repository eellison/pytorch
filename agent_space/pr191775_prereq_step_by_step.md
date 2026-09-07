# Exact staged-dependency prerequisite: step-by-step review

Full path:
`/data/users/eellison/pytorch/agent_space/pr191775_prereq_step_by_step.md`

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt`

This is the first review split. It changes fusion legality for the existing
factor-2 sub-parent path; it does not add MXFP6 codegen.

To inspect only this split:

```bash
cd /data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt
git diff --stat
git diff
```

The same diff is staged in the layered worktree, so this also works:

```bash
cd /data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt
git diff --cached
```

Steps marked **HIGH PRIORITY** contain the main correctness contract.

## Step 1 - Establish the old and new contracts

Read `SubParentSourceLayout` and the new `ProjectedSourceAccess` record:

- [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:586)
- [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1770)

Previously the plan retained only a buffer name and layout. It now retains the
raw source and consumer `MemoryDep`s which the planner actually proved.

Checkpoint: the record is evidence for fusion legality; current codegen still
uses compatibility views derived from it.

## Step 2 - Follow the existing interleaved proof

Read:
[NestedReduction._try_get_sub_parent_source_projections](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:874).

This is #190595's parent-to-lane proof, changed to retain exact accesses rather
than returning only `(name, layout)`.

Checkpoint: for every consumer, locate the explicit
`parent_r = factor * child_r + lane` comparison. No generic normalization is
allowed to substitute for this proof.

Fusion legality also checks the raw access frame in
[`_sub_parent_access_preserves_x_boundary`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:850).
An access using multiple loop variables must retain a whole-axis `X | R`
boundary. A one-active-variable access is allowed because its other loop axes
are broadcast-invariant; indirect accesses always reject.

## Step 3 - Review reduced-value broadcast proof **HIGH PRIORITY**

Read:
[NestedReduction._sub_parent_broadcast_projections](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1061).

The retained source axes must match the consumer prefix in the same order and
with the same extents. Dependency extraction already removes unused trailing
broadcast axes. Any extra axis still present therefore affects the address and
rejects. Same-name `StarDep` or `WeakDep` reads remain outside this proof.

Concrete distinction:

```text
source:       [2, 4],    index 4*s0 + s1
valid child:  [2, 4, 2], index 4*d0 + d1
invalid:      [4, 2, 2], index 2*d0 + d1
```

With raw dependencies, the invalid form uses a different logical frame. If
dependency normalization has already merged both forms to one flat axis, the
proof intentionally accepts their identical row-major address mapping.

## Step 4 - Review the stage compatibility views

Read:
[SubParentEpilogueStage](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1803).

`source_layouts` and `broadcast_source_names` are derived from the exact
records. They are temporary adapters for the current name-based codegen, not
independent planner state.

## Step 5 - Review append replanning

Read `FusedNestedReductions._plan_fusion_with`, `_plan_append`, and `fuse_with`:
[scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:3892).

The legality check plans the complete prospective topology, verifies every
appended node received exactly one domain, and selects the stage that produces
its dependencies. `_can_fuse` recognizes `FusedNestedReductions` directly and
runs that locally derived plan through the same legality path used by initial
fusion. `fuse_with` then replans from the current nodes rather than consuming
cached pre-fusion state.

Checkpoint: no plan survives across mutable fusion or loop-merging phases.

## Step 6 - Review residual dependency coverage **HIGH PRIORITY**

Read:
[Scheduler._prove_staged_fusion_dependencies](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9021).

For each unmet consumer `MemoryDep` whose name is a producer output:

1. ordinary strict matching passes normally;
2. a sub-parent read must be an exact pair recorded by the plan;
3. only a read owned exclusively by the inherited grouped stage may use the
   pre-existing nested equivalence proof;
4. ambiguous writes, temporary indices, and synchronized writes reject.

Before checking residual producer outputs, this method validates raw
`INTERLEAVED` access frames. Nested consumers use the grouped child frame, not
the outer tensor frame; this is what rejects a regrouped/transposed consumer
without rejecting broadcast-invariant weights or flat post-merge accesses.

Checkpoint: sub-parent ownership is checked before grouped-stage ownership, so
equal `MemoryDep` values cannot escape through the legacy branch.

## Step 7 - Review staged-plan recognition **HIGH PRIORITY**

Read the staged-plan block near the end of:
[Scheduler._can_fuse](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9216).

The implementation keeps the three semantic outcomes local rather than
returning a tri-state object:

```text
unrelated pair                 no plan; continue through ordinary fusion
valid staged pair              retain StagedReductionPlan and prove its deps
recognizable invalid candidate reject before ordinary rewrites or legality
```

The last branch is load-bearing. Allowing a recognizable transposed candidate
to fall through to ordinary fusion previously produced a wrong fused kernel.

## Step 8 - Review vertical legality and loop-mutation suppression

Continue in:
[Scheduler._can_fuse](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9216).

`plan is not None` suppresses optional dimension expansion, loop
reordering, and inversion. The dependency proof may return an empty match tuple;
the plan still exists and must still freeze those rewrites.

Then review the explicit ordinary/staged choice and exact-pair removal in
`_can_fuse_vertical_impl` immediately below it. The ordinary
`can_fuse_vertical` API remains unchanged.

## Step 9 - Review scoring

Read:
[Scheduler._score_staged_fusion_memory_for_can_fuse](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9970).

`_score_staged_fusion_memory_for_can_fuse` handles normalized strict reuse and
plan-approved reuse without adding parameters to ordinary scoring APIs. It
scores at most once per producer write and skips a raw exact dependency already
counted by ordinary scoring. An empty staged tuple still invokes it.

## Step 10 - Review tests

Scheduler contract tests start at:
[test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_inductor_scheduler.py:379).

Pay particular attention to:

- broadcast frame and normalization behavior;
- non-Memory, multiwrite, TMP, and atomic rejection;
- sub-parent versus grouped-stage ownership;
- empty-plan rewrite suppression;
- complete append-domain assignment.

End-to-end shifted/transposed regressions start at:
[test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_nested_reduction.py:1186).

Expected result: existing NVFP4/factor-2 kernels remain fused, while regrouped
or shifted relations decline rather than being authorized by a buffer name.
