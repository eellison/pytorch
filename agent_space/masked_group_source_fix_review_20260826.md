# Masked group-source early-fallback review

Date: 2026-08-27

Reviewed worktree:
`/data/users/eellison/pytorch/agent_space/masked_group_source_diag_wt`

Baseline:
`/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

## Findings

No blocking findings.

## Verdict

**Approve.** The 25-line change is a narrow fail-closed fix for the observed
fusion/codegen phase mismatch. It adds no accepted dependency relation, does
not alter codegen, and has a positive end-to-end fallback oracle.

## Correctness review

The existing raw-frame proof remains the first requirement at
`torch/_inductor/scheduler.py:1419`. The new check at line 1424 only rejects a
relation that stops satisfying the same trailing-broadcast predicate after
normalization. It therefore cannot broaden fusion or hide an invalid raw X/R
relationship.

The configuration gate is correctly scoped. GPU scheduler nodes retain raw
dependencies initially only when `loop_ordering_after_fusion` is enabled
(`SchedulerNode._compute_attrs`, line 3473), and `Scheduler.merge_loops()` then
refreshes those dependencies with normalization after fusion (lines 6566-6582).
When the option is disabled, dependencies were normalized during initial
construction and there is no later merge phase to predict, so repeating the
check would only reject more candidates without protecting a phase boundary.

`MemoryDep.normalize()` uses the same `_RecordLoadStoreInner._normalize`
canonicalization used by normalized dependency extraction. Staged candidates
also skip the generic reindex/reorder attempts after their plan is built. Thus
checking both source and consumer normal forms at relation construction is the
right pre-fusion approximation of the accesses from which codegen will rebuild
the plan.

The check is placed in `_sub_parent_broadcast_projections`, before the
projection is recorded. Both standalone and nested plans use this helper, and
both codegen paths rebuild their plan after scheduler loop merging, so applying
the invariant at the shared relation boundary is narrower and safer than a
special case in `_can_fuse` or codegen. A failed plan remains a hard rejection
for a recognizable staged candidate, allowing ordinary scheduling to emit the
two-kernel fallback rather than creating a `FusedStagedReduction` that codegen
cannot lower.

The new condition is limited to group-resolution broadcast sources. It does not
change interleaved source planning, internal forwarding, generic dependency
matching, or behavior when loop ordering after fusion is disabled. The full
nested-reduction suite retained every existing fusion assertion.

## Regression oracle

`test_standalone_sub_parent_masked_group_source_falls_back` checks all three
observable requirements:

1. Compiled output matches eager for both returned tensors.
2. `metrics.codegen_nested_reduction == 0`, proving the unstable staged plan was
   declined.
3. `metrics.generated_kernel_count == 2`, proving compilation reached the
   intended ordinary fallback rather than merely avoiding the staged metric.

The shared test base runs the case in persistent and nonpersistent modes. On the
frozen baseline, the same graph reaches
`AssertionError: sub-parent reduction plan was lost before codegen`; on the
fixed snapshot, both variants pass with two kernels.

## Exact scope

Only two tracked files differ from the frozen worktree:

```text
torch/_inductor/scheduler.py             | 10 ++++++++++
test/inductor/test_nested_reduction.py   | 15 +++++++++++++++
2 files changed, 25 insertions(+)
```

Normalized unified-diff SHA-256, scheduler then test:

```text
89747fb4156cfa7ca403d5f9e758fb0a8ad342f5ad023e17e98e80c05f1bf1bb
```

Changed-file SHA-256 values:

```text
3f209aa54a502b38d20761b14b73d1cdb165228599beec7567669d74d81244a5  torch/_inductor/scheduler.py
c20cf9988a9c31325351024fe0e7b95460511b35aaecb06665c1fc2df72ff4de  test/inductor/test_nested_reduction.py
```

## Reviewer-run validation

```text
focused fallback regression: 2 passed
full test_inductor_scheduler.py: 108 passed, 6 skipped
full test_nested_reduction.py: 397 passed, 8 skipped
spin quicklint: pass
python -m py_compile: pass
git diff --check: pass
frozen direct reproducer: fails at the expected lost-plan assertion
```
