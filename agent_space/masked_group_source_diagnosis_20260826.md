# Masked group-source plan-loss diagnosis

Date: 2026-08-26

## Verdict

This is an inherited scheduler/codegen phase-consistency bug, not an F2 lazy-CSE
semantic regression. The topology is outside the current measured F2 target and
should conservatively decline staged fusion for now. The smallest fix is to
require every planned trailing-broadcast relation to remain valid after the
dependency normalization that `Scheduler.merge_loops()` performs.

The failure is reachable only with the experimental nested-reduction path
enabled, but it is still a real fail-to-fallback bug: a normal PyTorch graph is
accepted as `FusedStagedReduction`, then compilation aborts instead of emitting
the valid two-kernel fallback.

## First phase mismatch

The fusion-time plan sees unnormalized dependencies because
`loop_ordering_after_fusion=True`:

```text
op1 write: MemoryDep('buf1', 3*d0 + d1, {d0: 2, d1: 3})
op2 read:  MemoryDep('buf1', 3*d0 + Identity(d1), {d0: 2, d1: 3})
```

`_sub_parent_consumer_is_trailing_broadcast()` removes `Identity`, proves the
same `[2, 3]` prefix and address, and records `buf1` as a group-resolution
broadcast source. `_can_fuse()` proves the remaining dependencies and
`Scheduler.fuse()` creates a `FusedStagedReduction`.

After fusion, `Scheduler.merge_loops()` mutates each underlying scheduler node
and refreshes dependencies with normalization. The producer and consumer do not
normalize alike:

```text
op1 write: MemoryDep('buf1', c0, {c0: 6})
op2 read:  MemoryDep('buf1', 3*c0 + Identity(c1), {c0: 2, c1: 3})
```

The scale producer collapses `[2, 3]` to `[6]`. The padded consumer retains the
two axes because its masked index needs the group coordinate. Codegen then calls
`_find_sub_parent_epilogue_plan()` on these post-merge nodes. Candidate grouping
and the interleaved input relation still pass, but
`_sub_parent_broadcast_projections()` (F1/F2:
`_sub_parent_broadcast_access_relations()`) rejects the `6` versus `2` prefix.
`codegen_staged_reduction()` therefore raises:

```text
AssertionError: sub-parent reduction plan was lost before codegen
```

This is the first mismatch. F1 indexed forwarding and F2 lazy projection are
not reached.

Relevant frozen paths:

- fusion planning: `torch/_inductor/scheduler.py:9858-9917`
- broadcast relation proof: `torch/_inductor/scheduler.py:1343-1431`
- post-fusion normalization: `torch/_inductor/scheduler.py:6556-6572`
- codegen re-plan and assertion: `torch/_inductor/codegen/simd.py:3237-3247`

## Why codegen cannot simply fall back

The stable cross-phase marker is the `FusedStagedReduction` identity. Generic
SIMD codegen cannot emit its mixed parent and sub-parent domains, and the
intermediate scale buffer has already become internal to the fused scheduler
node. Splitting it back into two kernels during codegen would require reversing
scheduler fusion and memory planning. The safe place to decline is fusion time.

## Recommended fix

Keep the current raw-frame proof, then check that it survives the normalization
which will run after fusion:

```python
if config.loop_ordering_after_fusion:
    normalized_source = source.normalize()
    if any(
        not cls._sub_parent_consumer_is_trailing_broadcast(
            normalized_source, consumer.normalize()
        )
        for consumer in consumers
    ):
        return None
```

In F1/F2 this belongs in
`NestedReduction._sub_parent_broadcast_access_relations()`, immediately after
the existing raw relation proof. The same predicate can later be shared with
the cross-output-group trailing-broadcast path if desired, but that is not
required to close this reproducer.

This is fail-closed and does not add a new accepted index relation. The raw plan
now declines before `FusedStagedReduction` is created, so normal scheduling emits
the existing two-kernel implementation.

## Why not broaden the proof in this fix

A narrow support prototype removed semantically inert `Identity` wrappers,
reindexed the consumer into the flattened source frame with
`MemoryDep.normalize_with_ranges()`, and proved exact address equality. It
compiled this repro as one correct staged kernel on frozen #191775, F1, and F2,
in both persistent and looped modes.

That shows the topology is supportable, but accepting it changes the planner's
frame-equivalence contract. The current follow-up does not need masked
group-width execution, and the existing follow-up record already says that
nontrivial broadcast provenance should decline until normalization retains the
needed dimension information. Broadening this proof belongs with that dedicated
work, including transpose/regrouping adversaries. It should not be folded into
the lazy-CSE optimization merely because the generated kernel happens to be
correct for this example.

## Focused regression

Add one end-to-end test to `test/inductor/test_nested_reduction.py` using the
minimized `B=2, D=48, G=16` graph:

1. Reduce each 16-value group to `scale`.
2. Read `scale[:, :-1]`, pad the final group with scalar `7.0`.
3. Divide the even lanes by the padded scale.
4. Assert compiled numerics match eager.
5. Assert `metrics.codegen_nested_reduction == 0` and two generated kernels.

The shared test base runs it in both persistent and nonpersistent classes. The
old code raises before the assertions; the proposed guard passes both variants.

## Prototype validation

Scratch worktree only:
`/data/users/eellison/pytorch/agent_space/masked_group_source_diag_wt`

- Proposed early-decline guard, original standalone-sub-parent subset:
  `74 passed, 1 skipped`.
- New focused regression: `2 passed` (persistent and nonpersistent classes).
- Direct minimized repro with the guard:
  correct numerics, `2` kernels, `codegen_nested_reduction=0` in both modes.
- Exact-support experiment, not recommended for this fix:
  correct numerics, `1` kernel, `codegen_nested_reduction=1` on frozen, F1, and
  F2 in both modes.

No reviewed, F1, or F2 worktree was edited. The frozen worktree hashes remain:

```text
unstaged: 8a7f5eca32cfa865f4ea784b2b23858fd359bf7e66d72c2f5ea6b4e0c1e451c0
staged:   8b841b64605acd39e0cd3e3520b4842dccd197ded41db206b3437202bc29fb4f
```

## Standalone follow-up snapshot

The clean prototype remains uncommitted and unstaged in:

```text
/data/users/eellison/pytorch/agent_space/masked_group_source_diag_wt
```

Its delta relative to the frozen #191775 worktree is exactly:

```text
torch/_inductor/scheduler.py             | 10 ++++++++++
test/inductor/test_nested_reduction.py   | 15 +++++++++++++++
2 files changed, 25 insertions(+)
```

Combined normalized unified-diff SHA-256 (scheduler first, then test):

```text
89747fb4156cfa7ca403d5f9e758fb0a8ad342f5ad023e17e98e80c05f1bf1bb
```

Final changed-file SHA-256 values:

```text
3f209aa54a502b38d20761b14b73d1cdb165228599beec7567669d74d81244a5  torch/_inductor/scheduler.py
c20cf9988a9c31325351024fe0e7b95460511b35aaecb06665c1fc2df72ff4de  test/inductor/test_nested_reduction.py
```

Verification on that exact snapshot:

```text
spin quicklint:                                      pass
git diff --check:                                    pass
python -m py_compile scheduler.py + nested test:     pass
full test_inductor_scheduler.py:                     108 passed, 6 skipped
full test_nested_reduction.py:                       397 passed, 8 skipped
standalone_sub_parent subset:                         74 passed, 1 skipped
focused fallback regression:                          2 passed
```

The full-file commands used the editable-source redirect runner at
`agent_space/run_test_from_worktree_20260826.py` so imports came from the
scratch worktree rather than the installed editable checkout.
