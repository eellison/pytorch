# F1/F2 review handoff

Date: 2026-08-27

No commit, amend, ghstack submission, or GitHub update has been made.

## Stack order

1. Reviewed #191775 layered base, now including the conservative masked
   group-source fallback.
2. F1 exact indexed forwarding.
3. F2a narrow cast-before-broadcast CSE optimization.

The parent divide-before-split prototype is rejected and is not part of this
review stack. The older parent-to-grouped Phase 4 is also not included; its
generic normalization proof remains a separately scoped follow-up.

## Base delta

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

Review only:

- `torch/_inductor/scheduler.py:1411`: require one unambiguous writer, then
  prove each consumer in both the raw frame and the dependency-normalized frame
  that `merge_loops()` will produce.
- `test/inductor/test_nested_reduction.py:2252`: a masked/padded group source
  now declines staged fusion and emits the valid two-kernel fallback.

This is +10 production lines and +15 test lines. It does not change any working
one-kernel target case.

## F1: exact indexed forwarding

Worktree:
`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

Full step-by-step review guide:
`/data/users/eellison/pytorch/agent_space/f1_detailed_review_guide_20260827.md`

F1 is committed locally as `fc09fb38287` on top of the local review-base
snapshot `d4167dba5b8`. The reviewed masked-load ownership simplification is
folded into that commit, and the worktree is clean. Use `git show HEAD` for the
complete F1 change. It has not been submitted.

Recommended review order:

1. `torch/_inductor/scheduler.py:2095` - `SubParentAccessRelation`, the one
   exact per-consumer planner/codegen record.
2. `torch/_inductor/scheduler.py:1094`, `1266`, and `1373` - relation
   construction for parent, internal, and reduced/broadcast sources.
3. `torch/_inductor/scheduler.py:9387` - fusion requires exact relation
   membership for sub-parent reads.
4. `torch/_inductor/codegen/simd.py:1527` - reconstruct the current logical
   memory access, including its temporal name and node-local index frame.
5. `torch/_inductor/codegen/simd.py:2365`, `2407`, and `2492` - record exact
   unguarded sources, resolve the exact consumer read, leave concrete masked
   loads to ordinary Triton codegen, reload external misses, and fail loudly for
   required in-kernel misses.
6. `test/inductor/test_inductor_scheduler.py:518` - the replacement ownership
   test for masked loads.

F1 deletes the projection layout enum, grouped projection records, name-based
codegen views, forwarding-name sets, and the remapped-value map. The current
review state is net +58 production lines; removing `_AccessGuard`, its keyed
caches, and resolver-owned fill synthesis reduced the committed F1 by 88 net
production lines.

Masked loads now follow the existing tree contract. The resolver records only
unguarded sources. It forwards when `_load_other is None`, including the
`TritonKernelOverrides.masked()` outer-`where` path used for in-kernel values.
When `_load_other` is concrete, an external read takes the normal physical-load
fallback so Triton remains responsible for the mask and fill.

This simplification was measured before it was applied. Across the complete
nested suite, no successful forward used a guarded source and no required
relation had a concrete `_load_other`. The conservative rule passes all 399
nested tests and preserves all ten protected kernel hashes exactly. See
`agent_space/f1_access_guard_reachability_20260827.md`.

## F2a: cast before broadcast

Worktree:
`/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

The F2a worktree still sits on the committed pre-simplification F1 snapshot and
must be rebased over the current F1 worktree before its final review. Its
algebraic change and measured result are unchanged; the stale guard fields in
that worktree should disappear mechanically during the rebase.

Recommended review order:

1. The resolver's one narrow raw group-width exit after exact F1 lookup and CSE
   liveness. After rebase, it should consult the active `_load_other` ownership
   rule rather than a stored consumer guard.
2. `torch/_inductor/codegen/simd.py:2415` - `to_dtype` casts at group width and
   immediately broadcasts the result.
3. `torch/_inductor/codegen/simd.py:2401`, `2409`, and `2431` - every other
   operation and every store materializes eagerly.

The only algebraic change is:

```text
cast(broadcast(scale)) -> broadcast(cast(scale))
```

Production delta is net +67 lines. The 12-row NVFP4 matrix improves by 1.301x
geometric mean; MXFP4 and protected MXFP6 forms remain unchanged.

The existing policy unit test also directly verifies that a group-width value
is materialized before `store`; this adds five test-only lines and no production
code.

## Verification

Full-package runner with FX and Inductor caches disabled:

- Base: scheduler 108 passed / 6 skipped; nested 397 passed / 8 skipped.
- F1: scheduler 130 passed / 6 skipped; nested 399 passed / 8 skipped.
- F2a: scheduler 132 passed / 6 skipped; nested 399 passed / 8 skipped.
- Final F1 masked-load simplification: scheduler 118 passed / 6 skipped;
  focused `sub_parent` selection 18 passed; additional focused
  ownership/integration selection 4 passed. The nested suite ran 399 tests
  successfully with 8 skipped. An earlier AOT checksum error was traced to a
  broken system `openssl` selected under the Conda library path.
- Focused cat, mismatched-mask, and persistent/looped MXFP6 tests pass under the
  simplified ownership policy.
- Focused masked-source regression passes in persistent and looped modes at
  every layer and emits two kernels rather than raising.
- `spin quicklint` passes on the final F2a worktree.
- `git diff --check` passed before the final amend.
- Final protected-kernel re-attestation passes: F1 matches 10/10 published
  hashes after the masked-load simplification, while the pre-rebase F2a changes
  only the 12 intended NVFP4 forms. See
  `agent_space/f1_f2a_protected_kernel_reattest_20260827.md`.

Detailed references:

- `agent_space/indexed_forwarding_rebase_20260826.md`
- `agent_space/f1_access_guard_reachability_20260827.md`
- `agent_space/indexed_forwarding_complexity_audit_20260826.md`
- `agent_space/lazy_cse_narrow_cast_review_20260826.md`
- `agent_space/lazy_cse_f2a_results_20260826.md`
- `agent_space/lazy_cse_complexity_audit_20260826.md`
