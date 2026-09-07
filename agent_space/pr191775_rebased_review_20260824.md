# #191775 rebased review (2026-08-24)

Verdict: **approve**. The rebase resolutions are clean, the one P0-shaped
pattern was checked explicitly and is correctly filtered, the four-line
integration fix is correct (with one comment requested), and both suites pass
on an independent rerun.

## Target and state verification

- Worktree `agent_space/pr191775_review_wt` at `85d18ee9e06`, parent
  `a2f664ecf57` (the reviewed #190595 with lazy materialization + mask
  propagation amended in).
- Uncommitted four-line integration fix on disk; its `git diff` sha256
  matches the documented `pr191775_rebase_integration_fix.patch`
  (`ec56a38ca142...`). The full saved diff
  `pr191775_rebased_review.patch` matches its documented sha256
  (`5b4ce727a512...`).
- Guide (`pr191775_mxfp6_staged_packing.md`) and step-by-step checklist both
  reference the correct commit, worktree, and saved diff; the guide honestly
  scopes this tree as excluding the F1/F2 follow-ups.

## Method

Interdiff of the previously reviewed isolated diff
(`359b11d5aa1..8dc79f8803b`) against the rebased isolated diff
(`a2f664ecf57..85d18ee9e06`), normalized to added/removed lines. The delta is
small: hunk-position noise plus four real conflict resolutions.

## The four rebase resolutions (all verified correct)

1. **`_GroupInvariantBroadcast` threading removed.** The old #191775 diff had
   to thread `group_broadcast` through the nested emitter; the new base
   deleted that mechanism, so the rebased diff no longer touches it. Grep
   confirms nothing orphaned.
2. **`half{factor}` -> `lane{factor}` family-suffix rename.** Correct at
   factor 4 ("half" was a factor-2 name). Rename is complete in code
   (remaining "half" hits are factor//2 arithmetic and one docstring), and
   #190595's new indirect-index mask tests were carried through the rename
   (`half2_r0_index_mask` -> `lane2_r0_index_mask` in the FileChecks) rather
   than dropped.
3. **Pre-flush internal-source materialization migrated to the new API**
   (`materialize(name, required=True)` -> `materialize(name)` with
   `must_forward` derived from `kernel.store_buffer_names`). Semantics are
   identical: at the pre-flush call point (simd.py:3756) the internal
   source's producing chain has already emitted its store, so
   `must_forward=True` and loud-on-miss is preserved.
4. **`parent_source_names` construction adapted** to the base's
   `_dependency_names` helper refactor -- composition/spelling change only,
   same union.

## The P0-shaped pattern, checked explicitly

The flush gate `if not internal_source_names: kernel.codegen_body()`
(simd.py:3752) is the same shape as the upper-stack replay P0 (a spuriously
non-empty internal set suppressing the pass-boundary flush). Here the set is
derived correctly: `internal_source_names = OrderedSet(source_layouts) &
parent-written buffer names` (simd.py:3672) -- lane-projected AND
in-kernel-written. The filter whose omission caused the replay P0 is present.

## The four-line integration fix

`_DerivedIterationFamily.set_value_masks` now wraps mask assignment in
`ensure_active`, because #191775's pre-flush materialization path calls it
before the family is active (mask_vars_for_shape asserts activity and
filter_masks needs the family's trees).

Subtlety verified: activation emits derived headers ONCE (`ensure_headers`,
loop-local for looped trees), so activating at the pre-flush point could in
principle emit headers into the wrong loop pass. It cannot here: in the
internal-source branch the flush is skipped, so pre-flush materialization
shares the same pending loop body as the epilogue that consumes the headers;
in the no-internal branch the first activation happens inside the epilogue
emission itself. Correct -- but the argument is non-obvious.
**Requested (non-blocking): a two-line comment at the ensure_active wrap
explaining why header emission is safe at that call point.** Positive note:
the fix's docstring records the divisibility argument that makes the
target-family mask exact.

## Independent verification (this review's own runs)

Full-package overlay (`run_wt.py`), caches disabled, B200:

```text
test/inductor/test_nested_reduction.py   386 ran, OK (7 skipped)
test/inductor/test_inductor_scheduler.py  86 ran, OK (6 skipped)
```

Matches the guide's counts, with one strengthening: the guide's "2 known
kernel_num_gb harness failures" PASS under the full-package overlay --
third independent confirmation that those failures are an artifact of the
narrower preload runner, not the code. Recommend dropping the "known
failures" framing from future verification records and using run_wt.py.

## Remaining items (unchanged, live at the #190595 level)

- Commit-message materialization sentence (lazy on first use,
  store_buffer_names-derived must_forward).
- Nested cpp_wrapper test variant.
