# Padded-scatter follow-up (2026-09-02)

Work on `agent_space/pr191974_padded_wt`, uncommitted on `61b333901fa`.
Resolves the two hardening suggestions and the two hygiene notes from
`pr191974_followup_and_padded_review_20260902.md` section 2, plus the
main-checkout divergence.

Final diff: `decomposition.py +1`, `ir.py +31/-6`, `lowering.py +36/-3`,
`scheduler.py +9`, new `test/inductor/test_padded_scatter.py` (12 tests).

## Suggestion 1 (CSE store cache) - REJECTED, with evidence

The suggestion was to make `CSEProxy.store` skip `_update_store_cache` when
`self.kernel._load_mask is not None`, on the theory that it would make consumer
fusion legal because the consumer would physically reload.

Implemented it and removed the scheduler guard to test that theory. Result:
3 of 10 tests fail (`test_downstream_read_observes_masked_store`,
`test_downstream_read_observes_masked_input_mutation`,
`test_mask_and_value_mutated_after_store`).

The invariant itself works - the consumer does stop forwarding and emits a real
load - but the load is wrong anyway:

```
tmp5 = tl.load(out_ptr0 + (x0), xmask)          # consumer reloads
tmp6 = tl_math.sin(tmp5)
tl.store(out_ptr0 + (x0), tmp1, tmp0 & xmask)   # predicated store, emitted AFTER
```

Triton codegen hoists loads above stores, so the reload reads the pre-store
value. Store-to-load forwarding through `store_cache` is exactly the mechanism
that makes intra-kernel read-after-write correct; a physical reload cannot
substitute for it. So the change trades a wrong forwarded value for a wrong
stale value - no correctness gain, and it would cost an extra load on any
future masked store.

Conclusion: the correct invariant is not "a predicated store does not populate
the store cache" but "a predicated store is not fused with a consumer that
reads the destination", which is what the scheduler guard already enforces.
Reverted; guard kept as the sole mechanism.

## Suggestion 2 (register_users_of) - NOT NEEDED

`name_to_users` has exactly one consumer, `mark_buffer_mutated`, which calls
`user.realize()`. The scatter is a `ComputedBuffer` realized eagerly in
`index_put_as_masked_fill`, so realizing it is a no-op. Separately,
`register_users_of`'s inner `register` only handles `ir.TensorBox`, so passing
the `ComputedBuffer` would do nothing at all.

The hazard behind the note - an *unrealized* mask or value whose source is
mutated after the store, where the loader is inlined into the scatter - is
real but already handled by scheduler dependencies. Pinned with two new tests
rather than adding an inert call:

- `test_unrealized_mask_reads_source_mutated_after_store`
- `test_unrealized_value_reads_source_mutated_after_store`

Both pass.

## Main-checkout divergence - resolved

The main checkout held an older scatter variant with two pieces `padded_wt`
lacked. A/B'd both.

**Adopt `decomposition.py`: exclude `aten._unsafe_index_put` from decomps.**
Without it the op decomposes to the checked `aten.index_put` (`check=True`).
With static shapes the generated Triton is byte-identical either way, because
the indices are provably in bounds. With dynamic shapes it is not:

```
_to_blocked_128x4, dynamic=True    device_assert count
  without exclusion                          1
  with exclusion                             0
```

That is the production swizzle path, so the bogus bounds check matters. The
entry also mirrors `aten._unsafe_index` directly above it. Pinned by adding
`assertNotIn("tl.device_assert", code[0])` to the existing
`test_dynamic_masked_padding`, which previously only asserted it for the
static case - that gap is why the exclusion went missing. Verified
non-vacuous: reverting the one-line change fails that test.

**Drop `type_promotion_kind=None` on the `_unsafe_index_put` registration.**
Unobservable: eager `_unsafe_index_put` requires source and destination dtypes
to match, so promotion can never fire. Tested four mismatched-dtype combos;
all raise identically with and without it.

Note the main checkout also carries an unrelated nested-reduction guard
(`NestedReduction._sub_parent_has_duplicated_cross_domain_origins`, +36 in
`scheduler.py`) that exists in neither worktree and is not part of this work.
Still unresolved.

## Verification

- `test_padded_scatter.py`: 12/12 on `padded_wt`.
- Combined stack (correctness follow-up + padded, applied together in
  `pr191974_simplified_wt`): padded 12/12, scheduler 150 OK / 6 skipped.
- `git apply --check` of the padded diff onto the correctness follow-up is
  clean; the two `scheduler.py` edits are ~9000 lines apart.
- `lintrunner` clean on all five files (the new test file needed PYFMT, applied).

## Interaction with the `mutate_to` retarget bug (added later on 2026-09-02)

The `index_put` fuzzer surfaced a **pre-existing mainline** silent-wrong-answer
bug that lives right next to this change: `mutate_to`'s fast-path pointer swing
retargets an earlier `MutationLayoutSHOULDREMOVE`. Full analysis in
`agent_space/mutate_to_retarget_bug_20260902.md`.

Two consequences for this PR:

1. **This change incidentally fixes most of it.** Routing masked fills through an
   `ir.Scatter` with a `store_mask` instead of `mutate_to(self, where(...))`
   removes the swing. On a 300-seed random mutation-sequence campaign: plain main
   15 MISMATCH, this worktree 7 MISMATCH, `mutate_to` fix 0.

2. **The 7 survivors are all `accumulate=True`**, which still takes the
   `mutate_to` fallback in `index_put_as_masked_fill`. They are wrong on main too,
   so this is not a regression -- but the padded change is *not* a substitute for
   the `mutate_to` fix, and the two are independent PRs.

   Extending the predicated-store path to `accumulate` (inner_fn
   `self_loader(idx) + value` under the same `store_mask`) looks feasible but is
   scope creep here, and the win is smaller since accumulate must load every lane
   anyway. Not attempted.

New test pinning the interaction: `test_masked_store_after_scatter_mutation` in
`test/inductor/test_padded_scatter.py` (13 tests total now). Verified
non-vacuous: the same program MISMATCHes on `pr191974_simplified_wt`, the stack
base with no scatter changes. `lintrunner` clean.
