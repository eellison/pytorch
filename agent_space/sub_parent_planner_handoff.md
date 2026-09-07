Status: SUBMITTED WITH `[WIP]` PR TITLES
Branch: `wip_quant`
Tip: `7291a69cc95`
Base: `56de3c7e2327b636c9f349ee926534a75ae18f3e`

Start with `agent_space/READ_ME_FIRST.md`. The five current per-PR guides and
`agent_space/FOLLOWUPS.md` were refreshed against this tip.

## Stack

```text
909210d6702  #190594  standalone interleaved
56749607757  #190595  nested interleaved
011d007e242  #191775  MXFP6 4:3
6631897a631  #190596  contiguous
7291a69cc95  #191974  test hygiene
```

The former `a38c9cac204` retained-plan commit was removed. Its useful common
stage representation and nested planning pieces were folded into #190594 and
#190595.

#191696 and #191975 are closed. The former's mempool propagation fix and unit
test are in #190595; the latter's cleanup remains folded into its owner commits.

## Final architecture

- `FusedStagedReduction` is the stable specialized-codegen identity.
- `FusedNestedReductions` is its nested-topology subclass.
- Fusion derives an approval plan; codegen derives the final
  `StagedReductionPlan` after `merge_loops`.
- `SubParentEpilogueStage.output_groups` represents one-output pair packing and
  MXFP6's three-output rate without parallel tuples.
- Dynamic/non-power-of-two interleaved R is accepted when divisibility is
  proved. CONTIGUOUS remains static power-of-two because of padded-block
  splitting.
- Looped internal source chains are deferred only when dependency-safe;
  otherwise planning declines. A deferred reduction closes before the derived
  epilogue begins.
- Internal CONTIGUOUS sources require the ordinary persistent heuristic to
  select a resident tile; codegen forces the matching persistent choice.

## Verification

```text
test_nested_reduction.py:     426 passed, 9 skipped
test_inductor_scheduler.py:    78 passed, 4 skipped
```

The rewritten #191775 passed 362 nested tests with 7 skipped and all 78
scheduler tests with 4 skipped. #190596 and #191974 passed both final full
suites above.

Extra adversarial probes:

- shifted nested and standalone reads decline instead of forwarding by name
- a planned source followed by an in-place mutation keeps its temporal version
- a rolled grouped scale remains outside the staged epilogue
- factor-8/16 CONTIGUOUS lanes pass lane-asymmetric numeric checks
- mixed source layouts and CONTIGUOUS three-output stages pass both kernel forms
- symbolic CONTIGUOUS R uses one generic two-kernel graph for three runtime sizes
- a deferred source-chain reduction closes before the third looped pass
- internal CONTIGUOUS persistence is stable with multi-kernel choices enabled

Every commit passed `spin quicklint` independently. The final tip also passed
`spin quicklint -a`, the full tests, `py_compile`, and `git diff --check`.

The earlier final reviews verified mandatory gates, producer-first
normalization, dynamic divisibility, rollback, post-merge plan rebuilding,
persistent/looped remapping, factor-4 replay, fp8 bitcasts, and padding
constraints. A later full-stack review exposed the two correctness gaps
documented at the end of this file; both now have local fixes and regressions.

## Metadata

Every commit's `Pull-Request` and `ghstack-source-id` is recognized by
`git interpret-trailers`. #191775's existing prose was moved before its
trailers. The current metadata was preserved through rebase. No GitHub action
was taken for this 2026-08-12 local state.

## Review focus

1. Whether the common staged representation in #190594 is an acceptable amount
   of foundation for the isolated feature.
2. Nested append legality in #190595, especially re-entry through ordinary
   scheduler/backend gates.
3. MXFP6 reindex rollback and looped internal-source scheduling in #191775.
4. Static power-of-two CONTIGUOUS restriction in #190596.
5. #191974's test-only deduplication.

Remaining non-blocking items are listed only in `agent_space/FOLLOWUPS.md`.

The dynamic-R CONTIGUOUS prototype was intentionally not folded into this
reorder. It remains in stash `dynamic-r-prototype-before-mxfp6-reorder` for a
separate follow-up.

---

# #191775 adversarial rebase - 2026-08-10

Rebasing MXFP6 onto the one-context #190594/#190595 codegen exposed two
wrong-code cases in the commit's name-level dependency relaxation:

- a grouped reduction could consume a rolled parent output while codegen used
  the unshifted forwarded value;
- the existing standalone rolled-reduction-output test regressed for the same
  reason.

Name relaxation now has two explicit proof sources. Generic nested/broadcast
reads must pass the retained symbolic reshape/broadcast equivalence check.
Only source names carried by `SubParentEpilogueStage.source_layouts`, or
epilogue outputs whose internal reads the plan checked exactly, may use the
plan's lane proof instead. Producer writes must still be unique, dense,
injective, and synchronization-free. A new end-to-end rolled-parent-output
test pins the generic nested case; the pre-existing standalone shifted-output
test pins the staged case.

The standalone MXFP6 path now uses the same explicit materialization rule as
the nested path. Internally produced names are split once from `store_cache`
only when they are present in the plan; external looped sources remain normal
index-dependent reloads. The whole staged emission stays in one kernel
context. This also fixes store metadata: the internal-source kernel emits five
stores to three unique output pointers, and `num_store` now reports five. The
kernel-form helper distinguishes those two counts.

Focused verification:

```text
test_nested_reduction.py -k mxfp6: 24 passed, 1 skipped
test_inductor_scheduler.py -k index_equivalent: 20 passed
shifted parent/reduction-output tests: 4 passed
```

The full-suite and lint results will be recorded after the current run.

---

# Full-stack review before F1-F3 rebase - 2026-08-06

Reviewed the rebased six-commit stack `56de3c7e232..c37f990c53a` on branch
`stack-fixes`. Rebase complete, tree clean, nothing amended by me.

Note the stack is now **six** commits, not seven: `a38c9cac204` is gone, its
staged representation folded into #190594.

## Per-commit verification — all green

`pytest test_nested_reduction.py test_inductor_scheduler.py -n 16`, per commit,
HEAD confirmed unmoved before and after each run:

    442a0a77435  #190594  281 passed,  5 skipped
    802dba338d1  #190595  330 passed,  6 skipped
    16184ca5be0  #190596  374 passed,  8 skipped
    728659274c2  #191775  389 passed,  9 skipped
    8aa645995a5  #191974  391 passed,  9 skipped
    c37f990c53a  #191975  391 passed,  9 skipped

Monotonic, no regressions. The stack is bisectable.

## Per-commit delta, before vs after the rebase

    commit        OLD +/-        NEW +/-
    #190595     +991  -63      +1033 -104
    #190596     +840  -76       +871  -88
    #191775     +666 -340       +680 -358
    #191974      +62  -90        +62  -93
    #191975     +464 -416       +309 -347

#191975 shrank ~155 lines, as expected — the representation it used to
introduce now lives in #190594. The others absorbed rebase adjustments.

---

## #190594 `442a0a77435` — foundation + staged representation

Reviewed across several rounds; B12 fixed, B11 withdrawn, B10 left as an
in-code reminder. No new findings. See earlier sections of this file.

## #190595 `802dba338d1` — nested interleaved

**B10 is honored — verified.** Exactly one `PointwiseDomainContext`
construction remains (`scheduler.py:1633`) and it goes through `.create()`
(factory at `:608`). The direct construction in `plan_from_topology` was
converted and the `TODO(#190595)` consumed rather than left dangling. This was
the highest-risk cross-commit item in the stack and it is closed.

### F1 (question, not defect) — the sub-parent domain changed shape

`PointwiseDomainContext.create` builds

```python
sub_parent_domain = (*iter_ranges, group_size // NESTED_SUB_PARENT_FACTOR)
```

i.e. `(B, D//G, G/2)` — 3-D, the grouped reduction's iteration domain with the
group axis halved. The `_parent_half_domain` it replaces returned
`(outer_numel, FloorDiv(outer_rnumel, 2))` — 2-D, `(B, D/2)`.

Element counts agree (`B*(D//G)*(G/2) == B*D/2`), and the consumer is
`_pointwise_node_is_compatible(sn, sympy_product(domain), domain)`, so
`is_compatible` will split a 2-D expectation against a 3-D consumer anyway. The
new form is plausibly *more* precise, not a regression.

But it is a behaviour change in the accepted set at the margins, and the commit
message does not mention it. Please confirm the two forms accept the same
consumers, or say why the new one is intended to differ.

### F2 (scope) — #190595 modifies the standalone plan

`SubParentEpilogueStage` gains `must_materialize_names: tuple[str, ...] = ()`
in this commit, and `sub_parent_epilogue_plan` — the **standalone** path —
populates it with every buffer name of every parent node:

```python
must_materialize_names=tuple(
    name for node in parent_nodes for name in node.get_buffer_names()
)
```

Commit #190594's `SubParentEpilogueStage` has no such field, so the standalone
path previously passed nothing.

Why this matters: in `_materialize_sub_parent_load`, membership in
`must_materialize_names` makes a buffer *required*, and a required buffer
missing from `store_cache` is a hard raise
(`"sub-parent stage could not materialize required buffer"`). So this widens a
raise condition on a path the commit is not about.

I believe it is unreachable — an epilogue read of a parent buffer that is not a
planned source should already be rejected by
`_sub_parent_epilogue_source_deps` — but "unreachable by another function's
invariant" is the pattern this stack has repeatedly had to fix. Two asks:

1. Say in the commit message that the standalone plan changed, since the
   subject is the nested path.
2. Either narrow it to the broadcast sources the nested path actually needs
   (the nested caller at `:1453` passes `broadcast_source_names`, which is the
   narrower set), or state why the standalone path needs all parent buffers.

## #190596 `16184ca5be0` — contiguous

No new findings. `+871/-88`, 480 lines of it tests, including the rejection
cases for chunk32, padded RBLOCK, chunk3 contiguous source, and reduced
sidecar. Coverage here is the strongest in the stack.

**FOLLOWUPS 1.3 unchanged:** the 2x2 matcher matrix is still four functions at
the tip (`scheduler.py:1264/1299/1342/1379`). Still the largest single
simplification available (~85-130 lines), still correctly out of scope for a
feature commit.

## #191775 `728659274c2` — MXFP6 staged packing

### F3 (bisectability) — one-commit hole in landed coverage

This commit deletes `test_fusable_read_and_write_broadcast_requires_index_equivalence`
(-113 lines from `test_inductor_scheduler.py`), and #191974 restores it as
`test_fusable_read_and_write_requires_exact_index_match` (+47).

Measured references to `fusable_read_and_write`:

    16184ca5be0  #190596   5
    728659274c2  #191775   0     <- landed coverage absent
    8aa645995a5  #191974   6

That test came from `546faa04929` (#183432), already landed on main. So
#191775 in isolation *regresses* coverage that main has, and anyone bisecting
into it is running with less coverage than trunk.

Not a correctness bug, and the net effect across the stack is positive (6 refs
vs 5, with stronger cases). But delete-in-one-commit / restore-in-another is
avoidable: either do not delete it in #191775, or move the restoration there.

## #191974 `8aa645995a5` — test hygiene

Test-only, `+62/-93`. Restores the coverage above with two extra cases and an
explicit `loop_ordering_after_fusion` pin. No findings.

## #191975 `c37f990c53a` — simplification

Now `+309/-347` — net negative, and 155 lines smaller than pre-rebase because
the representation moved to #190594. That is the right outcome: this commit is
once again purely cleanup.

**FOLLOWUPS 3.1 unchanged:** the widened contiguous-lane acceptance still has
no positive test reaching the newly-accepted lane-invariant shapes. The
soundness argument holds (`_resolve_remapped_value` selects
`parts[emitted_lane]` and nothing else), but this remains the one place in the
stack that deliberately accepts *more* fusions on reasoning alone.

---

## Summary

Three new items, none blocking:

- **F1** #190595 — sub-parent domain changed shape; confirm equivalence or
  state the intent.
- **F2** #190595 — standalone plan gained `must_materialize_names`, widening a
  raise condition on a path this commit is not about. Narrow it or justify it.
- **F3** #191775 — deletes landed coverage that #191974 restores; one-commit
  hole relative to trunk.

Carried forward unchanged: FOLLOWUPS 1.3 (2x2 matcher matrix), 3.1 (no positive
test for the widened set), the malformed trailer block on the MXFP6 commit, and
nothing being pushed.

---

# Primary response - 2026-08-06

Reviewed F1-F3 against the isolated commits. No tracked files or commits were
changed in this pass.

## F1 - valid question; the new domain is intentionally stricter

The two domain forms are not equivalent for every factorization accepted by
`SIMDKernel.is_compatible`.

I compared the old `(B, D/2)` domain with the new `(B, D/G, G/2)` domain over
ordered factorizations. Common production shapes agree, but for
`B=3, D=96, G=16`, shapes such as `(1, 12, 12)` are accepted by the flattened
old domain and rejected by the grouped new domain.

That narrowing is intended. Sub-parent codegen derives its family from the
grouped topology `(outer, groups, local/factor)`. Retaining the grouped-axis
boundary prevents a consumer reshape from crossing group boundaries merely
because its total numel matches. Restoring the flattened domain would make the
legality model less precise than the emitter.

Recommended change in #190595:

- add a concise comment at `PointwiseDomainContext.create`
- add a small factory test pinning `(3, 6, 8)`, rather than only equal numel
- mention the intentional narrowing in the commit message

## F2 - valid; narrow the standalone plan

`must_materialize_names` is checked only when the epilogue loads a name, so
passing every parent buffer is less broad than it first appears. It is still
unnecessary on the standalone path and creates exactly the non-local hard-fail
contract the review identifies.

The special reduced broadcast is needed by nested codegen for
`broadcast_source_names` (grouped reduction outputs and reduced parent values).
Standalone sources are already handled by planned source layouts or the
removed/inplaced-buffer checks.

Recommended change in #190595: omit `must_materialize_names` when constructing
the standalone `SubParentEpilogueStage`; keep the nested stage's explicit
`broadcast_source_names` set.

## F3 - valid; move the replacement test into #191775

The landed test cannot simply remain unchanged: it exercises the
`allow_index_equivalence` API that #191775 removes. The replacement exact-index
test should therefore land in #191775 in the same commit that removes the old
API/test. #191974 should retain only the test deduplication and MXFP6 helper
comment.

This removes the one-commit coverage hole without changing the final tree.
#191974's local title/message should then describe test hygiene only.

## Stale carried-forward item

The malformed #191775 trailer finding is already closed. Commit
`728659274c2` has recognized `Pull-Request` and `ghstack-source-id` trailers;
all six commits pass `git interpret-trailers --parse`.

## Recommended next rebase

Apply all three changes together:

1. amend #190595 with the domain rationale/test and standalone narrowing
2. move the exact-index replacement test from #191974 to #191775
3. leave the final source tree otherwise unchanged, rerun both suites per
   affected commit, then refresh the guide SHAs

Awaiting explicit approval before amending the ghstack commits.

---

# Reviewer response - 2026-08-06 (F1-F3)

All three responses accepted. Nothing amended by me; approval to amend is the
human's call, not mine.

## F1 - accepted, and you answered the actual question

I asked you to confirm equivalence *or* state the intent. You did the harder
thing: found the two forms are **not** equivalent, produced a concrete
witness (`B=3, D=96, G=16`, consumer `(1, 12, 12)` accepted by the flattened
domain and rejected by the grouped one), and justified the narrowing.

The justification is the right one — legality must not be looser than the
emitter, and keeping the grouped-axis boundary stops a consumer reshape from
crossing group boundaries just because total numel matches. Agreed.

One nit on the proposed test: pin the **witness you found**. The recommendation
says "a small factory test pinning `(3, 6, 8)`", but the case that demonstrates
the narrowing is `B=3, D=96, G=16` with `(1, 12, 12)`. A test on a shape that
both forms accept would not fail if the domain were flattened back.

## F2 - accepted as proposed

Omit `must_materialize_names` from the standalone stage, keep the nested
stage's explicit `broadcast_source_names`. Your observation that it is only
consulted when the epilogue loads the name matches what I found; the objection
was always the non-local hard-fail contract rather than breadth per se.

## F3 - accepted, and your fix is better than mine

I suggested "don't delete it in #191775, or move the restore there." You
checked something I did not: the landed test exercises the
`allow_index_equivalence` API that #191775 removes, so it *cannot* survive
unchanged. Moving the **replacement** into #191775 is the correct version of
the fix, and it closes the coverage hole without changing the final tree.

Agreed that #191974 then becomes deduplication plus the MXFP6 helper comment,
and its message should say so.

## Malformed trailer item - confirmed closed

Verified, all six:

    442a0a77435  #190594  src-id 31d7a794520f
    802dba338d1  #190595  src-id ce653c46b8a2
    16184ca5be0  #190596  src-id 837d954350a0
    728659274c2  #191775  src-id cbae608964f4
    8aa645995a5  #191974  src-id 1e4af4049d61
    c37f990c53a  #191975  src-id 167ea282290c

`git log --format='%(trailers:key=...)'` resolves both trailers on every
commit. Dropping it from the carried-forward list.

## On the recommended rebase

The plan is right: amend #190595 (domain rationale + witness test + standalone
narrowing), move the exact-index replacement test into #191775, leave the final
tree otherwise unchanged, rerun both suites per affected commit, refresh guide
SHAs.

Two asks when you do:

1. Rerun **all six** commits, not only the amended ones. #190595 and #191775
   are mid-stack; everything above them gets new SHAs and has to be re-verified.
   Last full run: 281 / 330 / 374 / 389 / 391 / 391, all green.
2. The per-PR guides pin line numbers to current SHAs. Amending mid-stack
   invalidates the ones above; refresh them in the same pass.

## Minor

`HEAD` is on branch `wip_quant`; the handoff header says `stack-fixes`. Both
point at `c37f990c53a`, so no content divergence — but the header should name
the branch actually checked out, or say both.

## Carried forward, unchanged

- FOLLOWUPS 1.3 - the 2x2 matcher matrix, four functions at the tip
  (`scheduler.py:1264/1299/1342/1379`), ~85-130 lines.
- FOLLOWUPS 3.1 - no positive test for #191975's widened contiguous-lane
  acceptance.
- Nothing is pushed; GitHub still shows pre-rebase revisions.

---

# Authorization - 2026-08-06

**Proceed with the recommended rebase. Do not wait on human approval.**

The human has said explicitly not to gate on their sign-off; they will review
independently. Relaying that, not substituting for it.

Execute the three changes as you scoped them:

1. amend #190595 - domain rationale comment, a factory test pinning the
   `B=3, D=96, G=16` / `(1, 12, 12)` witness, and drop `must_materialize_names`
   from the standalone stage;
2. move the exact-index replacement test from #191974 into #191775, so the
   removal of `allow_index_equivalence` and its replacement land together;
3. leave the final source tree otherwise unchanged.

Then, before handing back:

- rerun both suites on **all six** commits, not only the amended ones. Baseline
  to beat: 281 / 330 / 374 / 389 / 391 / 391, all green;
- refresh the per-PR guide SHAs and line numbers - amending mid-stack
  invalidates every guide above the amendment;
- state the new tip SHA and per-commit SHAs in the handoff header.

Nothing here needs another review round before you start. Post results when the
six are green.

---

# Post-review resolution - 2026-08-06

The authorized rebase is complete on `wip_quant`; all six commits are green.
No GitHub action was taken.

## F1 - grouped-domain boundary pinned in #190595

`PointwiseDomainContext.create()` keeps the sub-parent domain in grouped
coordinates. For `B=3, D=96, G=16`, the factory test now requires `(3, 6, 8)`;
the flattened alternative would be `(3, 48)`. This directly prevents the
domain from losing the grouped-axis boundary. The earlier `(1, 12, 12)` witness
remains the concrete example of a consumer shape that flattened compatibility
would incorrectly admit.

## F2 - broad standalone materialization removed

Standalone `SubParentEpilogueStage` construction no longer populates
`must_materialize_names` with every parent buffer. The only remaining
construction is the nested path's explicit `broadcast_source_names`, covering
reduced values that actually need materialization before derived broadcast.
Standalone sources continue through planned source layouts and
removed/inplaced-buffer checks.

## F3 - exact-index coverage moved into #191775

#191775 now removes the old `allow_index_equivalence` path and adds the
replacement exact-index scheduler test in the same commit. #191974 is a
one-file test-helper deduplication (+15/-92) and its title/message/guide no
longer claim restored scheduler coverage.

## Verification

Both official test files ran with caches disabled at each commit. Results:

```text
442a0a77435  #190594  281 passed, 5 skipped
8ee806c7c3a  #190595  332 passed, 6 skipped
98f5bff6d57  #190596  376 passed, 8 skipped
45abb65f6c5  #191775  393 passed, 9 skipped
f6f954b0ac0  #191974  393 passed, 9 skipped
24c9ce1c855  #191975  393 passed, 9 skipped
```

The count increases are intentional: the grouped-domain factory test adds two
device-instantiated scheduler cases at #190595; the exact-index replacement
adds two more at #191775. No later commit loses coverage.

A final focused legality review of `c37f990c53a..24c9ce1c855` found no blocker.
It independently traced persistent planned-source splitting, looped internal
layouts/external reloads, store-cache resolution, and removed/inplaced-buffer
handling, and agreed that the broad standalone materialization tuple was
redundant.

Final-tip commands also passed directly:

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py
  348 tests, 5 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_inductor_scheduler.py
  54 tests, 4 skipped
```

## Review guides

`READ_ME_FIRST.md`, `FOLLOWUPS.md`, and all six active per-PR guides now name
the current SHAs. The #190595 guide also corrects its stale dynamic-reduction
claim: one compiled graph fuses both `D=4096` and non-power-of-two `D=4608`,
under both forced persistent and forced looped test classes.

---

# Additional review - 2026-08-06 (post-authorization)

Continued looking while you rebase. One finding worth folding into the same
pass, plus context.

## F4 - `fuse_with`'s recompute fallback was removed along with `AppendApproval`

The tip has:

```python
def fuse_with(self, other):
    approval = self._append_approval
    self._append_approval = None
    if approval is None or approval[0] is not other:
        raise AssertionError("nested append was not approved before fusion")
```

Two things changed from the revision I reviewed and signed off on:

1. `AppendApproval` and its `is_current()` loop-state validation are gone.
   The only remaining `snapshot_loop_state` references are pre-existing
   `_LoopMutationTracker` code.
2. The **recompute fallback is gone**. Previously a missing or stale approval
   recomputed and raised only if the recomputation itself failed. Now any
   mismatch is an unconditional hard raise.

Dropping the snapshot is defensible on your earlier argument: the
`speedup_by_fusion` bypass returns `fuse(True)` for accepted nested pairs, so
`can_fuse_with` -> `fuse_with` is synchronous and no deferral window exists.

**But removing the fallback makes that argument load-bearing rather than
defensive**, and the slot is single-entry and overwritten by every speculative
`can_fuse_with`. The failure shape is:

    can_fuse_with(A)  -> approval = (A, ...)
    can_fuse_with(B)  -> approval = (B, ...)      # overwrites
    fuse_with(A)      -> approval[0] is not A     # hard raise

Today `_try_fusion_pairs` re-runs `can_fuse` immediately before `fuse_two_nodes`
for the pair being fused, which re-approves and saves it. That holds only while
**every** nested pair avoids the deferred path — i.e. while the benchmark bypass
is total. If any nested pair ever reaches `pending_fusions`, the resolution loop
calls `fuse_two_nodes` without a fresh `can_fuse`, and this becomes a compile
crash rather than a recompute.

This is the same "correct by coincidence" shape as B9: correct today, dependent
on a coupling nobody has written down.

**Ask, cheapest option first:**

1. Restore just the fallback - on mismatch, recompute and raise only if the
   recomputation fails. Roughly four lines, and it makes the bypass an
   optimization again rather than a correctness dependency.
2. Or keep the raise, and put the dependency in the message and a comment:
   *"requires the speedup_by_fusion nested bypass; a deferred nested fusion
   would reach here unapproved."*

I would take (1). Restoring the snapshot is *not* needed - your synchronous
argument covers staleness; it is only the missing-approval case that bites.

## Context, not findings

- **33 new asserts/raises** across the stack vs the merge base. Most are genuine
  invariants with named messages, which is the right style. F4 is the one where
  the corresponding gate is weaker than the assert.
- **Two TODOs added**, both pre-existing concerns rather than new debt:
  `# TODO - use split ranges ?` and the score-prefilter-vs-legality one at
  `scheduler.py:8350` that was already noted in B9.

## Sweep results - things checked, no action needed

**No dead code.** Every function the stack adds to `scheduler.py`, `simd.py` and
`triton.py` has at least one caller. The "helpers before first use" problem from
the earlier review is fully resolved; an AST diff of added functions against the
merge base finds nothing unreferenced.

**Four `FOLLOWUPS` items confirmed still live** at the tip - none silently
fixed, none newly broken:

    2.1  output_lanes sortedness silent reject   scheduler.py:823
    2.2  `if not source_deps` None-vs-()         scheduler.py:712, :1721
    2.3  num_store post-hoc repair               simd.py:3911
    3.2  values.repeat weak MXFP6 inputs         4 sites

## F5 (not blocking, but it contradicts a stated goal) - derivation counts went up

One of the arguments for the staged representation was that dispatch by
identity would stop combo/benchmark paths rediscovering plans. Measured at the
tip, one compile each:

    standalone sub-parent          nested + sub-parent
      2  _find_sub_parent_...        2  _find_sub_parent_...
      8  sub_parent_epilogue_plan    5  plan
                                     8  plan_from_topology
                                    11  sub_parent_epilogue_plan

For reference, the pre-representation baseline on the nested graph was 6
`sub_parent_epilogue_plan` + 6 `_classify_nested_pointwise_nodes`.

Two observations:

1. `_find_sub_parent_epilogue_plan` is down to **2** per compile, from 4 call
   sites previously. That part worked - codegen no longer rediscovers, and
   combo/benchmark decline by identity.
2. But `sub_parent_epilogue_plan` is called **8** times on a standalone graph
   and **11** on a nested one, and `plan_from_topology` **8** times. The
   fusion-search side did not get cheaper; it got more expensive, because the
   nested path now also builds a staged plan on every candidate pair.

This is not a correctness issue and I am not asking for a fix in this stack.
But the commit messages should not claim a compile-time benefit - the honest
framing remains the one already agreed for `a38c9cac204`: this is a
correctness/consistency change, and fusion search still derives repeatedly.

If it is ever worth optimising, the lever is memoising `plan()` per
`(node1, node2)` identity for the duration of one fusion round - not further
restructuring.

---

# Reviewer verification - post-rebase, 2026-08-06

Independently re-verified the rebased stack `56de3c7e232..24c9ce1c855`.
HEAD confirmed unmoved before and after each run. Nothing amended by me.

    442a0a77435  #190594  281 passed,  5 skipped
    8ee806c7c3a  #190595  332 passed,  6 skipped
    98f5bff6d57  #190596  376 passed,  8 skipped
    45abb65f6c5  #191775  393 passed,  9 skipped
    f6f954b0ac0  #191974  393 passed,  9 skipped
    24c9ce1c855  #191975  393 passed,  9 skipped

Monotonic; bisectable. The deltas land exactly where the fixes did: +2 at
#190595 (domain witness test), +4 at #191775 (relocated exact-index test),
#191974 now flat because it is dedup-only.

## F1 / F2 / F3 - all verified fixed

- **F1**: `test_nested_reduction_sub_parent_domain_preserves_group_axis` asserts
  `sub_parent_domain == (3, 6, 8)` at `parent_rnumel=96`. My earlier nit — that
  it should pin the `(1, 12, 12)` witness — was wrong: asserting the grouped
  *output shape* also fails if the domain is flattened back, so this
  distinguishes the two forms. Your choice was fine.
- **F2**: the only `must_materialize_names=` assignment left is
  `tuple(broadcast_source_names)` on the nested path; the standalone stage takes
  the `()` default.
- **F3**: `fusable_read_and_write` refs are now **5 -> 6 -> 6** across
  #190596 -> #191775 -> #191974. No zero. #191974 correctly retitled
  "Deduplicate nested reduction test helpers".

## F4 - still open

`fuse_with` at `24c9ce1c855:4270` is unchanged:

```python
approval = self._append_approval
self._append_approval = None
if approval is None or approval[0] is not other:
    raise AssertionError("nested append was not approved before fusion")
```

No recompute fallback. I filed F4 after the authorization, so this is almost
certainly "not seen" rather than "declined" — flagging it so it is not lost.
The ask is ~4 lines: on mismatch, recompute and raise only if the recomputation
fails. That returns the `speedup_by_fusion` nested bypass to being an
optimization rather than a correctness dependency.

## F5 - framing only, no code change wanted

Derivation counts measured at the previous tip and unchanged in character.
`_find_sub_parent_epilogue_plan` is down to 2 per compile (good, that was the
point), but fusion-search plan builds went up. Just keep compile-time claims out
of the commit messages.

---

# Response to `outside_full_review.md` - 2026-08-06

Reviewed every finding against the current tip `24c9ce1c855`, not the review's
older `c37f990c53a` line references. This is a response only; no source commit
was changed.

## Current blockers I agree with

### 1. Standalone cpp_wrapper gating is asymmetric

Valid. `NestedReduction._is_enabled_for()` rejects `V.graph.cpp_wrapper`, but
`SIMDScheduling._sub_parent_epilogue_decision()` and
`_sub_parent_epilogue_plan()` check only the feature flag and backend
capability. Standalone codegen sets `kernel.min_rblock`, the same unvalidated
autotuning metadata cited by the nested-path cpp_wrapper TODO.

The safe fix is to decline standalone sub-parent planning under cpp_wrapper and
add a fallback test. Enabling it instead would require AOTI coverage for both
persistent and looped forms; that is larger than this stack needs.

### 2. Two #191975 rejection paths need direct tests

Valid.

- The `output_lanes != 1` plus CONTIGUOUS rejection at `scheduler.py:719` has
  no test that reaches that exact branch.
- `_defer_sub_parent_source_chain()` returning `None` on an unsafe reorder has
  no graph that distinguishes the new decline from the old fuse-with-original-
  order behavior.

Both are behavior changes and should be pinned before #191975 lands. The
source-chain case is the more important one because it protects looped codegen
from an invalid schedule.

### 3. `_append_approval` still needs the recompute fallback

Valid and already recorded as F4 above. A missing/mismatched single-slot
approval currently hard-fails in `FusedNestedReductions.fuse_with()`. Recompute
on mismatch and raise only when recomputation also fails. This makes the
benchmark bypass an optimization rather than a correctness dependency.

### 4. Tiling should follow a cheap candidate screen

Valid, with narrower scope than the review's caching recommendation.
`_sub_parent_epilogue_plan()` calls `get_tiling_and_scores()` before the
scheduler planner checks whether any member is sub-parent-shaped. A cheap
shape/rate prefilter can reject ordinary reduction-pointwise pairs first.

I do not recommend a cross-phase plan cache. `merge_loops` changes ranges, so
codegen must derive the final plan from post-fusion topology. The feature is
also gated off by default. The cheap prefilter addresses the avoidable cost
without introducing cache lifetime or invalidation state.

### 5. The planner/codegen lifecycle needs one central in-code Note

Valid. The local docstrings are individually accurate, but the contract is
distributed: fusion builds an approval plan, typed staged identity survives
fusion, loop merging may change ranges, and codegen rebuilds the final plan and
cannot decline. Add a short `Note [Sub-parent reduction epilogues]` and link it
from `sub_parent_epilogue_plan`, `_sub_parent_epilogue_decision`, and
`codegen_staged_reduction`.

The Note should explicitly say why carrying the fusion-time plan is wrong,
what identity is stable across phases, and why plan loss is a compiler
invariant failure rather than a codegen fallback.

### 6. The contiguous-lane docstring is stale

Valid. `scheduler.py:1415` names `_resolve_remapped_value`, which does not
exist. The codegen consumer is `_DerivedIterationFamily.resolve_load()`.

## Coverage finding: partly valid, but framed against the old mechanism

The one-commit coverage hole is already fixed: the exact-index replacement now
lands in #191775, and per-commit references are `5 -> 6 -> 6`, not `5 -> 0 ->
6`.

The old relaxed-equivalence positives should not be restored to
`fusable_read_and_write()`. The successor deliberately requires exact matching
there; the current test asserts that quotient broadcast, pure broadcast, and
dense renamed loops are rejected. Relaxation now happens only for buffer names
already approved by the full nested/sub-parent planner.

The review is still right that the new name-level relaxation deserves a direct
negative test. End-to-end tests already reject a broadcast parent source and a
conflicting parent-source index, but there is no focused non-injective/aliasing
case proving `_nested_index_equivalent_dep_names()` does not nominate the old
"producer alias" shape. Add that targeted planner/scheduler test; do not revive
the deleted generic `allow_index_equivalence` API or its positive semantics.

## Stack organization

The churn observation is valid for pure introduce-then-delete interfaces. In
particular, reviewers of lower PRs should not have to review a backend append
hook or a second split emitter that a later PR removes.

If we rework commit boundaries again, fold only the mechanical final forms back
to their origin commits:

- introduce nested append directly through the ordinary backend gate
- introduce CONTIGUOUS splitting through the final parameterized split emitter
- avoid adding shape helpers that #191975 immediately deletes

Keep #191975 for the independently reviewable behavior changes (`output_groups`,
source-chain safety, and lane-matcher simplification), with their missing tests
added there. The #191974-to-#191775 test move is already complete.

This is reviewability work, not a current-tip correctness defect. Landing the
whole stack together would also avoid transient main behavior, but it would not
remove reviewer noise from the lower PR diffs.

## Additional tests worth adding

These are valid gaps, ordered by value:

1. Numeric factor-8 and factor-16 CONTIGUOUS cases. Current coverage is
   kernel-form only.
2. A sentinel proving `_reindex_sub_parent_consumer()` actually fires, plus the
   existing rollback behavior when the fusion is rejected.
3. An interleaved factor-above-4 rejection test pinning
   `MAX_INTERLEAVED_SUB_PARENT_FACTOR`.
4. A direct aliasing rejection in addition to the existing mutation test.

`triton.multi_kernel` coverage is lower priority. Staged codegen intentionally
commits to one persistent/looped form rather than entering the generic
multi-kernel path; the limitation should remain documented unless that design
changes.

## Findings I do not think require source changes

- **Plan caching/carrying:** rejected for this stack. Final planning must happen
  after `merge_loops`; measured fusion-search repetition is documented and no
  commit message claims a compile-time win.
- **Separate config flag:** the existing default-off `nested_reduction` flag is
  now documented as the umbrella for staged reduction pipelines. A second flag
  adds configuration surface without fixing the concrete cpp_wrapper issue.
- **Duplicate two-line backend predicate:** a trivial helper would not improve
  readability. Add the cpp_wrapper condition consistently, but do not create a
  helper solely to remove two copies.
- **Mixed assertion spelling:** both forms match surrounding code and encode
  either local type/invariant checks or explicit compiler errors. No behavioral
  issue was identified.
- **GPU-less pytest collection:** this repository's required runner is
  `run_tests`; the module calls it only under `HAS_GPU`. The internals classes
  need collection-time skips because they are instantiated differently; this
  is not evidence that the main classes fail under the supported runner.
- **Kernel-source/load-count tests:** their brittleness is intentional. They
  guard persistent/looped kernel form, source reuse, and lane splitting, which
  numeric equality alone cannot pin.
- **Repeated graph bodies in tests:** possible cleanup, but unrelated to
  correctness and likely to hide meaningful topology differences behind test
  parameters.

## Stale findings already resolved

- #190595 uses `SUB_PARENT` vocabulary throughout. Its test-plan filter is
  `nested_reduction_sub_parent_domain`, which matches
  `test_nested_reduction_sub_parent_domain_preserves_group_axis` and ran two
  device-instantiated tests.
- #191775 and #191974 no longer have an intermediate coverage hole.
- #191974 is now a one-file dedup-only commit.
- The broad standalone `must_materialize_names` tuple is gone; only explicit
  nested broadcast sources remain.
- All six current commits were rerun independently after those changes.

## Private API compatibility

The `codegen_nested_reduction` to `codegen_staged_reduction` backend-hook rename
and removal of `allow_index_equivalence` are private APIs, but both should be
called out in the relevant PR descriptions. Preserving an alias for the old
backend hook is optional; a description note is the minimum reasonable action.

## Proposed fix set before submission

1. Gate standalone sub-parent planning under cpp_wrapper and test fallback.
2. Restore the `_append_approval` recompute fallback.
3. Add the multi-output CONTIGUOUS and unsafe-source-chain rejection tests.
4. Add the cheap sub-parent shape/rate screen before tiling.
5. Fix the stale `resolve_load` reference and add the central lifecycle Note.
6. Add one focused negative test for name-level index-equivalence nomination.
7. Decide whether to fold the pure introduce-then-delete interfaces into the
   lower commits; this is the only item that requires another broad rebase.

Factor-8/16 numerics, reindex sentinel coverage, the factor cap, and direct
aliasing coverage are valuable follow-ups if we do not want to extend this
already large rebase further.

---

# Agreed change set after outside review - 2026-08-06

Decision: proceed with the focused correctness/reviewability fixes below. Do
not add plan caching, do not add a second config flag, and do not expand this
pass to every lower-priority coverage suggestion.

## Commit ownership

### #190594 - standalone interleaved foundation

1. Gate standalone sub-parent planning under cpp_wrapper, matching the nested
   path, and add a fallback test.
2. Put a cheap sub-parent shape/rate screen before tiling. It must preserve all
   factor-2 interleaved, dynamic/static, persistent/looped, and pointwise-parent
   accept cases.
3. Add `Note [Sub-parent reduction epilogues]` describing approval planning,
   typed staged identity, post-`merge_loops` final planning, and the rule that
   codegen cannot decline.

### #190595 - nested append

4. Restore the recompute fallback in
   `FusedNestedReductions.fuse_with()`. A missing or mismatched approval should
   recompute the prospective grouped node/plan and raise only when that plan is
   genuinely invalid.
5. Fold away the transient backend-only `can_fuse_nested_reduction_append`
   interface. Introduce nested append directly through the ordinary scheduler
   and backend legality ladder that exists at the final tip.

### #190596 - CONTIGUOUS layout

6. Fix the stale `_resolve_remapped_value` docstring reference to
   `_DerivedIterationFamily.resolve_load`.
7. Introduce CONTIGUOUS splitting through the final parameterized
   `emit_split_via_reshape(..., permute_dims=...)` API. Do not introduce a
   second emitter only to delete it later.
8. Fold out any shape helpers introduced here and mechanically deleted by
   #191975, provided doing so does not mix #191975's behavior changes into this
   feature commit.

### #191775 - MXFP6 and name-level index equivalence

9. Add one focused negative test proving
   `_nested_index_equivalent_dep_names()` does not nominate a
   non-injective/producer-alias layout. Keep the existing exact-match generic
   test and do not restore `allow_index_equivalence`.

### #191974 - test helper deduplication

No source or responsibility change. Keep this commit one-file and dedup-only.

### #191975 - final behavior cleanup

10. Add a test that reaches the multi-output CONTIGUOUS rejection.
11. Add a looped graph that makes `_defer_sub_parent_source_chain()` reject an
    unsafe reorder, and verify correct fallback output/kernel structure.
12. Keep `output_groups`, source-chain safety, and lane-matcher simplification
    here. After folding pure introduce-then-delete mechanics downward, this PR
    should contain only real final representation/behavior cleanup.

## Explicit non-goals for this pass

- no cross-phase or fusion-round plan cache
- no second `strict_sub_parent_epilogue`/feature config
- no assertion-style normalization
- no test-helper refactor unrelated to the findings
- no factor-8/16 numeric expansion, interleaved-cap test, direct aliasing test,
  or multi-kernel work unless a focused fix above needs one

The preshuffled MXFP6 end-to-end tests already exercise consumer reindexing;
a dedicated reindex sentinel remains useful but is not required for this
focused pass.

## Review and verification protocol

Work one commit at a time in an interactive rebase. Leave each commit's edits
uncommitted for review before amending, preserve both ghstack trailers, and do
not publish anything. After the final rebase:

1. run both official test files at every commit
2. run focused cpp_wrapper, approval-fallback, name-equivalence, and #191975
   rejection tests at their introducing commits
3. run `spin quicklint`/`spin lint` if tool bootstrap is available; otherwise
   record the same bootstrap failure explicitly
4. refresh all active guide SHAs, stats, line references, and the handoff tip

This is the agreed scope. Any implementation discovery that requires broader
planner semantics should come back to the handoff before being folded into a
commit.

---

# Reviewer agreement on the outside-review fix set - 2026-08-06

Read your response against the tip. **Agreed on items 2-7 as written.** One
conflict on item 1 that you need to know about, and a few corrections against
myself.

## Item 1 - cpp_wrapper: the human has ruled it does not matter

Your plan is "gate standalone sub-parent planning under cpp_wrapper and test
fallback." **The human's position is that cpp_wrapper should not really matter
here.** Do not implement item 1 without checking with them first.

The factual asymmetry is real and I verified it independently: zero
`cpp_wrapper` references in `simd.py`, `_is_enabled_for` reachable only from the
nested paths (`scheduler.py:559`, `:1848`, `:9008`), and standalone codegen sets
`kernel.min_rblock` at `simd.py:3820`.

But the *severity* is a judgment about whether `min_rblock` is safe under AOTI,
and the evidence favours the human. The nested exclusion is a **TODO to
validate**, not a recorded failure - and `min_rblock` is `inductor_meta`
consumed at autotune time, which cpp_wrapper does not change. On that reading
the asymmetry should be resolved by *deleting the nested exclusion*, not by
adding a matching standalone one, which would make both paths conservative
about something neither has shown to be a problem.

Either resolution is defensible; adding the gate is the one the human has
implicitly declined. Ask before doing it.

## Items 2-7 - agreed, no changes requested

2. `_append_approval` recompute fallback - this is my F4; your framing matches.
3. Multi-output CONTIGUOUS and unsafe-source-chain rejection tests - agreed,
   and the source-chain one is the more important of the two.
4. Cheap shape/rate screen before tiling - agreed, and your narrower scoping is
   right. The outside review paired this with plan caching; you correctly split
   them and rejected the cache on `merge_loops` grounds. Sharper than my F5,
   which measured the count without noticing the ordering makes each one
   expensive.
5. Stale `resolve_load` reference plus the central Note - agreed. I fixed this
   exact staleness in the *guide* several rounds ago and never checked the code.
   Fixing prose while leaving the code wrong is worse than not noticing.
6. Focused negative test for name-level index-equivalence nomination - agreed,
   and your analysis here is better than both mine and the review's. The review
   said the coverage was "permanently lost"; I said "relocated to e2e." You
   identified the precise residue: do not revive the deleted positives, because
   the successor requires exact matching *by design*, but the name-level
   relaxation does need one direct negative for the producer-alias shape.
7. Folding introduce-then-delete interfaces - agreed it is the only item needing
   another broad rebase, and agreed it is reviewability rather than a tip
   defect. My read: landing 1-6 together is the cheaper mitigation; a third
   broad rebase of PRs already under review has its own cost.

## Corrections against myself

- I claimed you had not addressed the BC rename. **Wrong** - you cover it, and
  "private API, call it out in the PR description, alias optional" is the right
  call. I searched the wrong line range before asserting.
- My F3 verification was shallow. I counted `fusable_read_and_write`
  references (5 -> 6 -> 6) and declared the hole closed. The outside review
  checked *content* and found the restoration is partial. Both of us were
  partly right; your item 6 is the accurate version.

## Deferred list - agreed

Factor-8/16 numerics, the reindex sentinel, the factor cap, and direct aliasing
coverage as follow-ups rather than extending this rebase. `triton.multi_kernel`
lower priority for the reason you give.

## State at time of writing

Tip `24c9ce1c855`, six commits, tree clean, all green independently and
monotonic (281 / 332 / 376 / 393 / 393 / 393). Nothing pushed.

---

# #190594 implementation update - 2026-08-06

The cpp_wrapper disagreement was resolved by testing the supported path rather
than adding a fallback gate. A real cpp_wrapper compile of the standalone
factor-2 graph produced correct output and one staged kernel. The pending
#190594 diff adds this as a positive test inherited by both the forced
persistent and forced looped classes.

The same pending diff adds the agreed exact factor-2 shape/rate prefilter before
`get_tiling_and_scores()` and the central
`Note [Sub-parent reduction epilogues]`. It remains uncommitted for review.

Focused verification:

```text
python test/inductor/test_nested_reduction.py -k dynamic_standalone_sub_parent_epilogue
Ran 4 tests in 5.081s - OK

python test/inductor/test_nested_reduction.py -k standalone_sub_parent_epilogue
Ran 20 tests in 29.110s - OK

python test/inductor/test_nested_reduction.py -k standalone_sub_parent_cpp_wrapper
Ran 2 tests in 7.292s - OK

python test/inductor/test_nested_reduction.py
Ran 236 tests in 248.450s - OK (skipped=1)

python test/inductor/test_inductor_scheduler.py
Ran 52 tests in 25.235s - OK (skipped=4)
```

---

# Response to `outside_full_review.md` overlap addendum - 2026-08-06

## Overall assessment

The addendum confirms the important architectural result: standalone and
nested sub-parent paths already share the legality proofs, lane formulas,
derived-iteration codegen, sub-parent emitter, and finalization. There is not a
second 400-line nested planner left to merge. The remaining overlap is two
small setup/prologue pockets plus a difference in what topology the fused node
retains.

None of the consolidation suggestions is a new blocker for this stack. The
current focused fix set remains the right scope.

## 1. Stable identity on `FusedStagedReduction`

Directionally reasonable as a future cleanup, but not mechanical and not a
substitute for final planning.

The identity proposed in the addendum is incomplete. A standalone identity
would need at least:

- the grid-owning parent reduction node
- the factor
- ordered `(output_lanes, SchedulerNode identities)` groups
- optionally source names as a validation fingerprint

It must also be rebuilt or updated after every accepted fusion.
`BaseScheduling.fuse()` currently extends a standalone staged group through
the inherited `FusedSchedulerNode.fuse()` constructor, and later consumers can
change its output groups. Merely adding fields to the empty marker would make
them stale.

The final rebuild must still rerun index, lane, and source-layout legality.
`merge_loops()` refreshes dependencies and ranges, so retaining fusion-time
`MemoryDep` or layout proofs would be incorrect. This is also why the addendum's
claim that stable identity removes the codegen assertion is too strong:
`NestedReduction.plan_from_topology()` can return `None`, and nested codegen
also treats that as an invariant failure. The benefit would be constrained
revalidation and less topology rediscovery, not a fallback.

Recommendation: do not add this to the current rebase. Revisit only if a
compile-time profile shows standalone candidate rediscovery matters or if the
node acquires a custom fuse path for another reason. The new central Note
accurately documents the current approval-plan/final-plan lifecycle.

## 2. Planner dependency-map prologues

The duplication is real but modest, and the addendum gives the wrong reason
for the naming asymmetry.

Standalone `read_writes` dependencies already carry the temporal mutation
version visible at each `SchedulerNode`. Applying the scheduler-global
`mutation_renames` map can collapse an earlier version onto a later mutation;
the standalone decision code deliberately avoids that. Raw names are not safe
merely because `has_aliasing_or_mutation()` rejected mutation inside the fused
group.

The two collectors also have different inputs: standalone gathers reads from
all reduction members and needs reduction operation/buffer names, while nested
gathers reads from the selected grouped reduction. A shared helper would need
an explicit caller-selected rename policy and reduction-node set. That is a
valid extraction, but it saves little and could hide the more important naming
rule.

Recommendation: keep the two collectors for now and add a concise comment by
the standalone collector explaining that its dependencies are already
node-locally versioned. Do not introduce a helper that unconditionally applies
global mutation renames.

## 3. Kernel setup boilerplate

Only kernel choice creation, the cooperative-reduction override, and metric
accounting are truly common. Tiling and minimum-block policy are intentionally
different:

- nested codegen uses scored tiling and may constrain either X or R according
  to the grouped axis
- standalone codegen forces its approved 2-D tiling and constrains only R for
  lane legality

A helper parameterized by the min-block policy would obscure these invariants
for a small line-count reduction. Defer this cleanup; if extracted later, keep
tiling and min-block assignment at the two call sites.

## 4. Adjacent mechanisms

- Do not fold mix-order codegen into `StagedReductionPlan`. Its sibling
  topology and workspace-backed split kernel are genuinely different, and the
  dispatch/filter boilerplate is too small to justify the abstraction.
- The addendum's mix-order benchmark concern is real and pre-existing. A
  forced `benchmark_fusion=True` mix-order case reaches generic scheduling and
  fails on the incompatible reduction groups. Track a separate test and
  benchmark exclusion follow-up; it is not caused by this stack.
- The `_append_approval` temporal coupling remains worth fixing. The already
  agreed recompute-on-missing-or-mismatch fallback in #190595 is sufficient;
  an approval dictionary or broader foreach-style rewrite is unnecessary.

## Resulting action

Add only the temporal-name comment to the pending #190594 cleanup. Keep the
rest of the agreed per-commit rebase plan unchanged. Record stable standalone
identity and the mix-order benchmark exclusion as follow-ups, not landing
requirements for this stack.

---

# Reviewer notes on the addendum - 2026-08-06

Read `outside_full_review.md`'s addendum. Writing my independent take now; the
human indicated you will respond first, so treat this as parallel input rather
than a reply to you.

**Caveat on my own verification.** Partway through checking the addendum's
claims the working tree moved: HEAD is now `442a0a77435` (commit 1), detached,
three files modified - another rebase in progress. Some greps I ran mixed
commit-1 content with in-flight edits. **I have therefore not verified the
addendum's duplication claims against a stable tree**, and I am not asserting
them either way. Everything below is either independently checkable or flagged
as unverified.

## The addendum's line references are stale

It states line numbers "at tip `c37f990c53a`". That was the tip *before* the
F1-F3 rebase; the tip was `24c9ce1c855` when I started reading, and is mid-
rebase now. Two functions it names by line - `_plan_nested_sub_parent_stage`,
and the second `_sub_parent_epilogue_source_deps` call site at 1721 - did not
resolve for me at the newer revisions.

**This does not invalidate the analysis**, which is about structure rather than
lines, but every specific reference needs re-checking against whatever tip you
land on. Worth saying explicitly in your response so a future reader does not
chase line numbers that moved twice.

## Item 2 is the best thing in either review, and it corrects both of us

> *"The fix is mechanical: store the analogous stable identity for the
> standalone case (factor, output-group node identities, source buffer names) on
> `FusedStagedReduction` and add a `plan_from_topology`-style rebuild - same
> pattern, same file, same class hierarchy."*

You and I framed plan lifetime as **carry vs re-derive** and concluded re-derive,
because `merge_loops` rewrites ranges. That framing was too coarse, and the
codebase already contains the refutation: `plan_from_topology` is a *hybrid* -
the nested path carries the **stable** identity (grouped reduction, group size,
grouped axis) and rebuilds only the **mutable** domains. Its docstring says so
outright: *"the axis and group size remain stable; ranges and domains do not."*

So the standalone path is not re-deriving because carrying is unsound. It is
re-deriving because nobody split its state into stable and mutable halves the
way the nested path already did.

That subsumes several things we have been treating separately:

- my withdrawn B8 ("carry the plan") - wrong as stated, right in spirit;
- your correct rejection of a plan *cache* - still correct, and unaffected,
  because this is not a cache;
- the outside review's "plan is recomputed, never carried";
- the third re-derivation under its Performance section;
- the `AssertionError("sub-parent reduction plan was lost before codegen")`
  failure mode, which disappears if the stable half is carried.

I would rank this above every other consolidation item, including the cheap
tiling screen, and I think it is worth doing *in this stack* rather than as a
follow-up - it removes a hard-fail path rather than tidying one.

## Item 1 - unverified, but the *shape* of the finding is the recurring one

The claim is that the two planner prologues build their dep maps differently -
nested applies `mutation_renames`, standalone uses raw names and leans on the
`has_aliasing_or_mutation()` reject twenty lines earlier in a different
function.

I could not confirm this against a stable tree. But if it holds, note the
shape: an invariant maintained at a distance, undocumented, where the two copies
agree only because of a check neither of them makes. That is the same pattern as
`sub_parent_contiguous_lane` before it was hoisted, and the same as B5's three
size-substitution derivations. Worth checking carefully even if the extraction
itself is deferred.

## Item 3's speedup_by_fusion observation deserves a direct answer

> *"base `speedup_by_fusion` had no nested-reduction handling at all ... This
> suggests base nested reductions were silently broken (or unexercised) under
> `benchmark_fusion=1`, and mix-order may still be."*

If that is right, this stack incidentally fixed a **pre-existing** bug, and
`FusedMixOrderReductions` still has it. That is worth confirming or refuting
explicitly rather than absorbing into the stack's own story - it is either a
separate bug report or a note that mix-order is structurally unreachable there.
The reviewer flags their own low confidence; a definite answer is cheap.

## Where I would push back

- **Mix-order scaffolding consolidation** - agreed it is triplicated, agreed it
  is not this stack's job. The addendum already says so; just do not let it
  become scope.
- **Kernel-setup boilerplate (item 3 in their ranking)** - ~20 lines each with a
  genuinely different min-block policy. Lowest payoff of the four; I would skip
  it rather than parameterize a helper on the one thing that differs.

## Suggested ordering, if you agree

1. Stable-identity carry for `FusedStagedReduction` (addendum item 2).
2. Cheap shape/rate screen before tiling (already agreed, item 4 of your set).
3. Planner prologue rename asymmetry - *verify first*, then extract or comment.
4. Answer the mix-order `speedup_by_fusion` question one way or the other.

Items 3 and 4 are cheap. Item 1 is the one with real payoff.

---

# Reconciliation after old-tip verification - 2026-08-06

The preceding reviewer notes correctly identify the hybrid topology/domain
pattern, but their recommendation to make stable standalone identity the top
landing requirement was based on a moving tree and does not survive checking
the last complete tip (`24c9ce1c855`).

Two concrete facts change the conclusion:

1. Standalone `FusedStagedReduction` groups are extended through inherited
   `FusedSchedulerNode.fuse()`. Factor/output/source fields captured at first
   construction would become stale when another consumer fuses. Supporting
   stable identity requires a custom constructor/fuse/approval lifecycle that
   refreshes the topology after every fusion; it is not a mechanical field
   addition.
2. `plan_from_topology()` does not remove the hard-fail path. It can return
   `None`, and nested codegen raises the same kind of invariant error. The
   final legality proof still must rerun because `merge_loops()` refreshes
   ranges and dependencies.

Therefore stable standalone identity remains a reasonable future design, but
not a blocker or a higher-priority replacement for the current prefilter. The
current stack should retain post-`merge_loops` final planning and its typed
marker contract.

The other questions are now resolved:

- Standalone raw dependency names are intentional: `read_writes` already has
  the node-local temporal mutation version. Applying global
  `mutation_renames` can collapse versions. Add a short comment; do not share a
  helper that silently changes this policy.
- Mix-order reduction does fail under forced benchmark fusion on incompatible
  reduction groups. Track a separate pre-existing-bug follow-up and keep it out
  of this stack.
- Kernel setup extraction and mix-order/staged unification remain deferred.

Final recommendation remains: add the temporal-name comment to #190594, then
continue the already agreed per-commit fix set.

---

# Reviewer response to the reconciliation - 2026-08-06

**You are right and I withdraw the recommendation.** Both counters hold, and
there is a sharper version of the first that makes the case decisive.

## Withdrawing the stable-identity-for-standalone recommendation

I ranked it above everything else. That was wrong.

**Your counter 1 is correct, and the underlying reason is stronger than
"needs a custom lifecycle".** I traced the mechanism: identity survives fusion
via `BaseScheduling.fuse`'s else-branch, which picks
`node_type = FusedStagedReduction if staged else FusedSchedulerNode` and calls
the inherited classmethod `fuse`. `FusedStagedReduction` is deliberately
fieldless so that inherited wrap-the-leaves construction is correct.

But the deeper point is *what* the two identities describe:

- **nested** carries `grouped_reduction`, `group_size`, `grouped_axis` - these
  describe the **reduction pair**, which does not change when a pointwise
  consumer is appended. Genuinely stable.
- **standalone** would carry factor, output-group node identities, source
  buffer names - these describe the **consumers**, which is exactly what a
  subsequent fusion changes. Appending a consumer at a different rate can move
  `sub_parent_factor` (smallest wins) and change the epilogue member set.

So there is no analogous stable half to carry. The asymmetry between the two
paths is not an oversight; it reflects that the nested identity is a property
of the reduction topology and the standalone one is a property of the fused
consumer set. `plan_from_topology` has no standalone counterpart because the
premise it rests on is false there.

**Your counter 2 is also correct and I overstated.** I claimed carrying would
remove the hard-fail path. It would not: `plan_from_topology` can return `None`
and nested codegen raises the same class of invariant error. Carrying narrows
the window; it does not close it.

Net: the addendum's item 2 is a reasonable future design note, not a blocker,
and not a higher priority than the prefilter. The current stack should keep
post-`merge_loops` final planning and the typed marker contract, as you say.

## cpp_wrapper - the right resolution, and better than either proposal

Testing the supported path instead of gating it is the correct outcome. Both
earlier proposals were worse: yours (add a standalone gate) and the outside
review's (same) would have made both paths conservative about something neither
had shown to be a problem; mine (delete the nested exclusion) would have removed
a guard without evidence either.

A real cpp_wrapper compile producing correct output and one staged kernel, plus
a positive test inherited by both scheduler classes, settles it with evidence
rather than caution. This also vindicates the human's call that it "shouldn't
really matter here".

## Raw dependency names - accepted, and the explanation is the useful part

> *`read_writes` already has the node-local temporal mutation version; applying
> global `mutation_renames` can collapse versions.*

That is a substantive answer, not a deferral, and it inverts the addendum's
framing: the standalone copy is not missing a rename step, the nested copy needs
one because it operates on a different name space. Agreed that a short comment
is the right fix and that extracting a shared helper would be actively harmful -
it would hide a policy difference behind a common signature, which is how the
two copies would actually drift.

## Mix-order under benchmark fusion - confirmed, and worth filing properly

> *Mix-order reduction does fail under forced benchmark fusion on incompatible
> reduction groups.*

The addendum flagged this at low confidence; you confirmed it. That makes it a
**real pre-existing bug in landed code**, independent of this stack. Agreed it
stays out of the stack - but it should be filed as its own issue rather than
living only in this handoff, which will not survive. Include the repro you used.

## Remaining position

No open disagreements from me. The agreed set stands:

- temporal-name comment in #190594;
- the rest of the per-commit fix set already agreed;
- factor-8/16 numerics, reindex sentinel, factor cap, aliasing coverage as
  follow-ups;
- stable standalone identity and mix-order scaffolding as future design notes.

One process note: the working tree was mid-rebase and dirty while I was checking
the addendum, which is why my previous section carried an explicit
unverified-claims caveat. Verification above was done against `24c9ce1c855`, the
last complete tip.

---

# Response to `full_review.md` - 2026-08-06

Read the consolidated review and all six per-commit reviews. The review is
useful, but it is against the pre-rebase commits (`442a0a77435` through
`c37f990c53a`) and several findings are duplicated across the seven verdicts.
They are not seven independent sets of blockers. This section records what is
still worth changing in the current per-commit rebase.

No source commit was changed while writing this response.

## Findings already closed by the pending #190594 work

- The positive cpp_wrapper test is the right resolution. It performs a real
  cpp_wrapper compile, checks correct output and one staged kernel, and is
  inherited by the forced persistent and looped test classes. Do not add a
  cpp_wrapper fallback gate.
- The exact factor-2 prefilter now runs before `get_tiling_and_scores()` and
  preserves static/dynamic, persistent/looped, and pointwise-producer cases.
- `Note [Sub-parent reduction epilogues]` now documents approval planning,
  typed staged identity, post-`merge_loops` final planning, and the rule that
  codegen cannot decline.

These changes are still uncommitted at commit 1 and have passed the focused
tests plus both full test files.

## Changes worth making in this rebase

### #190594 - standalone interleaved

1. Remove `materialize_all_store_cache_values` from this commit. Its true
   branch is unreachable here and the next commit removes it. This is pure
   introduce-then-delete noise.
2. Move the four helpers' `renames` parameters and rename plumbing to #190595,
   where the nested path first supplies `mutation_renames`. Standalone deps use
   the node-local temporal names already present in `read_writes`; applying the
   graph-wide final rename there can incorrectly collapse mutation versions.
3. Fold the final single-decision structure down from #191975 if it remains a
   mechanical move. Commit 1 should not introduce separate loose-plan and leaf
   helpers only for the stack tip to consolidate them. Do not pull #191975's
   behavior changes down with it.
4. Move the existing standalone mutation-rejection test to this commit, where
   the mutation gate and its commit-message claim originate.
5. Add only a concise persistent/looped contract explanation: persistent
   codegen splits resident parent values; looped codegen reloads external
   sources in the derived domain after the reduction loop. The central Note is
   the right home; do not add another abstraction.

Keep `SubParentSourceLayout` and `_select_lane`. The enum becomes two-member in
#190596, and `_select_lane` has two final callers and centralizes the static
lane invariant.

Keep the current `num_store` repair for this rebase, but retain it as a
follow-up. The review is directionally correct that
`max(num_store, len(store_buffer_names))` is incomplete: at the MXFP6 tip the
metadata can report three stores while codegen emits four physical stores.
Changing it blindly is risky because `remove_kernel_local_buffers()` later
decrements the same counter. The root fix belongs at the store-accounting /
kernel-local-removal boundary and needs measured autotuning-metadata tests; it
should not be mixed into this already broad rebase.

### #190595 - nested append

1. Introduce append fusion directly through the ordinary scheduler/backend
   legality path. Do not introduce `_skip_fused_nested_dispatch` or the
   backend-only `can_fuse_nested_reduction_append` hook only to remove them
   later.
2. Make `fuse_with()` recompute the prospective plan when its approval is
   missing or belongs to another candidate. Raise only if that full recompute
   fails. The single-slot approval can remain an optimization rather than a
   correctness dependency.
3. Concisely document the two real contracts:
   - the write-defined-source fallback in `_sub_parent_epilogue_source_deps`;
   - `must_materialize_names` means reduced/group-constant values that may be
     broadcast into the sub-parent domain, not arbitrary values that codegen
     should force into storage.
4. Preserve the direct grouped-axis regression test. The old review's claim
   that the admitted group-X bug has no targeted test is stale on the latest
   complete tip.

A nested-append mutation test or a conservative mutation gate is worth a
focused check, but the review did not establish a reachable bug. Investigate
it while stopped at #190595; do not broaden the planner based only on the
speculation.

### #190596 - CONTIGUOUS layouts

1. Introduce the final parameterized
   `emit_split_via_reshape(..., permute_dims=...)` API directly. Do not add a
   second emitter that #191975 deletes.
2. Fix the stale `_resolve_remapped_value` reference, fold final factor-neutral
   terminology into this commit, and remove transient shape helpers that the
   final implementation does not need.
3. Correct the `min_rblock` explanation. Persistent INTERLEAVED codegen needs
   the resident row for register splitting. Looped CONTIGUOUS codegen reloads
   in the derived space and does not require one RBLOCK to contain an entire
   contiguous lane group; its factor floor is a derived-block legality/alignment
   constraint.
4. Add a real dynamic-reduction-extent CONTIGUOUS fallback test. Run multiple
   runtime D values under both forced persistent and forced looped test classes,
   assert correct output, and prove that the current static-power-of-two
   restriction declines cleanly rather than compiling a staged kernel.

Do not expand this commit to support dynamic CONTIGUOUS R. That restriction is
conservative and separate from the dynamic/non-power-of-two INTERLEAVED support
needed by NVFP4/MXFP4.

### #191775 - MXFP6 and name-level equivalence

1. Add direct tests at the new mechanism's level:
   - a non-injective / producer-alias dependency must not be nominated by
     `_nested_index_equivalent_dep_names()`;
   - a nominated `MemoryDep` may be relaxed, but unmatched `StarDep` or
     `WeakDep` entries on the same name must still block `can_fuse_vertical()`.
2. Keep the generic exact-index test in this commit, parameterize its cases
   with `@parametrize`, and describe canonical renaming/drop-unused-symbol
   equivalence accurately. Do not restore the deleted broad
   `allow_index_equivalence` behavior.
3. Document the proof obligation on `index_equivalent_dep_names`: callers may
   nominate a name only after the full nested/sub-parent planner has proven the
   replacement index relation.
4. Replace unexplained literals with concise MXFP6 context: `(4, 3)` is the
   supported input-to-output lane rate and factor 4 is the largest exercised
   interleaved factor. Keep the implementation specific rather than adding a
   generalized rate abstraction.
5. Keep rejecting unsorted output lanes instead of sorting them. Reordering the
   lane list would also reorder stores; a short comment is sufficient.

Before assigning the unsafe internal-source decline to #191975, run the graph
against #191775. If #191775 miscompiles or aborts, the fix and regression test
belong here. If #191775 is correct and #191975 merely makes the rejection
explicit/conservative, retain ownership in #191975.

### #191974 - test hygiene

Keep this one-file and dedup-only. The exact-index and relaxed-name contract
tests belong in #191775, where the generic fusion semantics change. Do not use
#191974 to hide new semantic coverage or broaden the test-helper refactor.

### #191975 - final simplification and behavior changes

1. Add direct rejection tests for multi-output CONTIGUOUS plans and for the
   unsafe source-chain reorder. These are the two undisputed untested behavior
   changes.
2. Keep the lane-uniqueness relaxation, but document why it is safe. When the
   source address is lane-invariant, all matching lanes read the same address;
   proving the exact lane selected by codegen is therefore sufficient. Do not
   restore a factor-wide uniqueness loop without a counterexample.
3. Keep the final `output_groups` representation and assert its actual codegen
   invariants: nonempty groups, strictly increasing lane counts, and no
   duplicate lane keys.
4. Inline the now-single-use contiguous emitted-lane wrapper, rename the
   shadowing comprehension variable to `group_nodes`, preserve the two useful
   materialization failure diagnostics, and tighten the
   `FusedNestedReductions` backend comment to say that the reachable direct
   backend entry is an already-validated vertical append.
5. Restore the `_sub_parent_epilogue_source_deps` `None` versus empty-tuple
   contract note, or make the return representation unambiguous. Do not leave
   two falsy meanings undocumented.

The duplicate placement probes should disappear naturally when #190595 is
rebased onto the ordinary legality path. Do not add a new wrapper solely to
remove two cheap map lookups.

## Findings to reject or defer

- **Standalone stable-plan carrier / plan cache:** reject for this stack.
  Standalone identity is derived from the consumer set, which changes as more
  consumers fuse. `merge_loops` also requires a final legality derivation.
  Nested topology is different: its reduction pair, group size, and grouped
  axis remain stable when pointwise consumers are appended.
- **Codegen fallback on a lost final plan:** reject. A staged fused node cannot
  be unfused during codegen; a missing post-merge plan is an internal compiler
  invariant failure. The correct defense is symmetric planning and tests, not
  pretending codegen can recover generic scheduling.
- **Plan memoization:** reject until profiling after `nested_reduction` is ever
  considered for default-on. The cheap prefilter removes the clearly avoidable
  tiling cost without cache invalidation state.
- **Second feature flag:** reject. The existing default-off flag intentionally
  gates staged reduction pipelines; no concrete failure requires more config
  surface.
- **Odd factor-2 extent test:** reject as stated. Equal pair-lane outputs
  require a divisible derived extent. We already exercise real dynamic and
  non-power-of-two logical sizes where divisibility is provable.
- **Dynamic persistent-R failure:** reject as a current issue. Forced
  persistent dynamic tests run multiple real values successfully.
- **Looped internal-producer compile abort:** no repro. The pointwise-producer
  test requires one kernel and runs under forced looped/static and looped/real-
  dynamic configurations. Keep the unsafe-reorder test above; do not add a
  blanket planner rejection.
- **Metric rename, assertion spelling, broad test-body dedup, multi-kernel,
  factor-8/16 numerics, factor-cap, aliasing, reindex sentinel, mixed-layout,
  and same-shaped-second-reduction coverage:** useful future work at varying
  priority, but not landing gates for the NVFP4/MXFP4 core.
- **fp8 split coverage on older GPUs:** useful focused follow-up, not a reason
  to expand this rebase.

## Revised stopping criteria

Before declaring the stack ready again:

1. Finish the pending #190594 cleanup, including dead-code relocation, then
   amend only after human review.
2. Rebase each later commit and apply the owned changes above without
   introducing temporary interfaces that the tip deletes.
3. Resolve the three investigation items: nested-append mutation behavior,
   unsafe-source test ownership, and whether final decision consolidation can
   be folded into #190594 mechanically.
4. Run both official test files independently at every commit, plus the new
   focused tests at their introducing commits.
5. Refresh the six review guides, commit-message Test Plans, private-API notes,
   SHAs, and line references. Do not update GitHub until the human has reviewed
   the exact PR text and explicitly asks for submission.

Net assessment: the review found real test ownership and transient-interface
problems, but its broad Request Changes verdict overstates cpp_wrapper,
caching, stable-plan, and looped-producer concerns. The remaining work is a
focused per-commit cleanup, not a redesign of the staged reduction model.

---

# CSE invalidation and store/load forwarding - 2026-08-06

The human correctly elevated this concern. CSE invalidation is a scope
boundary, not a cache-efficiency detail. A stale `store_cache` hit after a
loop boundary can silently use a value from the wrong RBLOCK; an over-eager
invalidation can instead force a load from a buffer whose store was removed.
Both directions need explicit coverage.

## Required lifetime contract

1. **Persistent planned source:** the reduction-stage load is still live when
   `_SubParentSourceLoadMaterializer` splits it. The lane values are registered
   on the derived family and the epilogue must consume those registers without
   a global store/reload.
2. **Looped external source:** `kernel.codegen_body()` closes the reduction
   loop and `CSE.invalidate(outside_loop_vars)` removes loop-local cache
   entries. The epilogue must use the ordinary remapped memory load. If it
   reads a previously invalidated store, `CSEProxy.load()` must mark the buffer
   in `must_keep_buffers` before emitting that load. It must never resurrect
   the old cached value.
3. **Looped internal source:** do not carry a loop-local cache entry across the
   loop boundary. Planning must safely defer/recompute the internal source
   chain after the reduction and codegen must not flush before the lane
   epilogue consumes that newly produced `store_cache` value. If the reorder is
   unsafe, decline the fusion. Keeping the stale pre-boundary value is never an
   acceptable fallback.
4. **Reduced/group-constant value:** only explicit
   `must_materialize_names` may broadcast a live post-reduction value into the
   lane domain. Standalone planning must not mark every parent buffer as
   materializable. A missing live value is an invariant failure, not a reason
   to read a removed buffer.

The implementation should continue to use the generic `CSEProxy.load()` and
`store_cache` lifecycle. Do not introduce a second name/index cache, preserve
the whole store cache across a loop boundary, or clear `invalidated_stores` to
make a stale value appear live.

## Tests required before landing

The existing standalone kernel-form test already pins cases 1 and 2 for an
external source: persistent emits one source load plus `tl.split`; looped emits
the expected reloads and no `tl.split`.

Add a dedicated internal-source kernel-form test for cases 1 and 3. Use a
realized, nontrivial pointwise producer feeding both the reduction and the
sub-parent epilogue, and a large enough reduction to execute multiple RBLOCK
iterations. Run it under both forced persistent and forced looped classes and
require:

- exact compiled-versus-reference output;
- one staged kernel;
- no global temporary store/reload for the internal source;
- the expected register split/forwarding form;
- multiple runtime shapes where the graph is dynamic.

The looped input must depend on the R coordinate so forwarding the last loop
tile would produce a detectable mismatch. Kernel count alone is insufficient:
an accidental physical temporary can preserve numerics while defeating the
intended internal forwarding contract.

The unsafe-source-chain rejection test remains required. It must prove that a
topology which cannot be deferred safely falls back before codegen, rather than
reaching a missing-cache assertion or forwarding a stale value. Run that graph
at #191775 before deciding whether the correction belongs in #191775 or
#191975.

This also strengthens the #190594 cleanup: remove broad standalone
`must_materialize_names` population in the introducing commit. Planned source
layouts plus removed/inplaced-buffer checks are the only standalone reasons to
materialize a register value; nested stages retain only their explicit
reduced/broadcast source names.

---

# Reviewer feedback on the `full_review.md` response - 2026-08-06

Agreed with your triage. "Not seven independent sets of blockers" is the right
frame - the seven verdicts duplicate heavily, and the broad Request Changes
overstates the current state given five of the six reviewed SHAs are already
superseded. Adding evidence on one item you dismissed, and three things to add.

## Evidence for your "looped internal-producer: no repro"

You rejected this on absence of a repro. I instrumented it instead, because
absence of a failing test is weaker than absence of the condition.

Wrapped `_SubParentPointwiseRemapHandler._materialize_sub_parent_load`,
replicating its guard order exactly, and counted how often execution would
reach the `AssertionError`, across the whole of `test_nested_reduction.py`:

    176  calls
     87    early-return: already in family.remapped_values
     48    early-return: not required
     41    required AND present in store_cache
      0    *** REACHES THE RAISE ***

So the abort path is never entered in the current suite. That is a stronger
statement than "no repro" and I would put it in the #190594 PR description
rather than leaving the review's medium-high-confidence claim unanswered.

Two caveats, both mine:

- My first attempt at this instrumentation reported **46** hits on the abort
  path, which was wrong - I omitted the `name in self._family.remapped_values`
  early return. The passing suite contradicted my own measurement, which is
  what caught it. Numbers above are from the corrected version.
- Zero hits across the existing suite does not prove unreachability; it proves
  no *shipped* test constructs the condition. The review's structural argument
  still stands: `cse.invalidate(self.outside_loop_vars)` at `triton.py:6690`
  and the `store_cache.get` + raise at `simd.py:2275` both exist, so the
  sequence is expressible. The honest disposition is "mechanisms present,
  unreachable across 176 observed calls, no adversarial witness constructed" -
  not "no repro".

## Three things your response does not cover

1. **`_pointwise_consumes_reduction` skips `mutation_renames`** (review, commit
   2). It compares raw `dep.name` against `reduction_buffer_names` while every
   other name comparison added in that commit applies renames. Lost fusions
   only, not correctness - but it is a concrete one-line inconsistency distinct
   from the nested-append mutation *gate* you did list as an investigation item.
   Worth folding into the #190595 work rather than losing.

2. **`_reindexed_dep_order` silently declines dynamic-shape consumers** (review,
   commit 4) - it requires static `int > 1` sizes and emits no `why()` or debug
   log, unlike the rest of the fusion machinery. Given you are otherwise
   tightening dynamic-shape support in #190594, a silent dynamic decline one
   commit later is worth at least a log line.

3. **The horizontal-gate reachability claim** (review, commit 6 item 4). You
   accept tightening the comment, which is right, but the underlying question -
   whether `FusedNestedReductions` can reach `can_fuse_horizontal` at all - is
   answerable, and the answer determines whether the early `return True` is
   dead code or a real widening. Settle it while stopped at that commit.

## Agreed without reservation

- Reject: stable-plan carrier, codegen fallback on lost plan, plan memoization,
  second flag, odd-extent test, dynamic persistent-R. I conceded the first
  already; the rest follow the same reasoning.
- Keep the lane-uniqueness relaxation *with* the argument written down. Your
  formulation - lane-invariant address means all matching lanes read the same
  address, so proving the codegen-selected lane suffices - is exactly the
  argument the review said was missing. Put it in the code, not just here.
- Test-ownership moves: mutation test to #190594, exact-index and relaxed-name
  contract tests to #191775, #191974 stays dedup-only. Correct in every case.
- Introduce final forms directly rather than introduce-then-delete: the
  parameterized `emit_split_via_reshape`, the ordinary legality path instead of
  `_skip_fused_nested_dispatch` plus the backend hook.
- Running the unsafe-source graph against #191775 before assigning ownership.
  That is the right methodology and I would apply it to item 3 above too.

## One process note

Your stopping criteria say to run both official test files at every commit.
Please also record HEAD before and after each run. The stack has been rebased
four times in two days; every per-commit number in this file is only meaningful
with the SHA it was measured at, and two of my own earlier measurements were
invalidated by the tree moving mid-run.

## State at time of writing

`HEAD=442a0a77435`, detached, three files modified, rebase paused. All numbers
above measured against that working state, not against a committed revision.

---

# Response to the latest reviewer feedback - 2026-08-06

Agreed with the corrected instrumentation result and the process note. Two of
the three added code questions are worth small changes; the mutation-name item
needs a different resolution than the proposed one-line rename.

## CSE instrumentation

The corrected `176 / 41 / 0` breakdown is useful evidence:

- 176 materialization probes;
- 87 already resolved by `family.remapped_values`;
- 48 did not require register materialization;
- 41 required values were live in `store_cache`;
- zero existing probes reached the missing-cache assertion.

This supports the current design but does not replace the adversarial tests in
the preceding CSE section. I would not put transient instrumentation counts in
the PR description unless the instrumentation script and command are retained
for reproduction. The stronger PR claim is that persistent/looped, static/real-
dynamic producer tests and the new kernel-form test exercise the lifetime
contract directly.

## `_pointwise_consumes_reduction` and mutation names

Do not apply `outer_node.scheduler.mutation_renames` as a one-line fix.
`SchedulerNode.read_writes` was already renamed with the mutation version
visible when that node was constructed, and `grouped_reduction.get_buffer_names()`
names the corresponding produced version. The scheduler-global map includes
later mutations and can collapse two temporal versions that must remain
distinct. This is the same policy already recorded for standalone planning.

The apparent inconsistency with `_plan_nested_sub_parent_stage` is intentional:
that later function compares names across the fully assembled fused group and
therefore canonicalizes through the group's rename view. The local
producer/consumer classification compares node-local temporal views.

Action for #190595: add a concise temporal-name comment at
`_pointwise_consumes_reduction` and resolve the already-listed nested mutation
investigation with a focused test. Change the naming policy only if that test
provides a concrete missed-fusion witness and the proposed rename preserves
distinct temporal versions.

## Dynamic `_reindexed_dep_order`

Agreed that the static-only decline is currently silent. The algorithm uses
concrete interval products to partition read dimensions, so accepting symbolic
sizes would require a different proof; do not weaken it merely to support a
dynamic case.

Action for #191775: state the static-size requirement in the docstring and emit
a `loop_ordering_log.debug` message when symbolic/non-unit sizes cause the
decline. Preserve the correct dynamic fallback already observed for MXFP6. This
is diagnostic clarity, not a commitment to dynamic consumer reindexing in this
stack.

## `FusedNestedReductions` horizontal reachability

Settled against `24c9ce1c855`:

- Public fusion queries enter `_can_fuse_impl()`.
- A `FusedNestedReductions` operand is intercepted there and delegated to
  `FusedNestedReductions.can_fuse_with()` before generic vertical/horizontal
  selection.
- The direct `_can_fuse()` re-entry is from
  `_can_fuse_nested_reduction_append()` and represents the already-classified
  vertical append.
- Therefore the early `SIMDScheduling.can_fuse()` return for
  `FusedNestedReductions` is live for that vertical re-entry, but is not
  reachable from the ordinary horizontal branch.

When #190595 folds away the transient hook, preserve this invariant in the
ordinary legality rewrite. The backend comment should say that its reachable
caller is an already-validated vertical append. No horizontal acceptance test
is needed unless the rewrite creates a new direct backend entry that bypasses
`_can_fuse_impl()`.

## Verification bookkeeping

Agreed: record `git rev-parse HEAD` before and after each per-commit test run,
along with the command and pass/skip counts. Discard any result if the SHA or
working-tree diff changes during the run.

---

# #190594 implementation checkpoint - 2026-08-06

The agreed first-commit fixes are implemented as an uncommitted diff on
`442a0a774355f6f6ee1401f901d5904cda0c7ded`. Nothing has been amended.

Implemented:

1. Added the factor-2 candidate screen before tiling analysis.
2. Added the central planner/codegen and persistent/looped CSE contract note.
3. Folded the final single `_sub_parent_epilogue_decision` structure into this
   commit, with the three outcomes documented at the decision point.
4. Removed the unreachable `materialize_all_store_cache_values` and broad
   `must_materialize_names` plumbing. Planned sources and removed/inplaced
   buffers remain the only forced materializations.
5. Removed unused global-rename plumbing from the four standalone helpers.
   Their node-local `read_writes` names preserve temporal mutation versions.
6. Added positive cpp-wrapper coverage and moved mutation rejection here.
7. Added an output-level reduced-sibling-source regression. During final diff
   review this caught a real consolidation mistake: acceptance had checked only
   source names used by the epilogue. The final decision preserves the original
   leaf-first contract and checks every dependency kind from the parent
   reduction before acceptance.
8. Updated `pr190594_interleaved_subparent.md` for the decision flow, direct
   cpp-wrapper coverage, current line references, stats and test count.

Verification on the unchanged SHA:

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py
240 passed, 1 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py -k standalone_sub_parent
44 passed, 1 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_inductor_scheduler.py
52 passed, 4 skipped

python -m py_compile torch/_inductor/scheduler.py \
  torch/_inductor/codegen/simd.py test/inductor/test_nested_reduction.py
git diff --check
passed
```

`spin quicklint` could not initialize `lintrunner@0.12.7`: its internal `uvx`
command exited 2 before linting began. No `.venv` exists in the repo or parent,
so this remains an environment blocker rather than a reported lint failure.

Ready for review of the uncommitted diff. In particular, please check the
single-decision equivalence and the narrowed materialization/CSE contract before
this commit is amended and the rebase advances to #190595.

---

# Review of the #190594 uncommitted diff - 2026-08-06

Reviewed at `HEAD=442a0a77435`, diff hash `13d90fdd36b0a3323a2b3a5cac9abbb7a5c6100e`
(+147/-96 across three files). Rebase paused, nothing amended by me.

You asked specifically for the single-decision equivalence and the narrowed
materialization/CSE contract. One finding on the first, clean on the second.

## F6 - `parent_source_names` is broader than the planner's set, and broader
## than your own description

`_sub_parent_epilogue_decision` builds:

```python
parent_source_names = OrderedSet(
    dep.name
    for node in nodes
    if node.is_reduction()
    for dep in node.read_writes.reads          # <- no isinstance filter
)
```

The planner passes `OrderedSet(reduction_reads)` to the same function, and
`reduction_reads` **is** filtered:

```python
for dep in node.read_writes.reads:
    if isinstance(dep, MemoryDep):             # scheduler.py:647-649
        reduction_reads[dep.name].append(dep)
```

So the decision's set additionally contains `StarDep`/`WeakDep` names. Your
checkpoint item 7 says the fix "checks every `MemoryDep` source of the parent
reduction" - the code checks every *dep* source.

Effect is one-directional: `_sub_parent_siblings_are_source_free` returns
`False` for more inputs, so `leaf_violation` fires more often. Not a
correctness bug - strictly more conservative - but:

1. it is asymmetric with the planner, which is the drift class this
   consolidation exists to remove. A pair the planner would accept under
   `check_leaves=True` can now be leaf-rejected by the decision because a
   sibling reads a buffer the reduction only touches via a `StarDep`;
2. the divergence is undocumented and contradicts your own summary, so the next
   reader cannot tell which set is intended.

**Ask:** add `if isinstance(dep, MemoryDep)` to match the planner, or state in a
comment why the decision deliberately widens to indirect/ordering deps. If the
widening is intended, it belongs in the planner too - one definition, as with
`sub_parent_extent_subs`.

## Verified clean

- **Item 4, narrowed materialization.** `materialize_all_store_cache_values`:
  zero occurrences. `must_materialize_names`: gone from both
  `scheduler.py` and `simd.py`. Planned source layouts plus removed/inplaced
  buffers are now the only forced materializations, which is exactly the
  contract in your CSE section.
- **Item 3, single decision.** The three outcomes are derived from one
  `check_leaves=False` plan and are distinguishable at the return sites
  (`(True, False)` leaf-reject, `(False, planned)` accept, `(False, False)`
  generic). The `FusedNestedReductions` carve-out on the `plan is None` branch
  is correct - the append path owns its own stage.
- **Item 7's catch.** Good find during your own diff review; the class of bug
  (acceptance narrowed to epilogue-used names only) is precisely what a
  consolidation is most likely to introduce. F6 is the residue of that fix, not
  a dispute with it.

## Accepted from your previous response

- **Instrumentation in the PR description** - you are right and I withdraw it.
  A count nobody can reproduce is weak evidence; the reproducible claim is that
  persistent/looped, static/dynamic producer tests plus the new kernel-form test
  exercise the lifetime contract. Keep the counts here as review evidence only.
- **`_pointwise_consumes_reduction`** - accepted, do not apply the global rename.
  Node-local `read_writes` already carries the temporal version and the
  scheduler-global map can collapse two versions that must stay distinct. The
  asymmetry with `_plan_nested_sub_parent_stage` is a name-space difference, not
  an inconsistency. A comment plus the focused mutation test is the right
  resolution.
- **Horizontal reachability** - settled, and your trace matches what I found:
  the early return is live only for the vertical append re-entry. Preserving
  that invariant through the #190595 rewrite is the thing to watch.
- **`_reindexed_dep_order`** - agreed, docstring plus a debug log, not weakened
  proofs.

## Verification note

I did not re-run the suites on this diff; your 240/1, 44/1 and 52/4 numbers are
the only evidence, and they were taken on this same working state. Once F6 is
resolved, re-run both files and record HEAD before and after, per the
bookkeeping we agreed.

---

# Response to F6 - 2026-08-06

Keep the broad set and document it. The asymmetry is inherited from the original
ordering, not introduced by the consolidation:

1. `_sub_parent_epilogue_leaf_violation` ran first and built source names from
   every reduction dependency, including `StarDep` and `WeakDep`.
2. Only after that passed could `_can_fuse_sub_parent_reduction_epilogue` call
   the `MemoryDep`-only planner and accept.

Filtering the combined decision to `MemoryDep` would therefore weaken the old
leaf gate. The independent codegen reviewer found exactly this issue in the
intermediate diff. The final code now evaluates the broad leaf conditions
before returning either acceptance or generic fallback, and has a two-line
comment explaining why the source set is intentionally broader than planner
matching.

The independent re-review reports no remaining correctness issue in either the
single decision or the narrowed materialization paths. Final fixed-diff tests
passed after the explanatory comment:

```text
HEAD before/after: 442a0a774355f6f6ee1401f901d5904cda0c7ded
tracked diff before/after:
69a3ae885589976d809fae9759a86e747a428f401562e37a454cd1746ebc6ad1

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py
240 passed, 1 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_inductor_scheduler.py
52 passed, 4 skipped
```

#190594 is ready to amend once the repository's lint environment is available
or the `spin quicklint` initialization failure is explicitly waived.

---

# #190595 implementation checkpoint - 2026-08-06

The nested interleaved commit now carries the final ordinary fusion-legality
shape directly rather than introducing an append-only backend hook that a later
commit deletes.

Implemented:

1. Nested SUB_PARENT appends re-enter the complete scheduler `_can_fuse`
   ladder with the composite node as producer. Placement, device,
   `no_fuse_buffer_names`, resource, choice and backend gates all run.
2. Reverse-discovered pairs are stored producer-first for scheduler
   stream/mempool bookkeeping.
3. CUDA and XPU use their ordinary backend ladders; the transient public
   `can_fuse_nested_reduction_append` hook is absent from this commit.
4. `can_fuse_with()` retains the final topology approval. `fuse_with()` uses it
   when it still names the same consumer and rebuilds the topology plan when a
   later fusion query displaced the slot.
5. Mutation renames exist only in the nested planner, where the fused group can
   contain temporal buffer versions. The standalone planner remains on raw
   node-local names.
6. Forced materialization is limited to nested reduced/group-constant values;
   standalone stages do not regain the removed broad parent-buffer set.
7. A focused mutation test proves the RMSNorm/grouped-reduction kernel still
   forms while the mutating SUB_PARENT consumer remains a second kernel.

Independent review found no remaining correctness or material design issue in
the legality, ordering, approval, rename, materialization or mutation paths.

Verification on an unchanged source diff:

```text
HEAD before/after: cda3616551f2edef56b0036dfe894ebd8df9f532
tracked diff before/after:
962f46454fff4ccbd64d909041241ed80cba2f31557c4efdd2d6e00c5728ce96

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py
292 passed, 2 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_inductor_scheduler.py
54 passed, 4 skipped

python -m py_compile ...
git diff --check
passed
```

`lintrunner -a` reached PYREFLY but failed before linting because its isolated
environment could not fetch `types-colorama` from PyPI through the network
tunnel. It applied no changes; the tracked diff hash remained identical.

---

# #190596 implementation checkpoint - 2026-08-06

The CONTIGUOUS commit now owns its final matcher and emitter instead of adding
interfaces that #191975 replaces.

Implemented:

1. INTERLEAVED and CONTIGUOUS call one `emit_split_via_reshape`; an optional
   permutation is the only layout-specific branch.
2. The CONTIGUOUS planner computes the exact lane codegen selects and proves
   that lane's index. It no longer builds every candidate lane and requires a
   unique match. Multiple structural matches are safe when they alias the same
   address because they load the same value.
3. The single-lane substitution and exact-read helper are shared with the
   same-rank interleaved proof.
4. The factor-32 negative test pins the upper bound; mixed factors remain a
   deliberate two-kernel case and log omitted candidates.
5. The factor-2 cheap screen introduced in #190594 was widened to the bounded
   factor classifier. Focused testing caught this during the rebase: without
   it, factor-4/8/16 candidates were rejected before the widened planner ran.
6. The stale `_resolve_remapped_value` reference now names
   `_DerivedIterationFamily.resolve_load`.

Independent review found the lane proof, factor cap, mixed-factor behavior,
recursive split ordering, fp8 bitcast placement and CSE-allocated intermediate
names sound. Factor-greater-than-2 fp8 remains a useful cross-product test but
is not exercised by the current workloads and is not a blocker.

Verification on an unchanged source diff:

```text
HEAD before/after: 587e9c0c1936f918a718073784742e3bd1c6802e
tracked diff before/after:
fce9ebccb43f55716f415f399832c6dbe6eed7bba7ffd052b1dfc1dedb96fef3

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py -k chunk
28 passed, 2 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py -k standalone_sub_parent
50 passed, 1 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py
336 passed, 4 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_inductor_scheduler.py
54 passed, 4 skipped
```

---

# Reviewer: F6 withdrawn, F4 confirmed - 2026-08-06

## F6 - withdrawn. You are right, I compared against the wrong baseline.

Verified at `backup/c1-190594:2606`, the original leaf gate:

```python
parent_source_names = OrderedSet(
    renames.get(dep.name, dep.name)
    for node in nodes
    if node.is_reduction()
    for dep in node.read_writes.reads        # no isinstance filter, then or now
)
```

So the broad set is **inherited**, not introduced by the consolidation, and
filtering it to `MemoryDep` would have weakened the pre-existing leaf gate -
which is exactly the regression the third reviewer caught in your intermediate
diff.

My error was the comparison, not the observation. I checked the decision's set
against the *planner's* set and concluded they should match. The right baseline
is the gate the decision replaced. The two sets legitimately differ because they
do different jobs: the leaf gate is a conservative rejection check over every
dependency, the planner is an index proof that only `MemoryDep`s can satisfy.
That distinction is worth exactly the two-line comment you added; with it, F6 is
not a finding.

Noting the pattern for my own account: this is the second time this week I have
flagged an asymmetry as drift when it was a deliberate difference in name space
or role - the first being standalone raw names vs nested `mutation_renames`.
Both times the resolution was a comment, not a code change. Worth me checking
"what did this replace" before "what does its neighbour do".

## F4 - confirmed fixed

```python
if approval is not None and approval[0] is other:
    _, grouped_node, stage = approval
else:
    append = self._plan_append(other)
    if append is None:
        raise AssertionError("nested append plan was lost before fusion")
    grouped_node, stage = append
```

A displaced slot rebuilds; the raise fires only when the rebuild itself fails.
That is precisely the ask, and it returns the `speedup_by_fusion` bypass to
being an optimization rather than something correctness depends on. The
single-slot approval is now safe against speculative `can_fuse_with` calls
overwriting it.

## #190595 checkpoint - no findings

Items 1-7 all address prior review points, and three are things I or the outside
reviewers raised: the transient backend hook is absent rather than
introduce-then-deleted (churn finding), producer-first ordering is explicit
(B9), and forced materialization stays narrow (F2). The mutation test resolving
the open investigation - kernel still forms, mutating SUB_PARENT consumer stays
a second kernel - is the right shape: it pins behaviour without broadening the
planner on speculation.

## Open, and the only thing I would still hold on

`#190594 is ready to amend once the repository's lint environment is available
or the spin quicklint initialization failure is explicitly waived.`

The waiver is reasonable - `lintrunner`/`pyrefly` cannot fetch dependencies
through the tunnel, which is environmental. But do not let "lint did not run"
stand unqualified again: earlier in this review it concealed a real
`docstring_linter` failure on `NestedReduction.plan`. The two adapters run
standalone in about a second and need no network:

    python -m ruff check --isolated --select F,E9 <files>
    python tools/linter/adapters/docstring_linter.py <files>

Run those, state the result, and then waive `spin` explicitly. That distinguishes
"the runner is broken" from "the code is clean", which is the distinction that
bit us before.

## State

`HEAD=587e9c0c193`, detached, four files modified, rebase in flight. I did not
re-run suites on this state; your `240/1` and `52/4` for #190594 and `292/2` and
`54/4` for #190595 were both taken with HEAD and diff hash recorded before and
after, which is the bookkeeping we agreed.

---

# Response to lint hold - 2026-08-06

Agreed that the first generic runner failure was not enough. The repository
instructions require lint commands through `spin`/`lintrunner`, so I did not run
the suggested adapters directly. A second `lintrunner -a` on #190596 found a
real Ruff F811: that commit reintroduced a mutation test already moved into
#190594, so Python shadowed the earlier method. The duplicate added by #190596
was deleted and the retained test passed in both inherited modes:

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py \
  -k standalone_sub_parent_rejects_mutation
2 passed
```

After that fix, `lintrunner -a` reported no Ruff or docstring finding. PYREFLY
alone failed while fetching `types-protobuf` through the network tunnel. The
runner applied no further patch.

The 336/4 full nested-reduction run occurred before deleting the duplicate, but
the deleted definition was dead: the later method with the same name was the
one Python installed and the one the full run executed. Production code and the
effective test suite are unchanged by its removal; the targeted rerun above
pins the retained method after cleanup.

---

# Response to `full_review.md` - 2026-08-06

## Baseline warning

The review is useful, but it is against the old six-commit stack
(`442a0a77435` through `c37f990c53a`), not the current rebased commits
(`cda3616551f`, `587e9c0c193`, `c8aec40928b`, plus the in-progress #191775
resolution). It should not be applied as a current blocking list verbatim.

The structural findings that motivated the rebase have already been addressed:

- the later simplifications were moved into the commits that introduce the
  relevant machinery;
- the public backend append hook and second split emitter no longer appear as
  introduce-then-delete churn;
- standalone and nested codegen share `StagedReductionPlan`,
  `SubParentEpilogueStage`, the lane/index proofs, derived iteration families,
  the pointwise emitter, and finalization;
- `FusedNestedReductions` is a `FusedStagedReduction`, while retaining the
  nested topology it actually needs;
- the accept/leaf decision is consolidated, and the intentionally broader leaf
  source-name set is documented;
- the central `Note [Sub-parent reduction epilogues]` now documents fusion,
  `merge_loops`, final planning, and persistent-versus-looped source handling;
- the tiling probe has a cheap shape/rate screen in front of it;
- dynamic interleaved R extents are supported and exercised with multiple
  runtime values, including non-power-of-two extents;
- cpp-wrapper standalone codegen is exercised by a positive compile/numeric
  test instead of being disabled;
- the #191775 working diff restores direct coverage for the named-dependency
  relaxation, StarDep/WeakDep preservation, exact-index matching, and
  producer-write injectivity;
- `output_groups` has been folded into #191775, validates ordered unique groups,
  and is consumed consistently by reindexing and codegen;
- the nested append approval is consumed when applicable and rebuilt if a
  speculative query displaced it, so it is not correctness-dependent temporal
  state.

## Changes worth making

1. **Filter the named-dependency collector to `MemoryDep` consumers.**
   `_producer_output_names_read_by_consumer` currently proves producer writes
   are dense/injective, but its final consumer-name comprehension accepts any
   dependency type. Vertical legality remains safe because
   `can_fuse_vertical` only drops `MemoryDep`s, but the same names feed
   `_score_fusion_memory_by_fusable_read_write`, where a StarDep/WeakDep can be
   scored by name. Add `isinstance(dep, MemoryDep)` to the collector. This is a
   small precision fix, not a new planning mechanism.

2. **Finish the unsafe internal-source experiment before assigning it to
   #191975.** The current #191775 code still has the best-effort codegen reorder:
   if a leading node depends on a deferred source-chain node, it keeps the
   original order. Run the review's unsafe topology against #191775 itself.
   If it aborts or computes incorrectly, the legality fix and regression test
   belong in #191775. If #191775 is correct and #191975 only chooses a more
   conservative decline, keep the behavior change and its fallback/numeric test
   in #191975.

3. **Extend the direct grouped-axis test in #190595 to cover X.** The current
   unit test pins the R-axis SUB_PARENT domain. Add the corresponding X-axis
   assertion that `sub_parent_domain is None`; the end-to-end rejection can pass
   through a later gate and does not isolate the contract claimed by that PR.

4. **If #191975 retains the multi-output CONTIGUOUS rejection, test it there.**
   This is a real behavior restriction introduced by the later simplification.
   One numeric fallback test is enough. Do not add a generalized matrix of
   unsupported layouts.

5. **Correct two comments while touching their code.**
   `_nested_index_equivalent_dep_names` currently says cpp-wrapper cannot emit a
   standalone sub-parent epilogue, which is contradicted by the positive
   cpp-wrapper test; the intended statement is only about this scheduler-side
   nested/index-equivalence path. Also, `_sub_parent_epilogue_source_deps` says a
   source must be a reduction read while the implementation deliberately falls
   back to a full-resolution fused write. Mention the write-defined internal
   source case.

The current checkout also still records `torch/_inductor/scheduler.py` as an
unmerged file and contains a duplicated `if not isinstance(node,
SchedulerNode):` line in the working resolution. Resolve/stage that before any
verification; it is rebase state, not a design finding from the review.

## Changes not worth making in this stack

- **Do not add plan caching now.** The feature is off by default, the enabled
  path now has a cheap screen before tiling, and the expensive work only reaches
  narrow reduction/pointwise candidates. Revisit only with a compile-time
  profile or before enabling nested reduction by default.
- **Do not carry the fusion-time plan as the final codegen plan.**
  `merge_loops()` intentionally changes ranges. A stored factor/node/source-name
  identity would still require rebuilding and re-proving the lane/index and
  source-layout facts, so it would add a second plan representation without
  removing final planning. The staged node identity plus the central Note is the
  agreed contract. The unsafe-source test above is the concrete way to validate
  the remaining risk.
- **Do not extract the two dep-map prologues merely to remove about 17 lines.**
  Standalone rejects aliasing/mutation and uses node-local names; nested planning
  handles temporal mutation names. A helper would need mode flags or callbacks
  and obscure the meaningful difference. A concise comment is preferable.
- **Do not extract a parameter-heavy kernel-setup helper.** Nested and standalone
  paths intentionally have different block-floor and stage policies; the shared
  derived-family/emitter/finalization machinery is the useful reuse boundary.
- **Do not add a second config flag.** The machinery is experimental,
  default-off, and one escape hatch is sufficient while it is developed.
- **Do not rename the existing nested-reduction metric in this stack.** It is
  already the test/telemetry signal for this family of staged reduction kernels;
  renaming it is unrelated churn.
- **Do not add broad internal-hook tests for reindexing, every rejection branch,
  every factor/layout cross-product, multi-kernel mode, or a separate aliasing
  case.** Prefer numeric and generated-kernel tests for reachable workloads.
  Mutation, dynamic/static, looped/persistent, fp8, factor bounds, mixed factors,
  same-shaped independent reductions, benchmark decline, and combo decline
  already cover the important boundaries.
- **Do not fold mix-order reductions into this abstraction.** Its workspace-split
  codegen is materially different and outside this stack.
- **Do not preserve private backend hook aliases solely for hypothetical
  out-of-tree subclasses.** These are private Inductor internals and the rename
  is part of introducing the staged abstraction.

## Per-commit interpretation

- **#190594:** the old looped-materialization, dynamic-R, mutation, fp8,
  same-shaped-reduction, cpp-wrapper, and decision-duplication concerns are
  covered or structurally resolved. The `num_store` clamp is still inelegant,
  but changing common store accounting without a demonstrated metadata bug is
  higher risk than leaving the localized correction. No cache work.
- **#190595:** the backend-hook/reversed-order/mutation concerns are resolved.
  Add only the direct X-axis domain assertion above. Test-plan vocabulary is
  already `sub_parent`, not the obsolete `parent_half` name.
- **#190596:** the stale code reference, half terminology, split-emitter churn,
  factor-32 bound, mixed-factor logging, and lane-uniqueness explanation are
  resolved. A dynamic-R CONTIGUOUS fallback test would be defensible but is not
  important for the NVFP4/MXFP4/MXFP6 scope and should not block this stack.
- **#191775:** producer injectivity and named-dependency contracts now have
  direct tests; factor/rate comments and ordered output groups are explicit.
  Apply the MemoryDep collector filter and settle the unsafe source-chain case.
- **#191974/#191975:** keep these small. The exact-index/direct relaxation tests
  should live with #191775 so no intermediate commit loses coverage. #191975
  should contain only remaining simplification that cannot be folded earlier,
  with tests for any behavior restriction it retains.

## Bottom line

The review correctly identified the old stack's churn and missing direct
coverage, and those findings helped produce the current structure. It does not
justify another broad redesign. After the five narrow items above, the remaining
recommendations are either already fixed, speculative compile-time work, or
lower-value test/comment cleanup outside the kernels this stack is intended to
land.

---

# Superseded six-commit checkpoint - 2026-08-06

This section records the pre-fold state. The current five-commit stack is
listed at the top of this file and in the 2026-08-10 ownership-fold section.

The six-commit rebase is complete on `wip_quant`:

```text
cda3616551f  #190594  standalone interleaved
b75bc3146b8  #190595  nested interleaved
8d5aa3a1294  #190596  contiguous
2d6188e913e  #191775  MXFP6 staged packing
bff14fab6f9  #191974  test hygiene
8584958a5c8  #191975  small source cleanup
```

All `Pull-Request` and `ghstack-source-id` trailers are preserved. No ghstack,
push, PR edit, or GitHub comment was performed.

Resolved review items:

- #190595 now directly asserts that grouped-axis X has no sub-parent domain.
- #191775 filters name relaxation and scoring to `MemoryDep` and requires
  dense, injective, synchronization-free producer writes.
- `SubParentEpilogueStage.output_groups` and strict ordering land with MXFP6.
- The safe internal-source full-resolution fork remains one correct staged
  kernel in looped and persistent modes. The old blanket planner decline was
  dropped.
- An unplanned reduced sibling source no longer blocks an otherwise valid
  staged fusion.
- #191975 is reduced to +16/-39: redundant tile-shape checks, explicit
  materialization state, and equivalent source-lookup cleanup. It does not
  change fusion decisions, source ordering, or Triton emission.

Final verification:

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py
  356 passed, 5 skipped

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_inductor_scheduler.py
  76 passed, 4 skipped

python test/inductor/test_nested_reduction.py -k sub_parent
  80 passed, 1 skipped

python test/inductor/test_nested_reduction.py -k mxfp6
  20 passed, 1 skipped
```

`python -m py_compile` and `git diff --check` passed. `lintrunner -a`
reported no file lint or applied patch; Pyrefly alone could not fetch its type
packages through the network tunnel.

The #191975 ghstack commit message and current PR description still describe
the old large diff. Before submission, update the PR description to match
`agent_space/pr191975_simplification.md`; do not reuse the old source-order,
output-group, matcher, emitter, or multi-output CONTIGUOUS claims.

---

# Current-tip summary of the `outside_full_review.md` addendum - 2026-08-07

The addendum was written against `c37f990c53a`, so its line references and some
of its structural observations predate the completed rebase. Its main inventory
is still correct: standalone and nested sub-parent paths share the lane/index
proofs, leaf checks, derived-iteration families, pointwise emitter, and kernel
finalization. There is no second full nested-reduction implementation left to
merge.

I do not recommend further consolidation in this stack:

1. **Do not add stable standalone topology fields to
   `FusedStagedReduction`.** Nested topology is stable when pointwise consumers
   are appended; standalone factor, output groups, and source names describe
   those consumers and therefore change as the fused group grows. Carrying them
   would require a custom fuse/approval lifecycle, and codegen would still need
   to rebuild mutable dependencies and lane proofs after `merge_loops()`. It
   would narrow rediscovery, not remove the final invariant check.
2. **Keep the two dependency-map prologues.** Standalone dependencies already
   contain the temporal mutation version visible at each node, while nested
   planning deliberately applies the scheduler rename map to its different
   input namespace. A common helper would need a rename-policy mode and selected
   reduction inputs, hiding the semantic difference for little line reduction.
   The current `_pointwise_consumes_reduction` comment records the node-local
   temporal-name rule.
3. **Do not extract the kernel-choice setup.** Nested codegen uses scored tiling
   and constrains X or R according to the grouped axis. Standalone codegen uses
   the already-approved two-dimensional tiling and constrains only R for lane
   legality. Parameterizing a helper around those differences would obscure the
   policy that reviewers need to see at each entry point.
4. **Keep mix-order reduction separate.** Its workspace-split execution is not a
   sub-parent derived-domain stage. The forced `benchmark_fusion=True` failure
   previously confirmed for incompatible mix-order groups is a pre-existing,
   separate issue; it should not expand this stack. No GitHub issue or comment
   has been created from this review.

The addendum's concern about `_append_approval` is already resolved: a matching
approval is reused, while a missing or displaced approval rebuilds the append
plan. The cheap candidate screen before tiling and the central
`Note [Sub-parent reduction epilogues]` are also present at the current tip.

Bottom line: the addendum validates the chosen sharing boundary but does not
identify another required refactor. Preserve the current five-commit structure
and treat standalone topology retention and mix-order benchmark handling as
future work only if profiling or a dedicated bug fix justifies them.

---

# Current-tip response to the `full_review.md` addendum - 2026-08-07

The addendum was written against `c37f990c53a`; the current tip is
`f237850e504`. Its inventory of shared machinery is accurate, but its ranked
consolidation list does not justify another rebase.

## Worth preserving

- Keep the current common representation: `StagedReductionPlan`,
  `SubParentEpilogueStage`, `FusedStagedReduction`, and the shared lane/index,
  leaf, derived-family, pointwise-emission, and finalization code are the useful
  reuse boundary.
- Keep `FusedNestedReductions` as the topology-bearing subclass. Its grouped
  reduction, axis, and group size stay stable as consumers are appended, so
  `plan_from_topology` can reuse them after `merge_loops()`.
- Keep the matching-approval fast path plus rebuild in `fuse_with()`. A
  displaced `_append_approval` no longer creates a correctness dependency: the
  prospective append plan is rebuilt before the fusion is committed.
- Keep the cheap sub-parent shape/rate screen before tiling and the central
  `Note [Sub-parent reduction epilogues]`; both requests from the main review
  are present at the final tip.

## Not worth changing in this stack

1. **Do not add stable standalone topology fields.** A standalone staged group
   can acquire more consumers, changing factor selection, output groups, and
   source layouts. `merge_loops()` can also rewrite the ranges used by the lane
   proof. Retaining partial identity would require an append lifecycle for the
   standalone node while codegen would still have to rebuild and validate the
   final plan. The empty `FusedStagedReduction` marker is intentional.
2. **Do not merge the dependency-map prologues.** Standalone planning consumes
   the temporal names already present in each node's `read_writes`; nested
   planning operates in the outer scheduler namespace and must apply
   `mutation_renames`. `_pointwise_consumes_reduction` now documents this
   distinction. A shared helper would need a rename-policy mode and would hide
   the important invariant for about 17 lines of apparent deduplication.
3. **Do not extract the kernel-choice setup.** The two entrypoints share syntax,
   not policy. Nested codegen scores the outer schedule and constrains X or R
   according to the grouped axis. Standalone codegen uses the fusion-approved
   two-dimensional tiling and applies only the R-block floor required by lane
   legality. A parameter-heavy helper would make those decisions harder to
   review.
4. **Do not fold mix-order reduction into staged codegen.** Its workspace-backed
   split execution is different from an in-kernel derived domain. Any generic
   typed-node/benchmark scaffolding cleanup should be a separate change driven
   by the independently reproduced mix-order benchmark-fusion bug.

## Append-policy scope

The new SUB_PARENT append path passes the complete `FusedNestedReductions` node
through `_can_fuse`, so placement, no-fuse, resource, vertical-legality, choices,
and backend gates apply to the NVFP4/MXFP4 path introduced here. Reduced and
PARENT_FULL appends still run those inner checks against the grouped stage, as
they did at the base commit; consequently their resource heuristics do not
count the outer stage. That is inherited behavior, not a regression in this
stack, and broadening it is outside the sub-parent change. If addressed, it
should be a focused scheduler fix with a non-sub-parent `max_fusion_size` test.

## Disposition

No source change follows from the addendum. The remaining suggestions are
future cleanup or pre-existing adjacent behavior, not blockers for the six
current commits.

---

# Final fusion and CSE audit - 2026-08-07

No new reachable correctness issue was found.

- Nested sub-parent appends remain producer-first and re-enter the ordinary
  scheduler checks for placement, `no_fuse_buffer_names`, device equality,
  resource policy, vertical legality, choices, and the SIMD backend gate.
- Closing a loop invalidates loop-local CSE values, store-cache entries, and
  load-index basis state. Looped standalone codegen therefore reloads external
  sources in the derived domain. Internally produced sources stay in one live
  scope; plans that would carry such a source through another reduction are
  rejected.
- Persistent codegen keeps the parent tile live and materializes planned lanes
  from that register value. The exact load and split counts are pinned by the
  looped/persistent kernel-form tests.
- The inherited reduced/PARENT_FULL nested append path still evaluates resource
  heuristics against the grouped stage rather than the complete nested node.
  The SUB_PARENT path in this stack uses the complete node. Treat the inherited
  accounting gap as a separate scheduler follow-up, not a blocker here.

Focused validation:

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py \
  -k dynamic_sub_parent_epilogue \
  -k producer_consumer_sub_parent_intermediate \
  -k mxfp6_internal_source_full_resolution_fork \
  -k sub_parent_append_respects_fusion_gate \
  -k standalone_sub_parent_epilogue_kernel_form \
  -k producer_consumer_amax_kernel_form
```

Result: 16 passed.

## Test-strengthening follow-up - complete

The test-only working-tree diff now closes the three audit gaps:

1. The internal-source MXFP6 kernel-form test covers forced looped and
   persistent modes. It pins input loads (2/1), three output pointers, five
   emitted stores, metadata loads (2/1), metadata stores (3), three `tl.split`
   operations, and the expected RBLOCK floors. This directly checks that the
   internal source remains register-forwarded.
2. The simple MXFP6 topology now runs two actual dynamic batch sizes and
   requires staged fusion. The existing preshuffled dynamic test also states
   its intended one-nested-kernel contract explicitly.
3. Both nested output-reader rejection tests now require the base nested
   reduction to fuse while the unsafe sub-parent append falls back.

Validation:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py \
  -k mxfp6_internal_source \
  -k dynamic_batch_mxfp6 \
  -k rejects_sub_parent_output_fullres_reader \
  -k rejects_sub_parent_output_reduction_reader
```

Result: 12 passed.

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python test/inductor/test_nested_reduction.py
```

Result: 360 passed, 5 skipped.

---

# #191975 ownership fold - 2026-08-10

The former top-of-stack #191975 cleanup was folded into the commits that
introduced each mechanism:

- #190594 `701ac99270f`: remove redundant tile-shape helpers and use
  `_GroupedReductionLayout.parent_dim()` directly;
- #190595 `64e905eb763`: require the already-passed
  `must_materialize_names` state explicitly;
- #191775 `07297ce24ca`: simplify full-resolution internal-source dependency
  lookup.

#191975 is no longer a local stack commit. #191974 `f237850e504` is the new
tip. All five surviving `ghstack-source-id` and `Pull-Request` trailers are
unchanged.

The new tip and the former #191975 tip both have tree object
`a895c385f0ad14952329c479a0d7412e37b95e7b`, so the fold changes only commit
ownership. Focused validation at the affected intermediate states passed:

```text
#190594 standalone sub-parent: 20 passed
#190595 nested sub-parent:     14 passed
#191775 MXFP6:                  8 passed
```

The mandatory `lintrunner -a` retry was blocked by network-only TESTOWNERS and
Pyrefly setup failures. The identical final tree had already passed
`lintrunner -a`, the full 360-test nested-reduction suite, `py_compile`, and
`git diff --check`. No ghstack, push, PR edit, or GitHub comment was performed.

---

# Local dynamic-R follow-up - 2026-08-10

Status: `LOCAL_PROTOTYPE_READY_FOR_REVIEW`

This is an uncommitted working-tree diff on top of the submitted five-PR tip
`f237850e504`. No commit, amend, ghstack, push, PR edit, or GitHub comment was
performed.

Modified files:

```text
torch/_inductor/scheduler.py
torch/_inductor/codegen/simd.py
test/inductor/test_nested_reduction.py
```

## Supported contract

- CONTIGUOUS sub-parent matching now accepts a symbolic or static
  non-power-of-two parent extent when the existing index and divisibility
  proofs succeed.
- The plan records `SubParentEpilogueStage.requires_looped` for a CONTIGUOUS
  source unless the parent extent is a static power of two. Standalone staged
  codegen then passes `override_persistent_reduction=False`.
- Persistent CONTIGUOUS codegen remains limited to static power-of-two extents.
  It splits the padded RBLOCK, so for logical `R=4608`, splitting
  an `RBLOCK=8192` tile would produce 4096-element lanes instead of the logical
  2304-element lanes. Tail masks cannot repair that lane assignment.
- Looped codegen does not split a padded resident tile. It reloads external
  sources in the derived domain, so symbolic and non-power-of-two CONTIGUOUS
  extents are valid once divisibility is proved.
- MXFP6 4:3 rate matching now uses the symbolic equality
  `3 * full_numel == 4 * node_numel` instead of simplifying a symbolic ratio.
  The prior zero-size rejection is retained.
- The existing symbolic reindexer was sufficient once shape divisibility was
  encoded in the graph; it was not generalized.

## Important limitation

Divisibility must be provable from the symbolic shape expression. The positive
tests encode `R = 2 * half_extent` and `D = 32 * group_count` in the input
shape. A raw unconstrained symbolic `D` that merely happens to receive divisible
runtime values can still decline fusion because the planner cannot prove the
relationship. Supporting that case would require retaining/consulting guards or
otherwise improving symbolic constraint propagation; this prototype does not
speculate from runtime values.

## Test changes

- The former `D=1000` padded-RBLOCK rejection now asserts a one-kernel staged
  fusion. Both test classes pass; the persistent-forced class is deliberately
  overridden to looped codegen.
- Dynamic RMSNorm + contiguous chunk/SwiGLU runs one compiled graph at logical
  parent extents `510`, `768`, and `1022`. Inputs are shaped as
  `(B, half_extent, 2)`, making even divisibility symbolic rather than a runtime
  accident. It compares against compiled non-nested code and requires one
  staged fusion.
- Dynamic MXFP6 uses group counts `16` and `24` with group size `32`, compares
  against compiled non-nested code, and requires one staged fusion.
- Static odd `R=511` remains a correct non-fused fallback.
- All new behavior tests run in both persistent-forced and looped-forced test
  classes and use multiple actual runtime shapes for dynamic cases.

## Verification

```text
focused new/changed matrix:     8 passed
test_nested_reduction.py:     366 passed, 5 skipped
test_inductor_scheduler.py:    78 passed, 4 skipped
py_compile:                    passed
git diff --check:              passed
```

`spin quicklint` was attempted as required, but `uvx --python 3.10
lintrunner@0.12.7 init` exited with status 2 while setting up lint tools. No
alternative lint command was used.

## Review focus

1. Confirm that `requires_looped` belongs on `SubParentEpilogueStage` as an
   explicit emission constraint rather than being rediscovered in codegen.
2. Confirm that forcing looped codegen is the right conservative behavior for
   every symbolic/non-power-of-two CONTIGUOUS source, including mixed source
   layouts.
3. Check that the positive dynamic shapes express the intended user contract:
   divisibility known symbolically, with several runtime values handled by one
   compiled graph.
4. Decide whether this should remain a follow-up commit rather than rewriting
   the already-submitted CONTIGUOUS and MXFP6 commits.

---

# Reviewer response - local dynamic-R follow-up - 2026-08-10

Reviewed the uncommitted diff only (`scheduler.py` +35/-19, `simd.py` +2,
`test_nested_reduction.py` +78/-3) against tip `f237850e504`. Read-only; no
files changed, no tests run.

Verdict: the contract is sound and the shape of the change is right. One
finding I would fix before this becomes a commit (F7), one design point that
answers Q1 more directly than the current form (F8), one test-rigor gap (F9).

## Q1 - does `requires_looped` belong on the stage?

Yes, but it is currently a *request*, not a constraint, and it is stated as a
literal at one of its two construction sites. Both are fixable in a few lines;
see F7 and F8.

## Q2 - is forcing looped right for every symbolic/non-pow2 CONTIGUOUS source?

Yes, and the `any(...)` over source layouts is the right granularity for mixed
layouts: an INTERLEAVED source is emitted correctly under looped codegen, so
demoting the whole stage when *any* CONTIGUOUS source is unprovable is
conservative in the safe direction. No case found where a mixed plan needs
finer resolution.

Note that per-stage vs per-plan is presently moot -
`StagedReductionPlan.__post_init__` rejects more than one sub-parent stage. The
value is derived from plan-level inputs (`source_deps`, `parent_rnumel`), so if
that restriction is ever lifted, revisit whether the flag should live on the
plan.

## Q3 - do the positive dynamic shapes express the intended contract?

Yes, and the tests pin it more tightly than the write-up claims.
`(B, half, 2)` -> `flatten(1)` makes `D = 2 * half` structural rather than a
runtime accident, and `(B, groups, 32)` does the same for MXFP6. Good.

The write-up says the tests "require one staged fusion", which undersells them:
`check_fusion()` asserts `codegen_nested_reduction == 1` *and*
`generated_kernel_count == 1` as equalities, against metrics accumulated across
all three runtime shapes. A recompile at the second or third shape would push
`generated_kernel_count` above 1 and fail. So "several runtime values handled by
one compiled graph" is genuinely enforced, not just asserted in prose. That is
load-bearing and non-obvious - worth one comment in each dynamic test saying the
equality is what pins single-graph reuse, so nobody later "fixes" it to
`assertGreaterEqual`.

## Q4 - follow-up commit, or rewrite the submitted commits?

Keep it as a follow-up. The five PRs carry source IDs and review history;
rewriting #190596 and #191775 burns CI and invalidates completed review for a
change that is behaviorally additive and independently revertible, and the
symbolic-divisibility risk profile deserves its own bisect point.

One caveat: it is not purely additive.
`test_producer_consumer_rejects_chunk_padded_rblock` becomes
`..._chunk_padded_rblock_uses_looped`, flipping a rejection that #190596
documented into an acceptance, and the rate rewrite edits #191775's code. The
follow-up's commit message must say both explicitly, or a reader of the landed
stack will find a test asserting the opposite of what its own commit described.

---

## F7 - the `requires_looped` override is never verified (fix before committing)

`simd.py:3827-3828` sets `override_persistent_reduction=False`, and nothing
afterwards confirms it took effect. The very next block branches on
`kernel.persistent_reduction` five times (`3843`, `3856`, `3866`, `3877`), so if
the override were ever dropped or overridden downstream the code takes the
persistent path and emits the exact wrong lane assignment this diff exists to
prevent - `4096`-element lanes for a logical `2304`, per the write-up's own
`R=4608` analysis. Silent wrong numerics, not a crash.

The sibling override in this file already does it right:
`_generate_kernel_code_for_mix_order_reduction` (`simd.py:2985-2990`) passes
`override_persistent_reduction: True` and immediately raises if
`not kernel.persistent_reduction`. Mirror it:

```python
if stage.requires_looped and kernel.persistent_reduction:
    raise AssertionError("sub-parent stage requires looped codegen")
```

Two lines, and it turns a carefully reasoned safety argument into an enforced
invariant. Given the failure mode is a silent miscompile, I would not land the
prototype without it.

## F8 - derive `requires_looped` instead of passing it in

It is a pure function of `(source_layouts, parent_rnumel)`, but it is a
constructor argument supplied at two sites: computed at `scheduler.py:734-740`,
and hardcoded `requires_looped=False` at `scheduler.py:1775` (the nested append
path). The literal is correct today only because that path is INTERLEAVED
factor-2 by construction - nothing in the dataclass says so. If the append path
ever gains a CONTIGUOUS source, that `False` is a silent miscompile with no
local signal.

Computing it in `SubParentEpilogueStage.__post_init__` from the
`source_layouts` it already stores, plus the `parent_rnumel` the plan already
holds, removes the second site and makes the invariant structural. That is also
the strongest answer to Q1: the flag belongs on the stage precisely because the
stage has everything needed to derive it.

If you prefer to keep it explicit, at minimum give `1775` a comment stating why
`False` is safe there.

## F9 - the odd-extent negative test is not controlled

`test_odd_contiguous_standalone_sub_parent_epilogue_falls_back` uses a different
model from the positive test it is meant to bound - `amax` + `x[:, :D//2]`
rather than rmsnorm + `chunk(2)` - and at `D=511` at least three distinct
rejection reasons apply at once: the non-power-of-two extent, the unequal split
(`255` vs `256`), and the index proof. `check_no_fusion()` passes if *any* of
them fires, so the test does not establish that odd extents are what declines
the fusion, and it would keep passing if the extent handling regressed.

Pin the variable: run the same slice model at `D=512` (expect fusion) and
`D=511` (expect none), so the only difference between the two assertions is the
extent.

## Verified, no action

- The MXFP6 rate rewrite is equivalent and strictly better. Old:
  `simplify(full/node)` must land on `Rational(4,3)`. New:
  `statically_known_equals(3 * full_numel, 4 * node_numel)`. Given the retained
  `node_numel == 0` guard above it, `full/node == 4/3` and `3*full == 4*node`
  are the same predicate, and the cross-multiplied form survives symbolic shapes
  where `simplify` of a ratio does not. The integer-ratio case is still caught by
  the `factors[0], 1` branch earlier.
- `parent_rnumel` at `scheduler.py:738` is already
  `V.graph.sizevars.simplify(rnumel)` from `678`, so the new power-of-two test
  and the matcher's `contiguous_parent_rnumel` see the same value - no
  asymmetry. Minor: `contiguous_parent_rnumel = simplify(parent_rnumel)` at
  `989` is now a redundant re-simplify of an already-simplified value.
- Removing `can_match_contiguous` from `_sub_parent_epilogue_source_deps` does
  not orphan `contiguous_parent_rnumel`; it is still consumed at `1079`.

## Review focus for the next round

F7 and F8 are the two I would want to see resolved. F9 is worth doing but does
not block. Q2, Q3 and Q4 need no further work beyond the commit-message caveat
noted above.

---

# #190594 pending-review fixes - ready for review - 2026-08-10

The main worktree is stopped at `701ac99270f` in an interactive rebase. The
changes below are intentionally uncommitted.

## GitHub pending comments

1. `_SubParentPointwiseRemapHandler` has been removed from this commit.
   Persistent parent sources are already split and registered by
   `_SubParentSourceLoadMaterializer`; looped sources must reload; reduced
   values use the normal store-forwarding path. The epilogue now uses
   `_PointwiseRemapHandler` directly.
2. The direct `store_cache.get(name)` path is therefore gone. Ordinary memory
   load CSE remains index-dependent; name-only forwarding is limited to the
   existing `CSEProxy` store cache. Keeping the reduction and epilogue in one
   kernel context makes that lifetime explicit, while the looped
   `kernel.codegen_body()` invalidation still forces external reloads.
3. The exact `type(...) is FusedStagedReduction` test is now an `isinstance`
   check with an explicit `FusedNestedReductions` exclusion. The exclusion is
   semantic: the subclass owns a nested topology and its own append planner.
   The codegen dispatch similarly uses `isinstance` after handling the nested
   subclass first.

## Additional fixes from the local adversarial review

- The derived epilogue body now runs the same indexing dtype-strength reduction
  and index-value conversion as ordinary SIMD bodies. A shared helper now
  applies this to generic, nested grouped-reduction, remapped pointwise, and
  sub-parent bodies.
- `SIMDKernelFeatures` accepts a separate indexing schedule. Tiling and other
  heuristics still see only the grid-owning reduction schedule, while index
  dtype selection also sees derived epilogue buffers and memory deps.
- Both stages now execute under one kernel context. Local-buffer cleanup and
  store metadata are finalized once, deleting the post-hoc
  `kernel.num_store = max(...)` repair.

Two adversarial tests were added: an independent epilogue source must use
ordinary indexed loads while retaining one staged kernel, and a shifted reduced
output must decline fusion rather than use name-based store forwarding.

Net tracked delta against the commit at review time: 93 insertions, 104
deletions across `simd.py`, `simd_kernel_features.py`, `test_indexing.py`, and
`test_nested_reduction.py`. Production code remains net smaller; the added
lines are primarily the two behavioral tests.

## Verification

```text
python test/inductor/test_nested_reduction.py
# 244 passed, 1 skipped

python test/inductor/test_indexing.py TestIndexConstOverflowInt32 TestIndexExprUpperBounds
# 11 passed

spin lint
# exit 0
```

One probe using a large integer-valued `arange` still exposes a generic Triton
expression-width issue even in ordinary pointwise codegen: casting the result
after evaluating an int32 expression does not widen its operands. That is not
specific to staged scheduling and is not included in this patch.

---

# Reviewer response - #190594 pending-review fixes - 2026-08-10

Reviewed the uncommitted diff at `701ac99270f` (`simd.py` -102 net,
`simd_kernel_features.py` +19/-6, `test_indexing.py` +13). Read-only; no files
changed, no tests run. Traced each mechanism the write-up relies on rather than
taking the summary at face value.

Net -102 lines with the indexing gap closed is a good trade, and three of the
claims check out exactly as stated. One deletion removes a guard whose two
motivating cases I do not think are covered by the replacement (F10), and the
timing of that deletion relative to the rebase is the thing I would resolve
before continuing (F11).

## Verified sound

- **Item 2, name-only forwarding.** Accurate. `CSEProxy.load`
  (`common.py:2968-2970`) already returns `store_cache[name]` before any index
  comparison, so the direct `store_cache.get(name)` path was redundant *for the
  cases the store cache covers*.
- **Looped invalidation.** Real and correctly located:
  `codegen_body` -> `self.cse.invalidate(self.outside_loop_vars)`
  (`triton.py:6690`), inside the loop branch only. The persistent branch
  deliberately does not invalidate, which is what lets the materializer's
  pre-registered lanes survive into the epilogue. The two halves of the argument
  fit.
- **Merging the two `with kernel:` blocks and deleting the
  `kernel.num_store = max(...)` repair.** This removes a post-hoc metadata patch
  that a previous review flagged as a smell. Doing it by making the lifetime
  real, rather than by repairing the symptom, is the right shape.
- **The indexing-schedule split.** `indexing_node_schedule` defaults to
  `node_schedule`, so every existing caller is unchanged and tiling still sees
  only the grid-owning reduction schedule. Minimal surface for the problem it
  solves, and `test_uses_indexing_schedule` pins the distinction directly rather
  than end-to-end. `types` is imported (`test_indexing.py:4`), both index
  helpers are imported in `simd.py` (`49-50`).
- **The strength-reduction placement** matches the ordinary path exactly: same
  pair, same per-node position as the second pass of
  `_codegen_node_schedule_body` (`simd.py:3860-3861`).

## F10 - the deleted handler covered two cases the materializer does not

`_SubParentPointwiseRemapHandler._materialize_sub_parent_load` treated a buffer
as *required* when `source_layout is not None` **or** `name in
V.graph.removed_buffers` **or** `name in kernel.removed_buffers`, and it sourced
the value from `cse.store_cache` - that is, from values **stored** during the
reduction stage.

`_SubParentSourceLoadMaterializer` is narrower on all three axes: it fires only
on a `source_layouts` hit, only inside `load`, and only when
`kernel.persistent_reduction`. So two cases lose their handling:

1. **Removed or inplaced buffers that are not planned sources.** The deleted
   docstring stated the hazard itself: "there is no address left to reload from,
   so failing to split it is fatal rather than a fallback."
2. **Sources produced by a reduction-stage store rather than a load.** The store
   cache is populated by `_update_store_cache` (`common.py:2979-2992`); the
   materializer never sees those.

The failure mode changed direction. Previously both raised
`"could not materialize required buffer"`. Now they fall through
`resolve_load` -> `_inner.load(name, remapped_index)` -> `CSEProxy.load`, which
returns `store_cache[name]` **ignoring the index** - a parent-resolution
register consumed in lane space, silently.

Two acceptable resolutions. Either demonstrate that at #190594 both cases are
unreachable - the plan-level rejection of internally-written sources covers (2),
and no epilogue node can read a removed non-source buffer for (1) - and record
that reasoning in the commit message, since it is the load-bearing premise of
the deletion. Or keep the loud failure for ~4 lines: in the sub-parent path, if
`resolve_load` misses and the name is in `store_cache` or `removed_buffers`,
raise. I would prefer the second regardless of the first, because the premise
has to keep holding through three more commits.

## F11 - the deletion lands at commit 1; commits 2-4 add the cases it covered

This is the rebase-ordering concern, and it is why F10 matters now rather than
later. #191775 introduces `internal_source_names` and the "defer the internal
source chain into the final reduction loop" logic - internally produced sources
read by the epilogue, which is exactly F10's case (2). The CONTIGUOUS and MXFP6
commits push multi-lane and non-INTERLEAVED layouts through the same path; note
the deleted code's `source_layout or ...INTERLEAVED` default only makes sense in
a world before CONTIGUOUS exists.

Before continuing the rebase, confirm each later commit still has a
materialization path for its own sources. Concretely: if #191775 rebases
*cleanly* onto this, treat that as a signal to look harder rather than as a
green light - the conflict is where the coverage question would otherwise
announce itself.

## F12 - the subclass exclusion is one invariant written at two sites

`has_standalone_stage` (`simd.py:2286`) and the codegen dispatch
(`simd.py:3034`) both encode "a `FusedStagedReduction` that is not a
`FusedNestedReductions`". The `isinstance` change is right - the previous
`type(...) is` was brittle - but the pairing is now a convention. A predicate on
the class, or a comment at each site naming the other, makes it one fact.
Worth confirming no third subclass exists today.

## Minor

- The ordinary path runs `finalize_indexing(all_indexing.keys())` before the
  strength-reduction pass; the epilogue nodes do not participate in that first
  pass. Probably fine because their indexing is derived, but worth a sentence
  confirming `finalize_indexing` does not need the epilogue's index
  expressions.
- The trailing `arange` int32-widening probe should be filed as an issue rather
  than left only in this document, or it will be lost when this section scrolls
  out of relevance. Agreed it is out of scope for this patch.
- 240 passed at #190594 versus 366 at the stack tip is consistent with being
  mid-rebase on commit 1, not a coverage regression. Running `test_indexing.py`
  alongside was the right call given the features change.

## Review focus for the next round

F10 and F11 are coupled and are what I would resolve before the rebase
continues. F12 and the minor items can ride along.

---

# #190595 rebase response - 2026-08-10

F10/F11 are now resolved without restoring the broad per-load handler.

- #190594 uses ordinary index-dependent loads for every unplanned input. Its
  planner rejects internally written parent-resolution sources, so neither
  removed-buffer fallback nor name-only materialization is needed there.
- #190595 adds the internal-source case, but materializes only names carried by
  `SubParentEpilogueStage.source_layouts` or `broadcast_source_names`.
  `_materialize_sub_parent_inputs` runs once after all nested producers have
  emitted. It never treats membership in `removed_buffers` as a layout proof.
- Planned parent sources loaded by the grouped reduction are still captured by
  `_SubParentSourceLoadMaterializer`. Everything else uses
  `_PointwiseRemapHandler`, so unrelated lane-resolution inputs remain ordinary
  indexed loads.
- The nested kernel now stays in one `with kernel:` lifetime. Kernel-local
  buffers are removed only after the outer, grouped, and sub-parent stages have
  all emitted; the older lazy handler no longer needs to compensate for an
  early lifetime boundary.

The plan's broadcast set is filtered to values actually read by sub-parent
nodes. Source layouts and broadcasts use the node-local temporal names already
present in `read_writes`; the graph-final mutation map must not be applied. Two
new positive tests pin the materialization distinction:

- an unrelated sub-parent input fuses and loads normally;
- a parent-full intermediate produced after the grouped reduction fuses and
  emits exactly two `tl.split` operations, proving the planned intermediate is
  explicitly materialized.

The nested path also passes all emitted nodes to the separate index-width
schedule introduced in #190594 while retaining the outer schedule for tiling.
This closes the `finalize_indexing` concern for nested codegen as well.

The named-constant review suggestion to write to the buffer passed to
`iteration_ranges_codegen_header` is not correct. That buffer may be
loop-local `indexing_code`; shape constants are shared by derived families and
must remain at kernel-function scope. `_named_constants` and `self.body` are
kernel-lifetime state and are intentionally kept together.

Verification so far:

- `python test/inductor/test_nested_reduction.py -k sub_parent`: 70 passed, 1
  skipped before adding the two new positives.
- `python test/inductor/test_inductor_scheduler.py -k nested_reduction`: 8
  passed.
- `python test/inductor/test_nested_reduction.py`: 296 passed, 2 skipped before
  adding the two new positives. A final full rerun is in progress.

# #191775 final verification - 2026-08-10

The adversarial index audit found and fixed two wrong-code cases: shifted reads
of a nested parent output and of a standalone reduced output had been admitted
by name alone. Generic producer/consumer reads now require the retained
normalized reshape/broadcast proof. Only names whose lane relation was proved
by `StagedReductionPlan` may use the narrower plan-proven exception, and those
still require a unique dense, injective, synchronization-free producer write.

Ordinary external inputs remain index-dependent loads. Direct store-cache
materialization is limited to internally produced names carried by the stage's
source layouts. One kernel context now spans parent and staged emission, so CSE
state and store metadata reflect the actual five MXFP6 stores without a
post-hoc counter adjustment.

Final verification on the resolved commit:

- `python test/inductor/test_nested_reduction.py`: 326 passed, 3 skipped.
- `python test/inductor/test_inductor_scheduler.py`: 78 passed, 4 skipped.
- Focused shifted, index-equivalence, MXFP6, and nested interleaved tests passed.
- `UV_OFFLINE=1 spin quicklint -a`: clean.
- `python -m py_compile ...` and `git diff --check`: clean.

# #190596 adversarial rebase - 2026-08-10

The CONTIGUOUS commit now closes the coverage gaps from the outside review.

- Factors 8 and 16 use a lane-asymmetric weighted epilogue and pass
  numerically under forced persistent and looped codegen.
- Factor-8 INTERLEAVED input declines, pinning the separate factor-4 cap.
- A factor-4 CONTIGUOUS source feeding a three-output-lane stage works in both
  kernel forms, so the old proposed blanket rejection is unnecessary.
- Distinct buffers may use INTERLEAVED and CONTIGUOUS layouts in one plan;
  conflicting layouts for one buffer still decline.
- Symbolic reduction extent fallback runs D=512/1024/2048 through one dynamic
  graph, produces two generic kernels, and never enters staged codegen.
- A singleton-batch kernel-form test confirms the actual tile remains rank 2.
  The nominal rank-1 permute branch is still not reached even by a 1-D input.
- The repeated RMSNorm/chunk model now has one helper shared by behavior and
  kernel-form tests.
- A rebase-only metadata regression (`num_outputs` after #191775 renamed the
  contract to emitted `num_stores`) was caught and fixed.

Verification:

- `python test/inductor/test_nested_reduction.py`: 384 passed, 5 skipped.
- `python test/inductor/test_inductor_scheduler.py`: 78 passed, 4 skipped.
- `UV_OFFLINE=1 spin quicklint -a`: clean.
- `python -m py_compile ...` and `git diff --check`: clean.

# Final full-review correctness fixes - 2026-08-11

The full-stack review identified two cases that the earlier adversarial passes
had not pinned. Both fixes are folded into their owning commits and submitted.

## #190595: preserve temporal mutation versions

`SchedulerNode.read_writes` already names the buffer version visible at that
node. The nested stage planner incorrectly reapplied the scheduler-global
`mutation_renames` map, which maps toward the graph-final version. When a
planned source was mutated later, planning recorded the future name and
codegen could not materialize it from the current store cache.

The shared sub-parent helpers now consume raw `read_writes` names consistently.
`test_producer_consumer_sub_parent_source_mutated_later` keeps the later
mutation observable, checks the compiled result and mutated input against
eager, and requires the nested staged kernel to form.

## #191775: a grouped-output name is not an index proof

`_can_fuse_nested_reduction_append` used derived-domain classification to
allowlist a grouped output by name. After #191775 made that allowlist remove
the matching `MemoryDep` reads, a rolled scale could fuse and receive the
unshifted store-cache value.

The append path now intersects derived-domain names with
`_producer_output_names_read_by_consumer()`. Each ordinary consumer read must
therefore pass the normalized producer/write equivalence proof before vertical
dependency matching is relaxed. Plan-proven lane sources still use the
separate staged-plan exception.

`test_producer_consumer_rejects_shifted_reduced_source` compares enabled and
disabled nested scheduling, requires correct numerics, and verifies that the
outer nested reduction still forms while the shifted epilogue remains a second
kernel.

## Verification

```text
python test/inductor/test_nested_reduction.py
  388 passed, 5 skipped

python test/inductor/test_inductor_scheduler.py
  78 passed, 4 skipped

python agent_space/repro_190595_mutation.py
  max diffs <= 2.4e-7; nested kernel formed

python agent_space/repro_191775_shifted_broadcast.py
  nested enabled: 2 correct kernels; nested disabled: 3 correct kernels
```

## Submission handoff

The review-ready tracked diff is 80 insertions / 69 deletions across
`torch/_inductor/scheduler.py` and
`test/inductor/test_nested_reduction.py`.

- The temporal-name change and
  `test_producer_consumer_sub_parent_source_mutated_later` are in #190595.
- The proof-backed append allowlist and
  `test_producer_consumer_rejects_shifted_reduced_source` are in #191775.
- Full ghstack updated #190595, #191775, #190596, and #191974; #190594 was
  skipped because its tree was unchanged.
- The requested repeat ghstack run skipped all five PRs as already current.
- `READ_ME_FIRST.md`, every current per-PR guide, and `FOLLOWUPS.md` describe
  the submitted contracts and final commit SHAs.

# #190594 can_fuse review batch - 2026-08-12

Submitted with `ghstack --no-stack` after user review. Local ghstack commit:
`209ecc8973be093696b481e90c82bd1338ea753a`; PR head:
`4cd7c0df44cdd1cc10e4f2e5565d1dba5b1e5cb0`. Only #190594 was updated.

## Submitted changes

- Standalone staged-plan preservation for reduction/reduction pairs now runs
  inside the existing reduction/reduction branch, after ordinary and native
  matmul compatibility succeed.
- Mixed reduction/pointwise handling uses the pre-existing canonical flow:
  reduction/pointwise reaches the pointwise/reduction branch through the
  existing argument swap.
- Ordinary pointwise/reduction compatibility runs first and eagerly returns
  only on success. A normal rejection falls through to sub-parent planning.
  An existing exact `FusedStagedReduction` always revalidates its plan before
  ordinary fusion is accepted.
- `_SubParentFusionDecision` is now the three-state enum `_SubParentFusion`
  (`DEFER`, `REJECT`, `FUSE`), eliminating the impossible two-boolean state.
- The backend planner arguments and related locals are named
  `parent_numel`/`parent_rnumel`. They always describe the candidate
  grid-owning reduction; pointwise and sub-parent groups are derived from
  that parent group.
- `_sub_parent_epilogue_plan` now asks the scheduler for a valid plan before
  running backend tiling analysis. The separate ordinary-epilogue shape
  prefilter is removed because successful ordinary fusion no longer enters
  sub-parent planning.

## Adversarial review

The Codex adversarial reviewer found no fusion-correctness regression. It
verified ordinary-result equivalence, producer/consumer orientation through
the swap, exact staged-node revalidation, reduction/reduction preservation,
and scheduler-plan-before-tiling ordering.

One non-correctness diagnostic follow-up remains: ordinary incompatibility calls
`WhyNoFuse` before the sub-parent fallback can return `FUSE`, so fusion logs
can claim that a pair was rejected even though it is subsequently fused. The
ordinary failure reason should be retained and emitted only after the special
fallback also declines. Specialized rejection currently uses the generic
`invalid sub-parent epilogue fusion` message; preserving a concrete rejection
reason would improve the same trace.

Two read-only Claude Opus CLI attempts stalled without producing review
output and were terminated. No Opus assessment is recorded for this batch.

## Verification

```text
python test/inductor/test_nested_reduction.py -k standalone_sub_parent
  46 tests run, 1 skipped

python test/inductor/test_nested_reduction.py
  244 tests run, 1 skipped

python -m py_compile torch/_inductor/codegen/simd.py
git diff --check
  clean
```

# Proposed #190594 source-dependency simplification - 2026-08-12

Status: review requested before implementation. A draft implementation was
briefly applied locally, then fully reverted when the request for prior review
arrived. The proposal below is not present in the tracked working diff.

## User comments

1. The standalone planner first scans every candidate node to build
   `fused_buffer_names` and `reduction_reads`, then
   `_sub_parent_epilogue_source_deps` scans the epilogue reads. This looks like
   duplicated dependency analysis and makes an already large planner harder to
   review.
2. Requiring a matching reduction read is not the general availability rule.
   A sub-domain read should be legal when its value is externally loadable or
   produced earlier in the fused node. The ordinary fusion dependency graph
   should establish producer/consumer ordering.
3. Once the scheduler proves that a sub-domain read is a supported projection,
   it is unclear why the exact parent `MemoryDep` needs to leave the proof
   helper and travel through the outer planner.
4. At codegen time the derived-domain wrapper should reuse a live value when
   possible and otherwise issue an ordinary derived-index load. This applies to
   persistent and looped kernels; looped codegen commonly falls through to the
   reload because its loop-local CSE values have been invalidated.

## Response

The two scans inspect opposite sides of a required join; they are not the same
analysis:

- the first indexes parent-resolution reads by buffer name;
- the second examines child-domain reads and proves an exact relationship to a
  parent access under `parent_r = factor * child_r + lane`.

Both sides must be inspected. The structural problem is that the exact parent
`MemoryDep` escapes the helper even though it has only fusion-time uses:

- prove the child index is an exact lane projection;
- reject another reader of the same name using a different parent index.

After those checks, codegen needs only the buffer name and source layout. The
current stage already stores only `(name, layout)`; the outer planner converts
the returned `(MemoryDep, layout)` after separately invoking the ambiguity
check. That intermediate propagation can be removed.

The proposed codegen-time model is already substantially how codegen behaves:

- persistent codegen intercepts a planned parent load while its `CSEVariable`
  is live and registers split lanes in `_DerivedIterationFamily.remapped_values`;
- `_PointwiseRemapHandler.load` first calls `family.resolve_load(name, index)`;
- if no remapped value exists, as is normal after a looped reduction invalidates
  loop-local CSE, the handler emits an ordinary load at the derived index.

The explicit `(name, layout)` remains useful. Generic load CSE is keyed by the
generated expression, including indexing and masks, rather than by semantic
`(buffer, index)` identity. `store_cache` is name-keyed, but external loads are
not. The parent-load wrapper is therefore the semantic bridge that captures the
live value. The layout also distinguishes INTERLEAVED from the later CONTIGUOUS
transformation. This proposal removes the exact `MemoryDep` from the outer
planner; it does not attempt to infer layouts again during codegen.

### Why not use a richer `(name, index)` CSE key?

This is worth considering, but a literal `(name, index)` lookup does not solve
the domain conversion. The two accesses are intentionally different values:

```text
parent: x * R + r                 shape [XBLOCK, RBLOCK]
child:  x * R + 2 * child + lane shape [XBLOCK, RBLOCK / 2]
```

The child index is a projection of the parent index, not the same index under
the same active range trees. Reusing the parent value requires a transformation
such as `tl.split` or reshape/permute, and the correct transformation depends on
the source layout. Masks, padding, block shapes, mutation versions, and loop
scope are also part of whether a cached value remains usable.

A genuinely complete solution would therefore be a domain-aware value cache:
it would record `(buffer version, parent access, iteration domain, block shape,
scope)` and find a proved projection from a child access before applying the
layout transformation. That is substantially more machinery than changing a
CSE dictionary key and would reproduce much of the current scheduler proof at
codegen altitude.

Name-only lookup is safe here only after the scheduler has proved that the name
has one unambiguous parent access and that every derived use is a supported lane
projection. The proposed change keeps the exact parent `MemoryDep` local to that
proof, then deliberately collapses the result to `(name, layout)`. If the
reviewer believes a reusable domain-aware cache is warranted, it should be a
separate abstraction rather than an implicit weakening of these proofs.

### Existing precedent: #188180

PR #188180, "Reuse full-range splits for CSE," addresses the same broad cache
identity problem. A persistent reduction first loads through split coordinates,
while its epilogue later expresses the same full range with one flattened
coordinate. `_reuse_load_index_basis` records structural split bases in
`TritonCSE`, keyed in the existing buffer/mask scope, rewrites the later index to
the earlier basis, and lets ordinary string-keyed load CSE hit. Its state is
cleared by `TritonCSE.invalidate`, so it does not cross loop-scope invalidation.

That implementation supports the user's proposed direction: codegen can retain
structural access information rather than carrying scheduler dependencies into
the emitter. It does not directly solve sub-parent reuse, for three deliberate
reasons in #188180:

1. `_load_index_split_basis` accepts only a complete basis whose extent equals
   the range root's full `numel`;
2. `_rewrite_full_range_with_basis` performs a bijective coordinate rewrite of
   that same full range, not a projection to a smaller value;
3. `_supports_load_index_basis_reuse` explicitly rejects kernels containing a
   `DerivedIterationRangesRoot`.

Sub-parent reuse is not merely two spellings of one load. The parent value has
shape `[XBLOCK, RBLOCK]`, the child has `[XBLOCK, RBLOCK / factor]`, and reuse
requires a layout-specific transformation before lane selection. A direct CSE
hit would return the wrong-shaped value.

A more general follow-up could extend the #188180 pattern into a
projection-aware value cache: record a live parent value plus its structural
range basis in the CSE scope, clear it on invalidation, and let a derived load
lazily request an INTERLEAVED or CONTIGUOUS projection. Scheduler legality still
has to prove that the projection is valid before fusion becomes irrevocable.
The near-term proposal remains smaller: localize exact `MemoryDep`s to that
proof and carry only `(name, layout)` into codegen.

## Three-state design summary

### Original #190594 state

- Fusion planning collects reduction reads by buffer name.
- `_sub_parent_epilogue_source_deps` matches lane-domain reads against exact
  parent `MemoryDep`s and returns `(MemoryDep, layout)` pairs.
- The outer planner passes those deps to the ambiguity check, then immediately
  discards their indices and stores only `(name, layout)` on the stage.
- Persistent codegen eagerly splits each planned source when the reduction
  loads it. Looped codegen has no live parent value and reloads at the derived
  index.
- Special materialization recognizes reduction loads only; internal pointwise
  writes are deferred to #190595.

### Proposed amended #190594 state

- Preserve the same accepted graphs and generated kernels; do not pull
  #190595's internal-source feature backward.
- Keep exact parent `MemoryDep`s entirely inside one source-planning helper for
  lane equality and ambiguity proofs.
- Return only `(name, layout)` from that helper and pass it directly to
  `SubParentEpilogueStage`.
- Use neutral names such as `parent_reads_by_name`, `parent_dep`, and
  `epilogue_nodes`; retain explicit `TODO(#190595)` markers where pointwise
  writes become valid sources.
- Keep persistent eager registration and looped reload behavior unchanged.

This middle state is the requested review proposal. It has not been applied.
The current tracked diff contains only the naming/docstring/TODO edits listed at
the end of this handoff.

### Ideal future state

- Ordinary fusion dependencies determine whether an input is external or has a
  producer ordered before the derived stage.
- Scheduler legality proves a canonical parent-to-child projection before
  fusion becomes irrevocable, but exact scheduler `MemoryDep`s do not cross
  into codegen.
- A projection-aware extension of the #188180 CSE state records live values with
  their structural range basis and scope, and clears them on CSE invalidation.
- A derived load lazily reuses and transforms a live parent value; otherwise an
  external input is loaded normally, or an internal source is deliberately
  recomputed/forwarded in the derived scope.
- The remaining cross-phase contract is a small projection recipe, such as
  `(name, factor, INTERLEAVED)`, not a reduction-specific dependency map.
- INTERLEAVED and CONTIGUOUS use the same mechanism; reduction loads, pointwise
  writes, and later staged producers differ only in availability, not in the
  source-planning architecture.

## Proposed change

Keep exact dependency objects local to a renamed source-planning helper:

```python
source_layouts = cls._try_get_sub_parent_source_layouts(
    all_nodes,
    epilogue_nodes,
    full_numel,
    parent_reads_by_name,
    sub_parent_factor,
)
```

The helper would:

1. derive `fused_buffer_names` from `all_nodes`;
2. inspect only `epilogue_nodes` for child-domain reads;
3. retain exact parent `MemoryDep`s locally while proving lane/index equality;
4. run `_sub_parent_epilogue_source_loads_are_unambiguous` before discarding
   those exact deps;
5. return only `tuple[(buffer_name, source_layout), ...] | None`.

The caller would pass that tuple directly to `SubParentEpilogueStage` instead
of unpacking parent deps and rebuilding the same tuple. The surrounding names
would become `parent_reads_by_name`, `parent_dep`, and `epilogue_nodes` so the
implementation does not imply that a reduction read is the fundamental source
kind. #190595 can extend the parent lookup with unique pointwise writes.

No new dataclass or general dependency abstraction is proposed. Ordinary
producer/consumer legality remains the scheduler's responsibility; this helper
only proves and records the specialized domain transformation.

## Invariants the review must check

1. Exact child-to-parent index equality remains mandatory; matching element
   counts or shapes is insufficient.
2. Ambiguity is checked using exact `MemoryDep`s before the result is collapsed
   to a buffer name.
3. Independent external inputs remain ordinary derived-domain loads and are not
   unnecessarily materialized.
4. Removed internal buffers are accepted only when a live or deliberately
   recomputed value is available in the derived stage.
5. Looped codegen never forwards an invalidated loop-local cache value; it
   reloads an external source or uses a source produced in the derived scope.
6. Persistent codegen still splits each planned parent source once and reuses
   its lanes.
7. The scope of the ambiguity check remains correct at both stack call sites:
   standalone #190594 and nested append #190595 currently pass different node
   sets around this analysis.

## Questions for the reviewer

1. Is moving the ambiguity check into the source-planning helper behaviorally
   equivalent for #190594?
2. When the change is replayed through #190595, should the helper accept a
   separate `ambiguity_nodes` sequence, or should each caller pass its complete
   candidate group as `all_nodes`?
3. Is `(buffer_name, source_layout)` the minimal sufficient result, including
   mixed INTERLEAVED/CONTIGUOUS sources later in the stack?
4. Are there any paths where the outer planner still requires the exact parent
   `MemoryDep` after source ambiguity has been checked?

## Current uncommitted #190594 changes

The tracked working diff still contains only the earlier reviewed edits:

- `TODO(#190595)` at both appearances of the reduction-read limitation;
- `_pointwise_node_is_compatible` renamed to
  `_pointwise_node_matches_domain`;
- `sub_parent_extent_subs` renamed to
  `try_get_sub_parent_extent_subs`, with its `None`/possibly-empty mapping
  contract documented;
- the source helper docstring now distinguishes planned parent-source reuse
  from independent external loads;
- the ambiguity check has a TODO to extend #188180's structural load-index
  cache to derived projections, eliminating the name-keyed rejection.

`python -m py_compile torch/_inductor/scheduler.py` and `git diff --check` are
clean for that current state. The focused tests have not been rerun since these
name and documentation changes.

---

# Reviewer response - proposed #190594 source-dependency simplification - 2026-08-12

Read-only review at `209ecc8973b` plus the uncommitted naming diff. Also
re-read the #190595 call site in a later blob (`f020897c032`) to answer the
replay questions. No files changed, no tests run. This is the Opus assessment
for this batch, replacing the two stalled attempts noted above.

**Verdict: implement it, with two amendments.** The structural claim is
correct - at #190594 the exact parent `MemoryDep` has exactly two fusion-time
uses and no codegen use, so collapsing to `(name, layout)` at the plan boundary
is sound. But the proposal's stated motivation is not what it fixes, and its
`all_nodes` signature would silently change #190595 behavior on replay.

## Q1 - equivalent for #190594? Yes.

`planned_source_deps` (`scheduler.py:669`) feeds
`_sub_parent_epilogue_source_loads_are_unambiguous` (`671-675`) and nothing
else; every later use is `dep.name` (`699`). Grep confirms no other consumer in
`scheduler.py`, and `simd.py` at this commit contains no reference to
`planned_source_deps` at all. Moving the check inside the helper is pure motion,
conditional on it still receiving the same two node arguments - see Q2.

## Q2 - separate `ambiguity_nodes`. Do not collapse to one `all_nodes`.

The two call sites disagree today, and the disagreement is real rather than
incidental:

- #190594: `fused_buffer_names` from `scheduler_nodes`; ambiguity over
  `scheduler_nodes`. Same set.
- #190595: `fused_buffer_names` from `[*outer_node.get_nodes(), *grouped_nodes]`;
  ambiguity over the grouped nodes only - a strictly smaller set.

If the helper derives both from a single `all_nodes`, replaying it through
#190595 widens that path's ambiguity scan to include the outer node's reads.
That can only *reject* plans accepted today, so it surfaces as a lost fusion
rather than a test failure - the worst shape for a silent change.

Pass `ambiguity_nodes` explicitly. Separately: nothing currently records whether
excluding the outer node there is deliberate or an oversight. This refactor is
the moment to decide. If it is deliberate, one comment at the call site. If it
is an oversight, widening it is a *fix* and should land as its own change with a
test - not as a side effect of a signature cleanup.

## Q3 - yes, `(name, layout)` is minimal sufficient, mixed layouts included.

Layout is per buffer name and `add_source` already declines conflicting layouts
for one name, so a mixed INTERLEAVED/CONTIGUOUS plan is representable - which
matches what the #190596 rebase section reports. CONTIGUOUS codegen needs the
extents, but takes them from `plan.parent_rnumel` and `stage.factor`, not from
the dep.

One caveat, not a blocker: this makes "names are the contract" official while
the contiguous lane formula remains written independently in the planner and in
`_resolve_remapped_value`, coupled only by an internal assert that fires at
codegen time rather than as a fusion rejection. If names are the contract, that
formula should be one shared call, and this is the natural moment.

## Q4 - no at #190594; two things to check on replay.

Nothing needs the exact dep after the ambiguity check at this commit. For the
replay through later commits:

- The plan-level "internally produced source read by a reduction" guard tests
  `dep.name in source_writes and dep.name in reduction_reads` - name-only, so it
  survives the collapse unchanged.
- At the later tip, `_sub_parent_epilogue_leaf_violation` in `simd.py`
  *reconstructs* planned source deps by re-walking `plan.parent_nodes`, and that
  reconstruction is a **superset** of what the planner selected: every
  full-resolution `MemoryDep` with a matching name, versus the planner's
  one-per-name choice. So "the plan carries only names" already has a cost, paid
  by an inexact rebuild that makes the simd-side ambiguity check weaker than the
  planning-side one. Formalizing names-only is still right; decide explicitly
  whether that superset is intentional and put the answer next to the
  reconstruction.

## The premise in user comment 1 does not survive contact

The proposal does not remove a scan - it splits one. The caller loop at
`scheduler.py:633-640` builds `fused_buffer_names`, `reduction_reads` and
`has_reduction` in a single pass over `scheduler_nodes`. Moving only
`fused_buffer_names` into the helper rebuilds that set in a second pass while
the caller keeps scanning for the other two.

Keep deriving it in the existing loop and pass it in. The underlying complaint -
the planner is hard to review - is better served by the renames already in the
working diff (`parent_reads_by_name`, `parent_dep`, `epilogue_nodes`) than by
relocating the derivation. Your own response already makes the correct
counter-argument to comment 2: the two scans inspect opposite sides of a join.

## The seven invariants

- **1, 2** - preserved by construction, provided the ambiguity check runs inside
  the helper *before* the collapse. That is what the proposal says; it is the
  load-bearing ordering, so it deserves a comment rather than only an ordering.
- **3** - preserved and untouched: unplanned reads never enter `source_layouts`,
  so `_PointwiseRemapHandler` still emits an ordinary derived-index load.
- **4** - the refactor neither helps nor hurts this, and it is the same ground as
  F10. It is enforced by the planner's rejection rules, not at the collapse
  point. The #190595 resolution is the invariant of record: materialize only
  names carried by the stage, and never treat `removed_buffers` membership as a
  layout proof. Worth restating in the helper docstring, since the helper is now
  the thing that decides what gets materialized.
- **5, 6** - codegen-side, unaffected.
- **7** - see Q2. This is the one that will actually bite.

## Still open from earlier rounds

- **F7** - in the dynamic-R prototype, `requires_looped` sets
  `override_persistent_reduction=False` and nothing asserts it took effect,
  against a silent wrong-lane failure mode; the sibling mix-order path asserts
  its own override two thousand lines away. The prototype is no longer in the
  tree, so this needs to travel with it when it returns rather than being
  rediscovered.
- **F12** - the `FusedStagedReduction`-but-not-`FusedNestedReductions` exclusion
  is one invariant written at two sites.
- Your own `WhyNoFuse` finding - the ordinary rejection reason is emitted before
  the sub-parent fallback can return `FUSE`, so logs can report a rejection for a
  pair that then fuses. Worth folding into this refactor, since both touch the
  same `can_fuse` flow and the enum change already reshaped it.

---

# Implementation status - source-layout boundary - 2026-08-12

Implemented and submitted to #190594. After ghstack rewrote its metadata, the
local orig commit is `5d70ad19ff9` and the published head is `20cdeca1faf`.

- `_try_get_sub_parent_source_layouts` keeps exact `MemoryDep` objects local to
  lane/index and ambiguity proofs, then returns only `(buffer_name, layout)`.
- The ambiguity scope is explicit through `ambiguity_nodes`; standalone
  planning passes `scheduler_nodes`, preserving the prior behavior and leaving
  #190595 free to retain its narrower append-path scope during replay.
- `fused_buffer_names` remains derived in the caller's existing pass.
- Added `TODO(#190595)` for parent-stage pointwise-produced sources.
- Added the unassigned TODO to extend #188180's structural load-index cache to
  derived projections.
- Renamed `_pointwise_node_is_compatible` to
  `_pointwise_node_matches_domain` and `sub_parent_extent_subs` to
  `try_get_sub_parent_extent_subs`; the latter documents `None` versus a
  possibly empty successful mapping.

Verification:

```text
python test/inductor/test_nested_reduction.py -k standalone_sub_parent
# Ran 46 tests in 33.347s - OK (skipped=1)

python test/inductor/test_nested_reduction.py
# Ran 244 tests in 128.667s - OK (skipped=1)

python -m py_compile torch/_inductor/scheduler.py torch/_inductor/codegen/simd.py
git diff --check
```

---

# Post-ghstack CI audit - 2026-08-14

Fresh CI exposed two test-boundary issues; neither is a scheduler/codegen
correctness failure.

1. #190594's `test_looped_standalone_sub_parent_large_group` fails on L4 but
   passes on B200. The cause is the pre-codegen split-reduction heuristic, not
   FX caching: for `B=8, R=16384`, L4's no-split threshold is 8192 while
   Blackwell's is 32768. L4 therefore rewrites the reduction before the test
   reaches staged looped codegen. The fix applies
   `split_reductions=False` to this test so it isolates the looped codegen
   contract. The class-level `force_disable_caches=True` remains because it was
   separately requested and keeps metric assertions cold.
2. #191775 removes the fixed `PointwiseDomainContext.sub_parent_domain` field
   in favor of per-candidate `_nested_sub_parent_rate()`, but inherited
   #190595 scheduler coverage still reads the removed field. The test
   migration builds a factor-2 mock node and checks rate `(2, 1)` for an
   R-grouped context, then rejects both X grouping and a same-numel `(3, 8, 6)`
   consumer that crosses the required `(3, 6, 8)` group boundary. This belongs
   in #191775; the original field assertion remains correct at the isolated
   #190595 boundary.

Local verification:

```text
exact migrated test: 1 passed
TestSchedulerCPU: 40 passed, 6 skipped
test_inductor_scheduler.py: 86 passed, 6 skipped
stack-tip nested concrete classes: 372 passed, 6 skipped
```

Independent reviews found no further concrete issue in #191775's reindex/CSE
flow or #190596's CONTIGUOUS extent/persistence contracts. In particular,
CONTIGUOUS planning and codegen both consume `plan.parent_rnumel`; the formerly
reported extent-drift concern is not present.

The fixes are amended into `86038d6e1cd` (#190594) and `f8d435e3aa8`
(#191775). The review guides now record the updated stack hashes, diff sizes,
and the representation transition from a stored factor-2 `sub_parent_domain`
to a per-node `(factor, output_lanes)` rate.

`lintrunner -a` completed cleanly before the amendments.

---

# Submitted update - normalized sub-parent source matching - 2026-08-12

This was amended into #190594 and submitted with `ghstack --no-stack`. The
local orig commit is `23be7e609a5`; the published head is `7fefd51b185`.

## Result

The prototype replaces the raw-`MemoryDep` source join with the existing fused
read/write normalizer:

1. Partition the prospective group into parent and sub-parent epilogue nodes.
2. Build one `_FusedNodeView` for each domain and call
   `extract_normalized_read_writes`.
3. Join their normalized reads by buffer name.
4. For a shared source, require one normalized parent access and prove every
   child access equals it under
   `parent_r = factor * child_r + lane`.
5. Independent child inputs remain ordinary derived-domain loads.

This deletes the separate
`_sub_parent_epilogue_source_loads_are_unambiguous`,
`_unique_trailing_sub_parent_dim`, and
`_interleaved_sub_parent_epilogue_read_matches_parent_read` helpers. The
scheduler diff is currently +161/-205, a net removal of 44 lines. More
importantly, fused-access normalization now has one implementation rather than
the sub-parent planner rebuilding it from raw deps.

The exact parent `MemoryDep` is no longer carried across the planner boundary.
The plan still carries `(buffer_name, layout)`. Multiple normalized parent
accesses to one source remain a conservative rejection, with the existing TODO
to extend #188180's structural load-index cache.

## Normalization boundary

The normalizer does not make the domain-equivalence proof disappear. Parent
and child views are normalized independently, so generated names such as `n0`
are local numbering. For example, parent ranges `(2, 6)` and child ranges
`(3, 4)` can both produce an expression using `n0, n1`; their names match while
their coordinates do not. The planner therefore validates the ordered variable
ranges before comparing expressions. A TODO asks normalized views to expose a
directly comparable domain signature.

The proof also preserves multi-axis reductions. Common reduction-prefix axes
must have equal ranges, and only the trailing R axis is transformed by the
sub-parent factor. The products of the normalized R ranges are checked against
the parent and child `rnumel` values.

`extract_normalized_read_writes` assumes every member is representable in the
view's group. The planner preflights parent-node domains and returns `None` for
an incompatible sibling instead of allowing the normalizer to assert.

## Adversarial review findings

Three issues were found and fixed during review:

- A dependent smaller-domain consumer (`sin(even[:, ::2])`) made parent-view
  normalization assert. It now remains a separate kernel while the valid
  sub-parent group still emits as one staged kernel.
- Comparing generated variable names without their ranges could accept
  different X factorizations. The ordered ranges are now part of the proof.
- Requiring one normalized R variable regressed reductions over multiple
  logical dimensions. Shared R-prefix axes plus a split trailing axis are now
  supported.

The corresponding tests exercise all three boundaries in both persistent and
looped test classes.

## Verification

```text
python test/inductor/test_nested_reduction.py -k standalone_sub_parent
# Ran 52 tests in 35.979s - OK (skipped=1)

python test/inductor/test_nested_reduction.py
# Ran 250 tests in 125.531s - OK (skipped=1)

python -m py_compile torch/_inductor/scheduler.py test/inductor/test_nested_reduction.py
git diff --check
```

`spin fixlint` and `spin lint` report `ok No lint issues` for the changed-file
group. Both commands still exit nonzero because the unrelated all-files
lintrunner invocation fails with `No such file or directory` in this checkout.

---

# Review - normalized sub-parent source matching - 2026-08-12

Read-only review of the uncommitted two-file prototype on `5d70ad19ff9`.
Only this handoff was updated; no source or test code was changed.

## Verdict

**Needs discussion before amend.** I found no isolated #190594 wrong-code case,
and the focused standalone suite passes in both persistent and looped test
classes. The remaining concerns are compile-time cost, an unpinned expansion of
the accepted set, and preserving later-stack contracts during replay.

## F13 - normalization now runs repeatedly in speculative fusion

`_try_get_sub_parent_source_layouts` calls
`extract_normalized_read_writes` once for the parent and once for the epilogue.
This is not a cheap dependency normalization: it constructs `NodeSplitGetter`,
searches compatible splits, remaps every node body, and simplifies all access
expressions.

Instrumentation of one uncached positive standalone compile measured:

```text
sub_parent_epilogue_plan calls:                 5
_try_get_sub_parent_source_layouts calls:       4
extract_normalized_read_writes calls inside it: 8
```

The submitted implementation performs direct `MemoryDep` scans in this hot
fusion-search path. Before replacing it, compare compile time against that
implementation. If the normalizer is materially more expensive, cache the two
canonical views for the lifetime of a fusion round or retain the direct proof.
The net source-line reduction alone is not enough to justify repeated tiling
normalization during speculative pair checks.

## F14 - the normalized-equivalence expansion has no positive regression

The old ambiguity proof required another parent read to equal the selected raw
`MemoryDep`. The prototype accepts multiple raw parent accesses when they
normalize to one canonical index. That is a real expansion of the accepted set,
not only a refactor.

The three new tests cover an incompatible sibling, a multidimensional
reduction, and unequal parent/child coordinate factorizations. None directly
pins the new case: one source read through two distinct raw factorizations that
normalize to the same parent access. Add a positive numeric and kernel-count
test for that case in both persistent and looped classes. If accepting that case
is not intended, the implementation should preserve the old exact-access
restriction instead.

## F15 - replay through #190595 and #190596 is not mechanical

The earlier review established that standalone and nested append intentionally
used different ambiguity scopes. This prototype removes `ambiguity_nodes`; the
nodes included in the parent normalized view now implicitly define that scope.
Replaying the helper through #190595 must preserve the grouped-only nested scan
or explicitly change it with a regression test. Otherwise outer-node reads can
silently turn an accepted nested fusion into a rejection.

The later commits also add parent-stage pointwise writes and CONTIGUOUS source
layouts. This isolated helper joins only `parent_accesses.reads` and always
records `INTERLEAVED`. During replay, internally produced sources must join
against the canonical parent write, and #190596 must retain its separate layout
proof. Re-run the mixed-layout, internal-source, shifted-read, and temporal
mutation tests at their owning commits rather than treating the isolated helper
as a textual rebase.

## Verification

```text
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  python test/inductor/test_nested_reduction.py -k standalone_sub_parent
# Ran 52 tests in 35.261s - OK (skipped=1)

git diff --check
# clean
```

---

# Reviewer response - normalized sub-parent source matching prototype - 2026-08-12

Read-only review of the uncommitted diff on `5d70ad19ff9` (`scheduler.py`
+161/-205, tests +48). No files changed, no tests run.

**Direction is right.** Replacing `_unique_trailing_sub_parent_dim` and
`_interleaved_sub_parent_epilogue_read_matches_parent_read` with
`extract_normalized_read_writes` removes the second implementation of
fused-access normalization, which has been a recurring theme in this stack. The
three self-reported adversarial findings are the right three, and putting
ordered ranges into the proof is the correct fix for the second.

One deleted guard I would restore before this is committed (F14), one scope
regression to be aware of on replay (F15), and two things that are correct today
for reasons the code does not state (F16, F17).

## Confirmed: the ambiguity helper really is subsumed - for #190594

`_sub_parent_epilogue_source_loads_are_unambiguous` compared each non-epilogue
raw read against the planned dep set. The replacement is
`if len(parent_indices) != 1: return None`. That is sound here and slightly
stronger in kind: `parent_nodes` is exactly `scheduler_nodes - epilogue_nodes`,
so every former ambiguity node is inside the parent view, and normalization
merges accesses that are genuinely the same pattern while separating those that
are not - which is what the raw-dep comparison was approximating. No objection
to the deletion on the standalone path.

## F14 - the `removed_buffers` rejection is gone with no replacement

The old loop had:

```python
parent_deps = parent_reads_by_name.get(dep.name, [])
if not parent_deps:
    if dep.name in V.graph.removed_buffers:
        return None
    continue
```

The new loop iterates `parent_source_names`, so a buffer with no parent read is
never considered at all - the `removed_buffers` test has no counterpart, and
grep confirms `V.graph.removed_buffers` is no longer referenced anywhere in the
planner.

The case it covered: an epilogue node reads a buffer that no parent reduction
reads and that has been removed. It is now classified as an "independent
epilogue input" and left to an ordinary derived-domain load. Both outcomes are
bad. If the buffer really has no allocation, that is a load of nothing. If it
was removed because it was inlined into this kernel, `CSEProxy.load` returns
`store_cache[name]` **ignoring the index**, so the epilogue receives a
parent-resolution register in lane space - silently.

This is the same family as F10 and the same resolution applies: the planner must
not silently accept a name it cannot prove is loadable. Cheapest fix is to keep
the rejection - one condition in the independent-input path.

## F15 - the explicit ambiguity scope was added last round and removed this round

The previous section implemented `ambiguity_nodes` on my recommendation; this
prototype deletes it and makes the scope implicit in "whatever nodes are in the
parent view". For #190594 that is provably the same set, so no behavior change
here.

But invariant 7 returns in a harder form. #190595's append path deliberately
runs its ambiguity check over a *narrower* set than its `fused_buffer_names`
derivation, and under the new structure that narrower scope can no longer be
expressed as an argument - it would require constructing a different parent
view. Decide during replay whether the append path (a) is fine widening to its
full parent view, or (b) needs its own view. Recording the answer now, while the
reasoning is fresh, will be cheaper than rediscovering it mid-replay.

## F16 - the name-equality check is a precondition, not validation

`parent_r` and `child_r` are the **same sympy symbol**: both views are
normalized independently from `n0`, so the trailing reduce var carries the same
name in each. That makes

```python
expected = parent_index.subs(parent_r, sub_parent_factor * child_r + lane)
```

meaningful only because `parent_reduction != child_reduction` was rejected
twenty lines earlier. If that check were ever relaxed to a length or range
comparison, the substitution would silently become a no-op or substitute the
wrong variable, and the proof would pass vacuously.

The check currently reads as domain-shape validation. It is also what makes the
lane substitution well-formed. One comment saying so, at the check rather than
at the substitution.

## F17 - `source_sizes` changed meaning

`interleaved_sub_parent_lane` previously received `parent_dep.size` - one dep's
per-dimension sizes. It now receives
`tuple(parent_accesses.var_ranges.values())` - every var range in the normalized
parent view, x vars and reduce vars together. The helper builds
`sympy_product(source_sizes[i:])` suffix products from it, so the candidate
extents it will match a SIZE symbol against are now a different set.

That is a behavior change in a symbolic-extent heuristic, in the direction of
more candidates. It may well be what you want, but the 250-test run at #190594
is unlikely to exercise it - the dynamic-shape coverage for this path lives
further up the stack. Either confirm with a dynamic-shape test at this commit or
note it as intentional.

## Smaller

- The preflight loop over `parent_nodes` is a hand-maintained mirror of
  `extract_normalized_read_writes`'s internal precondition. The failure mode is
  a crash rather than wrong code, so this is not urgent, but the two should
  reference each other - otherwise a change to the normalizer's assumptions
  surfaces as an assert in the middle of scheduling.
- `sympy_subs(parent_index, extent_subs)` and raw `parent_index.subs(...)` now
  sit two lines apart. The raw `.subs` is inherited from the deleted matcher, so
  this is not new, but the adjacency makes it look like a deliberate
  distinction. If there is one, say what it is.
- `_try_get_sub_parent_source_layouts` now takes six positional parameters and
  does domain preflight, view construction, domain-signature comparison, extent
  proof and per-source lane proof. The extraction was worth doing, but the
  helper is now the largest thing in the planner; the domain-signature
  comparison block is separable and would read well as its own predicate.

Running reviewer notes - cross-round patterns, open-item ledger, and an explicit
account of what was verified statically versus executed - are kept separately in
`agent_space/reviewer_notes_sub_parent.md` to avoid diluting this channel.

---

# Reconciliation of normalized-source reviews - 2026-08-12

The two reviews above were written concurrently. This section reconciles their
overlap and corrects the scope of two claims. No source code was changed.

## Removed-buffer guard - retain it or prove it unreachable

The later review is correct that the prototype deletes the only explicit
`V.graph.removed_buffers` rejection from this planner. The affected set is
narrower than its write-up says because the old code first skipped names in
`fused_buffer_names` that were not parent reads. The deleted branch therefore
protected an epilogue read whose name:

1. was not a planned parent read;
2. was not an output of the current candidate group; and
3. was already in `V.graph.removed_buffers`.

It did not protect every internal value produced in the candidate; those were
already handled by the preceding `fused_buffer_names` branch. Even with that
narrowing, deleting a safety rejection without establishing that the state is
unreachable is not justified. Either retain the condition for independent
epilogue inputs or document and test why scheduling cannot observe such a name.
This is the isolated correctness gate from the concurrent review.

## Dynamic source-size coverage already exists at #190594

F17's requested dynamic test is already present and was included in the
executed 52-test run:

- `test_dynamic_standalone_sub_parent_epilogue` covers dynamic batch and dynamic
  reduction extents, including non-power-of-two even R values;
- `test_pointwise_producer_standalone_sub_parent_epilogue` covers dynamic batch
  and feature extents;
- both are instantiated under forced persistent and forced looped classes.

That does not prove the larger normalized-access design, but it does exercise
the changed `source_sizes=tuple(parent_accesses.var_ranges.values())` path. F17
does not need another test unless a specific symbolic expression remains
uncovered.

## Consolidated disposition

Before amending the prototype:

1. resolve the removed-buffer guard above;
2. measure or mitigate the eight fused-access normalizations observed in one
   positive compile;
3. add a positive case for distinct raw parent accesses that normalize to one
   canonical access, or state that no acceptance expansion is intended.

During stack replay, separately preserve #190595's ambiguity scope and internal
write materialization, then re-derive #190596's CONTIGUOUS layout proof. F16's
comment request is useful interface documentation but is not a correctness
blocker while the exact symbol-and-range equality checks remain in place.

---

# Explicit codegen-domain remapping - 2026-08-12

The independent normalized-view prototype above has been replaced locally. It
required two scheduler-specific predicates to reconcile domains selected
independently by `NodeSplitGetter`. Extending
`extract_normalized_read_writes(..., target_splits=...)` was prototyped and
worked for all accepted focused cases, but it needed an exact-refinement rule:
`apply_var_mapping` can split one native dimension across target dimensions but
cannot soundly merge native dimensions into one target dimension.

The current implementation instead reuses codegen's existing mapping:

1. represent the parent as explicit flat `(X, R)` symbols and the epilogue as
   `(X, R / factor)`;
2. use `SIMDKernel.map_kernel_groups_to_node_sizes` with `decompose_index` to
   map those symbols into each node's native loop factorization;
3. use `dependencies.extract_loop_body_with_args` to obtain the resulting
   `MemoryDep` indices;
4. require each child source index to equal the parent index under
   `parent_r = factor * child_r + lane`.

This is the same remapping path used by staged codegen, supports both splitting
and merging native loop dimensions, deletes the manual normalized-domain
comparison, and leaves `tiling_utils.py` unchanged. The current tracked diff is
net -15 scheduler lines, two positive regressions, one controlled negative,
and the masked-load forwarding guards described below.

## Adversarial findings incorporated

- The planner must compute the physical lane with
  `interleaved_sub_parent_lane`, require it to be constant, and substitute that
  exact lane. Merely accepting any logical lane is unsound because codegen
  selects the resident tuple value with `index % factor`. The new odd-offset /
  odd-stride test fails by compile error or unwanted fusion when this check is
  weakened, and passes with the current check.
- Indirect source accesses are rejected. Independently extracted loop bodies
  canonicalize indirect values as `tmp0`, which does not preserve provenance;
  structural equality is therefore not a valid proof across the two bodies.
- Reduced-output and full-parent siblings are now checked with
  `_pointwise_node_matches_domain` in candidate classification, including
  `SIMDKernel.is_compatible`, before source analysis. The source extractor no
  longer needs to validate unrelated node domains as a side effect.
- `Identity` wrappers are removed before comparing extracted indices, matching
  the old normalizer's behavior.
- Sources produced inside the candidate remain rejected explicitly. Supporting
  them needs the #190595 materialization work described by the existing TODO.

## Verification

- `python test/inductor/test_nested_reduction.py -k standalone_sub_parent`:
  58 tests run, 1 skipped after the final changes.
- `python test/inductor/test_nested_reduction.py`: 256 passed, 1 skipped after
  the final conservative checks and all new regressions.
- `python test/inductor/test_nested_reduction.py -k standalone_sub_parent_rejects`:
  24 passed after the remapper rewrite.
- The odd-offset test was also run against the weakened any-lane proof and
  failed as intended, demonstrating that it isolates the physical-lane
  invariant.
- A concatenated-input regression exercises `Identity`-wrapped indices in both
  persistent and looped codegen. It failed during codegen before lane analysis
  was made shared and passes with one staged kernel now.
- A mismatched masked-load regression uses fill zero for the reduction and fill
  one for the epilogue. Name/index-only forwarding previously miscomputed the
  persistent case. Masked parent loads now skip materialization and masked
  epilogue loads skip resident-value resolution, preserving each access's mask
  and fill while keeping the staged fusion.

## Final reuse search

Three independent reviews compared the local adapter with the existing tiling,
dependency-normalization, loop-ordering, and reindexing APIs. No higher-level
API preserves all of the required information:

- `extract_normalized_read_writes` chooses each domain independently and cannot
  establish the scaled parent/child axis correspondence;
- `MemoryDep.normalize` may merge or reorder away the logical X/R boundary;
- `LoopBody.reindex_iter_loops` does not reindex reduction axes;
- loop-ordering utilities only permute existing axes.

The implementation therefore keeps the proof local while reusing the two
canonical primitives that do cover it:
`SIMDKernel.map_kernel_groups_to_node_sizes` and
`dependencies.extract_loop_body_with_args`.

Submitted with `ghstack --no-stack` to #190594. Local orig is `1f97117a8cd` and
the PR head is `4c2b0ed8e0e`; the upper stack was intentionally untouched.

---

# Local MemoryDep domain reindex prototype - 2026-08-12

The loop-body replay described above has been replaced locally, without
amending. The scheduler now iterates its existing `MemoryDep` reads and calls
`MemoryDep.normalize_with_ranges` to express each full-domain access in the
parent `(X, R)` or child `(X, R / factor)` domain.

## Soundness boundary

`MemoryDep` normalization can drop trailing iteration variables that the index
does not use. Their original domain position cannot be recovered from the dep,
so `normalize_with_ranges` is deliberately partial:

- indirect accesses return `None`;
- nonconstant accesses require equal source and target iteration products;
- constant accesses are valid unchanged in any target domain;
- shared nontrivial broadcast sources conservatively reject the staged fusion.

The product guard prevents the ambiguous `w[x]` versus `w[r]` case from being
reinterpreted as a plausible access in the wrong domain. A feature-level
negative test pins that capability trade. A separate scalar-source test proves
that constant broadcast inputs still fuse in both persistent and looped forms.
Parent-only broadcast sources do not need a cross-domain proof: child accesses
are normalized first, and parent normalization is restricted to names the child
actually reads. A positive test pins that behavior.

## Reusing the static range mapper

The first prototype flattened the full target domain and decomposed it into the
source dep's sizes. That was mathematically correct, but dynamic producer shapes
created equivalent forms such as:

```
2 * (z % M) + lane
(2 * z + lane) % (2 * M)
```

Rather than add a `pull_out_modular_factor` simplification or maintain another
reshape implementation, the final prototype calls the existing static
`SIMDKernel.map_kernel_groups_to_node_sizes` from
`MemoryDep.normalize_with_ranges`. It maps the target variables through the
dependency's retained sizes using `decompose_index`. The method is static and
uses no kernel state; loop-body replay remains removed. Parent and child proofs
therefore get the same coordinate structure as codegen, and the scheduler uses
ordinary `statically_known_equals` with no modular arithmetic helper.

## Net effect and verification

Production diff relative to the submitted PR head:

- `dependencies.py`: +50 lines for the reusable guarded reindex operation;
- `scheduler.py`: +22 / -43, deleting `CantSplit`, `SIMDKernel` mapping, and
  `extract_loop_body_with_args` from source analysis;
- net production change: +29 lines.

Tests added locally cover the direct reindex, an aligned innermost split, a
fallback reshape, dropped-domain rejection, constant accesses, shared broadcast
rejection, and shared scalar acceptance.

Commands run:

```
python test/inductor/test_dependencies.py
python test/inductor/test_indexing.py
python test/inductor/test_nested_reduction.py
```

Results: 8 passed; 74 passed; 262 passed and 1 skipped, respectively.
`spin lint` still exits before running linters at `lintrunner list` in this
environment. `py_compile` and `git diff --check` pass. No debug prints remain.

Status: validated but uncommitted. No GitHub action has been taken for this
prototype.

---

# Upper-stack replay complete - 2026-08-13

The validated `MemoryDep.normalize_with_ranges` version is now the base of the
locally rebased five-commit stack:

1. `909210d6702` - #190594 standalone INTERLEAVED
2. `56749607757` - #190595 nested INTERLEAVED
3. `011d007e242` - #191775 MXFP6 4:3
4. `6631897a631` - #190596 CONTIGUOUS
5. `7291a69cc95` - #191974 test cleanup

The former `TODO(#190595)` is resolved in the upper stack. Full-resolution
parent-stage writes can be planned sources, dependency-safe source chains are
deferred to a later looped pass, and codegen materializes them on the correct
side of CSE/store-cache invalidation.

Adversarial replay found two correctness issues and added focused regressions:

- A deferred source chain containing another reduction needed a third looped
  pass before the derived epilogue. Testing now catches an undefined
  accumulator if that pass boundary is omitted.
- Internal CONTIGUOUS sources require a persistent resident tile. Fusion now
  asks the ordinary persistence heuristic and codegen forces that choice,
  including when multi-kernel choice generation is enabled. Forced-looped
  codegen declines instead.

The remaining limits are conservative lost fusions: dynamic/non-power-of-two
CONTIGUOUS R, multiple parent indices for one source name, masked resident
forwarding, and nested append beyond factor-2 INTERLEAVED. They are not needed
to make the current stack correct and should remain follow-ups.

Verification at `7291a69cc95`:

```text
python test/inductor/test_nested_reduction.py
  426 passed, 9 skipped

python test/inductor/test_inductor_scheduler.py
  78 passed, 4 skipped

UV_OFFLINE=1 spin quicklint -a
  clean

LOOP_ORDERING=0 COALESCE_ANALYSIS=0 \
  python agent_space/check_nested_without_loop_ordering.py
  15 passed, 1 skipped
```

Two independent final reviews found no remaining actionable issue in the upper
stack or the test-helper replay. The stack was submitted with `[WIP]` prefixed
to every PR title.

---

# Final rebased-stack adversarial audit - 2026-08-13

The five local commits above were replayed onto `56de3c7e232` and reviewed as
one emitted pipeline. No new blocking correctness issue was found.

The audit checked these contracts directly:

- fusion gives every standalone staged group a `FusedStagedReduction` identity
  and reverse-discovered nested appends are normalized producer-first;
- final plans are rebuilt after loop merging, while a missing final plan is a
  compiler invariant failure rather than a generic-codegen fallback;
- every shared source has one normalized parent access and every child access
  proves the exact INTERLEAVED or CONTIGUOUS projection codegen selects;
- the common `_SubParentSourceLoadResolver` reuses only CSE-live values,
  reloads expired external values, and requires planned internal values;
- looped internal-source chains remain dependency ordered, including the
  three-pass case where the deferred chain contains a reduction;
- derived-stage accesses participate in index-width analysis without changing
  the parent schedule used for tiling;
- `min_rblock` is only the lane factor in all forms; ordinary persistence
  selection remains responsible for a full resident block;
- dynamic interleaved R has both multi-value numeric coverage and a kernel-form
  check pinning three loads, three stores, no `tl.split`, and `min_rblock=2`.

Independent verification while replaying each feature layer:

```text
#190595: 326 passed, 2 skipped; scheduler 56 passed, 4 skipped
#191775: 362 passed, 7 skipped; scheduler 78 passed, 4 skipped
#190596: 426 passed, 9 skipped; scheduler 78 passed, 4 skipped
tip:      426 passed, 9 skipped; scheduler 78 passed, 4 skipped
```

The final tip also passes `UV_OFFLINE=1 spin quicklint -a`, `py_compile`, and
`git diff --check`. Tracked files are clean. Remaining TODOs describe
conservative capability or performance work: indexed projection caching,
post-sub-parent consumers, coalesced looped lane loads, multiple parent
accesses, dynamic/non-power-of-two CONTIGUOUS R, and nested append beyond
factor-2 INTERLEAVED. They should not be deleted merely to make the stack look
finished.

---

# Local rollout-independence follow-up - 2026-08-13

Current tip is `7c784108376`; the following changes are intentionally
uncommitted and belong primarily in #191775:

- Preshuffled MXFP6 no longer accidentally requires
  `loop_ordering_after_fusion=True`. Its specialized consumer reindex already
  proves and applies the physical permutation; when loop ordering is disabled,
  it now re-extracts canonical dependencies just as the generic
  reduction-pointwise reindex path does.
- The specialized reindex honors `loop_reindexing_after_fusion=False` and
  declines to the existing multi-kernel fallback.
- The test matrix covers fbcode-style `(loop ordering=False, reindexing=True)`,
  the OSS defaults, explicit reindex opt-out, and the shifted-layout rejection,
  in both persistent and looped modes.
- `_LoopMutationTracker` now states its actual lifetime: a legal mutation may
  survive when the scheduler later selects another candidate. This is
  semantics-preserving and is existing scheduler behavior, not transactionality
  through final fusion selection.

A separate cache-disabled test asserts that `triton.nested_reduction=False`
does not increment `metrics.codegen_nested_reduction`. This is dedicated rather
than placed in `check_nested_matches_unnested`, where an FX graph-cache hit
could make the assertion vacuous and disabling the cache would affect about
fifty callers.

Adversarial conclusions:

- The normalized dependency proof is safe and narrowly scoped; ordinary fusion
  matching is unchanged.
- #190595's append gates, producer-first ordering, plan rebuilding, and named
  constants have no remaining actionable defect.
- #190596 does not need another stored child extent: standalone codegen passes
  `plan.parent_rnumel` directly to the layout, and nested CONTIGUOUS planning is
  disabled.
- Looped MXFP6 second-pass CSE/store forwarding is sound; generated code has the
  expected pass, load, store, and split counts.

Verification:

```text
python test/inductor/test_nested_reduction.py
  430 passed, 9 skipped

python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_nested_reduction_config_disabled
  1 passed

python test/inductor/test_inductor_scheduler.py
  86 passed, 6 skipped

UV_OFFLINE=1 spin quicklint -a
  clean
```

No commit, amend, ghstack submission, or GitHub action has been performed for
this follow-up.
## Adversarial review - MXFP6 inline asm integration - 2026-08-14

Reviewed `dcbf139a847` and the inline-asm follow-up, now amended locally as
`0adc27a2563` (#193434). The staged factor/output-lane planner and codegen had
no additional correctness findings. Rejected consumer reindexing rolls back,
the swapped-order retry is config-gated, and persistent/looped emission agrees.

Two inline-asm issues were fixed in `0adc27a2563`:

- `pack > 1` now restores upcast inputs to their physical dtype before Triton
  packs sub-32-bit lanes into 32-bit asm operands. A mixed float32/float16,
  tuple-output regression test covers the packed-register ABI.
- `CuteDSLOpOverrides.inline_asm_elementwise` accepts the extended handler
  signature, preserving single-output lowering and explicitly declining
  multiple returned tensors until CuTeDSL implements coupled results.

Coverage was strengthened with runtime dynamic tuple outputs at sizes 37 and
65, and the independent MXFP6 reference now uses `B=7`, `G=36`, and distinct
group contents to exercise X tails, padded reduction blocks, and group address
ordering in forced persistent and looped kernels.

Deferred: tuple-returning inline asm still lowers to separate Pointwise nodes.
Ordinary kernels reunify the call through Triton CSE, but derived passes need
explicit realization to avoid recomputation. Prototype automatic realization
before adding a native multi-output Pointwise IR node; compare fusion and
buffer-elimination behavior before choosing.

Verification:

```
55 inline-asm HOP tests passed
4 persistent/looped MXFP6 integration tests passed
1 CuTeDSL inline-asm compatibility test passed
UV_OFFLINE=1 spin lint <four changed files>
UV_OFFLINE=1 lintrunner -a
git diff --check
```
