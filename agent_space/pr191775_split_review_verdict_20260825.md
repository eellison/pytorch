# Review verdict: #191775 prerequisite split (2026-08-25)

> Historical intermediate review. The swizzle blocker below was retracted
> after comparing complete generated kernels: both base and split emit one
> equivalent staged kernel, with only output-pointer ordering changed. Use
> [pr191775_split_review_20260824.md](/data/users/eellison/pytorch/agent_space/pr191775_split_review_20260824.md)
> for the current result.
>
> COUNTERSIGNED by the original reviewer (2026-08-25): the retraction is
> correct. Two independent confirmations: yesterday's traceback failed at
> extra_checks.run, i.e. PAST the single-kernel-form assertions, so fusion
> was intact; and the frame proof examines the consumer's READ of the
> broadcast source while the swizzled indexing lives in its STORE, which the
> proof never inspects -- my root-cause conflated the two. The pointer-
> agnostic FileCheck fix is right. The rest of this review (architecture
> approval, ownership boundary, P1 kill, semantics changes) stands and is
> superseded by the re-verification in the 2026-08-25 follow-up below the
> memory record.

Target: the split described in `pr191775_split_review_20260824.md` --
prerequisite `agent_space/pr191775_prereq_split_wt` (+825/-184,
scheduler+tests only) and upper MXFP6 layer
`agent_space/pr191775_layered_split_wt` (+1962/-448 cumulative), both
uncommitted on #190595 `6b8ef64bd37`. State verified: worktrees, bases, and
diffstats match the doc.

## Verdict

**Approve the architecture; one must-fix regression before either layer
ships.** The split boundary is right, the P1 fix is correct and verified,
and the exact-dependency mechanism landed in its final ownership-precedence
form. But the new broadcast frame proof is over-strict and silently
un-fuses the swizzled-scale epilogue, failing two checked-in kernel-form
tests in both layers -- missed by the doc's focused `-k` verification.

## What is right (verified in code and empirically)

1. **The ownership boundary closes P1 structurally**
   (`_plan_fusion_dependency_matches`): a residual read owned by the
   sub-parent stage is admitted ONLY as an exact plan record gated by
   `_memory_dep_supports_index_equivalence`; the legacy #190594 generic
   equivalence survives only for grouped-stage-owned reads; reads owned by
   neither decline. No fall-through exists. Both P1 attack families pass
   under my harness on the prerequisite (attack6 4/4, attack0 5/5).
2. **Broadcast projections are now plan records** with a fail-closed proof,
   and a non-MemoryDep (StarDep/WeakDep) read of a broadcast source
   declines the whole projection instead of being silently omitted --
   closing the caveat the old can_fuse_vertical comment documented.
3. **Intentional semantics changes are sound**: rewrite suppression now
   triggers on ANY staged plan (is-not-None, strictly safer than the pushed
   bool()); scoring moved from fallback to additive with an exact-read skip
   preventing double-counting (pinned by the same-write dedup test).
4. **Suite matrix** (run_narrow_plus, differential): scheduler suites show
   only the two known-environmental flop_counter float32 errors on all
   three trees (base/prereq/upper) -- zero scheduler regressions. Nested
   suite: base OK; both layers fail exactly the two swizzle tests below and
   nothing else.
5. Round-2 mutation holes have tests in the prerequisite (multiwrite, TMP,
   atomic rejection; empty-plan scoring; same-write dedup), and the P1
   regressions live in the correct (prerequisite) layer.

## MUST FIX: swizzled-scale fusion regression

`test_rmsnorm_block_scale_swizzle_kernel_form` fails in BOTH modes on BOTH
layers (base passes): the swizzle epilogue no longer fuses, so its
swizzled store leaves the staged kernel. Cause: the new
`_sub_parent_broadcast_projections` proof compares RAW per-node var
positions (consumer leading vars must positionally match the source write,
sizes positionally equal, no vars beyond source rank). The swizzle
consumer's reshaped iteration (`x0 // 32`, `x0 % 32` splits) legitimately
has a different var structure and fails those checks, so the whole staged
plan declines. Perf regression + two broken checked-in tests, not a
miscompile.

Fix direction: prove the broadcast in the SHARED frame the way the lane
proof does -- normalize both the source write and the consumer read with
`normalize_with_ranges` anchored to the SOURCE's frame (not each node's own
linearization), then require equality there. That accepts
reshaped-but-legal consumers (the swizzle read normalizes to the write
frame) while still rejecting P1's transposed node (whose read mixes frame
vars after normalization to the shared frame). The two swizzle form tests
are the regression gate; add them to the layer's own focused verification
list.

## Notes (non-blocking)

- The record no longer stores a per-consumer lane (reversal vs the pushed
  redesign). Fine for fusion legality (MemoryDepMatch pairs suffice), but
  the indexed-forwarding follow-up previously consumed the recorded lane;
  its rebase must re-derive lanes at codegen (the resolver already does,
  with a loud assert) or re-add the field. Record this in the F1 rebase
  notes so it is a decision, not a surprise.
- Process, fourth recurrence: the split doc's verification is focused `-k`
  lists only, and the one regression sat outside every selector. Full-file
  suite runs (or at minimum the Internals classes) belong in every
  verification block; the narrow-differential protocol makes them cheap.

## Empirical record (all run_narrow_plus, caches disabled, B200)

```text
nested:    base OK(2 skip) | prereq FAIL(2: swizzle x2) | upper FAIL(2: swizzle x2)
scheduler: base/prereq/upper all FAIL(2: env flop_counter float32) -- no delta
attacks:   prereq attack6 4/4 PASS, attack0 5/5 PASS
```
