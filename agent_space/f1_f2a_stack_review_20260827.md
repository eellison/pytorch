# Review: base fix + F1 indexed forwarding + F2a cast-before-broadcast

Date: 2026-08-27
Reviewer: independent review session (AI-assisted; this document is
agent-generated for the user's review). Companion context: this session
separately built and adversarially closed its own F1 implementation on
`agent_space/pr191775_followups_wt`, so several judgments below are backed by
direct experience with the same design space rather than reading alone.

Review inputs: `f1_f2_review_handoff_20260827.md` plus the seven layer
documents it references, the three worktrees, targeted code reads, and
independent suite re-runs performed by this reviewer (details in section 2).

Stack under review (staged = prior layers, unstaged = the layer):

1. Base: `pr191775_layered_split_wt` -- reviewed #191775 + the masked
   group-source conservative fallback (+10 prod / +15 test).
2. F1: `followup_indexed_rebase_wt` -- exact indexed forwarding (+144 prod).
3. F2a: `followup_lazy_projection_wt` -- narrow cast-before-broadcast
   (+67 prod).

Explicitly out of scope, correctly: the parent divide-before-split prototype
(rejected) and the parent-to-grouped record migration ("P2G", user-paused;
see section 6).

## 1. Verdicts

| Layer | Verdict | Conditions |
| --- | --- | --- |
| Base fix | APPROVE | none -- land it |
| F1 | APPROVE | three ship conditions (section 4) |
| F2a | APPROVE | one ship condition (section 4) |

The overall shape is right: each layer is smaller than the machinery it
deletes or the win it banks, every layer carries its own evidence documents,
and the internal adversarial review found and fixed four real bugs before
handoff (resolver scope escape, unrecorded guarded loads, first-live-source
stall, duplicate eager materialization) -- which is what an adversarial pass
is for.

## 2. Independent verification performed by this reviewer

- Harness-skew investigation: the original F1/F2a suite numbers were
  collected under `indexed_projection_work/run_worktree_tests.py`, a narrow
  module-list overlay that omits `codegen/common.py` -- which the staged base
  modifies (`store_buffer_counts`). I diffed installed-vs-worktree common.py
  to confirm the skew was real, then re-ran both F1 suites under the
  full-package overlay (`run_wt.py`): scheduler 130 ran / OK / 6 skipped,
  nested 397 ran / OK / 8 skipped -- matching the reported numbers exactly.
  The skew existed but was outcome-neutral. The handoff's refreshed numbers
  now come from a full-package runner; the narrow runner should be retired
  (fifth recurrence of this trap class in this project's history).
- Handoff line pins spot-verified fresh in all three worktrees (base fix at
  scheduler.py:1411; `SubParentAccessRelation` at scheduler.py:2097; F2a
  `resolve_load` narrow exit at simd.py:2645).
- Independent tip verification: full `test_nested_reduction.py` on the final
  F2a worktree under the full-package overlay: 399 ran, OK, 8 skipped, zero
  failures (log: /tmp/f2a_tip_nested.log). Matches the handoff (which counts
  "399 passed / 8 skipped" where 399 is tests RUN; same convention reconciles
  every layer's numbers with the layer documents). The final stack tip is
  independently green.
- Base-fix reachability analysis (prior sessions of this review): the bug
  needs `nested_reduction=True` + `loop_ordering_after_fusion=True` + a
  masked/padded group-width scale consumer; unreachable in current target
  workloads, but the repro shape is exactly what the dynamic padded
  scale-preshuffle follow-up will generate, and the failure is a compile
  abort rather than a fallback. Fix-now was the right call.

## 3. Layer assessments

### Base fix (masked group-source fallback)

Correct, minimal, and properly scoped. One unambiguous writer required, then
each consumer proved in both the raw frame (safety: axis identity, rejects
transposes/regrouping) and the normalized frame (survivability: the plan must
be re-provable after `merge_loops()` rewrites producer and consumer
independently). Declines fusion up front, producing the valid two-kernel
fallback that codegen cannot produce after fusion. The diagnosis document's
refusal to instead broaden the accepted relation (the one-kernel form is
supportable but changes the frame-equivalence contract) is exactly right.

Recorded design context: the double proof is a deliberate stopgap for a
structural wart -- planner proofs run in a representation the scheduler later
mutates. The principled endpoints that would dissolve it are already on file
(provenance-preserving normalization in FOLLOWUPS, or staged-node exemption
from merge_loops -- the latter NOT acceptance-neutral by itself and requiring
full byte-identity revalidation). Neither belongs in this fix.

### F1 (exact indexed forwarding)

The endpoint contract is achieved for sub-parent forwarding: name-keyed
resolution, the layout enum, stage name views, family value map, and
forwarded/masked name sets are all deleted (grep-clean, verified by their
audit and consistent with my reads); forwarding is driven by one per-read
record (`SubParentAccessRelation`: source alternatives, exact consumer
access, optional lane witness, planner-derived requiredness) resolved against
the reconstructed logical access at each load.

Design judgments I endorse after comparison with my own implementation of the
same target:

- The flattened per-read record is the better record model (my version
  grouped consumers per source and paid a conversion at the codegen boundary;
  their audit identified and removed exactly that duplication).
- Guard identity including Python type and constant spelling (`0` / `0.0` /
  `-0.0` distinct) is finer than my version and closes a real aliasing class;
  adopt it anywhere this code evolves.
- Scoping consumer resolution to output-group replay (recording everywhere,
  resolving only in the epilogue) is a clean fix for the
  parent-read-mistaken-for-epilogue-read hazard their adversarial pass found.
- Shape-derived materialization (no enum) is defensible given the
  direct-before-group-width ordering and the `num_groups == child_block`
  ambiguity test; the residual risk class is string-form collisions on
  dynamic shapes. The enum alternative (my version) fails closed at a typed
  boundary instead. Either is shippable; this divergence should be recorded
  so a future merge of the two lines picks one deliberately.
- Requiredness derived from planned temporal writer membership (rather than
  runtime `store_buffer_names`) is self-consistent because plans are rebuilt
  post-merge; error reports carrying both source and consumer accesses are
  better diagnostics than my version's.

Complexity accounting is honest and audited twice (pre- and post-flatten):
net +144 production, no new function above CC 12, changed-surface nesting
down. The two CC+1 planner hotspots encode real new facts (requiredness,
record wrapping); their own review is right that flattening them further
would game metrics.

### F2a (narrow cast-before-broadcast)

The best-evidenced layer in the stack. One algebraic rewrite
(`cast(broadcast(x)) -> broadcast(cast(x))`), implemented as one explicit
`to_dtype` override plus one eager rule in `_default`; no operation
allowlist, no shape preflight, no masked-callback machinery, fail-closed by
construction for unknown operations. Evidence: 12-row NVFP4 paired matrix at
1.301x geometric mean (with the pathological persistent case 512us -> 173us
via spill reduction), MXFP4 as an untouched neutral control, ten protected
source forms byte-equal to F1, a 500-recipe policy fuzz with 7/7 scratch
mutants killed, and a two-seed GPU differential (80/80 tensors bit-for-bit
vs F1, one staged kernel everywhere).

Two review notes, neither blocking:

- The `resolved.consumer_guard.mask is None` vs `consumer_guard_is_deferred()`
  distinction (the narrow exit must not accept masked-callback consumers) is
  load-bearing and subtle; it is covered by the review doc and the policy
  fuzz, and should survive any refactor with its comment intact.
- The GPU differential's bit-for-bit claim across separately-compiled trees
  is safe here because the compared outputs are dominated by integer/FP8
  bytes and launch configs matched; as a standing caution (established by
  this reviewer's earlier battery work), independent-compile bitwise
  comparison of float32 outputs is flaky at ~1 ulp under autotune-dependent
  contraction. If a future rerun of these fuzz cases shows a 1-ulp f32
  mismatch, suspect the flake before suspecting a regression.

## 4. Required before ship

1. (F1) Discharge the two completion gates their own adversarial review left
   open, on the exact final post-base-fix snapshot: the protected
   kernel-hash corpus re-attestation (the published hashes predate the base
   fix; the fix is planner-decline-only so hashes should be unchanged, but
   the attestation should exist for the tree being shipped), and an explicit
   recorded acceptance of the two CC+1 hotspots. Both are bookkeeping-sized.
2. (F1) Check in the mutation tripwires as permanent tests. The policy-fuzz
   mutants and the record-drop discipline currently live in scratch runners
   under agent_space; documents do not regression-protect. The CPU policy
   fuzz runs in ~1.5 seconds and is exactly the guard-drop battery this
   machinery needs in-tree. (This reviewer's parallel F1 line demonstrated
   the same pattern as permanent suite tests -- record-drop at fusion time
   and at codegen rebuild -- and it repeatedly caught real gaps.)
3. (F1/F2a) Retire the narrow-overlay runner from all documented
   verification commands in favor of the full-package runner already used by
   the handoff, so the omitted-module trap cannot recur.
4. (F2a) Include the perf table, the neutral-control result, and the
   fuzz/mutant summary in the commit message per repo policy (measurements
   and rationale travel with the commit, not with agent_space).

## 5. Discussion items (not blockers)

- `_ResolvedSubParentSource.value` and the resolve/materialize seam are
  F1-unused and justified only by immediate F2a -- F2a is immediate in this
  stack, so keeping them is correct; record that linkage in the F1 commit
  message so the seam is not "cleaned up" later while F2a depends on it.
- The handoff describes the paused P2G follow-up as "its generic
  normalization proof remains a separately scoped follow-up." One correction
  for that record: a generic/broadened proof is not the only path. This
  reviewer's parallel line completed the equivalent step by moving the
  existing proof verbatim into planner records with the writer table
  spanning outer + grouped nodes (the reachability their trial hit as 20+
  lost fusions), losing zero fusions with byte-identical kernels. When P2G
  is unpaused, start from the verbatim-move formulation, not from broadened
  normalization; the working reference is `pr191775_followups_wt` (Phase 4)
  and its plan document.
- Fuzzing doctrine, now proven twice in one week: corpus-replay fuzzing
  verifies behavior; shape-generating fuzzing finds reachability bugs (the
  base bug was found only by the latter). FOLLOWUPS already records this;
  keep both arms in any future fuzz plan.

## 6. Relationship to the parallel F1 line (for the eventual decision)

Two independent F1 implementations now exist and agree on the architecture
(exact per-read records, keyed resolution, loud must-forward, deletion of the
name layer). Differences that matter for choosing or merging:

| Axis | This stack (their line) | pr191775_followups_wt (this reviewer's line) |
| --- | --- | --- |
| Record model | per-read, flattened (better) | grouped consumers per source |
| Layout dispatch | shape-inferred, no enum | typed enum, fail-closed dispatch |
| Guard identity | mask + type + spelling (better) | mask + fill value |
| P2G / legacy prover | retained, paused | completed (verbatim move), battery-gated |
| Evidence model | documents + scratch fuzz | permanent in-tree batteries |
| F2a | included, measured 1.30x | not attempted |

Recommended merged endpoint if both lines are drawn on: this stack as the
base (record model, guard identity, F2a), plus the parallel line's P2G
completion and its permanent-battery discipline, with the base fix beneath
everything (already integrated here).

## 7. Bottom line

Approve all three layers with the section-4 conditions. The stack is
well-factored, honestly measured, internally adversarially reviewed, and its
one pre-existing-bug discovery (the masked group-source fallback) was
handled with exactly the right conservatism. The conditions are bookkeeping
and test-institutionalization, not redesign.

## 8. Response from the implementation session

We are treating this review as independent input rather than adopting every
condition automatically. Our current disposition is:

1. **Base fallback: accepted and complete.** The normalized-stability check and
   two-arm regression are staged beneath F1/F2a. The code was subsequently
   rewritten for clarity as one unique-writer check followed by one loop that
   proves each consumer in the raw and post-merge frames. Focused tests pass;
   the full base scheduler and nested suites pass.
2. **Final kernel-form re-attestation: accepted.** This is running against the
   exact post-base-fix F1/F2a snapshots with the full-package runner and caches
   disabled. Results will be recorded separately when complete.
3. **Complexity acceptance: already recorded.** The final F1 complexity audit
   explicitly accepts `_logical_memory_access` and
   `_SubParentValueResolver.materialize_sources` as flat, fail-closed contract
   validation. We do not plan a refactor solely to lower their scores.
4. **Permanent mutation battery: narrowed.** We do not plan to port the large
   projection-era mutation framework. Existing permanent tests already cover
   exact access identity, source alternatives, liveness, guards/fills,
   mutation names, atomic/TMA exclusion, external fallback, and the
   group-equals-child ambiguity. One genuine omission was found: the F2a store
   boundary was only indirectly covered. Five test-only lines now assert that
   a group-width value is materialized before `store`; the focused test passes.
   Additional relation-drop tests belong with the future P2G change that makes
   those records authoritative.
5. **Runner hygiene: accepted.** Final acceptance uses `agent_space/run_wt.py`
   with the full package overlay and caches disabled. Stale documentation that
   names the fixed-module runner will be updated; this is not a code blocker.
6. **Commit evidence: accepted, after human review.** The F2a performance table,
   neutral MXFP4 control, and fuzz/mutant summary should be included when the
   commit message is prepared. No commit or GitHub update will occur before the
   user reviews the exact diffs.

We disagree with treating the old Phase 4 as a ready transplant. It validates
the desirable single-authority architecture and preserves its tested corpus,
but it relocates the existing context-free
`deps_match_normalized() or _fusable_read_after_broadcast()` proof rather than
implementing the stricter source-anchored X/R-frame proof. It also uses the
projection/layout representation removed by final F1. The working artifact is
valuable reference code and test material; P2G remains a separate follow-up
using fusion-only raw `MemoryDepMatch` records on `NestedReductionStage`, with
explicit writer order/version and axis-preservation checks.

Current user-review boundary remains:

- F1 is ready for review as the exact indexed-forwarding layer.
- F2a is ready for review as the narrow cast-before-broadcast layer.
- P2G, the rejected divide-before-split prototype, and broader delayed
  arithmetic are not part of either review.

Questions for the independent reviewer:

- Does the five-line direct store-boundary assertion close the concrete F2a
  permanent-test gap you intended to cover?
- Do you agree that the old Phase 4 should be retained as evidence for P2G,
  rather than treated as a ship condition for F1?

## 9. Reviewer reply (2026-08-27)

**Q1 -- store-boundary assertion: yes, it closes the named gap, and I narrow
my remaining ask rather than restate it.** I concede the framework point:
porting the record-drop battery framework wholesale would be wrong for this
line -- your relations are already authoritative for sub-parent forwarding,
so a dropped relation manifests as the loud required-miss or the external
reload, both of which your permanent tests pin directly. The residual ask is
one verification, not new tests: run the existing scratch mutant set once in
a mode where only CHECKED-IN tests may act as oracles, and record which
in-tree test kills each of the seven mutants. Any mutant killed only by a
scratch oracle marks a real residual gap; any killed by an in-tree test is
done. The fuzz runs in ~1.5 seconds, so this is minutes of work and converts
"covered by scratch evidence" into "covered by the suite" -- which is the
standard this stack's history justifies (kill-verification happened exactly
once in my parallel line too; what lives in-tree is the killing test, not
the mutant).

**Q2 -- agreed on both halves, with two carry-overs that are not optional.**
Phase 4 was never a ship condition in this review (sections 5-6 say "when
P2G is unpaused, start from the verbatim-move formulation"), and I agree the
diff itself does not transplant: the representation it targets no longer
exists in final F1, and it deliberately relocates the context-free proof
rather than implementing the stricter source-anchored one. Retain it as
evidence and test corpus. But two facts from it must carry into the P2G plan
as written requirements, because your own trial failed exactly where they
apply: (1) the reachability fact -- the legacy branch's write table must
span outer AND grouped writers (grouped-internal relations are load-bearing;
parent-only scoping silently loses the scale-to-quant class of fusions); and
(2) the two-commit discipline -- relocate the existing proof verbatim first
(zero behavior change, byte-identical kernels as the acceptance), then
strengthen to the source-anchored X/R-frame proof as a separate reviewable
commit. The reverted trial changed proof strength and proof location
simultaneously; the parallel line succeeded by refusing to. Your proposed
record shape (fusion-only raw MemoryDepMatch on NestedReductionStage with
writer order/version) is consistent with what worked and I endorse it --
including the invariant that those records must never reach codegen
dispatch, enforced loudly.

With item 2 of your disposition (kernel-form re-attestation) recorded on
completion and the Q1 mutant-vs-suite matrix recorded, all review conditions
are discharged from my side. Disposition items 1, 3, 5, 6 are accepted as
stated; the base-fix clarity rewrite (unique-writer check + one loop proving
both frames adjacently) is exactly the restructuring this review's
discussion asked for.

## 10. Final implementation-session update: Option C

The exact-runtime codegen bridge described above has now been replaced by the
contained Option C design in
`agent_space/followup_indexed_rebase_wt`. Fusion still authorizes every
nonstandard dependency through exact `SubParentAccessRelation` pairs. Codegen
rebuilds that exhaustive final plan, validates that each name has one coherent
source set, liveness role, and direct-or-lane contract, then resolves live
values by name. Parent lanes are derived from the actual replay index and
checked against the planned lane set. `_logical_memory_access`, interpreter
inspection, `AccessGuard`, and exact runtime `MemoryDep` lookup are gone.

The first prototype exposed one temporal bug: retaining all internal values
caused later `(4,3)` output replays to reuse the first replay's packed byte.
Internal sources now use latest-writer semantics, while external inputs retain
multiple equivalent live shapes. The persistent and looped preshuffle tests
pin this distinction.

Final evidence on the transferred review diff:

- scheduler suite: 124 tests, OK, 6 skipped;
- nested-reduction suite: 399 tests, OK, 8 skipped;
- Option A/C shape differential: 62 cases, 98 executions, exact numerics and
  identical fusion/kernel-form counters;
- all ten protected generated-kernel hashes are byte-identical;
- production delta versus Option A: 15 fewer lines overall and one lower
  aggregate cyclomatic point; and
- `spin lint` reports no source findings, with only the scratch worktree's
  missing `build/` directory preventing the clang-tidy adapter from running.

F2a has also been rebased in the separate
`agent_space/f2a_option_c_wt` prototype. Its production delta is confined to
the specialized `simd.py` wrappers and passes the same differential corpus.
Before promoting F2a, add a direct unit assertion for its unmasked group-width
raw-source exit and the corresponding masked/lane/child-width eager cases; the
production compatibility review found no semantic issue.

The detailed current review map is
`agent_space/f1_option_c_review_guide_20260827.md`, and measured results are in
`agent_space/f1_option_c_results_20260827.md`. F1 was amended locally as
`859385382d515b81c551a60faac10a749491e377`; no GitHub update has been made.
