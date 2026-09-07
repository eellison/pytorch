# Reviewer notes - sub-parent epilogue stack

Running notes from reviewing this work across several rounds. Separate from
`sub_parent_planner_handoff.md`, which is the formal implementer/reviewer
channel - this is the stuff that does not belong in a per-round response:
patterns across rounds, open items at risk of being lost, and honest limits on
what I actually verified.

Last updated 2026-08-12, against local #190594 `5d70ad19ff9` plus the
uncommitted normalization prototype.

---

## The recurring failure mode: loud failure becomes silent fallback

This is the single most consistent pattern in the stack, and it has now shown up
four times in different clothing:

| Round | What happened |
|---|---|
| F10 | `_SubParentPointwiseRemapHandler` deleted; its `removed_buffers` "required" branch had no replacement. Loud `AssertionError` became a parent-resolution register in lane space. |
| F14 | The `removed_buffers` rejection in the source planner deleted with the raw-dep loop. Same shape, same silent outcome. |
| F7 | `requires_looped` sets `override_persistent_reduction=False` and nothing asserts it stuck, while five later branches key off `kernel.persistent_reduction`. |
| c4 (earlier pass) | `remaining.clear()` declared every unmet read satisfied without inspecting an index; later found and fixed as a real wrong-code bug. |

The mechanism is always the same: **a guard's justification lives in a docstring
or a comment, not in a test.** When the surrounding code is restructured, the
prose goes with the code and nothing fails. The refactors themselves have been
good - the problem is that a deleted guard is indistinguishable from a
simplification unless something red happens.

Two cheap habits would break the pattern:

1. When deleting a rejection, name the test that covers the case it rejected. If
   there is no such test, that is the finding.
2. Prefer converting a guard to an assertion over deleting it. An assert that
   never fires costs a line; the same reasoning re-derived six weeks later costs
   a day.

Worth noting the counter-example that went right: the #190595 resolution to F10
(materialize only names carried by the stage; never treat `removed_buffers`
membership as a layout proof) is a *better* answer than the guard I asked for,
because it removes the ambiguity rather than detecting it. That is the standard
to aim for - but it is also why F14 stings, since the same resolution was
available and the guard was dropped instead.

## The plan/codegen boundary keeps oscillating

Decisions here have been made and unmade several times:

- Carry exact `MemoryDep`s on the plan, or only `(name, layout)`? Named-only
  won, then the simd side grew a *superset* reconstruction to get deps back,
  then the proposal formalized names-only without addressing the
  reconstruction.
- Explicit `ambiguity_nodes` scope: absent, then added (last round), then
  removed again by the normalization prototype one section later.

Neither reversal was wrong in isolation. The cost is that each round spends
effort re-deriving what the boundary is *for*. A ten-line "what the plan carries
and why" note next to `StagedReductionPlan` would probably pay for itself - the
current answer is distributed across four review rounds and one deleted
docstring.

## Open items at risk of being lost

The handoff is 4000+ lines and append-only, which is excellent for provenance
and poor for "what is still true". These have not been closed and are now well
above the fold:

- **F7** - `requires_looped` override unverified. The dynamic-R prototype left
  the tree, so this must travel with it when it returns. Highest severity of the
  open set: silent wrong lanes.
- **F9** - the odd-extent negative test is not controlled (different model from
  the positive test; `D=511` conflates three rejection reasons).
- **F12** - the `FusedStagedReduction`-but-not-`FusedNestedReductions` exclusion
  is one invariant written at two sites.
- **F14/F15/F16/F17** - current round, not yet responded to.
- **Your own `WhyNoFuse` finding** - ordinary rejection reason emitted before the
  sub-parent fallback can return `FUSE`, so logs claim rejections for pairs that
  then fuse.
- **The `arange` int32-widening probe** - correctly scoped out of the patch, but
  it exists only in this document. It should be a GitHub issue or it evaporates.

Suggestion: a short living "open items" block at the *top* of the handoff,
rewritten in place rather than appended. Everything else can stay append-only.

## What I actually verified, and what I did not

Being explicit because several of my confirmations look stronger than they are.

**Verified by reading code and tracing callers:** the `CSEProxy.load` name-keyed
forwarding path; `cse.invalidate` being inside `codegen_body`'s loop branch;
`_producer_output_names_read_by_consumer` being namespace-consistent; the
`parent_nodes + epilogue_nodes == scheduler_nodes` argument that makes the
ambiguity deletion safe at #190594; `V.graph.removed_buffers` having no
remaining referent in the planner; the #190595 vs #190594 node-set asymmetry
(checked in two different blobs).

**Static reasoning only, no execution:** everything about reachability. I have
run zero tests across this entire review - the worktree has been shared with an
active session the whole time, and at two points earlier in the session
subagents of mine did disturb it. So when I say a branch is unreachable, that is
a guard-chain argument, not an observation. F14's severity in particular depends
on whether the removed-buffer case is reachable at all, and I did not establish
that it is - only that nothing now prevents it.

**Explicitly stale:** `STACK_SIMPLIFICATION_REVIEW.md`,
`STACK_SIMPLIFICATION_RECOMMENDATIONS.md` and `agent_space/simplify_c*.md`,
`agent_space/pass2_*.md`. Those reviewed a four-commit branch (`stack-fixes`,
tip `f020897c032`) that has since been reorganized into five ghstack PRs with
different commit boundaries and substantial rework. Individual findings there
may still be valid, but every line number is wrong and several items have been
overtaken. If anyone picks those docs up, they should re-locate each symbol
before acting. `SIMPLIFICATION_IDEAS.MD` already carries a SUPERSEDED banner.

## Smaller observations

- `check_fusion()`'s exactly-one equalities are doing more work than the prose
  around them claims: because metrics accumulate across a loop of runtime
  shapes, they pin "one compiled graph, one kernel, one nested fusion" for
  dynamic tests. That is a genuinely good pattern. It is also fragile against a
  well-meaning future relaxation to `assertGreaterEqual`, and nothing in the
  tests says so.
- The negative tests are the weakest part of the suite. Several assert only
  "did not fuse" for models where three independent rejection paths apply, so
  they would keep passing if the intended one regressed. Pinning the *reason* -
  or varying exactly one property between a positive and negative pair - would
  make them regression tests rather than smoke tests.
- The self-assessments in this document have been consistently accurate, with
  one systematic exception: they undersell in one direction and oversell in the
  other. "Lane-count generic" was described as dead generality that is actually
  exercised; the dynamic tests were described as requiring "one staged fusion"
  when they pin something stronger. Worth trusting the code over the summary in
  both directions.
