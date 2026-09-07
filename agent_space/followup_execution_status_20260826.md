# Sub-parent follow-up execution status

Last updated: 2026-08-26

## Frozen baseline

- Reviewed PR worktree: `agent_space/pr191775_layered_split_wt`
- Baseline commit: `6b8ef64bd37`
- Do not edit this worktree while the PR is paused.

## Workstream 1: per-read indexed forwarding

- Worktree: `agent_space/followup_indexed_rebase_wt`
- The frozen PR diff is staged. Follow-up edits remain unstaged.
- Replace name-keyed forwarding with exact emitted-load to planned-access
  matching.
- Delete compatibility machinery made obsolete by that relation, including the
  layout enum, public name views, resolver adapters, and duplicated layout
  proofs where possible.
- Preserve temporal writer identity, mutation-aware matching, mask/fill
  identity, external reload fallback, and loud failure for unavailable
  in-kernel values.

Current design checkpoint:

- Each consumer record carries its exact access and, only for a parent-width
  split, the statically proved lane.
- Direct forwarding and group broadcast are derived from the live value shape
  and active family rather than an `INTERLEAVED`/`BROADCAST`/`IDENTITY` enum.
- Required forwarding is derived from the temporal writer being in the active
  plan/kernel; it is not supplied as another name set.

The inherited parent-to-grouped scheduler equivalence is not part of this F1
rewrite. A first attempt to turn it into plan records used an unsafe broad
normalization and lost more than 20 existing fusions after plan reconstruction.
It was reverted. The existing proof stays visibly scoped to nested-stage reads;
`P2G` below owns its eventual strict replacement.

Acceptance requires a before/after inventory of production LOC, helper/type/API
surface, and complexity. Adding the new mechanism while retaining the old
layout/name scaffolding is not an acceptable result.

Current F1 checkpoint: `+450/-292` production lines (net `+158`), with the
layout enum, three stage compatibility views, family name cache, old resolver,
and forwarded/masked name plumbing removed. A shared output-group replay helper
keeps nested and standalone handler lifetimes identical.

## Workstream 2: lazy group-resolution projection and CSE

- Starts only after workstream 1 has passed review and tests.
- Rebase the useful historical F2 behavior onto exact per-read relations.
- Keep scale-side work group-sized until use when profitable.
- Derive masks from the target shape with `mask_vars_for_shape`; never copy a
  parent mask onto a lane-shaped value.
- Delete historical projection/cache machinery that exact indexed forwarding
  makes redundant.

This is split into F2a (required group-only delayed broadcast/CSE) and F2b
(optional persistent factor-2 divide-before-split). F2b proceeds only if a
fresh benchmark shows at least a 10% flagship gain without exceeding the
combined complexity budget.

## Deferred P2G proof cleanup

- Replace the inherited parent-to-grouped equivalence helper with raw exact
  plan records only after a source-anchored X/R-frame proof can preserve the
  existing fusion set.
- Reject transpose/equal-size axis swaps, ambiguous/out-of-order writes, TMP,
  indirect, and synchronizing accesses.
- Preserve the more than 20 existing accepted nested cases before deleting the
  legacy helper and ownership branch.

## Independent audit

- Maintain a correctness, mutation, fuzz, kernel-form, and performance matrix.
- Record per-function cyclomatic/branch complexity, maximum nesting, helper and
  type count, public API surface, and production LOC before and after.
- Every complexity hotspot must have either a concrete simplification or a
  short reason it is irreducible.

## Sequence

1. Finish the indexed-forwarding delta without touching the frozen PR.
2. Review for soundness and deletion of old scaffolding.
3. Run focused tests, full nested/scheduler files, mutation cases, and fuzzing.
4. Create a fresh F2 layer from the accepted F1 state.
5. Benchmark representative NVFP4, MXFP4, MXFP6, looped, persistent, and tail
   cases before preparing anything for submission.

No commits, ghstack submissions, or GitHub changes are part of this work until
the user returns and reviews the code.
