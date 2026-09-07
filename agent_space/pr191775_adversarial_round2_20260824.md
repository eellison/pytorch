# #191775 adversarial round 2 (2026-08-24)

Target: worktree `agent_space/stack_rebase_20260824/wt`, HEAD `140296df2441`
(exact-dependency redesign amended in) plus the uncommitted
`projected_access_pairs()` refactor. Three fresh-eyes lenses
(fusion-machinery, codegen, tests/docs) with live repros, mutation testing,
and pristine-base controls. Artifacts: `agent_space/adv_round_191775/
{sched,cg,tests}/`. Environment: full-package overlay unusable at this trunk
distance; everything ran via `run_narrow_plus.py` differentially.

## P1 -- PROVEN silent wrong fusion (BLOCKS #191775 landing)

A frame-mismatched consumer is admitted into the staged kernel and its
output becomes a bitwise copy of another node's values (~96% of elements
wrong vs both nested-off and eager). Repro
`adv_round_191775/sched/attack6_score_and_repro.py` case transposed_8_512
(B=8, D=512, G=16): a transposed sibling read linearizes, in that node's OWN
loop order, to exactly a lane-0 read; the per-node lane proof passes; the
distinguishing scale read (broadcast over the WRONG axis) is admitted by
`_fusable_read_after_broadcast` strategy 1, which drops index-absent vars
position-blind; `_reindex_sub_parent_consumer` is bypassed because the
combined plan succeeds directly; codegen then CSEs the two bodies. Both the
standalone path and the nested append path reproduce
(attack0_misclass_nested.py nested_transposed). Shape-dependent: requires
inductor to pick the mismatched loop order for the bad node (declines at
4x256 and 64x2048).

**Scope (verified by me on pristine bases):** NOT in landed trunk
(e812f6fc74c, all 4 cases pass) and NOT in #190595 alone (70d9ccee025, all
9 cases pass). The bug arrives with #191775's relaxation machinery and is
IDENTICAL in the committed name-based version and the uncommitted redesign:
it survives through the shared `_fusable_read_after_index_equivalence`
escape hatch, i.e. it is the redesign's target bug class escaping through
the one branch the redesign kept.

**Fix directions (either kills it):**
1. Frame-consistent validation: prove residual producer-output reads in the
   SAME normalized (x, child_r) frame the projection proof uses
   (normalize_with_ranges), requiring write(x)-broadcast form there,
   instead of frame-free pairwise broadcast equivalence. And/or require
   frame consistency of each epilogue node's writes before admission.
2. The "eventual model" from the design discussion, pulled forward: record
   broadcast/identity relations in the plan (the indexed-forwarding record
   vocabulary already has this shape) and require EVERY residual to be an
   exact plan-proved relation -- delete the generic-equivalence branch from
   `_plan_fusion_dependency_matches`. P1 turns that from a cleanliness
   argument into a correctness argument.
Either way, transposed_8_512 and nested_transposed become regression tests,
and S1 below folds into the fix.

## P2 -- the uncommitted refactor breaks 8 unit tests

`projected_access_pairs()` extraction: Mock-based plan fixtures lack the
new method (TypeError at scheduler.py:9204), breaking
require_injective_producer[dense|alias] and require_memory_consumer
[star|weak] x2 devices -- including the direct pins for the aliasing and
planned-match branches. Fixture-only: the full nested e2e suite passes
387/387 WITH the refactor active (corroborated by the codegen lens, whose
whole round ran with it loaded). Fix the fixtures or keep inline iteration.

## Coverage holes (mutation-tested: mutant survives both suites = unpinned)

1. Multi-write-producer decline (len(writes) != 1).
2. mutation_renames through a planned match (all fixtures use empty renames).
3. Atomic-mode write gate on the exact path.
4. TMP-symbol gates on the exact path.
   (3+4 = one parametrized unit test on _memory_dep_supports_index_equivalence.)
Softer: append-path None-decline is shadowed by ordinary legality
(defense-in-depth, unpinned); score-bridge zero side unpinned.
Strongly pinned (for the record): the unproven-second-read decline at both
layers -- the planning-layer mutant fails 3 e2e tests x2 modes with hard
numerics; suppression-of-rewrites has a direct test; Star/WeakDep isolation
pinned at three layers.

## Secondary findings

- S1: `_reindex_sub_parent_consumer`'s True path is unreachable in every
  constructible scenario and has zero coverage; its bypass is part of P1's
  mechanism. Resolve together with P1 (likely: delete or make load-bearing).
- S2: `_sub_parent_epilogue_candidate_nodes` runs the rate check before the
  reduced-domain check; with rnumel in {2,4} a reduced-domain consumer is
  classified as a lane node. No wrong fusion constructed; fragile ordering.
- F1 (perf): a lane epilogue reading ONLY the source never gets
  producer-first orientation (get_possible_fusions swap requires ancestor
  overlap; consumer-first is gated off) -- staged fusion silently never
  attempted. Fail-safe.
- Dynamic-D (perf): standalone sub-parent with symbolic D declines before
  the planner at generic scoring (dep normalization under symbolic D breaks
  score_fusion_memory). Repro cg/dyn_d_why.py.
- Codegen cleanups: `dict(stage.source_layouts)` silently collapses
  duplicate names (assert uniqueness given the multi-access TODO; also
  computed twice); codegen's re-derived internal_source_names could assert
  agreement with plan.deferred_parent_start; store_buffer_counts is a no-op
  outside nested kernels.

## Codegen verdict (attacked and held, all bitwise)

No proven codegen bugs. Codegen files are byte-identical between base and
rebased tree (redesign is scheduler-only); the plan is recomputed at codegen
and raises loudly on skew. Held under attack: lane agreement (permuted
non-contiguous lane sets, mixed lanes in one consumer, fp8/uint8 factor-4,
multi-hop internal chains); a crafted three-pass looped kernel verified
line-by-line; 4:3 emission with CSE-shared splits and exact store
accounting; masked forwarding (precise invariant: masked_forward_names only
contains in-kernel-written names, and store_cache membership forces the
explicit where -- worth documenting); dynamic batch/preshuffled through one
graph; hostile layouts decline cleanly (the redesign is strictly TIGHTER
than the name-based version here).

## Docs (mechanical fixes for the author)

pr191775_exact_dependency_redesign.md: "intentionally uncommitted" is stale
(amended into HEAD); all 7 scrutiny line-pins drifted; "26 passed" is
currently 18+8 errors (P2). Checklist: orig-commit hash and most
scheduler.py pins drifted; Step 5 checkpoint inverted (code requires FACTOR
to divide GROUP_SIZE). Guide: header staleness only; content verified
accurate, including every historical-compatibility bullet having a live
covering test. Focused-count nit: mxfp6 = 27 passed + 3 skipped.

## Bottom line

The redesign's core is sound and strictly tighter than what it replaced --
the exact-match machinery held every direct attack, rollback is leak-free
under instrumentation, and codegen is clean. But the round found exactly
what it was looking for: the one proof branch the redesign kept (generic
broadcast equivalence, frame-free) is the surviving instance of the bug
class the redesign exists to kill, with a proven silent-wrong-numerics
repro. #191775 should not land until P1 is fixed (both directions above
also subsume S1), the P2 fixtures are repaired, and the four mutation holes
plus the P1 repros are pinned as tests.
