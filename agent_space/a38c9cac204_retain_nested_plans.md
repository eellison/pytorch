# `a38c9cac204` — Retain nested sub-parent fusion plans

`a38c9cac204` · **no PR yet** (local commit, no ghstack trailer) · +394 / −349
across 4 files · 312 passed / 6 skipped (both suites, 2026-08-04)

| File | Δ | What it holds |
|---|---|---|
| `torch/_inductor/scheduler.py` | +342/−293 | `NestedReductionPlan`, `plan()`, `AppendApproval` |
| `torch/_inductor/codegen/simd.py` | +40/−44 | codegen reads the plan instead of re-deriving |
| `test/inductor/test_nested_reduction.py` | +9/-9 | `PARENT_HALF` -> `SUB_PARENT` and plan-based test updates |
| `test/inductor/test_inductor_scheduler.py` | +3/−3 | renames |

Read #190594 and #190595 first. This commit strengthens initial legality by
including source planning before construction, so some pairs that previously
asserted after acceptance now decline. For pairs accepted by both revisions,
the intent is unchanged emitted semantics plus a retained decision.

## Stack evolution and reservations

**Changed later:** #191975 reduces repeated *standalone* planning within each
phase, but it does not remove this commit's `NestedReductionPlan` or
`AppendApproval`. The retained nested-plan lifecycle described here survives at
the final tip.

**Remaining reservations at the final tip:** fusion search can still derive a
nested plan more than once; the stale-approval fallback is defensive and
untested under the current synchronous dispatch; and the standalone planner
remains a separate entry funnel. We intentionally did not add a memoization
cache merely to reduce call counts.

## The problem

The stack's recurring bug class in one sentence: **the fusion decision is
irrevocable, but the information behind it was thrown away and re-derived.**

`NestedReduction.can_fuse` returned a bool. `FusedNestedReductions.__init__`
then re-ran `_get_grouped_reduction_and_size`, `get_grouped_axis`,
`PointwiseDomainContext.create` and `_classify_grouped_pointwise_nodes` to
rebuild what `can_fuse` had just computed and discarded — raising
`AssertionError` if any came back `None`. Codegen derived the domains a third
time:

```
can_fuse            -> derive, return bool, discard
__init__            -> derive again, or raise
_codegen_nested_... -> derive a third time
```

That is not merely wasteful. `loop_ordering_after_fusion` and
`loop_reindexing_after_fusion` mutate scheduler node ranges *after* a fusion is
accepted, and `_LoopMutationTracker` only rolls back **rejected** fusions. So a
later re-derivation runs against ranges the original proof never saw, and
disagreement surfaced as a hard compile assert rather than a declined fusion.

## Design decisions

1. **`can_fuse` becomes `plan()`.** Returns `NestedReductionPlan | None`;
   `can_fuse` is a one-line wrapper. The plan carries `group_size`,
   `grouped_axis`, `domain_context`, `pointwise_domains`, the broadcast source
   names and the sub-parent source layouts.

2. **Construct only from a plan.** `BaseScheduling.fuse` dispatches with a
   walrus so the node cannot be built without one, and both `AssertionError`s in
   `__init__` are gone:

   ```python
   elif NestedReduction._is_dependent_reduction_pair(node1, node2) and (
       nested_plan := NestedReduction.plan(node1, node2)
   ) is not None:
       return FusedNestedReductions(node1, node2, nested_plan)
   ```

3. **Share the source planner.** `_parent_half_source_layouts` is promoted
   from a `FusedNestedReductions` method to a `NestedReduction` classmethod,
   `_plan_nested_sub_parent_sources`, so the initial and append paths run the
   same code. This is the actual dedupe in this commit.

4. **Validate the append handoff instead of trusting it.** `can_fuse_with`
   records an `AppendApproval`: the consumer, the prospective grouped node, the
   full plan, and a `snapshot_loop_state()` for every `SchedulerNode` in both
   stages. `fuse_with` consumes it only if it matches and `is_current()`;
   otherwise it recomputes and raises if legality changed.

   The current `speedup_by_fusion` bypass returns `FusionResult.fuse(True)` for
   accepted non-template nested appends, so the normal path is synchronous and
   does not exercise a deferral window. The snapshot is defensive against
   future dispatch changes and nonstandard callers, not evidence of a currently
   observed delay.

   Why the snapshot works: `apply_new_loop_order` and `apply_loop_reindexing`
   **rebind** (`self._body = self._body.reorder_iter_loops(...)`) rather than
   mutating in place, so the snapshot holds the old objects and comparison fails
   as intended. `snapshot_loop_state` is not new — it is what
   `_LoopMutationTracker` already uses for rollback.

## What it generates

For a pair accepted before and after, codegen consumes the retained values and
is intended to emit the same schedule. The commit does not compare generated
source byte-for-byte, and its stronger initial legality can change which pairs
are accepted.

## The flow: the plan's lifecycle

Before — three independent derivations, the second and third able to disagree
with the first:

```
  can_fuse(n1,n2) ──derive──> bool          (analysis discarded)
  FusedNestedReductions.__init__ ──derive again──> or raise AssertionError
  codegen_nested_reduction ──derive a third time──> or raise
```

After — the construction-time plan is carried; fusion search can still derive
more than once:

```
  INITIAL FUSION
  ──────────────
  Scheduler.can_fuse
    └─ NestedReduction.can_fuse -> NestedReduction.plan(n1, n2)
                                              (search result not retained here)

  BaseScheduling.fuse
    └─ NestedReduction.plan(n1, n2)
                                      ├─ _get_grouped_reduction_and_size
                                      ├─ get_grouped_axis
                                      ├─ PointwiseDomainContext.create
                                      │    └─ _nested_sub_parent_domain
                                      ├─ _classify_nested_pointwise_nodes
                                      │    └─ _classify_grouped_pointwise_nodes
                                      ├─ _pointwise_domains_are_compatible
                                      └─ _plan_nested_sub_parent_sources   ◄── shared
                                           └─ _sub_parent_epilogue_source_deps
                                                     │
                                            NestedReductionPlan | None
                                                     │
  BaseScheduling.fuse ── walrus, constructs only if not None ──┐
                                                               ▼
                                    FusedNestedReductions(n1, n2, plan)
                                      └─ self.plan            ◄── stored, not rebuilt

  APPEND  (approval handoff; normally synchronous in current dispatch)
  ──────
  FusedNestedReductions.can_fuse_with(other)
    ├─ classify candidate and run scheduler legality
    ├─ tentatively fuse the grouped node
    ├─ NestedReduction.plan(outer, new_grouped_node)
    │    └─ _plan_nested_sub_parent_sources     ◄── same helper as above
    └─ records AppendApproval(consumer, grouped_node, plan,
                              snapshot_loop_state() for every SchedulerNode)
                    │
        current non-template dispatch fuses immediately; snapshot is defensive
                    │
  FusedNestedReductions.fuse_with(other)
    ├─ approval matches AND approval.is_current()  → use it
    └─ else recompute; raise only if the recompute itself fails

  CODEGEN
  ───────
  codegen_nested_reduction
    └─ node.plan.pointwise_domains       ◄── read, never re-derived
```

The two `_plan_nested_sub_parent_sources` boxes are the dedupe: initial and
append legality run the same code, where the append path used to have its own
hand-rolled copy on `FusedNestedReductions`.

## How to read it

1. **`NestedReductionPlan`** — the dataclass. Small; read it first.
2. **`NestedReduction.plan`** — formerly `can_fuse`. It now performs explicit
   classification/compatibility plus full source planning before constructing
   the retained result; it is more than a return-type rewrite.
3. **`_plan_nested_sub_parent_sources`** — the promoted, now-shared planner.
4. **`AppendApproval` + `fuse_with`** — the validated handoff in decision 4.
5. **`simd.py` -> `codegen_nested_reduction`** — now reads
   `node.plan.pointwise_domains` where it called
   `_classify_nested_pointwise_nodes`.

## What to pay attention to

- **This is a correctness change, not a compile-time win, and the commit message
  says so.** The plan is still derived more than once during fusion search —
  measured at 6–7 derivations per compile of one fusable graph. Any claim that
  this reduces work is wrong. (#191975 later cuts the standalone planner from
  three builds to one at fusion and two to one at codegen.)

- **`snapshot_loop_state` carries a sync obligation.** Its docstring says it must
  be kept in sync with the loop transforms. A future transform mutating state
  outside that tuple blinds rollback *and* this validation together. Not a
  regression — same exposure the existing mechanism has — but there are now two
  callers depending on it.

- **The benchmark bypass is load-bearing here.** `speedup_by_fusion` returns
  `fuse(True)` for accepted nested pairs, making the current handoff
  synchronous. It is guarded by `not is_multi_template` so it
  cannot skip template finalization, and keyed on
  `_is_dependent_reduction_pair(...) and can_fuse(...)` rather than the cheap
  `is_candidate` filter — measured, 70 of 487 `is_candidate` firings are pairs
  that fail `can_fuse` and would fuse generically.

- **Source writes remain part of the matcher input.** The promoted helper
  preserves #190595's behavior of collecting writes from the outer and grouped
  nodes, including sub-parent consumers. This is factoring, not a new behavior
  change in this commit.

## Test coverage

This commit adds no new test cases; its +9/-9 test diff is renaming and adapting
existing tests to the plan API. It inherits the intermediate-read, leaf, and
benchmark coverage from #190595.

Not covered: no test checks plan-object identity, constructor/codegen
non-rederivation, or forces `AppendApproval.is_current()` to fail. Under the
current synchronous benchmark bypass, the stale-approval path appears
defensive rather than normally reachable.

## Changed during the stack rewrite

Four review rounds; see `sub_parent_planner_handoff.md` and
`sub_parent_planner_review_criteria.md`.

- `can_fuse_with` now computes full legality and records the approved plan.
  `fuse_with` still raises if a missing or stale approval is recomputed and
  legality has changed; the current approval handoff is what avoids that path
  in normal operation.
- The approval was first an unbounded identity-keyed dict that leaked an entry
  per non-fused candidate; now a single slot cleared on consumption.
- The benchmark bypass initially keyed on `is_candidate` and sat *after* the
  `is_multi_template` check, so it could skip template finalization. Both fixed.
- The commit message originally described this as deduplication; corrected to
  correctness / plan consistency, since derivation counts went up, not down.
