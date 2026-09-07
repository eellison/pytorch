# Sub-parent planner: standing review criteria and context

Companion to `sub_parent_planner_handoff.md`. That file churns every round --
each side deletes the other's text -- so anything durable lives **here** and is
not deleted by either side. Append; do not rewrite.

## Protocol

1. Implementer: `Status: READY_FOR_REVIEW` + diff summary + test results +
   diff hash + questions. Then stalls.
2. Reviewer: replaces body with `Status: REVIEWED` + findings + the hash of the
   diff actually reviewed.
3. Implementer: consumes, sets `Status: WORKING`, repeats.

**`Owner` names who holds the token, not who wrote last.** Reviewer sets
`Owner: primary` when handing findings back; implementer sets `Owner: reviewer`
on `READY_FOR_REVIEW`. Getting this backwards deadlocks both sides silently --
it happened once (2026-08-03, five idle polls) because the reviewer wrote
`Owner: reviewer` meaning "written by me".

**Always include the diff hash**, from `git diff | git hash-object --stdin`.
A review of a revision that already moved is worse than no review. Reviewer
states the hash reviewed; if it does not match, the findings are stale.

## Standing criteria (checked every round)

1. **Deletes the duplicated funnel, not renames it.** A rename that leaves two
   parallel implementations is not progress.
2. **Every commit independently green.** `pytest -n 16` per commit. The stack is
   bisectable today and that was expensive to get.
3. **State the plan lifetime precisely.** The nested plan is carried on
   `FusedNestedReductions`; its append approval is validated before reuse. The
   standalone plan is rebuilt once per phase. "Stored on the node", "derived
   once per phase", and "memoized across phases" are different claims.
4. **No new `raise` where the matching `can_*` check is weaker.** Every
   rejection in this file returns `False`. An assert on a path whose gate
   validates a subset is a compile crash where a declined fusion belongs. This
   is the single recurring bug class in this work.
5. **Behavior changes inside moved code get called out.** Moving a function and
   changing it in the same diff hides the change. Say it, comment it, test it.
6. **Docstrings: contract, not narration.** No step-by-step replay of the body.
   Document invariants, return-value contracts (e.g. `None` rejects vs `()`
   means nothing to do), and why something lives where it does. `__init__` that
   is pure assignment gets a class docstring instead.

## Rebuild-count metric

How to measure: wrap the classmethods on `NestedReduction` with a counter, run
one `torch.compile` of the nested rmsnorm+pair graph
(`B=32, D=1024, G=16`, `triton.nested_reduction=True`,
`loop_ordering_after_fusion=True`, `fx_graph_cache=False`).

```
                                    baseline   2026-08-03 proposal
_classify_nested_pointwise_nodes       6            6
sub_parent_epilogue_plan               6            6
plan                                   -            6
_plan_nested_sub_parent_sources        -            8
```

This table is a historical 2026-08-03 measurement of an earlier proposal, not
the current stack. #191975 later reduced the standalone planner to one build per
fusion phase and one per codegen phase; the nested plan is retained separately.

Why it is correctness and not only cost: `loop_ordering_after_fusion` and
`loop_reindexing_after_fusion` mutate node ranges *after* a fusion is accepted,
and `_LoopMutationTracker` only rolls back *rejected* fusions. A codegen-time
rebuild therefore runs against ranges legality was never proven on.

## Settled decisions -- do not re-litigate

- **Combo kernels and benchmark fusion need not support this path.** Declining
  is correct; still emitting the nested kernel is required. Already fixed in
  commit 1 via `BaseScheduling.has_sub_parent_epilogue`, forwarded by
  `CUDACombinedScheduling`. A retained plan does **not** fix these -- they came
  from `generate_node_schedule` being unable to represent the derived group.
- **The lane proof is irreducible.** `_sub_parent_read_matches_lane` plus the
  substitution machinery (~130 lines) is real work. Do not count it as savings.
- **Refactors land as their own commit** at the end of the stack, like #191975.
  Not folded into a feature commit.
- **Not done deliberately**, because the premises proved false on inspection:
  folding `_float_to_mxfp6_e2m3` (its op count is load-bearing for the fusion
  decision) and dropping `allow_reduced_broadcast` (removing it turns a loud
  assert into a silent wrong broadcast).

## Known-open, not owned by this refactor

Tracked in `agent_space/FOLLOWUPS.md`:

- **Matcher matrix — separate from planner unification.** Four functions, 130
  lines, differing
  by one lane-expression line. `_sub_parent_read_matches_lane` is already the
  shared core. Layout and dep-shape are orthogonal; they should be two
  parameters. ~85-130 lines removable.
  If taken: resolve the asymmetry deliberately -- when `lane_dim is None`,
  interleaved falls back to the flat matcher and contiguous returns `False`,
  and nothing explains why.
- **1.4 — `SIMDKernelFeatures` excludes epilogue nodes** from
  `select_index_dtype()` (`simd.py`). The parent element count covers ordinary
  contiguous epilogue outputs; the remaining concern is a narrow, unproven
  strided-output address-range case.
- No positive test reaching the shapes #191975's relaxation newly admits.
- `values.repeat` test inputs hide group-indexing and permutation bugs;
  `torch.randn` catches both.

## Environment

- `pytest -n 16` is ~8x faster than serial (387s -> 47s on
  `test_nested_reduction.py`). Note xdist reports `304 passed, 4 skipped` where
  unittest reports `Ran 308 tests ... OK (skipped=4)` -- same total.
- **Do not parallelize per-commit testing with `git worktree`.** There is one
  compiled `_C`, in the main tree. A fresh worktree has no build, and pointing
  `PYTHONPATH` at the main tree silently pairs the worktree's *test files* with
  the main tree's *torch source* -- a wrong-commit result that looks like a
  pass. Check out commits serially in the one tree; parallelize *within* a run.
- Print `git rev-parse HEAD` before **and** after any per-commit run. Two
  background jobs both doing `git switch` in that tree will race.
- `/home/eellison/local/pytorch` is a symlink to `/data/users/eellison/pytorch`
  (same inode). `torch.__file__` reporting the latter is not a mismatch.
- `test_standalone_sub_parent_declines_benchmark_fusion` deliberately does not
  assert `codegen_nested_reduction`: benchmarking times real kernels, so noise
  can change whether an unrelated pair fuses. It failed once and then passed 7
  consecutive runs; treat as suspect if it fails again.

## 2026-08-10 ownership update

The earlier rule that cleanup must remain in a final standalone #191975 commit
is superseded. The final #191975 diff had become source-only cleanup of
mechanisms introduced within this stack, so keeping it separate created
add-then-delete churn. Its three pieces now land with their owners: tile-shape
cleanup in #190594, explicit materialization state in #190595, and internal
source lookup cleanup in #191775. #191975 is no longer in the local stack.

The old open item about shapes admitted by a #191975 relaxation is also
obsolete: the final cleanup changed no fusion decision, and the five-commit tip
is tree-identical to the previously tested six-commit tip.
