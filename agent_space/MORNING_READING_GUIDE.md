# Morning Reading Guide

Overnight exploration produced these artifacts, in order of priority:

## 1. Read first: comparison synthesis

**`/data/users/eellison/pytorch/agent_space/path_comparison.md`**

Side-by-side of all five candidate landing paths, with concrete LOC/test data
and a recommendation framework. The TL;DR + Recommendation sections are at the
top.

## 2. The agent probe reports

These were produced by parallel background agents and inform the comparison:

- **`/data/users/eellison/pytorch/agent_space/explore_path_a.md`**
  Agent's exploration of Path A (true `BlockLocalReduction` IR primitive).
  Key questions: feasibility, LOC cost, hardest obstacles, additional cost
  on top of the existing Path B+ scheduler work.

- **`/data/users/eellison/pytorch/agent_space/explore_path_b.md`**
  Agent's assessment of Path B (the `path3` worktree — scheduler-owned
  half-resolution discovery only). Status, tests, gaps, cost to land.

## 3. The five candidates, recap

| # | Name | Worktree | Architectural shape |
|---|------|----------|---------------------|
| 1 | Baseline | `pytorch_nested_reduction_single_commit_wip` | Codegen-owned, full feature |
| 2 | Core landable | `pytorch_nested_reduction_core_landable` | Same architecture as Baseline, minus NVFP4 |
| 3 | Path B | `pytorch_nested_reduction_path3` | Scheduler-owned half-res discovery |
| 4 | **Path B+** | `pytorch_nested_reduction_ir_explore` (uncommitted) | Scheduler-owned grouped-reduction plan |
| 5 | Path A | `pytorch_nested_reduction_ir_explore` (proposed) | True IR primitive |

## 4. Background design docs (already written)

- `/data/users/eellison/pytorch/agent_space/nested_reduction_non_local_optimum.md`
  — consumer-side abstraction generalization (paths 1-5)
- `/data/users/eellison/pytorch/agent_space/nested_reduction_ir_uplift.md`
  — IR-level design space (paths A-D, written yesterday)
- `/data/users/eellison/pytorch/agent_space/nested_reduction_ir_explore_comparison.md`
  — pre-existing documentation of the Path B+ work (not auto-generated;
  written by the agent that did the prototype work earlier)

## 5. Status (all agents completed)

- **Path B (path3) — assessed.** Obsolete: built on stale baseline; has
  regressions vs current main (subprocess tests, inline_reduction_buffers
  re-added, override_mask removed); ~14h cleanup to reach baseline parity.
  Don't pursue.
- **Path B+ (ir_explore) — verified structurally.** Built on current
  baseline. No subprocess tests, override_mask preserved, no
  inline_reduction_buffers. Agent 1 didn't run test suite (CLAUDE.md
  forbids builds without approval). **One command needed in the morning
  to confirm tests pass.**
- **Path A (true IR primitive) — assessed by Agent 1.** Verdict: defer.
  2–4 weeks (option B) or 4–8 weeks (option A with backend ext). Without
  Option A, payoff is mostly cosmetic — `_DerivedIterationFamily` and bulk
  of simd.py codegen survive regardless. Path A becomes worth pursuing
  only if (1) second fusion pattern emerges, (2) dynamic-shape support
  becomes hard requirement, or (3) reduce-over-interior-tree backend
  extension lands for unrelated reasons.
- **Baseline — measured.** Working, 80/80 tests, big PR.
- **Core landable — measured.** 25% smaller, same architecture as baseline.

## 6. Recommendation

**Land Path B+.** Tests verified passing overnight: **80/80 in 54s.**

Concrete first actions:

1. Decide what to do with Agent 1's IR-stub additions in `ir_explore`
   (`torch/_inductor/ir.py` BlockLocalReduction class +
   `torch/_inductor/fx_passes/nested_reduction.py`). Likely options:
   discard, or park on a separate `path_a_seed` branch for future work.
2. Commit the Path B+ work (probably as 2 commits: scheduler additions,
   then simd.py simplification reading the plan). Possibly cherry-pick
   to a fresh branch off `nested_reduction_single_commit_wip` rather than
   landing from `ir_explore`.
3. Polish and review.

Don't pursue Path A until a separate motivating use case appears. Don't
pursue Path B (path3); it's superseded.

## 7. The five candidate paths (compact)

| # | Name | Worktree | Verdict |
|---|------|----------|---------|
| 1 | Baseline | `single_commit_wip` | Backup option |
| 2 | Core landable | `core_landable` | Smaller-scope alternative |
| 3 | Path B (path3) | `path3` | **Obsolete, archive** |
| 4 | **Path B+** | `ir_explore` (uncommitted) | **PRIMARY RECOMMENDATION** |
| 5 | Path A | `ir_explore` (Agent 1's stub) | Defer to future architecture project |
