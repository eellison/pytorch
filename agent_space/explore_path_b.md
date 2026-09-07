# Path B Assessment: Scheduler-Owned Half-Resolution Discovery

## Current state

- **Worktree**: `/data/users/eellison/pytorch_nested_reduction_path3`
- **Branch**: `nested_reduction_path3`
- **HEAD**: `1c958ec0816 [inductor] Nested reduction: path-3 — scheduler owns half-resolution discovery`
- **Commits unique to path-3 vs baseline tip (`03baa04b984`)**: 2
  1. `901fc956a97` — "import single_commit_wip dirty state as path-3 baseline" (the bulk of the change: `inline_reduction_buffers` codegen primitive + handler simplification + test rewrite)
  2. `1c958ec0816` — "scheduler owns half-resolution discovery" (the actual path-3 refactor; `-43` net lines)

The branch reuses the baseline up through `[inductor] Nested reduction: collapse stack for iteration`. Two commits sit on top.

## Test results

`python test/inductor/test_nested_reduction.py` — **80/80 pass** in 86s.

```
Ran 80 tests in 86.494s
OK
```

25 unique `test_*` methods, instantiated across two `TestCase`s (`NestedReductionTest` with nested-reduction enabled, `NoNestedReductionTest` with it disabled, plus a few B=1 / size variants). No failures, no skips, no xfails. The path-3 commit's own message claims test parity with baseline (39/40, same single environmental `test_dynamic_shapes_pattern2` failure) — current run shows 80/80 so even that environmental flake is gone.

## Diff vs baseline

```
 test/inductor/test_nested_reduction.py | 269 ++++++++++++------------
 torch/_inductor/codegen/simd.py        | 362 ++++++++-------------------------
 torch/_inductor/codegen/triton.py      | 116 ++++++++---
 torch/_inductor/scheduler.py           | 130 ++++++++++++
 4 files changed, 429 insertions(+), 448 deletions(-)
```

**Net: -19 LOC across all four files.** Per-file:

- **`scheduler.py` (+130 / -0)** — Pure addition. New `_collect_half_resolution_consumers` (62 lines) and helper `_half_resolution_bfs` (60 lines) on `FusedNestedReductions`, plus a stored field `half_resolution_consumers: list[BaseSchedulerNode]` populated in `__init__`. No deletions; no existing scheduler logic was touched.
- **`simd.py` (+80 / -282; net -202)** — Largely a removal. Two codegen-side discovery helpers (`_collect_half_resolution_consumers_early`, `_collect_half_resolution_consumers`) and their shared BFS (`_collect_half_resolution_consumers_from_available`) are deleted (~150 lines), along with the early-vs-late "agreement guard" (~15 lines). The `_GroupedReductionOpsHandler` got simpler: stage-load capture moved out of the handler into kernel state (`captured_load_values`); `store_reduction` lost its mutation-aware bookkeeping in favor of the `inline_reduction_buffers` mechanism. Some defensive comments and a `removed_buffers` early-return in `_DerivedIterationFamily.store` were stripped.
- **`triton.py` (+116 / -36; net +80)** — Two new mechanisms, one trivial regression. `inline_reduction_buffers: dict[str, CSEVariable]` field on `TritonKernel` (with intercepts in `load`, `store`, `store_reduction` — about 20 lines plus 17 lines of documenting comment). `captured_load_values` / `captured_load_range_trees` capture-window mechanism in `load` (~25 lines including invariant assertions). Plus a partial revert of a small refactor: `_mask_name_for_symbol` is undone — the inline `prefix_str`/`mask_name` branch returns to two callers.
- **`test_nested_reduction.py` (+135 / -134; net +1)** — Refactored kernel-form assertions from `FileCheck` against an in-process compile to a JSON dict produced by a **subprocess-spawned** Python child. Each pattern (`nvfp4`, `amax`, `fullres`) checks substring counts in the captured Triton source.

## What got moved/added

**Added to `scheduler.py FusedNestedReductions`:**
- `self.half_resolution_consumers` field (computed at fusion time)
- `_collect_half_resolution_consumers()` — orchestration, gates on `config.triton.nested_reduction`, `small_dim_in_r`, `rnumel1` static, `rnumel2 % 2 == 0`, and a *predicted* persistent-reduction condition (`is_producer_consumer or rnumel1 <= 8192`)
- `_half_resolution_bfs()` — BFS that walks consumers, validates per-read constant-lane modulo-2 indexing for graph inputs

**Removed from `simd.py SIMDScheduling`:**
- `_collect_half_resolution_consumers_early()` (~33 lines)
- `_collect_half_resolution_consumers()` (~33 lines)
- `_collect_half_resolution_consumers_from_available()` (the shared BFS, ~75 lines)
- The `early_half_resolution_names != late_half_resolution_names` agreement-or-RuntimeError guard (~13 lines) — its existence was the smoking-gun motivation for the refactor

**Added to `triton.py TritonKernel` (independent of the scheduler refactor):**
- `inline_reduction_buffers: dict[str, CSEVariable]` — a generic in-register-instead-of-memory primitive intercepting `load`/`store`/`store_reduction`. Replaces the "node1 store_cache invariant" pattern from baseline.
- `captured_load_values` / `captured_load_range_trees` — per-stage load-snapshot mechanism with active-range-tree invariant. Replaces the per-handler `_stage_load_values` dict in `_GroupedReductionOpsHandler`.

## What did NOT get cleaned up

The non-local-optimum design doc lists 5 migration steps. Path-3 implements **only step 3** (and only for half-resolution, not the broader family construction):

1. Land the current local optimum — ✓ (baseline)
2. **Unify reduced-output and half-resolution into one generic derived-family object** — *not done*. `_DerivedIterationFamily`, `_GroupReductionLayout`, `_GroupedReductionOpsHandler`, `_PointwiseRemapHandler` are all still distinct classes in `simd.py`. The "one consumer path for all remapped pointwise" goal is unmet.
3. **Move legality and family construction onto `FusedNestedReductions`** — *partial*. Only the half-res *consumer list* moved. `_GroupReductionLayout` construction, `make_reduced_output_family`, `make_half_resolution_family`, full-res legality, `small_dim_in_r` decision (well, that one already lived on the node) — all still happen inside `codegen_nested_reduction`.
4. Affine masked subregions — *not done*.
5. Reuse beyond nested reduction — *not done*.

`codegen_nested_reduction` in `simd.py` is still the same shape it was in baseline. The simd.py LOC reduction (-202) comes mostly from deleting the duplicated discovery passes, not from a structural simplification of codegen.

The `_RemappedOpsHandler` unification mentioned in the project memory ("unified handler for pass 2 epilogue, full-res epilogue, and pass 3") is in the *baseline*, not added by path-3 — path-3 inherits it.

The dual early/late half-res discovery — yes, that's gone. That was the headline cleanup.

## Code smells / TODOs

No `TODO`/`FIXME`/`XXX`/`HACK` markers were added by path-3 in nested-reduction code (all matches in `simd.py`/`triton.py`/`scheduler.py` predate this work).

That said, several smells are visible in the diff:

- **`torch/_inductor/codegen/triton.py:4010-4020`** — `captured_load_values` is a global mutable kernel field guarded by a runtime `assert all(active is expected ...)` to detect range-tree drift. It's effectively a context-managed scratch buffer leaked onto the kernel. A real `with kernel.capture_loads(...)` context manager would be cleaner. The path-3 worktree wraps it in `try/finally` at the call site (`simd.py:2714-2725`) but the field itself is on the kernel.

- **`torch/_inductor/codegen/triton.py:2841-2864`** — `inline_reduction_buffers` is the path-3 author's most general-purpose addition (17-line docstring describing it as a generic register-communication primitive). It's used *only* by nested reduction. Adding a generic kernel-wide primitive that has one caller is suspicious — either it should be specialized down to nested-reduction state (consistent with how baseline kept things), or it should be exercised by something else to justify its generality. Currently it's documented as generic but isn't.

- **`torch/_inductor/scheduler.py:2627`** — The new `_collect_half_resolution_consumers` predicts the persistent-reduction decision with a hard-coded `int(rnumel1) <= 8192` mirroring `override_persistent_reduction` heuristics in codegen. This is a knowledge-leak from codegen up into the scheduler. If the codegen heuristic ever changes, the prediction silently desynchronizes — and unlike baseline's early/late agreement check, there's no longer any cross-validation. The commit message itself describes this as "static persistent-reduction predictor that mirrors codegen's override_persistent_reduction logic." The mirror is brittle.

- **`test/inductor/test_nested_reduction.py:24-128`** — Kernel-form assertions now spawn a fresh Python subprocess per assertion (3 patterns × multiple batch sizes per test). This was likely done to work around test-isolation issues with `inductor_config` / `metrics` / `_dynamo` state leaking. It works, but it's slower (test suite is 86s; 80 tests × subprocess startup is non-trivial), more brittle (`subprocess.check_output(... cwd=REPO_ROOT)` makes assumptions about repo layout), and harder to debug. The `_run_and_capture_kernel_source` in-process helper from baseline is gone.

- **`test/inductor/test_nested_reduction.py:98-103`** — The subprocess test wraps the compile in a try/except that swallows `AssertionError` matching `"expected" in msg and "outputs, got" in msg`. That's working around a known assertion mismatch in the kernel-form test path; the comment doesn't explain it. The code captures the kernel anyway and continues, which means kernel-form tests can pass even when the actual numerical run is broken.

- **`torch/_inductor/codegen/simd.py:1429-1440`** — `_resolve_remapped_value` lost its explanatory comment about the modulo-2 lane-selection being NVFP4-specific. The behavior is unchanged, but the deleted comment was useful documentation of legality scope.

- **`torch/_inductor/codegen/triton.py`** — Several baseline-commit comments explaining tricky derived-tree / family / removed-buffer interactions were stripped (e.g. `_DerivedIterationFamily.is_active_on`, `store`, `ensure_headers`, `DerivedIterationRangesRoot.is_loop`). The behavioral changes are real (e.g. the `is_loop=False` change at line 405 of simd.py) but are now under-commented.

- **`torch/_inductor/codegen/triton.py:3069`** — `override_mask` parameter on `indexing()` was removed. This was a generic feature added by baseline; path-3 reverts it. No nested-reduction tests need it, but anything else that did is now worse off. Worth checking whether anything outside the nested-reduction worktree depends on it (the commit doesn't claim to).

## Cost to make landable

Treating "landable" as "merge candidate that doesn't actively make the codebase worse":

- **Subprocess-based kernel-form tests** — needs replacement with in-process capture and proper test isolation. ~3-4 hours to root-cause the leak that motivated the subprocess workaround and rewrite the helper. **~4 hours.**
- **Persistent-reduction predictor sync risk** — needs either (a) codegen to *consume* the scheduler's decision rather than re-derive, removing the duplication; or (b) a structural test that asserts the prediction matches actual codegen behavior across configurations. ~2-3 hours. **~3 hours.**
- **`inline_reduction_buffers` generality justification** — either narrow it down to nested-reduction state (cleaner) or actually use it from another caller. The 17-line "general-purpose" docstring overpromises for a single-call-site mechanism. ~1-2 hours to either narrow or document the asymmetry honestly. **~2 hours.**
- **`captured_load_values` lifecycle** — promote to a context manager rather than try/finally on raw kernel fields. ~1 hour. **~1 hour.**
- **Re-add or justify removal of `override_mask`** — confirm nothing outside the worktree depends on it; restore if so. ~1 hour. **~1 hour.**
- **Re-add stripped explanatory comments** — moderate; these were useful. ~1 hour. **~1 hour.**
- **Subprocess test cleanup of `AssertionError` swallow** — fix the underlying issue rather than papering over it. ~2 hours. **~2 hours.**

**Subtotal: ~14 hours** to clean up the existing scope.

If the goal is also to actually deliver path-3 as advertised in the design doc (steps 2–4), that's much larger:
- Unifying `_GroupedReductionOpsHandler` and `_PointwiseRemapHandler` into one — ~1 day
- Moving `_GroupReductionLayout` construction onto the scheduler node — ~2-3 days
- Affine masked subregions — separate project, weeks

## Honest recommendation

**Path-3 in its current form is a partial cleanup, not a structural rewrite.**

What it actually delivers:
- The half-resolution discovery now happens once, at fusion time, on the scheduler node. The early/late agreement check (a clear smell in baseline) is gone.
- A new `inline_reduction_buffers` codegen primitive replaces a subtler `cse.store_cache` invariant in baseline.
- Net -19 LOC overall (-202 in simd.py, +130 in scheduler.py, +80 in triton.py, +1 in tests).

What it does *not* deliver:
- The non-local-optimum design's central goal (one generic derived-family object consumed by one remapped pointwise path) is unaddressed. The design doc treats path-3 as step 3 of 5; the prototype delivers ~30% of step 3.
- `simd.py` is still ~4408 lines and the `codegen_nested_reduction` shape is unchanged. The "~1500 LOC in `simd.py`" framing in the comparison is not meaningfully addressed (the absolute reduction is ~5%).

**Is it landable now?** No. The subprocess-based test infrastructure and the swallowed-AssertionError pattern are land-blockers; the persistent-reduction-predictor duplication is a correctness risk.

**In 1 day?** Yes for the cleanup items above (~14h). After that you'd have a clean, small refactor that fixes one real smell (early/late discovery duplication) and adds two well-tested primitives.

**In 1 week?** Could attempt step 2 of the migration (unify the two pointwise handlers), which would make this a more substantive structural improvement.

**Better or worse than baseline as a landing target?**

- **As a landing target right now**: *worse than baseline*. Baseline is shipping, well-tested, and clean apart from the early/late discovery smell. Path-3 introduces a subprocess test harness, a brittle codegen-mirror predictor, and an under-justified generic primitive — in exchange for fixing one smell.
- **As a landing target after ~14h of cleanup**: *roughly equivalent to baseline*. You've fixed the early/late smell at the cost of adding `inline_reduction_buffers` complexity in triton.py. Net +0 to maintenance burden, with a cleaner separation of concerns.
- **As a stepping-stone toward path A (IR primitive)**: *neutral-to-slightly-helpful*. The scheduler now owns the consumer-list decision; that's directionally consistent with lifting more legality up. But path A's value comes from doing this at the IR level, where path-3's scheduler refactor is largely orthogonal scaffolding.

Recommendation: if the choice is between landing baseline as-is vs. landing path-3 as-is, ship baseline. If the goal is to incrementally reduce codegen-side complexity, path-3 needs ~2 days of polish before it's a clear improvement. If the goal is the design doc's full vision, path-3 isn't close — invest in path A or in expanding path-3 to actually unify the family/handler classes.
