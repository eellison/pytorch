# Nested Reduction: Landing Path Comparison

*Generated overnight from parallel Path A / Path B exploration agents
plus baseline measurement.*

---

## TL;DR

**Land Path B+** (the uncommitted work in `pytorch_nested_reduction_ir_explore`).

- It's a clean architectural improvement over Baseline: grouped-reduction
  *semantics* moved up to scheduler, consumer-side machinery stays in codegen.
- Built on current main, all baseline cleanups preserved (override_mask,
  no inline_reduction_buffers, in-process FileCheck tests).
- Already prototyped, documented, and per docs tests pass.
- Estimated 1–2 days commit + polish to land.

**Don't pursue Path B (path3)** — built on stale baseline, has real
regressions vs current main, ~14h cleanup just to reach baseline parity.
Its one good idea (scheduler-owned half-res discovery) is independently
absorbed by Path B+.

**Defer Path A (true `BlockLocalReduction` IR primitive).** Estimated 2–4
weeks (option B) or 4–8 weeks (option A with backend extension). Without a
real backend extension, the LOC delta is approximately neutral and the
payoff is mostly cosmetic — `_DerivedIterationFamily` and the bulk of
simd.py codegen survive regardless. Not worth bundling with this landing.

**Path A becomes worth pursuing** only if: (1) a second fusion pattern
lands needing similar IR machinery, OR (2) dynamic-shape support for
nested reduction becomes a hard requirement, OR (3) reduce-over-interior-
tree backend extension lands for unrelated reasons.

**Core landable** is a viable smaller-scope alternative if you want to
ship without NVFP4/half-res first, but it has the same architectural
shape as Baseline — it doesn't address layering.

The strong recommendation is: **commit Path B+ → polish → ship.** Write up
Path A in a follow-up architecture doc and let it bake until a second
customer materializes.

---

## The five candidates

The exploration uncovered that there are actually FIVE candidate paths,
not three. They differ along two axes: **architectural depth** (how much
of the work moves up the stack) and **scope** (full feature vs. trimmed).

| # | Name | Worktree | Status |
|---|------|----------|--------|
| 1 | **Baseline** | `pytorch_nested_reduction_single_commit_wip` | Done, 80/80 tests |
| 2 | **Core landable** (baseline minus half-res) | `pytorch_nested_reduction_core_landable` | Done, smaller scope |
| 3 | **Path B** (scheduler-owned half-res discovery only) | `pytorch_nested_reduction_path3` | Obsolete — stale baseline + regressions |
| 4 | **Path B+** (scheduler-owned grouped-reduction plan) | `pytorch_nested_reduction_ir_explore` (uncommitted) | **Recommended landing target** |
| 5 | **Path A** (true `BlockLocalReduction` IR primitive) | `pytorch_nested_reduction_ir_explore` (Agent 1's stub) | Defer — 2–4 weeks min, mostly cosmetic without backend ext |

The interesting realization: **Path B+ already exists as documented prior work**
in the `ir_explore` worktree. It's a partial answer to "lift work upward" — moves
*grouped-reduction semantics* up to the scheduler, while keeping
`_DerivedIterationFamily` in codegen.

Path A would be the *next* step beyond Path B+: replacing the plan dataclasses
with actual IR nodes.

---

## Baseline (1): current branch

### Footprint vs main

| File | LOC | Notes |
|------|-----|-------|
| `simd.py` | +1469 | Largest single change |
| `scheduler.py` | +350 | `NestedReduction`, `FusedNestedReductions` |
| `triton.py` | +274 | Mask plumbing, emit helpers |
| `test_nested_reduction.py` | +724 | Coverage including NVFP4/half-res |
| `runtime/triton_heuristics.py` | +55 | |
| Other | +21 | config, metrics, cuda_combined |
| **Total** | **+2793** | 8 files |

### Tests
80/80 passing. Numerics + form-checks.

### Architectural shape
Codegen-owned. `simd.py` carries:
- pattern re-recognition (which buffers are internal, fullres vs reduced)
- `_GroupReductionLayout` (grouped-reduction semantics)
- `_GroupedReductionOpsHandler` (reduction emit)
- Half-res BFS + early/late dual pass
- `_DerivedIterationFamily` (consumer remapping)

### Outstanding concerns
- Early/late dual pass + `RuntimeError` agreement check
- Singleton-numel substitution is global
- Out-of-band kernel state (`stage_load_values`, `_node1_cse_vars`,
  `nested_reduction_max_xblock`)
- No perf CI
- Half-res only tested for NVFP4

### Cost to land
- Quality: already there.
- Shape: 1700-LOC PR with new core abstractions.
- Mitigations: extract to `nested_reduction.py`, delete dual pass, stack PRs.
- **Estimated**: 3–5 days of polish + 1–2 weeks of review.

---

## Core landable (2): baseline minus half-res

### Footprint vs main

| File | LOC | Notes |
|------|-----|-------|
| `simd.py` | +1008 | −460 vs baseline (no half-res) |
| `scheduler.py` | +350 | Same |
| `triton.py` | +228 | −46 vs baseline |
| `test_nested_reduction.py` | +556 | −170 vs baseline |
| Other | +84 | Same as baseline |
| **Total** | **+2118** | 8 files, **−675 vs baseline** |

### Tests
*[Need to verify, but per commit message: half-res NVFP4 tests removed,
core tests retained]*

### Architectural shape
Same as baseline. Just smaller scope — landing without the NVFP4/half-res
path means simpler reviewable surface.

### Trade-off
- ✅ Easier to land (smaller PR, less surface area)
- ❌ Loses the NVFP4 perf win (the marquee result)
- ❌ Half-res path lives on a "saved branch" awaiting follow-up
- ✅ Sets up cleaner follow-up where half-res is its own PR

### Cost to land
- **Estimated**: 2–4 days polish + 1 week review.

---

## Path B (3): scheduler-owned half-res discovery only

*Agent assessment of `pytorch_nested_reduction_path3` complete. Full report:
`agent_space/explore_path_b.md`. Summary below.*

### Critical context: Path B is on a STALE baseline

Path-3 was committed on top of the *older* baseline `03baa04b984`. Since then,
the current baseline `06ef55bbeef` has independently added:
- `inline_reduction_buffers` removal
- `override_mask` preservation
- `_stage_load_values` cleanup
- `is_loop=parent.is_loop` fix
- `_mask_name_for_symbol` extraction
- Form-check tests via in-process FileCheck (not subprocess)

Path-3 lacks all of these. As code, **path-3 is "old baseline + scheduler-owned
half-res discovery."** The half-res discovery refactor itself is a clean win;
the surrounding code is regressed vs current main.

### Tests
80/80 passing in 86s. Numerics good.

### Footprint vs old baseline `03baa04b984`

```
test/inductor/test_nested_reduction.py | +135 / -134 (net +1)
torch/_inductor/codegen/simd.py        |  +80 / -282 (net -202)
torch/_inductor/codegen/triton.py      | +116 /  -36 (net  +80)
torch/_inductor/scheduler.py           | +130 /   -0 (net +130)
                                              Total: -19 LOC
```

### What got delivered
- Half-res consumer discovery moved from late codegen onto
  `FusedNestedReductions._collect_half_resolution_consumers`
- The early/late dual pass + `RuntimeError` agreement check is GONE
- That's it — substantively. ~150 lines of duplicate BFS code deleted from
  codegen.

### What did NOT get delivered (per design doc step-3 promise)
- `_GroupReductionLayout` construction still in codegen
- `make_reduced_output_family`, `make_half_resolution_family` still in codegen
- Full-res legality still in codegen
- `_GroupedReductionOpsHandler`, `_PointwiseRemapHandler`, `_DerivedIterationFamily`
  all still distinct
- `codegen_nested_reduction` shape unchanged
- `simd.py` still ~4408 lines (only ~5% reduction)

### Critical smells the agent flagged
1. **Subprocess-based kernel-form tests** — each pattern check forks Python.
   Slower, more brittle than baseline's in-process FileCheck.
2. **`AssertionError` swallow in test code** — wraps compile in try/except
   matching specific error strings, with no comment explaining why. Means
   form-check can pass even if numerical run is broken.
3. **Hard-coded `int(rnumel1) <= 8192`** in scheduler-side persistent-reduction
   predictor. Mirrors codegen heuristic without cross-validation now that the
   agreement check is gone.
4. **`inline_reduction_buffers` documented as generic but has 1 caller** —
   17-line docstring describes it as a generic kernel primitive but it's used
   only by nested reduction.
5. **`override_mask` parameter REVERTED from `indexing()`** — anything outside
   nested-reduction depending on it would be silently regressed.

### Cost to land
Per agent: **~14 hours of cleanup** to bring path-3 to roughly-parity-with-
current-baseline. Even after cleanup, only fixes the early/late smell at the
cost of adding `inline_reduction_buffers` complexity. Net architectural improvement
is small.

### Better or worse than baseline as a landing target?
- **As-is**: *worse than baseline*. Subprocess tests, brittle predictor,
  under-justified generic primitive.
- **After 14h cleanup**: *roughly equivalent to baseline*. Fixes one smell,
  adds another. Net wash.
- **As stepping-stone to Path A**: *neutral-to-slightly-helpful*. Path A's
  value comes from IR uplift; path-3's scheduler ownership is orthogonal
  scaffolding.

### Verdict on Path B
Not the right landing target. The good idea in path-3 (scheduler-owned half-res
discovery) has been independently and more cleanly absorbed into the broader
**Path B+** work in `ir_explore`. Path B is essentially obsolete.

---

## Path B+ (4): scheduler-owned grouped-reduction plan

### What it does
Lifts grouped-reduction semantics up to the scheduler via:
- `BlockLocalReductionSpec` (scheduler.py)
- `NestedReductionPlan` (scheduler.py)
- `FusedNestedReductions.plan` (scheduler.py)

Codegen reads from `node.plan` instead of re-recognizing everything in
`codegen_nested_reduction`.

### Footprint vs baseline

| File | Δ vs baseline |
|------|--------------|
| `scheduler.py` | +237 (on top of baseline's +350) |
| `simd.py` | −289 (relative to baseline's +1469) |
| Net | −52 across these two files |

### What dissolves from baseline's `simd.py`
Per the existing comparison doc:
- `codegen_nested_reduction`: 229 lines → 177
- All early/late half-resolution discovery helpers
- Internal-buffer ownership checks
- Persistent-reduction prediction logic
- `shared_reads` re-derivation
- `group_size` re-derivation

### What stays in codegen
- `_DerivedIterationFamily`
- `DerivedIterationRangesRoot`
- `_GroupReductionLayout` (still constructs the layout, just from plan inputs)
- `_GroupedReductionOpsHandler` / `_PointwiseRemapHandler`
- `codegen_nested_reduction` (now thinner, just executes the plan)
- Half-resolution / NVFP4 codegen

### Tests
Tests pass per the documented exploration.

### Architectural value
Real layering improvement: grouped-reduction *semantics* are scheduler-owned;
*remapped consumer execution* stays in codegen. This is the cleanest answer
that doesn't require adding IR nodes.

### Verification (vs Path B's regressions)

I directly verified Path B+ does NOT have the smells found in Path B:
- ✅ `override_mask` parameter preserved (5 occurrences in `triton.py`)
- ✅ `inline_reduction_buffers` is GONE (relies on `cse.store_cache`)
- ✅ Test file uses in-process `_run_and_capture_kernel_source` helpers,
  same as baseline — no subprocess machinery
- ✅ Built on the current baseline `06ef55bbeef`, not the older one

This makes Path B+ a clean extension of baseline, not a regression.

### Cost to land vs baseline
- The work is **already done** in `ir_explore` worktree (uncommitted)
- Needs: commit, polish, possibly extract to `nested_reduction.py`
- **Estimated**: 1–2 days commit + polish, then standard review

### Tests verified
**80/80 tests pass in 54s** on the current uncommitted state of `ir_explore`,
verified directly. This is *with* Agent 1's IR-class additions present
(those are inert — defined but not wired into any lowering path). The
underlying Path B+ work is good.

---

## Path A (5): `BlockLocalReduction` IR primitive

*Agent assessment complete. Full report: `agent_space/explore_path_a.md`.
The agent built on the Path B+ state in `ir_explore` rather than starting
fresh, and structured the cost analysis around the Path B+ → IR primitive
delta.*

### What got implemented (delta on top of Path B+)
- `BlockLocalReduction(Reduction)` class in `torch/_inductor/ir.py`
  (~50 lines, compiles clean, not yet wired into lowering)
- Stub `maybe_recognize_nested_reductions(gm)` graph rewrite pass in
  `torch/_inductor/fx_passes/nested_reduction.py` (~160 lines, tags FX
  nodes but lowering hook not wired)

### What didn't get done
- `make_reduction` lowering dispatch
- `BlockLocalReduction.create()` (the IR-creation logic)
- Replacing scheduler-side `BlockLocalReductionSpec` reads with IR reads
- Replacing `_GroupedReductionOpsHandler.reduction()` with standard reduction codegen
- Tests (worktree has no build artifacts; CLAUDE.md forbids unprompted builds)

### Four critical challenges identified

**1. FX recognition is graph-aware, not local.** A purely local
`Reduction(reshape(x))` rewrite is wrong — `BlockLocalReduction` is only
useful when paired with an outer reduction over the same input. The
existing scheduler-time `NestedReduction.can_fuse` does ~110 lines of
shared-input detection, `MemoryDep.ranges` stride analysis, and coalescing
analysis. Replicating this at FX time requires either reimplementing
dependency analysis without `MemoryDep` (non-trivial) or making the FX
pass a "candidate marker" with legality staying in scheduler — in which
case the IR primitive is mostly a *rename* of `BlockLocalReductionSpec`.

**2. Codegen complexity doesn't collapse.** `TritonKernel.reduction()`
reduces over `range_trees[-1]`. For "reshape then reduce over interior
dim" via standard reduction codegen, you need either:
- *Option A* — add an extra range tree for the group axis, ~150-200 LOC
  backend change, plus regression testing of every existing reduction
  codepath
- *Option B* — special-case `BlockLocalReduction` in `store_reduction`,
  doing reshape internally — which is essentially calling
  `_GroupedReductionOpsHandler.reduction()` from a different entry point.
  The handler doesn't actually dissolve.

**3. Iteration-family infrastructure is orthogonal.**
`_DerivedIterationFamily`, `DerivedIterationRangesRoot`,
`_PointwiseRemapHandler`, and the half-resolution discovery all stay in
codegen *regardless of Path A*. They're consumer-side; Path A is
producer-side. The simd.py LOC most quoted (~1500) is dominantly
consumer-side machinery that Path A doesn't address.

**4. Dynamic shapes.** The codegen-only path rejects non-static `rnumel2`
at fusion time. Path A's IR rewrite either inherits this rejection
(regressing dynamic-shape support) or has to support symbolic
`group_size` — the latter is non-trivial new work.

### LOC estimate (Path B+ → IR primitive, Option B)

| File | Adds | Removes | Net |
|---|---|---|---|
| `ir.py` | +120 | 0 | +120 |
| `lowering.py` | +30 | 0 | +30 |
| `fx_passes/nested_reduction.py` | +250 | 0 | +250 |
| `fx_passes/post_grad.py` | +5 | 0 | +5 |
| `scheduler.py` | +20 | -50 | -30 |
| `codegen/simd.py` | +30 | -50 | -20 |
| **Total** | **+455** | **-100** | **+355** |

Option A adds another ~+200 LOC + significant testing risk for ~90 LOC
savings in simd.py (`_GroupedReductionOpsHandler` truly dissolves).

**Net: Path A doesn't reduce total LOC.** It redistributes ~50-100 LOC
from simd.py into ir.py/fx_passes while adding ~250 LOC of FX
recognition.

### What would dissolve (option B, conservative)
- `BlockLocalReductionSpec` (~10 LOC)
- Parts of `FusedNestedReductions._build_plan` (~25 LOC)
- `NestedReduction.is_small_dim_in_r` + parts of `can_fuse` (~80 LOC)

### What would survive (regardless)
- `_DerivedIterationFamily` (~108 LOC)
- `DerivedIterationRangesRoot` (~46 LOC)
- `_PointwiseRemapHandler` (~50 LOC)
- Half-resolution consumer discovery (~110 LOC)
- `_codegen_group_reduction_epilogue`, half-res codegen (~500 LOC)
- All NVFP4 / full-resolution / half-resolution codegen support

### Backend implications
- **Triton**: option B = no backend changes; option A = ~150-200 LOC and
  full reduction-codepath regression testing
- **CPP, MPS, Halide**: each gets a new IR node to dispatch on. Today CPP
  doesn't even see nested reductions (gated in `can_fuse`); with Path A,
  every backend explicitly handles or rejects `BlockLocalReduction`.

### Cost estimate
**2-4 weeks of focused work** for an experienced Inductor contributor
(option B). **4-8 weeks** if pursuing option A. Dominant costs:
1. FX-level recognition that's actually correct (~50%)
2. Codegen routing (~25%)
3. Dynamic shapes / symbolic group_size (~15%)
4. Backend dispatch hygiene (~10%)

### Agent's recommendation
**Defer Path A.** Three reasons:
1. Path B+ already moved the semantically meaningful pieces. The remaining
   simd.py is mostly `_DerivedIterationFamily`-shaped and isn't addressed
   by Path A.
2. Path A's payoff is mostly cosmetic without Option A — type system
   improves, executable behavior is unchanged.
3. Option A is a real backend project and should be scoped on its own
   merits, not bundled with nested-reduction landing.

Conditions that would tip toward pursuing Path A:
- A second fusion pattern lands needing similar IR machinery (amortizes the cost)
- Dynamic shape support for nested reduction becomes a hard requirement
- Reduce-over-interior-tree backend extension lands for unrelated reasons

---

## Side-by-side comparison

| Dimension | Baseline | Core landable | Path B | Path B+ | Path A |
|-----------|----------|---------------|--------|---------|--------|
| Total LOC added | +2793 | +2118 | +2774 | ~+2741 | ~+3096 |
| `simd.py` LOC | +1469 | +1008 | +1267 | ~+1180 | ~+1160 |
| `scheduler.py` LOC | +350 | +350 | +480 | ~+587 | ~+557 |
| `triton.py` LOC | +274 | +228 | +354 | +274 | +274 |
| New file: `fx_passes/nested_reduction.py` | — | — | — | — | +250 |
| New file: `ir.py` additions | — | — | — | — | +120 |
| Tests passing | 80/80 | n/a (smaller) | 80/80 | **80/80 (verified)** | n/a (untested) |
| Built on current main | ✅ | ✅ | ❌ stale | ✅ | ✅ |
| `override_mask` preserved | ✅ | ✅ | ❌ removed | ✅ | ✅ |
| `inline_reduction_buffers` removed | ✅ | ✅ | ❌ re-added | ✅ | ✅ |
| In-process FileCheck tests | ✅ | ✅ | ❌ subprocess | ✅ | ✅ |
| Early/late dual pass | Present | Present | Removed | Removed | Removed |
| Reviewability | Hard | Medium | Hard + smells | Medium | Hardest |
| Time to land | 3-5d + review | 2-4d + review | ~14h + review | 1-2d + review | 2-4 weeks + review |
| `_GroupedReductionOpsHandler` dissolves | No | No | No | No | Only with Option A backend ext (~+200 LOC) |
| `_DerivedIterationFamily` survives | n/a | n/a | n/a | ✅ | ✅ |
| Future generality | Low | Low | Medium | Medium-High | High (with Option A) |
| Risk | None (done) | Low | Medium (regressions) | Low | High (unknown) |

---

## Decision framework

The right answer depends on what dominates:

### If "ship soon" wins
**Pick Core landable (#2)**. Smaller scope, faster review, NVFP4 follows
later. Trade-off: lose the marquee NVFP4 perf result on this PR.

### If "land the full feature" wins
**Pick Path B+ (#4)** if it's actually as polished as the docs say. Same
feature surface as Baseline, cleaner architecture, similar timeline.
Path B+ may be the sweet spot.

### If "architecturally pure" wins
**Pick Path A (#5)** *if and only if* the agent probe shows it's ≤ a few
weeks. Otherwise the cost dominates and you'd be better off shipping
Path B+ now and pursuing IR uplift as a separate Inductor project.

### If "minimize Inductor risk" wins
**Pick Baseline (#1)** as-is with extraction + dual-pass cleanup.
No Inductor architecture changes, just feature-local code.

---

## Final Recommendation

**Land Path B+** as the primary plan. Test suite is **verified passing
overnight: 80/80 in 54s.** No fallback needed.

Concretely:

1. **Discard Agent 1's IR-stub additions in `ir_explore`** (the inert
   `BlockLocalReduction` class in ir.py and the new
   `fx_passes/nested_reduction.py`). They're not part of Path B+ — they
   were the Agent 1 probe, and they're not wired into anything. Two
   commands: `git checkout torch/_inductor/ir.py && rm
   torch/_inductor/fx_passes/nested_reduction.py`. Or keep them on a
   separate branch as a Path A starting point for future work.
2. **Commit the Path B+ work** to a real branch. Likely two commits:
   (a) the scheduler-side plan dataclasses (`BlockLocalReductionSpec`,
   `NestedReductionPlan`, `FusedNestedReductions.plan`); (b) the simd.py
   simplification reading from the plan.
3. **Polish and review.** Optionally extract codegen-side nested-reduction
   to a `nested_reduction.py` module per the earlier discussion (further
   shrinks simd.py footprint).
4. **Stack into 2–3 reviewable PRs** if appropriate.

**Path A** belongs in a follow-up architecture doc, not this branch:
- It's 2–4 weeks of work minimum
- The payoff without backend extension is mostly cosmetic
- It's worth pursuing when a second fusion pattern needs similar machinery

**Path B (path3) should be archived.** It's been independently superseded
by Path B+ and Baseline's own cleanup. The work is not lost — the design
insight (scheduler-owned half-res discovery) is preserved in Path B+.

**Core landable** is available if you want to ship without NVFP4 first,
but doesn't address layering. Use it only if review-bandwidth pressure
dominates other concerns.

---

## Open questions for the morning

Things the data answered:
- ✅ Path B+ tests pass (80/80, verified directly overnight)
- ✅ Path A is not worth pursuing as part of this landing
- ✅ Path B (path3) is obsolete

Things you still need to decide:

1. **How do you want to commit Path B+?** The uncommitted work is currently
   in two intertwined files (scheduler.py +237, simd.py −289). Options:
   - One commit (simpler, harder to review)
   - Two commits (scheduler additions, then simd.py simplification reading
     from plan)
   - Stack of N small commits (most reviewable)

2. **Is the current `ir_explore` branch the right place to land from?**
   Or should you cherry-pick the uncommitted state onto a fresh branch
   based on `nested_reduction_single_commit_wip`? My weak preference:
   cherry-pick to a fresh branch, since `ir_explore` also accumulated
   Agent 1's stub additions which you don't want to land.

3. **What to do with Agent 1's IR-stub work?** The inert
   `BlockLocalReduction` class in ir.py and the new
   `fx_passes/nested_reduction.py` are interesting future-work seed
   material. Options:
   - Discard (fastest)
   - Park on a separate branch (`nested_reduction_path_a_seed`) as
     starting material for future architecture work
   - Include in a follow-up doc/branch but not the landing PR

4. **Do you want the codegen-extraction step (`nested_reduction.py`) on
   this PR, or as a follow-up?** Path B+ + extraction would reduce
   simd.py to ~+400 LOC instead of ~+1180. Bigger reviewability win but
   adds churn.

5. **What's the fate of the half-res / NVFP4 path?** If you go with
   Core-landable's strategy of shipping without it first, that's a
   different scope decision than the architecture decision. Could be
   layered onto any of the candidate paths.
