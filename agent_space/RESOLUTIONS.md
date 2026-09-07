# Sub-parent epilogue stack: what was found, and how each item was resolved

> **Historical snapshot.** This records an earlier stack shape and intentionally
> keeps its old SHAs and conclusions. It is not the current worklist. Start with
> `READ_ME_FIRST.md` and use `FOLLOWUPS.md` for current status.

Companion to the feature-PR docs. This one is the changelog: every issue raised,
what the evidence was, and what was done at that time.

Stack as of this writing:

```
0f77d8af6e0  legality-path simplification (batch 4)     308 tests OK
bd543b559dc  plan state explicit          (batch 3)     308 tests OK
aa693b96327  bounds + contiguous flag     (batch 5)     308 tests OK
c9b74512737  mechanical cleanup + 6.2     (batch 2)     308 tests OK
0837972e9f6  StarDep/WeakDep checks kept                308 tests OK
56df9eadea2  test hygiene                 (batch 1)     308 tests OK
04a665ecf11  MXFP6 staged packing         #191775       308 tests OK
d1d827b05dd  contiguous sub-parent        #190596       290 tests OK
d8d74f97585  interleaved nested           #190595       244 tests OK
94eb87b4fa0  interleaved sub-parent       #190594       210 tests OK
```

The six commits above the four PRs are cleanup, staged on top rather than
folded down to avoid a rebase per fix. Each commit message names the PR its
change logically belongs to, for redistribution before submitting.

Every commit is green **individually** — that was not true at the start.
Backups: `backup/c1-190594` … `backup/c4-top`, `backup/wip`, and
`wip/peak-memory-guard` for work removed from the stack.

---

## A. Real bugs found and fixed

### A1. `NameError: group_reduction_vars` — fatal, #190595

`_codegen_nested_grouped_schedule` referenced a name bound only in its
*caller*. Verified with pyflakes across the stack:

```
6a7c1d8ec56 (c1): clean
f3599046d4c (c2): undefined name 'group_reduction_vars'  :3349
c1f9eee6309 (c3): undefined name 'group_reduction_vars'  :3371
20448685ad5 (c4): undefined name 'group_reduction_vars'  :3388
```

`simd.py:3156` asserts the grouped reduction is in the schedule, so the branch
holding the reference always executes: **every `FusedNestedReductions` codegen
raised**. #190595 and #190596 were non-functional as pushed, and the Test Plans
in their commit messages could not have passed.

**Resolution.** Threaded the `_GroupedReductionVars` bundle through the callee's
signature (3 hunks), landed in #190595 where it was introduced. Recommend adding
`F821` to lint for `torch/_inductor/codegen/` — an undefined name on a mandatory
path survived three commits.

### A2. `min_xblock` non-power-of-two → compile crash, #190594

`min(int(numel_static), 128)` can produce a floor that doesn't divide
`TRITON_MAX_BLOCK["X"]`. Reproduced:

```
B= 3  FAIL  AssertionError: XBLOCK divides max_block["X"] but XBLOCK=3 and max_block["X"]=4096
B= 5  FAIL  ... XBLOCK=5 ...
B= 6  FAIL  ... XBLOCK=6 ...
B= 8  OK
```

**Resolution.** Initially clamped to a power of two; ultimately the whole floor
was **deleted** from this path (see B3b), which removes the crash class rather
than patching it. A `(3, 16, 16)` parametrization was added to
`test_standalone_sub_parent_epilogue`; all four shapes compile with correct
numerics.

### A3. Full-resolution reader joining a sub-parent group, #190594

`test_standalone_nvfp4_inline_asm_rejects_fullres_reader` failed with
`expected pointwise sizes to match pointwise_numel * red_numel`. Instrumented
the group:

```
op0 (8192,16) reduction | op1 (8192,1) | op2 (65536,1) half-res | op3 (131072,1) full-res reader
legal=False  loose=False
```

`op3` reads `op2`'s output but joins the group by generic numel matching
(`131072 == 8192*16`), leaving a half-resolution member that generic coalesce
tiling cannot model. `_sub_parent_epilogue_leaf_violation` bailed early — with
`op3` present the loose plan is `None`, so it returned "no violation" and
permitted exactly the fusion it exists to block.

**Resolution.** `_is_sub_parent_shaped` — when no plan covers the combined set,
reject if any member sits at a fraction of the parent tile. A first attempt was
too broad and broke the nested-append path (14 failures at c2/c3, caught and
narrowed to exclude `FusedNestedReductions`, which owns its own sub-parent
stage). **This helper is load-bearing**: doc item 4.3 must preserve it.

---

## B. Design issues resolved

### B1. `min_xblock` carve-out for MXFP6 — removed

The MXFP6 path forced `min_xblock = 16` "to limit registers". Backwards:
register pressure from casts argues for *permitting* smaller blocks, not
mandating one, and if 16 is optimal the autotuner finds it. Measurements support
this — the binding constraint for these kernels is the fixed `num_warps=2` in
the default candidate set, not the floor.

**Resolution.** Deleted; `min_xblock` is now uniform. One kernel-form test
pinned the unjustified `16` and was updated.

**Followed through in B3b:** the standalone `min_xblock` was a performance
heuristic riding on a correctness mechanism, and is now gone entirely.

### B2. Unrelated peak-memory heuristic — removed from the stack

`can_fusion_increase_peak_memory` taking the caller's `shared_data_score`,
`_pointwise_reduction_broadcast_orders`, and a `test_loop_ordering.py` test were
riding inside the MXFP6 commit. `test_loop_ordering.py` has **zero** references
to `triton.nested_reduction`, so this changes `memory_overhead > 32 * score` for
*every fusion in the compiler*.

Initially split into its own commit below MXFP6, per recommendations item 6.1.
That was wrong: the doc's conditional ("if the preshuffled test depends on the
broadcast reorder…") was never tested. Dropping the commit entirely gives
**304/304 passing**, so MXFP6 has no dependency on it.

**Resolution.** Removed from the stack, preserved at tag
`wip/peak-memory-guard`. It is a genuine general improvement and deserves a
standalone PR with a test showing its effect on the generic path.

### B3. Rate function: enumeration replaced by derivation, #191775

`_sub_parent_epilogue_rate` had a power-of-two branch plus a hardcoded 4→3
branch. Two attempts to improve it were both wrong — naming the constants
`MXFP6_LANES_IN/OUT` put a format name in the scheduler, and a `(1, 3)`
allowlist just renamed the enumeration.

The relation has a closed form. Since
`node_numel * factor == output_lanes * full_numel`, the pair *is* the reduced
fraction `full_numel / node_numel`:

```python
ratio = V.graph.sizevars.simplify(full_numel / node_numel)
factor, output_lanes = int(ratio.p), int(ratio.q)
```

Pair packing reduces to `2/1`, chunked gating to `4/1`, MXFP6 to `4/3`. No list,
no format names.

**Validation.** An `(8,3)` rate (8 three-bit values into 3 bytes) — a shape
nobody wrote code for — is admitted by the derivation, **declined by the
matchers**, and falls back to correct unfused code. Three findings:
the allowlist was redundant (the matchers already gate it); rejection now
happens at the component that genuinely cannot handle the shape; and an
unmodelled input degrades rather than miscompiles. 308 tests still pass.

Symbolic ratios (`1024*B / 768*B → 4/3`) now work for free; the old
`isinstance(..., Integer)` guard rejected them.

### B3b. `min_xblock` deleted from the standalone path, #190594

Following B1 to its conclusion. `min_rblock` is the legality constraint on this
path -- the lanes derive from the parent's R axis, so the tile must hold a whole
lane group. `min_xblock` was throughput only. Deleting it removes, in one
change: the non-power-of-two crash class, the `CONTIGUOUS` carve-out that
existed solely to opt chunk consumers out of the floor, and a dynamic-shape
`optimization_hint` dependence that changed behaviour for dynamic graphs.

Three kernel-form tests now assert the floor's absence. The nested path's
`min_xblock` (`simd.py:3236`) is untouched -- there it picks whichever axis
carries the grouped reduction and is load-bearing.

### B3c. One lane formula, #190596

The contiguous lane was derived twice: the planner from `parent_rnumel`, codegen
from `local_reduction_size`, equal only because the standalone path passes the
former as the latter. Both now call
`NestedReduction.sub_parent_contiguous_lane(index, factor, parent_extent)`, and
`_ContiguousSubParentRemappedValue` carries the parent extent rather than a
pre-divided child extent, so the division that differed happens once.

The unvalidated second lane candidate went with it (instrumented: 0 hits across
304 tests). A lane that does not resolve is now a loud assert.

### B4. Tiling: asserted after the fact → decided at fusion time, #190594

`assert len(kernel.range_trees) == 2` ran *after* tiling was chosen and after
`can_fuse` had already committed. A 3-D tiling (`native_matmul` on a dot
reduction, or `tile_reductions`) was therefore an uncaught `AssertionError`
mid-compile rather than a declined fusion.

**Resolution, in two steps.** First `_sub_parent_tiling_is_2d` in
`_sub_parent_epilogue_plan` computes the tiling and checks `len(tiling) == 2`,
rather than reasoning about which config combinations widen it. Then codegen
*forces* `create_tiling([numel], [rnumel])` instead of re-running the heuristic,
so the fusion-time answer and the codegen-time answer cannot disagree at all --
the first draft had the two sites calling the heuristic with different node sets
(`reduction_nodes` vs `parent_schedule`), reintroducing the very divergence
class being fixed.

Relatedly, the coalesce guard in `codegen_node` is now an **assertion**: a group
holding a sub-parent-shaped member must have a plan, because fusion is what put
the member there. Losing the plan by codegen means a fusion was wrongly
admitted, which is a bug to surface rather than a state to tolerate -- and it is
exactly what the nvfp4 failure (A3) was.

---

## C. Reviewer comments (drisspg)

| comment | resolution |
|---|---|
| #190595 `scheduler.py:1149` — G=2 makes REDUCED and SUB_PARENT share a numel; could a pair consumer be misclassified? | **Doesn't reproduce.** G=2 fuses to 1 kernel *with* a `tl.split` and is bit-exact vs nested-off. A misclassification would produce no split and different values (element-0-of-pair ≠ amax-over-pair). Added the requested regression as `@parametrize("G", [2, 16])`. |
| #190595 `:770`, #190596 `:928` — "do we have any correctness test?" | **Yes** — `check_nested_matches_unnested` compiles with `nested_reduction=False` and `assertEqual(atol=1e-2)`. But the instinct was right that on-vs-off only validates the *transform*; added an eager-**exact** test (see C1 below). |
| #190596 `triton.py:6217` — doc block, since `tl.split` requires the split axis last | **Added.** Documents that `tl.split` only splits the trailing axis into two, that both layouts reduce to "get the lane axis last", and that the single `tl.permute` is the entire codegen difference between them. |
| #190596 `simd.py:1513` — nit: description | **Added** to `_resolve_remapped_value`, covering why the two layouts recover the lane differently. |
| #190596 `scheduler.py:817` — doc how we got these numbers | **Superseded by B3** — the numbers are now derived, so there is nothing to justify. |
| #190595 `:582`, `:1111`, #190595 `:8497` (argument reversal) | **Open.** The reversal question wants a regression presenting the pair in both orders, not a comment. |

### C1. An eager-exact correctness test

Comparing packed output against eager is normally unstable. Measured on the real
MXFP6 graph: **583 / 32768 codes differ (1.78%)**, with 95% of mismatches
sitting within `1e-2` of a `round()` tie.

Chased the cause and got it wrong twice before testing properly:

| suspected | verdict |
|---|---|
| `exp` in silu | **no** — an `identity` pre-op diverges just as much |
| `sum` reduction order | **no** — same |
| bf16-vs-fp32 chain | contributes ~half |
| **`/7.5` → `*0.1333…` reciprocal rewrite** | **dominant cause** |
| this stack | **not involved** — nested on/off bit-identical, both 234 off eager |

`emulate_precision_casts` halves it (1.79% → 0.92%) but cannot reach exact. A
power-of-two divisor takes it to 1–3 in 32768.

**Resolution.** `test_producer_consumer_mxfp6_four_to_three_pack_exact` sidesteps
the problem instead of tolerating it: integer inputs whose group max *is* the
power-of-two divisor give a scale of exactly `1.0`, so no rounding occurs on
either side and the comparison is `atol=0, rtol=0` against eager. Still
exercises the full factor-4 / three-output-lane path, and the output has 22
distinct byte values so it is not degenerate.

**On the "252" scare.** `high = ((v2 >> 4) & 0x03) | (v3 << 2)`, so a code
differing by 63 (the `& 0x3F` wraparound at a tie, 63→64→0) gives `63 << 2` =
252. The mask is discontinuous at the boundary, converting the smallest possible
float disagreement into the largest possible byte disagreement. Not a bug.

---

## D. Generated-code audit (four auditors over 13 captured kernels)

**Correctness: clean.** No mask/index-family mismatches, no bound errors,
`min_rblock ≥ lane factor` in every looped kernel (hand-checked at the floor,
where a straddling lane group would corrupt silently), both factor-4 split trees
assign lanes correctly, mxfp6 bit-packing matches the reference bit-for-bit.

**CSE: working.** Zero duplicated subexpressions across all 13 kernels,
including mxfp6's three output-lane passes — which answers the design question
behind the lane-generic codegen: it is *not* paying 3× for shared work.

**Open, perf only:**
- looped kernels re-read with 2–4 strided loads where one contiguous load plus a
  split would do (`standalone_looped`, `mxfp6_looped`)
- the split is applied to the raw input rather than the reduction's intermediate,
  so the elementwise chain runs twice (largest single item: ~45% of
  `mxfp6_persistent`'s arithmetic is a duplicated `exp`/`div`)
- `evict_last` on streamed data never re-read
- `num_store` reports 3 where mxfp6 emits 4 — the
  `max(num_store, len(store_buffer_names))` repair cannot count multiple lane
  stores into one buffer. **Not fixed**: it feeds autotune heuristics and no test
  pins it, so a blind change was not worth the risk.

---

## E. Persistent vs looped — the residency rule

Worth recording because it was initially stated wrongly. The rule is about
**source provenance, not persistence**:

| source | looped behaviour |
|---|---|
| **external** (epilogue reads a kernel input) | must re-read — the scale isn't final until the loop ends and the row has been streamed |
| **internal** (recomputed in loop 2, e.g. RMSNorm's `y`) | `tl.split`s in the final loop, **zero extra reads** |

Verified on a D=16384 looped RMSNorm + pair packing: one kernel, `x` read twice
total (exactly what plain RMSNorm costs), packing free. An earlier claim that
looped kernels never `tl.split` came from a kernel-form test for the
*standalone* (external-source) case and is not a general contract.

---

## E2. The duplicated elementwise prefix (investigated, not fixed)

`mxfp6_persistent` emits **five `libdevice.exp`** where one would do: the
reduction computes `silu(x)` over the full tile for the amax, and the epilogue
then re-derives it per lane. The load *is* shared -- one `tl.load`, four lanes
off it via `tl.split` -- so the tile-reuse the feature exists for works. What is
not shared is the elementwise chain between the load and the reduction.

**This is not a CSE failure.** The reduction computes `silu(tmp0)`, the epilogue
`silu(tmp1..tmp4)`. Different operands, genuinely different expressions. The
issue is *where the split sits*: today it is immediately after the load, so the
chain runs per lane. `split(silu(x)) == silu(split(x))` for elementwise `silu`,
and only the second form is emitted. The lanes truly diverge at the first
lane-*combining* op (`low = v0 | (v1 & 3) << 6`); everything before that could
run once at parent resolution.

The planner reasons about `MemoryDep`s, so it can only name buffers, and
`silu(x) * 1.125` is a fused intermediate with no buffer name. That is also why
"just realize it" is not the fix.

**Cost: unmeasured.** Three A/B attempts, three different confounds:

| probe | confound |
|---|---|
| silu vs no-silu graph | different graphs, different autotuned configs |
| `_inductor_test.realize` + time | +134 MB traffic, 1 kernel becomes 2 |
| realize + remove the internal-source guard | `_realize` is `x.realize(); return clone(x)` -- the **clone** was the blocker, not any guard |

Two guards were suspected and both cleared by experiment: removing the
"internally produced source" rejection changed nothing (and broke
`test_producer_consumer_mxfp6_rejects_source_read_by_reduction`, which exists
for it), and neutralising `_is_sub_parent_shaped` changed nothing either.

**Recommendation.** Treat as a standalone follow-up, not a cleanup. The
implementation is a deferred-elementwise layer over the ops handler: the
materializer returns a lazy lane reference, elementwise ops accumulate at parent
resolution, and the split is forced only when a lane-combining op appears --
order 100-200 subtle lines. Size the prize first with two hand-written kernels,
because no in-compiler A/B constructed so far holds everything else fixed.

## F. Still open

Batches 1-5, 6.1 and 6.2 of the recommendations plan are all done, as are the
`remaining.clear()` and `min_xblock` items. What remains, by value:

1. **`SIMDKernelFeatures(parent_schedule, …)`** excludes epilogue nodes, so
   `select_index_dtype()` cannot see their buffers → possible silent int32
   address overflow. Not fixed: it changes codegen broadly and wants more
   validation than was available.
2. **The shared elementwise prefix** (see E2) — five `exp` where one would do.
   A standalone feature, cost unmeasured.
3. **A positive test for 4.1's relaxation.** Landed deliberately widening the
   accepted set on a local correctness argument, with no test reaching the
   newly-accepted lane-invariant shapes.
4. **`num_store`** undercount (see D).
5. **Perf, deferred by choice**: the split applied to the raw input rather than
   the reduction's intermediate; strided reloads in looped kernels; `evict_last`
   on streamed data.
6. drisspg's argument-reversal regression, and two doc nits.
7. `wip/peak-memory-guard` as a standalone PR.
8. **Nothing is pushed.** GitHub is ten commits behind.

Two items from the plan were deliberately **not** done, both because their
premise turned out to be false on inspection: 1.5 (folding
`_float_to_mxfp6_e2m3` into a one-liner — its op count is load-bearing for the
fusion decision) and 4.6 (dropping `allow_reduced_broadcast` — removing it turns
a loud assert into a silent wrong broadcast).

## G. Process notes

Two measurement errors worth recording, both the same shape: I reported findings
from runs executed on the **wrong commit** after a `git switch` I did not
re-verify. Once produced a false "the tests don't exist" (they exist from c2),
once a false "RMSNorm + packing doesn't fuse" (it does). Both were caught by
re-running with `HEAD` printed. Every measurement in this document now states
the commit it ran on.
