# Adversarial verification of `STACK_SIMPLIFICATION_RECOMMENDATIONS.md`

Verified read-only against tip `f020897c032` (`git show <blob>`; no checkout, no writes to
tracked files). Line numbers below are from the `f020897c032` blobs.

**Premise correction up front.** The dispatch said the tip adds a
`_reads_planned_sub_parent_output` method. That symbol does **not exist** anywhere: not in
`f020897c032`, not in the working tree (`git status --porcelain` is clean for `simd.py`,
`scheduler.py`, `test_nested_reduction.py`), and not in any reachable commit
(`git log -S --all` finds nothing). The actual `bdf385fc2ed -> f020897c032` delta to
`simd.py` is +35/-1: a new `_is_sub_parent_shaped` static method plus a new branch in the
`plan is None` arm of `_sub_parent_epilogue_leaf_violation`. Details in item 3.

---

## Summary

| # | Item | Claim | Verdict |
|---|---|---|---|
| 1 | 4.1 | Contiguous lane search is redundant; uniqueness check adds nothing ("strict equivalence") | **REFUTED as stated** (relaxation, not equivalence); safety argument CONFIRMED |
| 2 | LOC | "~750 LOC removable, ~400 of it tests" | **REFUTED** (inflated ~25-30%; the plan's own batches sum to ~540) |
| 3 | Sequencing | Uncommitted `_reads_planned_sub_parent_output` collides with 4.3 | **REFUTED** (symbol does not exist; worktree is clean) |
| 4 | 4.3 | leaf_violation's first build "re-implements `_can_fuse_sub_parent_reduction_epilogue`" | **REFUTED** (node ordering differs; sharing them is not behaviour-preserving) |
| 5 | 4.3 | Plan buildable once; three leaf checks pure; `leaves_ok` computable unconditionally | CONFIRMED |
| 6 | 4.4 | CUDA template ladder is already a no-op for a `FusedSchedulerNode` | CONFIRMED (weakened: 2 of 5 predicates test node2, not node1) |
| 7 | 4.6 | `allow_reduced_broadcast` only turns a crash into working code | CONFIRMED literally; **materially weakened** (turns an invariant assert into a possible silent miscompile) |
| 8 | 2.1 | `_is_axis_tile_shaped` == `parent_dim(v) == block`; `is_parent_tile_shaped` branch dead | CONFIRMED |
| 9 | 2.4a | `assert source_layout is INTERLEAVED` dead | CONFIRMED |
| 10 | 2.4b | `if domain_context.parent_half_domain is None` in PARENT_HALF arm dead | CONFIRMED |
| 11 | 2.4c | `expected_numel: ... \| None` never assigned None | CONFIRMED |
| 12 | 2.4d | `must_materialize_names: ... \| None = None` always passed | CONFIRMED |
| 13 | 2.4e | `_nested_index_equivalent_dep_names` fall-through unreachable | CONFIRMED |
| 14 | 3.1 | `assert sub_parent_source_layouts is not None` dead | CONFIRMED |
| 15 | 5.x | Support matrix rows (layout x factor x lanes) | CONFIRMED (all 6 rows) |
| 16 | 5.3 | CONTIGUOUS/4/3 reachable but untested | CONFIRMED statically, with a refinement |
| 17 | 5.1 | Rate relation written twice; `(lanes==1 or factor==4)` clause needed | CONFIRMED |
| 18 | 5.2 | `PARENT_HALF_FACTOR` borrowed as a literal 2; `<= 4` is equivalent to `in (2,4)` | CONFIRMED |
| 19 | 5.4 | `parent_rnumel=None` is a hidden feature toggle | CONFIRMED |
| 20 | 4.5(a) | Moving the four dispatches into `_can_fuse_impl` kills the flag | CONFIRMED (caveat: reorders 3 guards) |
| 21 | Do-not-do | `MAX_SUB_PARENT_FACTOR=16` is a tested boundary | CONFIRMED |
| 22 | Do-not-do | `_sub_parent_epilogue_internal_reads_match` not redundant, has a negative test | CONFIRMED |
| 23 | Do-not-do | `_SubParentSourceLoadMaterializer` stacks over two different inner handlers | CONFIRMED |
| 24 | 6.1 | Loop-ordering work is separable and lives in commit 4 | CONFIRMED |
| 25 | 6.2 | Deleted test was landed; `_same_index_with_prefix_size` survives untested | CONFIRMED |
| 26 | 2.2 (b1) | Second lane-resolution candidate in `_resolve_remapped_value` is dead | **UNVERIFIABLE** statically (see item 26) |
| 27 | 2.5/2.6 | Noise items (re-wrap, `omitted_nodes`, `min_xblock`, stale docstring) | CONFIRMED (docstring is incomplete, not wrong) |

**Tally: 22 CONFIRMED, 4 REFUTED, 1 UNVERIFIABLE.**

---

## 1. Item 4.1 — index the contiguous lane directly. REFUTED as stated.

`scheduler.py:1094-1111`:

```python
matching_lanes = [lane for lane, expected in enumerate(expected_by_lane)
                  if V.graph.sizevars.statically_known_equals(dep.index, expected)]
return len(matching_lanes) == 1 and V.graph.sizevars.statically_known_equals(
    NestedReduction._contiguous_sub_parent_epilogue_emitted_lane(dep, sub_parent_factor, parent_rnumel),
    matching_lanes[0])
```

**The safety half of the argument is sound.** I verified the codegen side: for a
`_ContiguousSubParentRemappedValue`, `_resolve_remapped_value` (`simd.py:1539-1560`)
computes exactly one lane from `FloorDiv(Mod(offset, factor*child_extent), child_extent)`
and returns `parts[lane]` — nothing else. `_unique_trailing_sub_parent_dim` (1046-1066)
plus the entry guard `factor * dep.size[-1] == parent_rnumel` (1190-1192) force
`dep.size[lane_dim] == child_extent` and `lane_dim == len(dep.var_names)-1`, so
`expected_by_lane[L]` is precisely the address set held by `parts[L]`. Therefore
`dep.index == expected_by_lane[emitted_lane]` alone establishes that the read is served
correctly.

**But the plan (and `simplify_c3` #1, which calls it "a strict-equivalence argument") is
wrong that the change is behaviour-preserving.** The two predicates differ on exactly one
input class, and there the new one is *more permissive*:

- Today: reject when `len(matching_lanes) > 1`.
- Proposed: accept when `dep.index == expected_by_lane[emitted_lane]`, regardless of how
  many other lanes alias.

Multi-match is reachable. Concretely, whenever `reduction_dep.index` is invariant in the
lane variable — a source read broadcast along the parent reduction axis (`bias[x]` read by
the reduction over `(x, r)`, index `x`, while the child epilogue reads it at
`(x, r')` with the same index `x`) — every one of the `factor` substitutions produces the
same `expected`, so `len(matching_lanes) == factor` and the plan is rejected today. The
same holds for periodic reduction indices such as `Mod(r, child_extent)`, and for a
constant-index scalar read of a source buffer whose reduction read also has a constant
index (cf. `test_standalone_sub_parent_rejects_same_buffer_scalar`,
`test_nested_reduction.py:1579`, which today rejects for a *different* reason — zero
matching lanes — so it will not catch the change).

I could not construct a case where accepting is *wrong*: in every multi-match case I built,
`parts[emitted_lane]` still holds the values the epilogue wants. So the recommendation is
probably fine to land. But it must be described as **"relax the contiguous matcher, which
also lets us delete the search"**, not as a redundancy removal. Consequences:

- It can enable fusions that never previously happened, on a path with no test coverage.
  A `test_nested_reduction.py` sweep proves nothing here — nothing currently exercises it.
- The doc's own risk framing ("Pure legality-check restructuring, no codegen change, and
  the argument above is a strict-equivalence argument") understates it.

**Implementation hazard.** If `_contiguous_sub_parent_lane(...) -> int | None` is written
as "simplify, then require a literal `sympy.Integer`", it is *stricter* than today, because
today's `statically_known_equals(emitted_expr, matching_lanes[0])` can prove equality via
the shape env when structural simplification cannot. Preserve behaviour by looping
`for L in range(factor): if statically_known_equals(emitted, L)` — i.e. the same shape as
`_select_lane` (`simd.py:1523-1530`).

**Recommendation:** keep the item, restate the justification, and add a positive test for
the broadcast-source case so the newly-admitted plans are covered.

---

## 2. LOC aggregate — REFUTED (inflated).

Measured stack size (base `8b054943180^` .. `f020897c032`, `torch/_inductor` + `test/inductor`):
3717 changed lines, of which `test_nested_reduction.py` is 1736.

The "~750, ~400 tests" headline comes from summing the four per-commit estimates
(151 + ~160 + ~364 + 87 = ~762). That sum **double-counts** at least:

- `emit_split_via_reshape` merge: c1 #6 (~13) *and* c3 #6 (~10) — one edit, counted twice.
- Kernel-form assertion helper: c1 #4 (~22) *and* c3 #7 (~70) — recommendation 1.3 does
  both in one change, worth ~70 total, not ~92.
- c3 #7's ~70 is the *delete both tests* option; the plan (1.3) picks the *fold* option the
  same doc prices at ~55.

The recommendations doc's own sequencing table is the honest number and contradicts the
headline: `300 + 90 + 60 + 90 + 0 + 0 + 0 = 540`, and batch 6.2 *adds* ~20 lines back.

My independent bottom-up on batch 1 (NVFP4 body 3-4x ~66; PTX literal 9 sites ~8; RMSNorm
chunk model 10 sites ~63; interleaved-pair model ~14; 36 local `import ... as F` ~35;
parametrize 3+3 chunk tests ~60-80; kernel-form fork fold ~70; `run_nested_and_unnested`
~30; `_float_to_mxfp6_e2m3` ~32) lands at **~340-360**, not 400.

**Defensible restatement: ~520-600 LOC total, ~320-360 of it tests.**

---

## 3. Sequencing collision on `_reads_planned_sub_parent_output` — REFUTED.

The doc's first collision bullet ("The uncommitted `_reads_planned_sub_parent_output`
change in the worktree edits `_sub_parent_epilogue_leaf_violation` … Land or drop that
change before starting 4.3") is stale. The symbol does not exist in the worktree, the tip,
or history. The worktree is clean against `f020897c032` for all three stack files. **Drop
the bullet.**

What *did* land between `bdf385fc2ed` and the tip is `_is_sub_parent_shaped`
(`simd.py:2620-2640`) plus this new arm inside `_sub_parent_epilogue_leaf_violation`:

```python
plan = self._sub_parent_epilogue_plan(nodes, numel, rnumel, check_leaves=False)
if plan is None:
    if isinstance(node1, scheduler.FusedNestedReductions) or isinstance(node2, ...):
        return False
    return any(self._is_sub_parent_shaped(node, numel, rnumel) for node in nodes)
```

**This does not invalidate item 4.3.** The new branch lives entirely in the `plan is None`
arm, which the `leaves_ok` rewrite leaves untouched. It also does not affect 4.4, because
the append path is explicitly routed away from it.

---

## 4. Item 4.3, sub-claim "the first build re-implements `_can_fuse_sub_parent_reduction_epilogue`" — REFUTED.

The two are *not* the same call:

- `_can_fuse_sub_parent_reduction_epilogue` (`simd.py:2608-2618`) builds on
  `[*reduction_node.get_nodes(), *consumer_node.get_nodes()]`.
- `_sub_parent_epilogue_leaf_violation` (`simd.py:2667`) builds on
  `[*node1.get_nodes(), *node2.get_nodes()]` — **argument order**, which is the pointwise
  first when `can_fuse` is called with the consumer as node1.

Node order is load-bearing: `_sub_parent_epilogue_candidate_nodes` collects candidates in
`nodes` order and then rejects the plan outright if
`output_lanes != tuple(sorted(output_lanes))` (`scheduler.py:797-799`). A pointwise-first
vs reduction-first ordering can therefore produce a plan in one order and `None` in the
other whenever a multi-stage (1-lane + 3-lane) epilogue is involved — i.e. exactly the
MXFP6 shape commit 4 exists for.

So: "builds it twice on identical arguments" inside `leaf_violation` is **true** (both use
the same `nodes`), but "its first build re-implements `_can_fuse_sub_parent_reduction_epilogue`"
is **false in general**. Implementing 4.3 by having `leaf_violation` call
`_can_fuse_sub_parent_reduction_epilogue` would be a behaviour change. Keep the local
`nodes` list.

## 5. Item 4.3, main claim — CONFIRMED.

`check_leaves` gates exactly three terminal `return None`s (`scheduler.py:711`, `717`,
`721`) and nothing else; everything upstream is identical. Hence
`plan(check_leaves=True) == plan(check_leaves=False) if leaves_ok else None`, and
`leaves_ok: bool` on `SubParentEpiloguePlan` is a faithful encoding.

Purity of the three checks:
- `_sub_parent_epilogue_source_loads_are_unambiguous` (990-1015): reads `read_writes.reads`,
  `dep.rename`, dict/`OrderedSet` construction. Pure.
- `_sub_parent_epilogue_outputs_unread` (967-988): same shape. Pure.
- `_sub_parent_siblings_are_source_free` (1017-1043): adds `node.group` and
  `statically_known_equals`. Pure (`statically_known_true` does not install guards).

None of them mutates scheduler or node state, so computing `leaves_ok` unconditionally
cannot change any result — only cost. Two cost notes worth keeping in the item:

- Today the `check_leaves=False` build **skips** those three scans; after the change every
  build pays for them. The net win (3 builds -> 1) is real but smaller than "delete two
  builds" suggests.
- The gate really is hot: `_sub_parent_epilogue_leaf_violation` is the first thing
  `SIMDScheduling.can_fuse` does (`simd.py:2421`) for every pair in every Triton compile,
  and `_sub_parent_epilogue_parent_rnumel` only fast-exits on non-power-of-two `rnumel`.
  The plan already asks for a measurement; keep that requirement.

---

## 6. Item 4.4 — delete the `can_fuse_nested_reduction_append` hook. CONFIRMED, with a caveat.

Verified each rung of `CUDACombinedScheduling.can_fuse_vertical` for `node1 =
FusedNestedReductions` (a `FusedSchedulerNode`, `scheduler.py:3755`, and **not** a
`SchedulerNode` — `SchedulerNode` and `FusedSchedulerNode` are sibling subclasses of
`BaseSchedulerNode`, lines 2936 / 3394):

| Rung | Result | Why |
|---|---|---|
| `cutlass.can_fuse_vertical(n1, n2)` | False | both arms need `is_cutlass_template(n1)` or `is_cutlass_fused_template(n1)`; the latter is `isinstance(FusedSchedulerNode) and is_cutlass_template(...)`, and `is_cutlass_template` needs `SchedulerNode` -> always False |
| `is_cutlass_template(node1)` | False | needs `SchedulerNode` |
| `is_cutedsl_template(node1)` | False | needs `SchedulerNode` |
| `is_nv_universal_gemm_template(node1)` | False | early `if not isinstance(node, SchedulerNode): return False` |
| `is_nv_universal_gemm_fused_template(node1)` | False | passes the `FusedSchedulerNode` test, then `_is_nvgemm_ir_buffer(node.get_template_node())`; `FusedSchedulerNode.get_template_node()` (3586-3590) returns `None` because no snode `is_template()`, and `_is_nvgemm_ir_buffer(None)` is False |

So the ladder falls through to `_triton_scheduling.can_fuse_vertical`. The `xpu` twin is
the same shape.

**Caveat the item should absorb:** two rungs test **node2**, not node1
(`is_cutlass_template(node2)`, `is_cutedsl_template(node2)`). The review's blanket "the
ladder is already a no-op for a `FusedSchedulerNode`" does not cover them. Those two would
newly return `False` for a template node2. That direction is conservative (a lost fusion,
not a wrong one), and it is very likely unreachable anyway since
`FusedNestedReductions.can_fuse_with` requires `_classify_grouped_pointwise_nodes` to
succeed on `other`. Same argument applies to the `ForeachKernelSchedulerNode` check at
`simd.py:2412`, which the hook currently bypasses.

**The horizontal-branch argument is CONFIRMED.** `can_fuse_horizontal = can_fuse`
(`simd.py:2595`), so an early return inside `can_fuse` would also cover horizontal fusion —
but the horizontal branch (`scheduler.py:9316-9319`) is unreachable for `node1 =
FusedNestedReductions`: the only way to reach `scheduler.py:9276` with an FNR node1 is via
`_can_fuse(..., _skip_fused_nested_dispatch=True)` from `_can_fuse_nested_reduction_append`,
and `can_fuse_with` already required `self.node2.get_operation_names() & other.ancestors`
(`3860`), which implies `node1.get_operation_names() & node2.ancestors` is non-empty. Also
confirmed the only backend `can_fuse_vertical`/`can_fuse_horizontal` call sites in
`scheduler.py` are 9291, 9312 and 9319.

Also confirmed: `backend_can_fuse` picks the hook only when
`producer_node=self` is passed, i.e. only when `half_resolution_nodes` is non-empty
(`scheduler.py:3920`).

---

## 7. Item 4.6 — drop `allow_reduced_broadcast`. CONFIRMED literally, materially weakened.

The mechanical claim holds. Both callers turn a `False` return into a raise:

- `_SubParentPointwiseRemapHandler._materialize_sub_parent_load` (`simd.py:2342-2348`):
  `if materialized: return` else `raise AssertionError(...)`.
- `_SubParentSourceLoadMaterializer.load` (`simd.py:2386-2390`): same.

So removing the flag can only convert a raise into a success. **But the plan's framing
("never the reverse") hides the real trade.** The one call site that *never* passes the
flag is `_SubParentSourceLoadMaterializer`, and it only reaches `materialize_...` when
`source_layout is not None` (`simd.py:2376`) — i.e. for a buffer the **planner explicitly
matched as INTERLEAVED or CONTIGUOUS**. If such a value shows up at
`parent_dim == num_groups_str` (reduced resolution), today you get a loud
`"sub-parent planner invariant violated: could not materialize planned parent-tile load"`.
After the change you silently get
`_broadcast_value_to_axis_resolution(...)` — a value broadcast across the child tile,
which is *not* what an INTERLEAVED/CONTIGUOUS source dep means.

That is trading an internal-invariant crash for a potential silent miscompile. Either
(a) keep the flag, or (b) drop it but keep a rejection when `source_layout is not None`, or
(c) drop it and state explicitly in the commit message that this path is now unchecked.

Confirmed the supporting detail: in the standalone path `must_materialize_names` is all
parent-node buffer names (`simd.py:3769-3773`), so the flag is already `True` there.

---

## 8-14. The "provably dead" set — all CONFIRMED.

**2.1 (`_is_axis_tile_shaped`).** Compare `parent_dim` (`simd.py:1792-1802`) with
`_is_axis_tile_shaped` (1804-1816) case by case: `shape is None` -> `None` vs `False`;
`len==2` -> `str(shape[parent_axis])` vs `== block`; `len==1 and passthrough==1` ->
`str(shape[0])` vs `== block`; everything else -> `None` vs `False`. Since `block` is
always a `str`, `_is_axis_tile_shaped(v, b) == (parent_dim(v) == b)` on every input.
The dead branch also confirmed: `materialize_value_at_sub_parent_resolution` computes
`parent_dim` at 2040, returns at 2041 / 2044, and at 2047 the `!= self.parent_block` arm
returns on both paths — so line 2057 `if not self.is_parent_tile_shaped(value)` is reached
only when `parent_dim == parent_block`, i.e. the predicate is unconditionally True.
Confirmed `is_parent_tile_shaped` / `is_child_tile_shaped` each have exactly one caller
(2057 / 2044). Folding 2044 into `parent_dim == self.child_block(factor)` is safe because
2041 already excluded `None`/`"1"`.

**2.4a (`assert source_layout is INTERLEAVED`, `simd.py:2085`).** `SubParentSourceLayout`
has exactly two members (`scheduler.py:593-599`), `source_layout is None` already returned
at 2059-2060, `CONTIGUOUS` is the `if` arm at 2074. Dead.

**2.4b (`parent_half_domain is None` in the PARENT_HALF arm, `scheduler.py:1472`).**
`_classify_grouped_pointwise_nodes` only emits `PARENT_HALF` under
`has_parent_half_domain` (1388, 1424). Both callers of
`_pointwise_domains_are_compatible` (1331 via `_pointwise_nodes_match_nested_domains`,
3871 in `can_fuse_with`) pass the *same* context object used for classification. Dead;
keep an `assert` only if mypy needs it.

**2.4c (`expected_numel: sympy.Expr | None`, `scheduler.py:1457`).** Four arms assign
non-`None` (1459, 1462, 1467, 1475); the fifth raises. Confirmed.

**2.4d (`must_materialize_names: ... | None = None`).** Two defaults (`simd.py:2306`
handler, `3538` `_codegen_sub_parent_pointwise`). `_codegen_sub_parent_pointwise` has
exactly two callers (3317, 3806) and both pass the kwarg; the handler has exactly one
construction site (3540) which always forwards it. Confirmed.

**2.4e (`_nested_index_equivalent_dep_names` fall-through, `scheduler.py:8751-8766`).**
`is_candidate` -> `_is_dependent_reduction_pair(node1, node2)` -> requires
`grouped_node.is_reduction()`, i.e. `node2.is_reduction()` (536-544, 557-565). The next
block returns `None` when `node2.is_reduction()`. So falling out of the `is_candidate`
block always hits `return None`. Making it explicit is behaviour-preserving. Confirmed.

**3.1 (`assert sub_parent_source_layouts is not None`, `simd.py:3316`).** The variable is
annotated `dict[str, SubParentSourceLayout]` and initialized to `{}` at 3280-3282; the
only reassignment (3291) is `node.parent_half_source_layouts`, itself a non-optional
`dict` (`scheduler.py:3838-3840`). Dead. Confirmed.

---

## 15-19. Batch 5. Matrix CONFIRMED; 5.3 refined.

Gates re-read: `_sub_parent_epilogue_rate` (`scheduler.py:816-843`),
the interleaved gate `sub_parent_factor in (cls.PARENT_HALF_FACTOR, 4)` (934), the
contiguous gate `parent_rnumel is not None` (949-950), and the append path passing
`PARENT_HALF_FACTOR` with no `parent_rnumel` (`3950-3958`).

| Layout | Factor | Lanes | Verdict |
|---|---|---|---|
| INTERLEAVED | 2 | 1 | correct — rate branch 1, gate 934 admits 2 |
| INTERLEAVED | 4 | 1 | correct — rate branch 1, gate 934 admits 4 |
| INTERLEAVED | 4 | 3 | correct — rate branch 2 (837-842) |
| CONTIGUOUS | 2,4,8,16 | 1 | correct — factors 8/16 can *only* be contiguous, since 934 caps interleaved at 4 |
| CONTIGUOUS | 4 | 3 | see below |
| append | 2 | 1 | correct — `_parent_half_source_layouts` omits `parent_rnumel`, so the contiguous arm at 949 is unreachable there |

**5.3, refined.** The "reachable" claim survives, but not by the route the doc implies. A
3-lane node cannot itself contribute a CONTIGUOUS source dep: its `node_numel` is
`3*full/4`, and on GPU with the default `loop_ordering_after_fusion=True` deps are
*unnormalized* (`scheduler.py:2976-2988`), so `product(dep.ranges) == node_numel` and the
guard at `scheduler.py:901` (`factor * product(ranges) == full_numel`, i.e. `3*full ==
full`) rejects every one of its reads as a source. So CONTIGUOUS/4/3 is reachable only in a
*mixed* plan: 1-lane contiguous stages fixing the layout at factor 4, plus a 3-lane
consumer of them (whose `output_lanes` must sort last, per 797-799). Contrived but not
blocked by any guard I could find.

The concern is also *stronger* than the doc says: `sub_parent_iteration_values`
(`simd.py:1965-1989`) hardcodes the multi-lane output mapping as
`pair_var * output_lanes + output_lane` — interleaved — and nothing in the planner
validates the epilogue's *output* layout (the rate function only compares numels;
`SIMDKernel.is_compatible` only compares ranges). A `cat`-style rather than `stack`-style
3-lane consumer would be mis-mapped. That affects INTERLEAVED/4/3 too. Recommend making
5.3 "reject `lanes == 3` unless the output layout is validated", not just "add a test".

**5.1** confirmed: branch 1 checks `factor * node_numel == full_numel`; branch 2 checks
`4*group == full and 3*group == node` with `group = FloorDiv(full, 4)`, which is
algebraically `4*node == 3*full` (gcd(3,4)=1 forces divisibility). `parent_rnumel >= 4`
(837) is branch 1's `factor <= min(16, parent_rnumel)` specialized to factor 4. The
`(lanes == 1 or factor == 4)` clause is genuinely needed: a naive `for lanes in (1,3)` loop
would admit (8,3)/(16,3), which the interleaved gate at 934 rejects and only CONTIGUOUS
would pick up.

**5.2** confirmed, with one note: `in (PARENT_HALF_FACTOR, 4)` is a set membership, not a
bound. Replacing it with `factor <= MAX_INTERLEAVED_SUB_PARENT_FACTOR` is equivalent only
because the rate function guarantees the factor is a power of two `>= 2` (825-835) or
literally 4 (842) or literally 2 (3956). It is — so the rename is safe.

**5.4** confirmed: `parent_rnumel: int | None = None` at `scheduler.py:874`, documented
only by the body comment at 877, gating at 949-950; caller 698 passes it, caller 3956
omits it.

---

## 20. Item 4.5(a) — CONFIRMED, one caveat.

`self._can_fuse(...)` has exactly two callers: `_can_fuse_nested_reduction_append`
(`scheduler.py:8922`) and `_can_fuse_impl` (8966). Moving the four dispatches
(`scheduler.py:9002-9012`) up into `_can_fuse_impl` does make the append path skip
re-dispatch, so `_skip_fused_nested_dispatch` disappears.

Caveat: the dispatch currently sits *after* three guards inside `_can_fuse` — `node1 is
node2` (8982), the multi-stream check (8991-8995), and the mempool check (8996-9000).
Hoisting it runs `can_fuse_with` before those. In practice the recursive `_can_fuse` call
from `_can_fuse_nested_reduction_append` re-applies them, but with a possibly different
node1 (`self` vs `self.node2`), so `get_node_stream(FNR)` vs `get_node_stream(node2)` could
in principle differ. Low risk; mention it in the item.

---

## 21-23. "Do not do" spot checks — all three CONFIRMED.

**`MAX_SUB_PARENT_FACTOR = 16` is a tested boundary.** `test_rmsnorm_chunk16_kernel_form`
(`test_nested_reduction.py:2734`, accept) and
`test_producer_consumer_rejects_rmsnorm_chunk32` (1075, reject) both exist. Correct to keep.

**`_sub_parent_epilogue_internal_reads_match` is not redundant.** It checks
*epilogue nodes reading each other's outputs* with an exact
`dep.normalize() == writes[0].normalize()` (`scheduler.py:845-862`);
`_sub_parent_epilogue_source_loads_are_unambiguous` checks *non-epilogue* readers of the
planned source deps (990-1015). Disjoint. The dedicated negative test is
`test_producer_consumer_mxfp6_rejects_shifted_intermediate` (1502-1526), which inserts
`torch.roll(low, 1, -1)` between the `low` stage and the stacked stage — precisely the
normalize mismatch. Keeping it is correct. (The nit about moving the
`isinstance(dep, MemoryDep)` check above the two `dep.name` uses is also correct: 854 and
857 both touch `dep.name` before the isinstance test at 858.)

**`_SubParentSourceLoadMaterializer` stacks over two inner handlers.** Confirmed: 3513
(over `_GroupedReductionOpsHandler`) and 3777 (over the raw `V.get_ops_handler()`).
Folding it in would duplicate it. Correct to keep.

**"Hardcode factor = 2" is wrong at the tip** — corroborated by everything in section 15.

---

## 24-25. Batch 6 — CONFIRMED.

**6.1.** `test/inductor/test_loop_ordering.py` contains zero references to
`nested_reduction`. `_pointwise_reduction_broadcast_orders` (`git log -S`),
`test_square_block_broadcast_vertical_fusion`, the `test_loop_ordering.py` +24, and the
`choices.py` one-liner (`can_fusion_increase_peak_memory(node1, node2)` ->
`(node1, node2, shared_data_score)`) are all introduced by `f020897c032` (the MXFP6
commit). The new parameter is consumed by `Scheduler.can_fusion_increase_peak_memory`
(`scheduler.py:7812-7817`), which now takes the caller's score globally. Split, or at
minimum say so in the commit message — as the plan states.

**6.2.** `546faa04929` (#183432) is an ancestor of `origin/main`, so the deleted 112-line
test was landed code. `_same_index_with_prefix_size` survives (`scheduler.py:9459`, used at
9486 and 9497), as does the `config.loop_ordering_after_fusion and read.num_vars !=
write.num_vars` branch at 9489. `git grep` at the tip finds `fusable_read_and_write` and
`_same_index_with_prefix_size` **only** in `torch/_inductor/scheduler.py` — no test file
references either. Confirmed exactly as written.

---

## 26. Item 2.2, first bullet — UNVERIFIABLE statically.

The claim that the second lane-resolution candidate in `_resolve_remapped_value`
(`simd.py:1544-1553`) is never taken rests on an instrumentation run plus a full-file pass,
which I cannot reproduce here (read-only, shared worktree — running the suite would write
`__pycache__` and inductor caches into the tree). Static reasoning is inconclusive: the
fallback exists precisely because codegen's `index` need not be structurally identical to
the planner's `dep.index`, and both candidates agree only if the constant term survives
whatever rewriting happened in between.

Note the failure mode if the empirical result is wrong: deleting the fallback converts a
working compile into a hard `AssertionError`, not a miscompile. That is acceptable, but the
item should say so, and it pairs naturally with 2.2's third bullet (call the planner's
`_contiguous_sub_parent_epilogue_emitted_lane`, `scheduler.py:1256-1266`), which makes the
first candidate provably the planner's own formula. **Evidence needed:** re-run the
instrumented sweep on `f020897c032` in a private worktree.

---

## 27. Batch 2.5 / 2.6 — CONFIRMED (one nuance).

- The 5-line re-wrap of `codegen_node_schedule_with_kernel` is real and gratuitous
  (`git show 8b054943180` turns `def codegen_node_schedule_with_kernel(self, node_schedule, kernel):`
  into 5 lines with identical parameters).
- The `omitted_nodes` block is 11 lines building a name list eagerly to emit one
  `fusion_log.debug` (`scheduler.py:786-796`), on a path reached from `can_fuse`.
- `min_xblock` carve-out (`simd.py:3740-3743`) is the stated double negative; `all(...)`
  and `not any(...)` agree on the empty dict, so the positive rewrite is safe.
- **Nuance on the stale docstring** (`_score_fusion_memory_by_fusable_read_write`,
  `scheduler.py:9545-9557`): the docstring is *incomplete*, not wrong. `fusable_read_and_write`
  is still called at 9576, so "normalized equivalent read/write deps" remains accurate for
  that arm; what is undocumented is the new `index_equivalent` arm (9571-9575), which
  matches on buffer name only. Reword rather than replace.
