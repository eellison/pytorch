# Simplification review: `dcd46c19a41` "[inductor] Fuse contiguous sub-parent reduction epilogues"

Commit 3 of 4 (+786/-107: `simd.py` +159/-59, `triton.py` +30/-16, `scheduler.py` +224/-30,
`test_nested_reduction.py` +480). It adds a second `SubParentSourceLayout` (`CONTIGUOUS`,
i.e. `chunk`/`split` consumers) alongside commit 1's `INTERLEAVED` (NVFP4/MXFP4 packing),
generalizes the sub-parent factor from 2 to any power of two <= 16, adds a reshape+permute
split codegen path, and adds chunked SwiGLU / gating tests.

I read `git show dcd46c19a41`, then re-read every cited region in the **stack HEAD blob**
(`c3a8446b78e`, identical to `bdf385fc2ed` apart from one test line) because commit 4
rewrites much of `scheduler.py`. All `file:line` below are verified against stack HEAD.
Note: the shared working tree was checked out to commit 1 by another process mid-review,
so late verification used `git show c3a8446b78e:<file>`; earlier line numbers were captured
while the tree was at stack HEAD and match.

Empirically validated (before the tree moved): idea 4 was confirmed by instrumentation +
a full 304-test run of `test/inductor/test_nested_reduction.py` (300 passed, 4 skipped).

---

## Ranked ideas

### 1. Index the contiguous lane directly instead of searching all lanes and cross-checking

**Summary:** The contiguous matcher builds an expected index for *every* lane, finds the
unique lane whose expected index matches, and then asserts that lane equals the lane codegen
will actually emit. The second condition alone is sufficient, so the whole search and its
per-lane plumbing is redundant. Notably the interleaved matcher already does it the simple
way (compute the lane, verify one substitution).

**Where (stack HEAD):**
- `torch/_inductor/scheduler.py:1094-1111` `_matches_unique_sub_parent_lane` (the search).
- `torch/_inductor/scheduler.py:1068-1092` `_sub_parent_lane_substitutions` — commit 3
  widened this from one substitution dict to `list[dict]` keyed by lane *only* to feed the
  search; the interleaved caller at `1132-1134` passes a 1-element list and immediately
  indexes `[0]`.
- `torch/_inductor/scheduler.py:1207-1221` (`lane_exprs` / `expected_by_lane` in
  `_contiguous_sub_parent_epilogue_read_matches_reduction_read`).
- `torch/_inductor/scheduler.py:1246-1253` (same pattern in the flat variant).
- The lane that actually matters: `scheduler.py:1256-1266`
  `_contiguous_sub_parent_epilogue_emitted_lane`.

**Why uniqueness is redundant:** codegen (`simd.py:1539-1556`) picks `parts[emitted_lane]`
and nothing else. If `dep.index == expected_by_lane[emitted_lane]`, the read is satisfied by
that part regardless of whether some other lane happens to alias the same address. If the
emitted lane is not a constant in `[0, factor)`, `_select_lane` returns `None` today and the
planner must reject — same outcome either way.

**Concrete change:** turn `_matches_unique_sub_parent_lane` into
`_contiguous_sub_parent_lane(dep, factor, parent_rnumel) -> int | None` that returns the
constant emitted lane (or `None`). Both contiguous matchers then compute that lane first and
build exactly one `lane_expr` / one substitution dict, and
`_sub_parent_lane_substitutions` reverts to returning a single `dict`.

**Est. LOC removed:** ~35-45.

**Risk:** medium. Pure legality-check restructuring, no codegen change, and the argument
above is a strict-equivalence argument — but it is a legality check, so it wants the full
`test_nested_reduction.py` run plus the mxfp4/nvfp4 kernel-form tests. (I had to abandon
in-tree validation when the worktree was re-checked-out.)

**Verdict:** yes.

---

### 2. Unify the interleaved and contiguous read matchers behind the layout enum

**Summary:** After idea 1, `_interleaved_..._read_matches_reduction_read` and
`_contiguous_..._read_matches_reduction_read` differ only in (a) the lane expression built
for the lane dim and (b) how the lane is derived from `dep.index`. Everything else — rank
check, flat fallback, `_unique_trailing_sub_parent_dim`, substitution, final
`statically_known_equals` — is copied.

**Where (stack HEAD):**
- `scheduler.py:1113-1138` interleaved (same-rank) vs `scheduler.py:1183-1222` contiguous
  (same-rank).
- `scheduler.py:1140-1181` interleaved flat vs `scheduler.py:1224-1254` contiguous flat
  (the interleaved flat variant is commit 4's; both delinearize `dep` into a parent index
  and substitute into `reduction_dep.index`).
- Both are dispatched from the same `if/elif` in
  `_sub_parent_epilogue_source_deps` (`scheduler.py:933-963`), where the interleaved arm
  additionally precomputes `lane = Mod(dep.index, factor)` and gates on
  `sub_parent_factor in (PARENT_HALF_FACTOR, 4)` — i.e. its own bespoke version of "the lane
  must be a constant in range".

**Concrete change:** one
`_sub_parent_epilogue_read_matches_reduction_read(dep, reduction_dep, factor, parent_rnumel, layout)`
parameterized by two tiny layout-dependent pieces:
`lane_expr = factor*var + lane` vs `var + lane*size`, and
`lane = Mod(offset, factor)` vs `FloorDiv(Mod(offset, parent_rnumel), child_extent)`. The
caller's `lane`/`Mod` precomputation and the `factor in (2, 4)` gate at `scheduler.py:930-938`
then fold into the shared helper. Same for the flat pair.

**Est. LOC removed:** ~45-60 (two of the four matchers plus the caller's lane preamble).

**Risk:** medium-high. The interleaved path derives the lane from the *full* index
(`Mod(index, factor)`, letting sympy cancel the symbolic multiple of `factor`) while the
contiguous path zeroes free symbols first; unifying on the constant-offset form is *less*
conservative for interleaved and must not be done blindly. The safe unification keeps
the lane derivation per-layout and shares only the matcher body.

**Verdict:** yes-if — do it after idea 1, and keep the per-layout lane derivation.

---

### 3. Collapse the duplicated rmsnorm-chunk model in the tests

**Summary:** The 480 test lines are dominated by one copy-pasted model. The same 4-line
RMSNorm body, the same 3-line bf16 input triple, and a function-local
`import torch.nn.functional as F` are repeated in 9 test methods plus the capture helper.

**Where (stack HEAD, `test/inductor/test_nested_reduction.py`):**
- Identical model+inputs at `1006`, `1024`, `1057`, `1075`, `1095`, `1114`, `1133`, `1164`,
  `1219`, and again inside `_capture_rmsnorm_chunk_kernel_sources` at `2208`.
- `grep -c "variance = h.pow(2).mean(dim=-1, keepdim=True)"` -> 11 at HEAD;
  `grep -c "import torch.nn.functional as F"` -> 36 (commit 3 added 11 of them).

**Concrete change:**
1. Module-level `_rmsnorm_chunk(x, residual, weight, chunks)` and
   `_rmsnorm_chunk_inputs(B, D)` helpers; reuse them in
   `_capture_rmsnorm_chunk_kernel_sources` too. Hoist `import torch.nn.functional as F` to
   module scope.
2. `@parametrize("chunks, D", [(2, 1024), (4, 1024), (2, 8192)])` merges
   `test_producer_consumer_rmsnorm_chunk_swiglu` / `..._chunk4_gating` /
   `..._chunk_large_rnumel` (`1006`, `1057`, `1219`).
3. `@parametrize("D, chunks", [(1024, 32), (1000, 2), (960, 3)])` merges the three
   reject tests at `1075`, `1095`, `1114`, which are byte-identical apart from shape.
4. Use the existing `_check_rejected` (`1753`) everywhere instead of the
   `check_nested_matches_unnested` + `check_no_fusion` + `assertGreater` triple — commit 3
   uses both idioms for the same intent (`1185`/`1204` vs `1075`/`1095`/`1114`/`1164`).

**Est. LOC removed:** ~130-160 of the 480 added, with no loss of coverage.

**Risk:** low. Test-only; `check_nested_matches_unnested` and `_check_rejected` differ in
whether the reference is eager or unfused-compiled, so pick per test rather than blanket-swap.

**Verdict:** yes.

---

### 4. Delete the never-taken second lane-resolution candidate in `_resolve_remapped_value`

**Summary:** The contiguous load path tries the constant-offset lane, then falls back to
`kernel.simplify_indexing(index)`. The fallback is dead, and it is the only reason the
function takes a `kernel`.

**Where:** `torch/_inductor/codegen/simd.py:1539-1560`; the `kernel` parameter at `1537`;
the sole callsite `simd.py:2282` (`_PointwiseRemapHandler.load`).

**Verification:** I instrumented the second arm to log on hit and ran the 74
chunk/contiguous/sub_parent/mxfp6 tests -> zero hits. I then removed the arm entirely and
ran the whole file: `300 passed, 4 skipped`.

**Concrete change:** drop the two-element `for candidate, simplify in (...)` loop, compute
`lane` once from `offset`, drop the `kernel` parameter and the `self._kernel` argument at
the callsite. Also rewrite the comment: "fall back to symbolic simplification for equivalent
forms that keep the offset in the index" documents the branch being deleted.

**Est. LOC removed:** ~12.

**Risk:** low. Verified by full-suite run.

**Verdict:** yes.

---

### 5. Share the contiguous lane formula between the planner and codegen

**Summary:** The lane-selection rule is written twice, in two modules, with no link between
them. The planner's `_matches_unique_sub_parent_lane` exists *solely* to check that the
planner agrees with the codegen copy; if they drift, the only signal is an internal
`AssertionError` at codegen time.

**Where:**
- `scheduler.py:1256-1266` `_contiguous_sub_parent_epilogue_emitted_lane`:
  `FloorDiv(Mod(sympy_subs(dep.index, free_symbols->0), parent_rnumel), child_extent)`.
- `simd.py:1543-1553`: `FloorDiv(Mod(sympy_subs(index, free_symbols->0), factor*child_extent), child_extent)`
  — the same expression, since `factor * child_extent == parent_rnumel`.

**Concrete change:** `simd.py` already does `scheduler.NestedReduction.SubParentSourceLayout`
lookups, so have `_resolve_remapped_value` call the planner's helper (renamed to something
non-private-ish, e.g. `NestedReduction.contiguous_sub_parent_lane(index, parent_rnumel, child_extent)`)
instead of re-deriving it. Store `parent_extent` rather than `child_extent` on the remapped
value so the `factor * child_extent` reconstruction disappears too.

**Est. LOC removed:** ~8, plus one silent cross-module invariant becomes a call edge.

**Risk:** low. Same expression on both sides; combine with idea 4 in one edit.

**Verdict:** yes.

---

### 6. Merge `emit_split_via_reshape_permute` into `emit_split_via_reshape`

**Summary:** Two `TritonKernel` methods that differ by three lines, one caller each.

**Where:** `torch/_inductor/codegen/triton.py:6206-6215` and `6217-6229`. Callers:
`simd.py:2077` (permute) and `simd.py:2086` (no permute) — the two arms of the same
`if source_layout is CONTIGUOUS` branch in `materialize_value_at_sub_parent_resolution`.

**Concrete change:** `emit_split_via_reshape(value, reshape_shape, part_names, *, permute_dims=None)`;
when `permute_dims` is set, wrap the reshaped expression in `tl.permute` and reorder
`reshape_shape` before calling `_emit_recursive_split`. (Commit 3's `_bitcast_reshape_expr`
/ `_emit_recursive_split` extraction already did the hard part — this is the leftover.)

**Est. LOC removed:** ~10.

**Risk:** low. Purely mechanical; `_emit_recursive_split` is unchanged.

**Verdict:** yes.

---

### 7. Drop the two `*_default_kernel_form_large_d` tests (or fold them into `assert_single_kernel_form`)

**Summary:** `assert_default_rmsnorm_chunk_kernel_form` is a 46-line re-implementation of
`assert_single_kernel_form` whose only difference is that it picks the expected numbers by
sniffing the emitted kernel name. Its two callers pass 7 keyword arguments of magic numbers
each and both skip half the time.

**Where:** `test_nested_reduction.py:2737-2781` (helper), `2783-2794` and `2796-2807`
(callers, both `skipTest` when `force_persistent_outer_reduction is False`); compare
`assert_single_kernel_form` at `2441-2485`.

**Coverage overlap:** `test_rmsnorm_chunk_swiglu_kernel_form` (`2666`) and
`test_rmsnorm_chunk4_gating_kernel_form` (`2689`) already pin *both* the persistent and the
looped kernel shape for the same models; `test_producer_consumer_rmsnorm_chunk_large_rnumel`
(`1219`) already covers D=8192 numerics + `check_fusion()`. What is uniquely asserted is
"at D=8192 the default heuristic lands on one of those two shapes".

**Concrete change:** either delete both tests plus the helper, or give
`assert_single_kernel_form` a `force_persistent_outer_reduction=None` mode that resolves
`looped_or_persistent` from `_kernel_name(kernel_code)` — which would also let the
pre-existing `assert_standalone_nvfp4_inline_asm_kernel_form` (`2809`) collapse.

**Est. LOC removed:** ~70 (delete) or ~55 (fold, and more if the nvfp4 twin follows).

**Risk:** low-medium. Deleting loses the "heuristic picks a nested kernel at D=8192" signal;
folding keeps it. Prefer folding if the nvfp4 twin is also cleaned up.

**Verdict:** yes-if (fold, not delete, if the nvfp4 helper is in scope).

---

### 8. Store the planned source *deps* on `SubParentEpiloguePlan`, not just names

**Summary:** `SubParentEpiloguePlan.source_layouts` is `tuple[(name, layout)]`, so
`_sub_parent_epilogue_leaf_violation` has to reconstruct the `MemoryDep`s the planner already
selected by re-walking `plan.parent_nodes`.

**Where:** plan field `scheduler.py:632`; the reconstruction `simd.py:2659-2672` (14 lines);
the planner's own copy of the same list, `scheduler.py:702`
(`planned_source_deps = tuple(dep for dep, _layout in source_deps)`).

**Concrete change:** make the plan carry `tuple[(MemoryDep, layout)]` and derive names where
needed; `simd.py` becomes
`planned_source_deps = tuple(dep.rename(renames) for dep, _ in plan.source_deps)`.

**Est. LOC removed:** ~12.

**Risk:** medium. The reconstruction is a *superset* of the planner's per-name selection (it
keeps every full-resolution MemoryDep in `parent_nodes` with a matching name, whereas the
planner dedups to one per name), so `_sub_parent_epilogue_source_loads_are_unambiguous` is
currently more permissive on the simd side. If that is deliberate, keep the reconstruction
and add a one-line comment saying so; if it is accidental, this is also a behaviour fix.

**Verdict:** yes-if — only after deciding whether the superset is intentional.

---

### 9. Collapse the `omitted_nodes` debug block

**Summary:** 11 lines to emit one debug log line, in a function called from `can_fuse`
(i.e. hot on compile time), building the name list eagerly.

**Where:** `scheduler.py:786-796`.

**Concrete change:**
```python
if any(factor != sub_parent_factor for _n, factor, _l in candidates):
    fusion_log.debug("sub-parent factor %s omits candidates %s from this plan", sub_parent_factor,
                     [n.get_name() for n, factor, _l in candidates if factor != sub_parent_factor])
```
or just delete it — the surrounding code has no comparable per-rejection logging.

**Est. LOC removed:** ~7 (collapse) / ~11 (delete).

**Risk:** low.

**Verdict:** yes.

---

### 10. Drop the unreachable `INTERLEAVED` assert and de-verbose the invariant messages

**Summary:** Over-defensive assertions added by this commit.

**Where:**
- `simd.py:2085` `assert source_layout is source_layout_kind.INTERLEAVED  # noqa: S101`.
  Unreachable: `SubParentSourceLayout` has exactly two members (`scheduler.py:593-599`),
  `source_layout is None` already returned at `simd.py:2059-2060`, and `CONTIGUOUS` is the
  `if` arm at `2074`.
- `simd.py:1557-1560` and `1567-1570`: commit 3 prefixed both `AssertionError`s with
  `"sub-parent planner invariant violated: "`, splitting each message across two lines.
  Every `AssertionError` here is an internal invariant; the prefix restates the obvious and
  costs the one-line form.

**Concrete change:** delete the assert (or replace the `if/else` with `if CONTIGUOUS: ... else: ...`
unchanged, which is what it already is); drop the prefixes and keep each message on one line.

**Est. LOC removed:** ~5.

**Risk:** low.

**Verdict:** yes.

---

### 11. Fold the two remapped-value representations into one dataclass

**Summary:** `RemappedRangeValue` is now a 3-arm union where a bare `tuple[CSEVariable, ...]`
means "interleaved split" and a dataclass means "contiguous split" — two encodings of the
same concept, distinguished by `isinstance(..., tuple)`.

**Where:** `simd.py:1512-1520` (dataclass + union), `1533-1570` (`_resolve_remapped_value`
branches on the type), producers at `simd.py:2080` and `2089`.

**Concrete change:** one frozen dataclass
`_SubParentSplitValue(parts, layout, parent_extent)`; `RemappedRangeValue = CSEVariable | _SubParentSplitValue`;
`_resolve_remapped_value` switches on `value.layout` for the lane formula. The `tuple`
sentinel disappears and the union stops encoding layout in the Python type.

**Est. LOC removed:** ~5 net, but removes a type-as-tag idiom that CLAUDE.md's "explicit
state" rule points away from.

**Risk:** low.

**Verdict:** yes-if — bundle with ideas 4/5, which touch the same function.

---

### 12. State the `min_xblock` layout carve-out positively and say why

**Summary:** A double negative with an unexplained layout exception.

**Where:** `simd.py:3707-3716`.

```python
if not V.graph.sizevars.statically_known_equals(numel, 1) and not any(
    layout is scheduler.NestedReduction.SubParentSourceLayout.CONTIGUOUS
    for layout in sub_parent_source_layouts.values()
):
```

The comment underneath explains the floor ("amortize the parent-tile load ... for the
standalone packing pattern") but says nothing about why contiguous plans opt out.

**Concrete change:**
```python
interleaved_only = all(
    layout is scheduler.NestedReduction.SubParentSourceLayout.INTERLEAVED
    for layout in sub_parent_source_layouts.values()
)
if interleaved_only and not V.graph.sizevars.statically_known_equals(numel, 1):
```
plus one sentence on why chunk-style consumers do not want the XBLOCK floor.

**Est. LOC removed:** 0 (net +1 comment line).

**Risk:** low. `all(...)` and `not any(...)` agree on the empty dict.

**Verdict:** yes.

---

### 13. Replace the `parent_rnumel: int | None` sentinel used as a feature toggle

**Summary:** `_sub_parent_epilogue_source_deps` takes `parent_rnumel: int | None = None`
where `None` silently means "do not consider the contiguous layout" — documented only by a
comment on the first line of the body.

**Where:** `scheduler.py:874` (param), `scheduler.py:877` (the comment),
`scheduler.py:949-956` (the guard). Callers: `scheduler.py:698` (passes it) and
`scheduler.py:3950-3958` (`FusedNestedReductions._parent_half_source_layouts`, omits it).

**Concrete change:** either pass `parent_rnumel` from the append path too and gate on an
explicit `allow_contiguous: bool`, or rename to make the meaning local
(`contiguous_parent_rnumel: int | None`) and move the comment onto the parameter.

**Est. LOC removed:** 0; removes an implicit-state smell.

**Risk:** low (rename only).

**Verdict:** yes-if — the rename is free; actually enabling contiguous on the append path
is a behaviour change and out of scope.

---

## Considered and rejected

- **`MAX_SUB_PARENT_FACTOR = 16` (`scheduler.py:587`) as dead generality.** Not dead: it is
  the boundary exercised by `test_rmsnorm_chunk16_kernel_form` (accept) and
  `test_producer_consumer_rejects_rmsnorm_chunk32` (reject). Keep.
- **`reject_group_mismatch` flag on `_sub_parent_siblings_are_source_free`
  (`scheduler.py:1024-1038`).** A bool that changes semantics is a smell, but it removed a
  9-line duplicated loop from `simd.py` and there are only two callers with genuinely
  different needs. Net win as written; leave it.
- **Unifying `check_leaves=False` + the manual re-checks in
  `_sub_parent_epilogue_leaf_violation` (`simd.py:2651-2694`).** Tempting — the same three
  leaf checks run in both places — but the simd copy passes `mutation_renames`, a different
  source-name set (all reduction reads, not planned sources), and `reject_group_mismatch=True`.
  Threading three more parameters through the plan API to share ~20 lines would make the
  planner harder to read, not easier. See idea 8 for the one piece worth extracting.
- **Hardcoding the sub-parent factor.** The older review (SIMPLIFICATION_IDEAS.MD idea 1)
  argued `factor` was always 2. Commit 3 made that false: factors 4/8/16 are all exercised.
  Settled in the opposite direction; do not re-litigate.
- **`_select_lane` (`simd.py:1523-1530`) as a trivial helper.** Two callers within
  `_resolve_remapped_value`; keep (and it survives idea 11).
- **`test_reduced_epilogue_uses_generic_fusion` (`test:1152`) as off-topic.** It looks
  unrelated to chunking, but it is the regression guard for commit 3 generalizing
  `_sub_parent_epilogue_rate` beyond factor 2: a reduced-resolution epilogue now looks like a
  factor-8 sub-parent candidate at D=8. Keep.
- **`test_rmsnorm_chunk8_kernel_form` as redundant with chunk16.** Both are 2 lines thanks to
  the shared `assert_rmsnorm_chunk_many_kernel_form`, and factor 8 vs 16 exercise different
  `_emit_recursive_split` recursion depths. Keep.

---

## Possible correctness concerns

(Out of scope for this review; a separate pass covers correctness. Flagging only what I
tripped over.)

1. `simd.py:2659-2672` reconstructs `planned_source_deps` as a superset of the deps the
   planner actually selected (see idea 8), making
   `_sub_parent_epilogue_source_loads_are_unambiguous` weaker on the fusion-veto path than on
   the planning path. Possibly intentional, possibly not.
2. `_matches_unique_sub_parent_lane` (`scheduler.py:1094`) is the only thing keeping the
   planner's lane choice in sync with codegen's independently-written formula at
   `simd.py:1543-1553`. Any future change to one side fails as an internal `AssertionError`
   at codegen time rather than as a fusion rejection.
