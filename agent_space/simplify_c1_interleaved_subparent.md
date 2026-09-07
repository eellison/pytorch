# Simplification review — `b85cab9930d` "[inductor] Fuse interleaved sub-parent reduction epilogues"

Commit 1 of 4 (+1397/-61 across `simd.py`, `triton.py`, `scheduler.py`, `utils.py`,
`test_nested_reduction.py`). It teaches SIMD scheduling to fuse a pointwise epilogue that
consumes interleaved lane pairs of a parent reduction tile (standalone NVFP4 packing), adds
the `_SubParent*` handlers + `_GroupedReductionLayout` tile-shape predicates, the
`NestedReduction.sub_parent_epilogue_plan` legality pass, and `emit_split_via_reshape`.

I read the full diff, then re-read every cited region in the **working tree at HEAD**
(`bdf385fc2ed`) plus `git diff b85cab9930d..HEAD` for all four files, and grepped every
symbol I call single-use/dead across `torch/`, `test/`, `benchmarks/`. Ideas whose code
commits 2-4 already deleted (e.g. the never-called `_materialize_sub_parent_source_load`
free function, `materialize_all_store_cache_values`) are dropped. All `file:line` refs are
against `git show HEAD:<file>` — note the working tree currently carries an unrelated
debug probe in `_resolve_remapped_value` that shifts `simd.py` by ~+7 lines after 1541.

---

## Ranked ideas

### 1. Delete `_is_axis_tile_shaped` and its two wrappers; one branch is provably dead

**Summary:** `_is_axis_tile_shaped(value, block)` is *exactly* `parent_dim(value) == block`
for every input, and its two 2-line wrappers each have a single caller. One of those
callers (`is_parent_tile_shaped`) sits behind a guard that already proves it True, so the
branch is dead.

**Where:**
- `torch/_inductor/codegen/simd.py:1792-1802` `parent_dim`
- `torch/_inductor/codegen/simd.py:1804-1816` `_is_axis_tile_shaped` (only callers are the
  two wrappers below)
- `torch/_inductor/codegen/simd.py:1818-1819` `is_parent_tile_shaped` (only caller: 2057)
- `torch/_inductor/codegen/simd.py:1821-1822` `is_child_tile_shaped` (only caller: 2044)
- `torch/_inductor/codegen/simd.py:2040-2058` the callsites in
  `materialize_value_at_sub_parent_resolution`

**Equivalence proof:** both read `getattr(value, "shape", None)`. `parent_dim` returns
`None` for shape-less, rank-0 and rank>=3 values and for rank-1 values whose passthrough
tree is non-trivial; otherwise it returns `str(shape[parent_axis])` (rank 2) or
`str(shape[0])` (rank 1, trivial passthrough). `_is_axis_tile_shaped` returns `False` in
exactly the `None` cases and `parent_dim == block` otherwise. Since `block` is always a
`str`, `None != block`, so the two agree on every input.

**Dead branch:** at 2047 the code does `if parent_dim != self.parent_block:` and *every*
path inside that block returns (2049 `return False`, 2056 `return True`). So line 2057
`if not self.is_parent_tile_shaped(value): return False` is only reachable when
`parent_dim == self.parent_block`, i.e. when the predicate is True. `parent_dim` is a pure
function of `value` and nothing mutates `value` in between.

**Concrete change:** delete `_is_axis_tile_shaped`, `is_parent_tile_shaped`,
`is_child_tile_shaped`; fold 2044 into the 2041 condition
(`if parent_dim is None or parent_dim in ("1", self.child_block(factor)):`); delete
2057-2058. `child_block` keeps two callers (the folded condition and 2053), so it stays.

**Est. LOC removed:** ~20

**Risk:** low. Pure deletion backed by a textual equivalence argument; the removed
`is_parent_tile_shaped` call is a no-op. Covered by every `test_standalone_sub_parent_*`
and the kernel-form goldens.

**Verdict:** yes.

---

### 2. Collapse the three `sub_parent_epilogue_plan` computations per `can_fuse` into one, and drop the `check_leaves` flag

**Summary:** `_sub_parent_epilogue_leaf_violation` builds the plan twice (once with leaf
checks, once without) on identical arguments, and its first build re-implements
`_can_fuse_sub_parent_reduction_epilogue` inline. `can_fuse` then builds it a third time.
Making the plan always carry a `leaves_ok: bool` removes the mode parameter and two of the
three builds.

**Where:**
- `torch/_inductor/codegen/simd.py:2608-2618` `_can_fuse_sub_parent_reduction_epilogue`
- `torch/_inductor/codegen/simd.py:2646-2653` — 2646-2650 is a verbatim re-implementation
  of 2615-2618; 2651 is the same call again with `check_leaves=False`
- `torch/_inductor/codegen/simd.py:2589` third build in `can_fuse`
- `torch/_inductor/codegen/simd.py:2696-2711` `_sub_parent_epilogue_plan` (pass-through of
  the flag)
- `torch/_inductor/scheduler.py:641,645-646,711,717,721` the `check_leaves` parameter,
  its docstring, and the three guarded checks

**Concrete change:** add `leaves_ok: bool` to `NestedReduction.SubParentEpiloguePlan`
(`scheduler.py:626-632`); always run the three leaf checks and record the conjunction
instead of returning `None`. Delete the `check_leaves` parameter from
`NestedReduction.sub_parent_epilogue_plan` and `SIMDScheduling._sub_parent_epilogue_plan`.
`_can_fuse_sub_parent_reduction_epilogue` and `_find_sub_parent_epilogue_plan` add
`and plan.leaves_ok`. `_sub_parent_epilogue_leaf_violation` makes one call and replaces
2646-2650 with `if plan.leaves_ok and all(...): return False`.

**Est. LOC removed:** ~12 (plus a compile-time win: `_sub_parent_epilogue_leaf_violation`
runs at the very top of `can_fuse` for every reduction/pointwise pair whose `rnumel` is a
power of two, which is most of them)

**Risk:** medium. Behaviour is preserved only if the three leaf checks are side-effect free
(they are: pure scans over `read_writes`), and `leaves_ok` must be computed on the same
`epilogue_node_set`/`planned_source_deps` the strict path used. Verified those are already
in scope at `scheduler.py:702-726`.

**Verdict:** yes-if (needs the full `test_nested_reduction.py` + `test_inductor_scheduler.py`
sweeps green, since this touches the fusion gate for all Triton compiles).

---

### 3. Hoist the triplicated NVFP4 pack body into a module-level helper

**Summary:** the ~22-line NVFP4 `f(x)` (view -> amax -> scale -> even/odd ->
`inline_asm_elementwise`) is copy-pasted three times by this commit, differing only in
whether `scale` is inlined and whether an extra output is returned. The file already
establishes the right pattern with `MXFP4_RECIP_UE8M0_ASM` and `_rmsnorm_mxfp4`.

**Where:**
- `test/inductor/test_nested_reduction.py:1339-1377` `test_standalone_nvfp4_inline_asm`
- `test/inductor/test_nested_reduction.py:1381-1421`
  `test_standalone_nvfp4_inline_asm_rejects_fullres_reader`
- `test/inductor/test_nested_reduction.py:2246-2279`
  `_capture_standalone_nvfp4_kernel_sources`
- Existing precedent: `MXFP4_RECIP_UE8M0_ASM` at line 30, `_rmsnorm_mxfp4` at 2156-2186
- The `cvt.rn.satfinite.e2m1x2.f32` string literal appears 9 times at HEAD and 0 times in
  the pre-stack file

**Concrete change:** add `NVFP4_PACK_ASM` next to `MXFP4_RECIP_UE8M0_ASM` and a
`_nvfp4_pack(x, B, D, G) -> (packed_uint8, scale_fp8)` helper next to `_rmsnorm_mxfp4`.
The three sites become `packed, scale = _nvfp4_pack(x, B, D, G)` plus their distinguishing
line (the `rejects_fullres_reader` variant additionally needs `even`, so return it or have
the test recompute the `repeat_interleave` from `packed`). Later commits in the stack add
more copies that would collapse too.

**Est. LOC removed:** ~45 in this commit's tests (~80 across the stack)

**Risk:** low. Test-only; the goldens key on emitted Triton, not on Python source shape.
Only care needed is that the three variants really are the same graph — verify by
re-running the two numeric tests and the kernel-form test.

**Verdict:** yes.

---

### 4. Make `assert_standalone_nvfp4_inline_asm_kernel_form` use `assert_single_kernel_form`

**Summary:** this 34-line helper re-implements, statement for statement, the
`assert_single_kernel_form` helper that every sibling kernel-form test uses. It only forked
because `assert_single_kernel_form` hardcodes `self.force_persistent_outer_reduction`.

**Where:**
- `test/inductor/test_nested_reduction.py:2809-2843`
  `assert_standalone_nvfp4_inline_asm_kernel_form` (including a local
  `looped_or_persistent` that shadows `self.looped_or_persistent` at 2368)
- `test/inductor/test_nested_reduction.py:2441-2485` `assert_single_kernel_form`
- The sibling that does it right:
  `test/inductor/test_nested_reduction.py:2875-2890`
  `test_standalone_sub_parent_epilogue_kernel_form`

**Concrete change:** give `assert_single_kernel_form` an optional
`force_persistent_outer_reduction` kwarg (sentinel default = "use `self.`"), then replace
the 34-line helper with an `assert_single_kernel_form(...)` call and let the two tests at
2846/2856 pass `self.force_persistent_outer_reduction` / `None`.

**Est. LOC removed:** ~22

**Risk:** low. `assert_single_kernel_form` already performs the same four checks in the same
order with the same defaults (`num_deallocs=1` and `num_allocs=2` must be passed
explicitly, as the current helper does).

**Verdict:** yes.

---

### 5. Share the "run with nested off, then on" dance instead of hand-rolling it

**Summary:** `TestBase.check_nested_matches_unnested` already encodes
`patch(nested_reduction=False) -> ref -> reset -> act -> assertEqual`, but the two NVFP4
numeric tests re-inline it because they need per-output tolerances (fp8 outputs must be
compared as `.float()`).

**Where:**
- `test/inductor/test_nested_reduction.py:83-90` the existing helper
- `test/inductor/test_nested_reduction.py:1367-1377` and `1413-1421` the two hand-rolled
  copies added by this commit (two more pre-existing copies at 971-981 and 996-1004)

**Concrete change:** extract `run_nested_and_unnested(self, f, args) -> tuple[Any, Any]`
returning `(ref, act)`; have `check_nested_matches_unnested` call it and assert, and have
the four hand-rolled sites call it and keep their custom assertions.

**Est. LOC removed:** ~14 in this commit (~26 across the file)

**Risk:** low. Only nuance is that the hand-rolled sites use `fullgraph=True` and order
`torch._dynamo.reset()` before `metrics.reset()`; give the helper a `fullgraph` kwarg and
keep the helper's ordering (both orderings are equivalent).

**Verdict:** yes.

---

### 6. Merge `emit_split_via_reshape` and `emit_split_via_reshape_permute`

**Summary:** the permute variant (added by commit 3 on top of the function this commit
introduced) is the same six statements plus one `tl.permute` line; both are single-caller.

**Where:**
- `torch/_inductor/codegen/triton.py:6206-6215` `emit_split_via_reshape`
  (caller: `simd.py:2086`)
- `torch/_inductor/codegen/triton.py:6217-6229` `emit_split_via_reshape_permute`
  (caller: `simd.py:2077`)

**Concrete change:** one `emit_split(value, reshape_shape, part_names, *, permute_dims=None)`
that bitcast-reshapes, optionally permutes (recomputing `shape` from `permute_dims`), and
calls `_emit_recursive_split`.

**Est. LOC removed:** ~13

**Risk:** low. Both callers are in `materialize_value_at_sub_parent_resolution`'s two arms;
the emitted text is unchanged. Crosses a commit boundary (touches commit-1 and commit-3
code), so it may be cleaner to land on commit 3.

**Verdict:** yes-if (fine if the stack is being reflowed; skip if commits must stay
independently reviewable).

---

### 7. Rename `flat_index_derived_tree`; drop the `sub_parent_tree()` accessor

**Summary:** the field name says "flat index", the accessor says "sub parent", and the
string `flat_index` appears nowhere else in the codebase. Two names for one concept, plus
a 2-line accessor whose only content is an assert.

**Where:**
- `torch/_inductor/codegen/simd.py:1592` field declaration
- `torch/_inductor/codegen/simd.py:1606-1608` `sub_parent_tree()`
- `torch/_inductor/codegen/simd.py:1962` the only writer (`make_sub_parent_family`)
- Readers: `simd.py:1972`, `simd.py:2062`

**Concrete change:** rename the field to `sub_parent_derived_tree`. Keep the accessor only
if the `Optional` narrowing genuinely helps the type checker at the two callsites; if
`assert family.sub_parent_derived_tree is not None` reads fine inline, delete it.

**Est. LOC removed:** ~3 (the value is naming, not lines)

**Risk:** low. Mechanical rename, four sites.

**Verdict:** yes.

---

### 8. Revert the gratuitous re-wrap of `codegen_node_schedule_with_kernel`

**Summary:** the commit exploded a one-line signature onto five lines without changing it.
CLAUDE.md explicitly asks for the single-line form.

**Where:** `torch/_inductor/codegen/simd.py:4004-4008`. Diff hunk:
```
-    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
+    def codegen_node_schedule_with_kernel(
+        self,
+        node_schedule,
+        kernel,
+    ):
```

**Concrete change:** put it back on one line. (The sibling `_codegen_node_schedule_body`
extraction in the same hunk *is* load-bearing — `_codegen_reduction_with_sub_parent_epilogue`
at `simd.py:3753` calls it inside its own `with kernel:` — keep that.)

**Est. LOC removed:** 4

**Risk:** none.

**Verdict:** yes.

---

### 9. Merge the two AssertionError arms in `_SubParentPointwiseRemapHandler._materialize_sub_parent_load`

**Summary:** two differentiated internal-invariant messages ("could not materialize
required buffer" vs "planner did not provide a usable layout for required buffer") for the
same user-visible outcome; no test distinguishes them.

**Where:** `torch/_inductor/codegen/simd.py:2326-2348`.

**Concrete change:** one `raise AssertionError(f"sub-parent stage could not materialize
required buffer {name!r}")` covering both the missing-`store_cache` and
`materialized is False` cases; the `if materialized: return` / `raise` pair becomes
`if not materialized: raise ...`.

**Est. LOC removed:** ~6

**Risk:** low. These are internal-compiler asserts, not user diagnostics. Grep confirms no
test asserts on either message.

**Verdict:** yes.

---

### 10. Stop computing the sub-parent plan twice per `codegen_node`

**Summary:** `codegen_node` calls `_find_sub_parent_epilogue_plan(nodes)` purely to decide
whether to skip coalesce analysis, then `_codegen_nodes` immediately calls it again and
uses the result.

**Where:**
- `torch/_inductor/codegen/simd.py:3806,3812-3814` (`has_sub_parent_epilogue`)
- `torch/_inductor/codegen/simd.py:3607-3612` (the real use)
- `_codegen_nodes`' other caller is `simd.py:3088`, which passes no coalesce analysis

**Concrete change:** compute the plan once in `codegen_node` and pass it through as an
optional third argument to `_codegen_nodes`, replacing the boolean.

**Est. LOC removed:** ~4 (the point is the removed duplicate scan, not the lines)

**Risk:** low, but honestly this is close to a wash: the alternative (just deleting the
gate and letting `_codegen_nodes` discard an unused `coalesce_analysis`) trades one plan
build for one `_analyze_memory_coalescing` call, which is probably the more expensive of
the two. Only worth doing as the "pass the plan down" variant.

**Verdict:** yes-if (do the thread-the-plan version, not the delete-the-gate version).

---

### 11. `test_standalone_nvfp4_inline_asm_default_kernel_form` runs identically twice

**Summary:** it passes an explicit `None` for `force_persistent_outer_reduction`, so it
ignores the class attribute — and both concrete `_InternalsBase` subclasses
(`NestedReductionInternalsPersistentTest`, `NestedReductionInternalsNonPersistentTest`)
therefore execute byte-identical assertions, including two full compiles of a 128x4096
NVFP4 kernel on SM100+.

**Where:**
- `test/inductor/test_nested_reduction.py:2856-2862` the test
- `test/inductor/test_nested_reduction.py:2916-2923` the two subclasses (only ever
  `True`/`False`, never `None`)

**Concrete change:** move it to a standalone `TestCase` (or drop it — the autotuned-default
path is only meaningfully different from the two forced variants in which config it picks,
and both forced variants are already asserted).

**Est. LOC removed:** ~8, plus one redundant SM100 compile per run

**Risk:** low, test-only.

**Verdict:** yes-if (keep the coverage, just don't run it twice).

---

## Considered and rejected

- **Hardcode `factor = 2` / drop the `factor` parameter.** The previous review (idea #1 in
  `SIMPLIFICATION_IDEAS.MD`) was right for its snapshot, but is now stale: at HEAD
  `_sub_parent_epilogue_rate` (`scheduler.py:816-843`) admits any power-of-two up to
  `MAX_SUB_PARENT_FACTOR = 16`, and `_emit_recursive_split` genuinely recurses. The
  generality is live.
- **Drop the `renames` / `parent_rnumel` optional params on
  `_sub_parent_epilogue_source_deps`.** Both look like dead generality from commit 1 alone,
  but commit 2's producer path (`scheduler.py:3950-3958`
  `FusedNestedReductions._parent_half_source_layouts`) passes `renames=` and deliberately
  omits `parent_rnumel` to disable contiguous matching. Live.
- **Merge `_SubParentSourceLoadMaterializer` into `_SubParentPointwiseRemapHandler`.** They
  run in separate `with kernel:` scopes and source values from different places (live
  `_inner.load` vs `cse.store_cache`). Commit 1 actually shipped a shared free function for
  this (`_materialize_sub_parent_source_load`) that it never called; a later commit deleted
  it. Re-introducing it would be going backwards.
- **Trim the scheduler legality helpers** (`_sub_parent_epilogue_source_deps`,
  `_interleaved_..._read_matches_reduction_read`, `_sub_parent_epilogue_outputs_unread`,
  `_sub_parent_epilogue_source_loads_are_unambiguous`, `_sub_parent_siblings_are_source_free`).
  Each guards a distinct miscompile and each has a dedicated negative test
  (`..._rejects_output_fullres_reader`, `..._rejects_output_reduction_reader`,
  `..._rejects_same_buffer_scalar`, `..._rejects_ambiguous_source_load`). Irreducible.
- **`_sub_parent_epilogue_plan` / `_find_sub_parent_epilogue_plan` as thin wrappers.** Four
  and two callers respectively, and the former carries the two-condition backend/config
  gate. Keep.
- **`TRITON_FLOAT8_DTYPES` (`utils.py:173-179`).** Genuine de-duplication of a 4-tuple that
  was spelled out twice. Keep as-is.
- **`_codegen_node_schedule_body` extraction.** Load-bearing (see idea 8).
- **`_select_lane` / `_resolve_remapped_value`.** Two callers each at HEAD and the lane
  logic is non-trivial; not 1-2 LOC helpers.

---

## Possible correctness concerns (not part of the simplification set)

1. **Rank>=3 values are silently accepted unremapped.**
   `materialize_value_at_sub_parent_resolution` (`simd.py:2040-2043`) treats
   `parent_dim is None` as "already at the right resolution" and stores `value` into
   `family.remapped_values` verbatim. `parent_dim` returns `None` not only for
   shape-less/scalar values but also for any rank-3+ tile and for rank-1 tiles with a
   non-trivial passthrough axis — cases where the value is *not* known to be at child
   resolution. The comment at `simd.py:2067` ("parent_dim() only accepts rank-1/2 tiles")
   acknowledges the restriction but the early return does not reject.

2. **`kernel.num_store` is patched after the fact.**
   `simd.py:3785` does `kernel.num_store = max(kernel.num_store, len(kernel.store_buffer_names))`
   to repair metadata that removed reduction-stage stores left undercounted. Worth checking
   whether the undercount also affects anything else keyed off `num_store` (autotune
   heuristics, `inductor_meta`) rather than only the value repaired here.

3. **`_sub_parent_epilogue_leaf_violation` runs before every other `can_fuse` check.**
   `simd.py:2421` gates on `config.triton.nested_reduction` (default `True`), so every
   reduction/pointwise fusion candidate in every Triton compile now walks
   `sub_parent_epilogue_plan` up to three times (idea 2 reduces this to one). Fast-exits
   only when `rnumel` is not an integral power of two, which is the uncommon case. Worth a
   compile-time measurement on a large benchmark before landing.
