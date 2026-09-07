# Adversarial Simplification Review: commit 4 — "Fuse staged MXFP6 packing epilogues"

Commit under review: `bdf385fc2ed` (== `c3a8446b78e` on branch `stack-fixes`, identical
apart from one test constant). +738/-356 across `scheduler.py` (+582/-...),
`simd.py` (+153/-...), `choices.py` (2 lines), and three test files. It teaches the
sub-parent epilogue planner/codegen the 4-input -> 3-output MXFP6 packing rate, keeps the
internally-produced source chain inside the final reduction loop for looped kernels,
adds a broadcast-driven loop reorder to `_try_reindex_pointwise_for_reduction`, and
**deletes** the pre-existing `allow_index_equivalence` dep-matching machinery
(`_fusable_read_after_index_equivalence` / `_fusable_read_after_broadcast`, ~110 lines)
plus its 112-line unit test.

I read the full diff, then the working-tree state at HEAD of every touched region:
`scheduler.py:536-1200`, `7810-7870`, `8400-8560`, `8730-8890`, `9200-9400`, `9520-9610`;
`simd.py:1900-2060`, `2290-2400`, `2590-2720`, `3250-3300`, `3455-3535`, `3600-3790`;
and the three test files. Every "single use"/"unreachable" claim below was grepped.
Note: the env has no working triton, so no test could be executed; all claims are static.

---

## Answer to (a): how much machinery exists only for lane counts the planner refuses?

**Almost none — the commit message oversells its own dead generality.** Precisely:

* The planner can return `output_lanes` of 1 or 3 only (`scheduler.py:835`, `842`).
  A *single plan mixes both*: the MXFP6 graph yields three 1-lane stages (`low`,
  `middle`, `high`, each `full_numel/4`) and one 3-lane stage (the `stack`,
  `3*full_numel/4`). So `epilogue_node_output_lanes`, the per-candidate 3-tuples
  (`scheduler.py:751-804`), the `* output_lanes` term in the `is_compatible` check
  (`scheduler.py:768-771`), the `groupby` staging loop (`simd.py:3762-3772`), and the
  `output_lanes == 1: continue` skip in `_reindex_sub_parent_consumer`
  (`scheduler.py:8807`) are all **exercised**, not dead. None of them would shrink if you
  hardcoded 3.
* The only literally lane-generic code is two expressions in
  `sub_parent_iteration_values` (`simd.py:1979`, `1984`). They cannot be specialised to
  a constant because both 1 and 3 flow through them; specialising to "1 or 3" would be
  *longer*. Cost of the generality: **~2 lines**.
* The `output_lanes: int = 1, output_lane: int = 0` defaults (`simd.py:1969-1970`) are
  live — the parent-half producer path calls with neither (`simd.py:3262`).

So deleting "lane generality" buys ~2 lines and costs correctness. **The real smell is
the inverse**: the *planner* hardcodes the 4-to-3 rate as a bolted-on second branch
(`scheduler.py:837-842`) expressing the same `node_numel == lanes * full_numel / factor`
relation a second time, in a form (`group_numel`, `parent_rnumel >= 4`) that shares no
code with the first branch. See idea 2. Second-order smell: the sortedness guard at
`scheduler.py:797-799` is an undocumented coupling to `itertools.groupby` 1100 lines away
in `simd.py`. See idea 1.

Related, and genuinely *not* dead: `MAX_SUB_PARENT_FACTOR = 16` — factors 8 and 16 are
covered by `test_rmsnorm_chunk8_kernel_form` / `test_rmsnorm_chunk16_kernel_form` (added
by commit 3) via the CONTIGUOUS layout, and factor 32 has a negative test
(`test_producer_consumer_rejects_rmsnorm_chunk32`).

---

## Ranked ideas

### 1. Store epilogue *stages* in the plan instead of a parallel `lanes` tuple
**Summary:** The plan carries two parallel tuples that codegen must re-zip and re-group
with `itertools.groupby` + `operator.itemgetter`, and the planner enforces a sortedness
precondition for that grouping with no comment saying why. Make the grouping the plan's
representation.

**Where:**
- `torch/_inductor/scheduler.py:627-628` (`epilogue_nodes` + `epilogue_node_output_lanes`)
- `torch/_inductor/scheduler.py:781-804` (build + the bare `if output_lanes != tuple(sorted(output_lanes)): return None`)
- `torch/_inductor/codegen/simd.py:3757-3772` (`zip(..., strict=True)` + `groupby` + `itemgetter`)
- `torch/_inductor/scheduler.py:8804-8808` (second `zip(..., strict=True)` of the same pair)

**Concrete change:** Replace the two fields with
`epilogue_stages: tuple[tuple[int, tuple[SchedulerNode, ...]], ...]` (lanes, nodes) plus a
flattened `epilogue_nodes` property (4 callsites still want the flat set:
`simd.py:2617`, `2648`, `scheduler.py:8784`, `8790`). Do the grouping once in
`_sub_parent_epilogue_candidate_nodes`, and put the reason for the ordering constraint
there in one comment ("a lower lane count is a producer of a higher one; codegen emits
stages in order"). Codegen collapses to
`for output_lanes, stage_nodes in plan.epilogue_stages:`.

**Est. LOC removed:** ~10 (9 in `simd.py`, ~3 in `scheduler.py`, +2 for the property).

**Risk:** low. Pure representation change; the sortedness guard stays, it just moves next
to the thing it enables. `groupby` on a sorted-by-construction sequence and an explicit
group build are equivalent.

**Verdict:** yes.

---

### 2. Unify the two branches of `_sub_parent_epilogue_rate`
**Summary:** The 4-to-3 rate is expressed as a magic second branch that re-derives the
same relation with different arithmetic and a `parent_rnumel >= 4` guard that duplicates
the `factor <= parent_rnumel` bound of branch one.

**Where:** `torch/_inductor/scheduler.py:816-843`.

**Concrete change:** One loop over the accepted lane counts:

```python
for lanes in (1, 3):
    factor_expr = V.graph.sizevars.simplify(FloorDiv(lanes * full_numel, node_numel))
    if not isinstance(factor_expr, (int, sympy.Integer)):
        continue
    factor = int(factor_expr)
    if (
        lanes < factor <= min(cls.MAX_SUB_PARENT_FACTOR, parent_rnumel)
        and is_power_of_2(factor)
        and (lanes == 1 or factor == 4)  # only the MXFP6 4-to-3 rate is supported
        and V.graph.sizevars.statically_known_equals(factor * node_numel, lanes * full_numel)
    ):
        return factor, lanes
return None
```

This keeps acceptance *exactly* where it is today (the `lanes == 1 or factor == 4`
clause is the "required 4-to-3 rate" restriction, now stated in one line rather than
encoded in a branch shape) and makes the invariant `node_numel * factor == lanes *
full_numel` visible.

**Est. LOC removed:** ~6, plus the `group_numel` temporary and the redundant
`parent_rnumel >= 4`.

**Risk:** low-med. Behaviour is identical for lanes=1. For lanes=3 the only reachable
`factor` is 4 either way, but you must keep the `(lanes == 1 or factor == 4)` clause or
you silently start accepting (8,3)/(16,3), which the interleaved matcher at
`scheduler.py:934` would reject and only CONTIGUOUS would pick up — untested territory.

**Verdict:** yes-if (keep the explicit rate restriction).

---

### 3. Split the generic loop-ordering work out of this commit
**Summary:** `_pointwise_reduction_broadcast_orders` (+54), the rollback restructure in
`_try_reindex_pointwise_for_reduction` (+8/-8), the `can_fusion_increase_peak_memory`
signature change (`choices.py` +1/-1, `scheduler.py` +5/-4), and
`test_square_block_broadcast_vertical_fusion` (+24) are a *generic* loop-ordering /
fusion-heuristic improvement. `test_loop_ordering.py` never enables
`triton.nested_reduction` (grepped: zero hits in the file), so this stands on its own and
is reviewable on its own.

**Where:**
- `torch/_inductor/scheduler.py:8446-8467` (rollback move), `8503-8555` (new helper)
- `torch/_inductor/scheduler.py:7812-7817` + `torch/_inductor/choices.py:674`
- `test/inductor/test_loop_ordering.py:724-747`

**Concrete change:** Move all four hunks into a commit below this one in the stack. If
the MXFP6 preshuffled test depends on the broadcast reorder (it plausibly does — its
6-D pointwise reads a broadcast 5-D `scale_exponent`), that commit becomes a
prerequisite, which is exactly what a stack is for. At minimum, the current commit
message must say the peak-memory heuristic now consumes a *possibly relaxed/reordered*
score for **all** fusions, not just nested ones — that is a global behaviour change
buried in a "staged MXFP6 packing" commit.

**Est. LOC removed:** 0 net from the codebase; **-81 from this commit's diff**.

**Risk:** low (mechanical split). The only coupling is ordering within the stack.

**Verdict:** yes.

---

### 4. Reuse `_can_fuse_sub_parent_reduction_epilogue` instead of re-implementing it
**Summary:** The new early-out in `_sub_parent_epilogue_leaf_violation` is a verbatim
re-implementation of a method sitting 30 lines above it.

**Where:** `torch/_inductor/codegen/simd.py:2646-2650` duplicates
`torch/_inductor/codegen/simd.py:2608-2618`.

**Concrete change:**
```python
if self._can_fuse_sub_parent_reduction_epilogue(reduction_node, consumer_node):
    return False
```
The only textual difference is the order of `nodes` (`node1,node2` vs
`reduction_node,consumer_node`); `sub_parent_epilogue_plan`'s boolean result is
order-independent (order only affects the emitted `parent_nodes` ordering, which this
call discards). Keep the local `nodes` list — it is still used at 2674/2678/2683/2688.

**Est. LOC removed:** ~4, plus one fewer copy of the
`plan is not None and all(node in plan.epilogue_nodes ...)` idiom (which appears 4x —
see idea 1's property).

**Risk:** low.

**Verdict:** yes.

---

### 5. Extract the "defer the internal source chain" reorder into a named method
**Summary:** `_codegen_reduction_with_sub_parent_epilogue` is now ~165 lines and opens
with a 36-line topological reshuffle built from five intermediate `OrderedSet`s before
any codegen concept appears. That is a scheduling decision living at codegen altitude.

**Where:** `torch/_inductor/codegen/simd.py:3631-3670`.

**Concrete change:** `parent_nodes = self._defer_internal_source_chain(parent_nodes,
internal_source_names, numel, rnumel)` with the existing comment as its docstring. While
moving it, fold `source_nodes` + `source_ancestors` (3638-3647) into one comprehension
and drop `deferred_node_set` in favour of testing against `deferred_names`:

```python
source_chain = OrderedSet(
    name
    for node in parent_nodes
    if internal_source_names & node.get_buffer_names()
    for name in (*node.ancestors, *node.get_operation_names())
)
```

**Est. LOC removed:** ~7, plus a real altitude win in the caller.

**Risk:** low. Pure motion + two collapsed comprehensions.

**Verdict:** yes.

---

### 6. Delete the `full_resolution_source_deps` / `reduction_deps` alias
**Summary:** The commit renames a local, then immediately re-binds the old name to it, so
the same list has two names for the next 8 lines.

**Where:** `torch/_inductor/scheduler.py:913-929`.

**Concrete change:**
```python
full_res_source_deps = [...]
if not full_res_source_deps:
    if dep.name not in fused_buffer_names and dep.name in V.graph.removed_buffers:
        return None
    continue
if len(full_res_source_deps) != 1:
    return None
source_dep = full_res_source_deps[0]
```
(The current `if dep.name in fused_buffer_names and not ...: continue` followed by
`if not reduction_deps:` is exactly the merged condition above.)

**Est. LOC removed:** ~5.

**Risk:** low. Verified the two `continue` paths and the `removed_buffers` early return
collapse without changing which case wins.

**Verdict:** yes.

---

### 7. Trim the MXFP6 test fixtures: drop `_float_to_mxfp6_e2m3`
**Summary:** 30 lines of bit-exact e2m3 float->fp6 emulation in the test file. Every
test that uses it asserts `nested_reduction=True` output equals `nested_reduction=False`
output of the *same* function, so the numeric fidelity of the quantizer is irrelevant —
only the graph shape (a per-group scale, a `& 0x3F` int producer, then the 4-to-3 pack)
matters, and `_mxfp6_four_to_three_quantize` already demonstrates that with a one-liner.

**Where:** `test/inductor/test_nested_reduction.py:177-206` (definition), sole caller
`test/inductor/test_nested_reduction.py:162-164`.

**Concrete change:** Replace the call with the same value producer the other helper uses:
`values = (blocks / torch.pow(2.0, scale_exponent).unsqueeze(-1)).round().to(torch.int32)`.
`_mxfp6_preshuffled_quantize` keeps its distinguishing feature (the 7-D
permute/reshape preshuffle), which is what the test is actually about.

**Est. LOC removed:** ~32.

**Risk:** low-med. The e2m3 chain contributes extra `where`/`clamp` pointwise ops; if the
fusion decision turns out to depend on them the test was testing the wrong thing anyway.
Needs a test run to confirm (not possible in this env).

**Verdict:** yes-if (tests stay green).

---

### 8. De-duplicate the "dynamic batch, nested-off reference" test harness
**Summary:** `test_dynamic_batch_mxfp6_preshuffled_four_to_three_pack` hand-rolls the
exact loop that `test_dynamic_batch_rmsnorm_chunk_swiglu` (added by commit 3) hand-rolls:
patch config off, compile with `dynamic=True`, collect refs, `metrics.reset()`,
`torch._dynamo.reset()`, recompile, compare. `check_nested_matches_unnested` already
exists for the static case.

**Where:** `test/inductor/test_nested_reduction.py:1482-1500` (new) and `1024-1055`
(existing).

**Concrete change:** Add `check_nested_matches_unnested(self, f, args_list, *, dynamic=False)`
next to the existing helper at `test/inductor/test_nested_reduction.py:83`, accepting a
list of arg tuples; both dynamic tests shrink to 3 lines of body each.

**Est. LOC removed:** ~20 across the two tests (net ~14 after the helper grows).

**Risk:** low.

**Verdict:** yes.

---

### 9. Make the dead fall-through in `_nested_index_equivalent_dep_names` explicit
**Summary:** The commit turned `if not is_candidate(...): return None` into
`if is_candidate(...): ...` with a fall-through, which reads like "if the nested-append
relation fails, try the sub-parent relation". That fall-through is **unreachable**:
`is_candidate` requires `_is_dependent_reduction_pair` (`scheduler.py:540-544`), which
requires `grouped_node.is_reduction()`, and the next block bails on
`node2.is_reduction()` (`scheduler.py:8764`).

**Where:** `torch/_inductor/scheduler.py:8751-8766`.

**Concrete change:** `return None` inside the `is_candidate` branch when
`can_fuse` fails, so the two relations read as mutually exclusive (which they are).

**Est. LOC removed:** ~1; the value is removing a phantom path a reader must disprove.

**Risk:** none.

**Verdict:** yes.

---

### 10. Hoist the `output_lane` loop into `_codegen_sub_parent_pointwise`
**Summary:** For a 3-lane stage the caller calls `_codegen_sub_parent_pointwise` three
times with identical arguments except `sub_parent_source`, re-entering
`sub_parent_family.activate(kernel)` and rebuilding an identical
`_SubParentPointwiseRemapHandler` each time.

**Where:** caller `torch/_inductor/codegen/simd.py:3766-3782`; callee
`torch/_inductor/codegen/simd.py:3492-3520`.

**Concrete change:** Pass `sub_parent_sources: Sequence[_IterationSpace]` and loop inside,
under one `activate` and one handler. The other caller (`simd.py:3281`) passes a
one-element sequence.

**Est. LOC removed:** ~6.

**Risk:** med. The handler is configuration-only (no per-lane mutable state), and
`remapped_values` lives on the family, not the handler — so this should be a no-op — but
it changes the nesting of `activate`, so it needs the golden kernel-form tests
(`test_mxfp6_four_to_three_pack_kernel_form`) to confirm byte-identical output.

**Verdict:** yes-if (golden kernels unchanged).

---

### 11. Drop the dead `| None = None` defaults on `must_materialize_names`
**Summary:** Both construction sites always pass it, so the `or OrderedSet()` fallback is
unreachable and the `Optional` type is a lie.

**Where:** `torch/_inductor/codegen/simd.py:2313` + `2317`, and `3501`. Callers:
`simd.py:3289` and `simd.py:3779` (grepped — those are the only two).

**Concrete change:** Make it a required `OrderedSet[str]` in both signatures.

**Est. LOC removed:** ~2; removes one nullable field from the handler's state.

**Risk:** none.

**Verdict:** yes.

---

### 12. Name the interleaved-factor bound instead of `(cls.PARENT_HALF_FACTOR, 4)`
**Summary:** `PARENT_HALF_FACTOR` is documented at `scheduler.py:585-586` as "the fused
nested-reduction *append* path supports only factor-2 interleaving" — a different
subsystem. Using it here as a synonym for the literal `2`, next to a bare literal `4`,
makes the guard read as if two unrelated concepts are being OR'd.

**Where:** `torch/_inductor/scheduler.py:934`.

**Concrete change:** `if sub_parent_factor <= cls.MAX_INTERLEAVED_SUB_PARENT_FACTOR` with
`MAX_INTERLEAVED_SUB_PARENT_FACTOR = 4` next to `MAX_SUB_PARENT_FACTOR`. (`<= 4` and
`in (2, 4)` are equivalent here: `_sub_parent_epilogue_rate` only ever returns powers of
two `>= 2`.)

**Est. LOC removed:** 0 (naming only).

**Risk:** none.

**Verdict:** yes.

---

### 13. Document the insight in `_reindexed_dep_order` rather than restructuring it
**Summary:** The 20-line interval partition (`scheduler.py:8869-8883`) computes, given
that `write` is contiguous, exactly `sorted(range(len(read.size)),
key=read_strides.__getitem__, reverse=True)` — the write's dims already have strictly
descending strides, so partitioning by write-dim interval and sorting descending within
each partition *is* a global descending sort. The loop's remaining work is validation
(each read dim fits inside one write dim, and each partition's sizes multiply to the
write size).

**Where:** `torch/_inductor/scheduler.py:8839-8888`.

**Concrete change:** Either (a) add one comment line stating "with a contiguous write this
is a descending-stride sort of the read dims; the loop also validates that no read dim
straddles a write dim", or (b) actually compute the sort and validate coverage with a
`while` over the sorted list. I recommend (a) — I tried (b) and it is not shorter.

**Est. LOC removed:** 0-5.

**Risk:** none for (a).

**Verdict:** yes for (a), no for (b).

---

### 14. Refresh the now-stale docstring on `_score_fusion_memory_by_fusable_read_write`
**Where:** `torch/_inductor/scheduler.py:9552-9557`. It says the helper scores
"normalized equivalent read/write deps"; after this commit the `index_equivalent` branch
(`9571-9575`) matches on **buffer name only**, with no index comparison at all. One
sentence.

**Est. LOC removed:** 0. **Risk:** none. **Verdict:** yes.

---

## (b) Test coverage: moved, or lost?

**Correction to the brief:** the deleted 112-line
`test_fusable_read_and_write_broadcast_requires_index_equivalence` was **not** added by
commit 2 of this stack. `git log -S` puts it in `546faa04929` "[inductor] Add scheduler
index equivalence for nested reductions (#183432)", which is **already landed in
`origin/main`** and is an ancestor of the stack base (`b85cab9930d~1`). Commit 2 added
only 27 lines to that file (`test_nested_reduction_parent_half_domain`), which still
exists at HEAD (`test/inductor/test_inductor_scheduler.py:326`). So this commit is
deleting a *landed* test, not walking back its own.

Breaking the deleted test into what it covered:

| What it covered | Status |
|---|---|
| `_fusable_read_after_index_equivalence` and `_fusable_read_after_broadcast` (the `expected_relaxed` column: quotient broadcast, quotient-tail-remains, pure broadcast, dynamic dense, producer broadcast, producer alias) | **Correctly deleted** — this commit deletes both helpers and the `allow_index_equivalence` parameter. Coverage removed with its subject. |
| The `expected_default` column: exact-gapped dep fuses (`_same_index_with_prefix_size` fast path), and 5 negative cases | **Lost.** That code path survives at `scheduler.py:9360` (post-refactor line: `if self._same_index_with_prefix_size(read, write): return True`). |
| The final `loop_ordering_after_fusion=True` block: a gapped read with an extra trailing dim fuses only after `normalize()` | **Lost.** That path survives at `scheduler.py:9481-9487`. |

Grep confirms: after this commit **no test in `test/` references `fusable_read_and_write`
or `_same_index_with_prefix_size` at all.** There is no unit-level coverage of the
surviving dep-matching logic anywhere.

The 24 added lines in `test_loop_ordering.py`
(`test_square_block_broadcast_vertical_fusion:724`) are **not a relocation** — they cover
`_pointwise_reduction_broadcast_orders`, a helper this same commit introduces, in a file
that never enables `triton.nested_reduction`. Likewise the 221 new lines in
`test_nested_reduction.py` are end-to-end coverage of new MXFP6 behaviour.

**Verdict:** ~85% of the deletion is legitimate (subject removed), but two assertions
covering *surviving* behaviour were dropped with no replacement, and no coverage was
relocated. Recommended fix: keep a ~20-line trimmed version of the test with only the
`expected_default` cases and the `loop_ordering_after_fusion` case, dropping the
`allow_index_equivalence` column and the `Mock`/`SizeVarAllocator` cases that only
existed for the deleted helpers. Also: the new
`test_square_block_broadcast_vertical_fusion` is the only test in its neighbourhood
without a docstring — its sibling at `test_loop_ordering.py:690` explains the exact index
pattern under test; this one should too, since nothing in the body says which scheduler
path it pins.

---

## Considered and rejected

* **Fold `_reindex_sub_parent_consumer`'s manual `snapshot_loop_state`/`restore_loop_state`
  into `_LoopStateSnapshot`.** Looks like it would unify three rollback mechanisms, but
  `_LoopStateSnapshot.create` exists for multi-node/`FusedSchedulerNode` groups and this
  path has already narrowed to a single `SchedulerNode` (`scheduler.py:8800`). Net zero
  lines, slightly more indirection.
* **Delete `_sub_parent_epilogue_internal_reads_match`** (`scheduler.py:845-862`, new,
  18 lines) as redundant with `_sub_parent_epilogue_source_loads_are_unambiguous`. Not
  redundant: the latter checks *non*-epilogue readers of source deps; this one checks
  epilogue nodes reading *each other's* outputs, and has a dedicated negative test
  (`test_producer_consumer_mxfp6_rejects_shifted_intermediate`). Keep. (Nit only: the
  `isinstance(dep, MemoryDep)` check at 858 runs after `dep.name` is already used at 855
  and 857 — move it up.)
* **Drop the `shifted=True` parametrization of
  `test_producer_consumer_mxfp6_preshuffled_four_to_three_pack`** as duplicating
  `test_producer_consumer_mxfp6_rejects_shifted_intermediate`. They roll at different
  levels (final packed `subs` axis vs. the `low` intermediate) and reach the rejection
  through different guards. Both cost only 2 lines in the fixture. Keep.
* **Collapse the assertion pair in `_SubParentPointwiseRemapHandler`
  (`simd.py:2335-2337` and `2352-2355`).** The earlier review recommended this for the
  older code; at HEAD there are only two assertions with genuinely different meanings
  ("no cached value at all" vs. "no usable layout"), and `must_materialize_names` is now
  load-bearing (it drives both `required` and `allow_reduced_broadcast` at 2329/2347),
  so the older review's ideas #2/#3 no longer apply.
* **Add a `_repeat_pattern(B, D)` test helper** for the 4 copies of
  `values = torch.tensor([...]); x = values.repeat(B, D // values.numel())`. ~8 lines,
  but CLAUDE.md discourages trivial single-purpose helpers and the pattern is readable
  inline. Marginal.

---

## Possible correctness concerns (not the focus of this review)

1. **`remaining.clear()` is strictly weaker than what it replaced.**
   `scheduler.py:9355-9360`: when a producer write name is in
   `index_equivalent_dep_names`, *every* unmet consumer read of that buffer is declared
   satisfied without inspecting a single index. Previously each read went through
   `_fusable_read_after_index_equivalence`, which required the write to be dense and
   non-broadcasting and the read to be a conservative reshape/broadcast of it. All
   legality now rests on `_nested_index_equivalent_dep_names` having validated the whole
   plan — worth confirming that a consumer with *two* reads of the same producer buffer
   (one legal, one shifted) is rejected by the plan rather than waved through here.
2. **`can_fusion_increase_peak_memory` now consumes a relaxed score, globally.**
   `scheduler.py:7812-7817` + `choices.py:674`: it used to recompute
   `score_fusion_memory(node1, node2)` itself; it now takes the caller's
   `shared_data_score`, which may have been boosted by
   `_score_fusion_memory_by_fusable_read_write` (name-only match) or by
   `shared_data_after_reordering_loop` / `shared_data_after_inverting_indexing`. Since
   the guard is `memory_overhead > 32 * score`, a boosted score makes the peak-memory
   guard *less* likely to fire — for all fusions in the compiler, not just nested ones.
   This is a global heuristic change with no dedicated test.
3. **Looped kernels with an external (non-internal) source.**
   `simd.py:3727-3740`: `epilogue_source_layouts` is filtered down to
   `internal_source_names` when the kernel is not persistent, but `must_materialize_names`
   still contains *all* parent buffer names. In `_materialize_sub_parent_load`
   (`simd.py:2321-2355`) such a name is `required` yet has `source_layout is None`, so it
   must succeed via `allow_reduced_broadcast` or raise. Worth checking that the
   looped + external-source combination is actually reachable (the plan-level guard at
   `scheduler.py:705-709` may already exclude it).
