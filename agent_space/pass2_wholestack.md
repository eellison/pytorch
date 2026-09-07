# Sub-parent epilogue stack: whole-stack simplification pass

Second pass, reading the four commits as one squashed change instead of one agent per commit.

Read first (findings in these are treated as worth zero here):
`STACK_SIMPLIFICATION_RECOMMENDATIONS.md`, `STACK_SIMPLIFICATION_REVIEW.md`.

Then read: `git diff 56de3c7e232 f020897c032 -- torch/ test/` in full, and the final state of
`torch/_inductor/codegen/simd.py` (1496-2400, 2394-2760, 3160-3560, 3600-3860),
`torch/_inductor/scheduler.py` (514-1660, 3790-3995, 8735-8930, 9200-9610),
`torch/_inductor/codegen/triton.py` (6126-6272, 7820-7860).

All line numbers below are against blob `f020897c032` (`git show f020897c032:<path>`), verified
by re-reading the blob, not the worktree.

Note on the parent's brief: `_reads_planned_sub_parent_output` does not exist at
`f020897c032`. The uncommitted change described in the earlier review landed in commit 4's
latest amend in a different shape -- as `_is_sub_parent_shaped` plus a new `plan is None`
arm in `_sub_parent_epilogue_leaf_violation` (`git diff bdf385fc2ed f020897c032`). That is
what item 6 below reviews.

The stack is already well covered. Most of what a whole-stack read adds is about the two
*ends* of the pipeline -- the enablement gate and the number of times the plan is built --
which no per-commit reviewer could see because those sites live in different commits and
different files. Seven new items, plus one observation I am recommending against.

---

### 1. Give the sub-parent feature one enablement predicate

**Summary.** There are three different spellings of "is sub-parent fusion on", added in three
different commits, and they disagree on exactly the checks that matter. The scheduler-side one
is the weakest and runs the whole planner on every reduction-to-pointwise pair on *every*
backend, including CPU and cpp-wrapper builds.

**Where.**
- `scheduler.py:546-554` -- `NestedReduction._is_enabled_for`:
  `config.triton.nested_reduction and not V.graph.cpp_wrapper and _is_gpu_triton_backend(...)`.
- `simd.py:2737-2741` -- `SIMDScheduling._sub_parent_epilogue_plan`:
  `supports_sub_parent_epilogue and config.triton.nested_reduction` (no cpp-wrapper check).
- `scheduler.py:8761-8766` -- `Scheduler._nested_index_equivalent_dep_names`:
  `config.triton.nested_reduction` and a numel-shape test only. No backend check, no
  cpp-wrapper check. `Scheduler._can_fuse` calls this at `scheduler.py:9222-9225` for every
  fusion pair, before any backend is consulted.

So on a CPU graph, or a Halide/Cutlass CUDA backend, or any cpp-wrapper compile, every
`(reduction, pointwise)` candidate pair still runs `NestedReduction.sub_parent_epilogue_plan`
end to end (it only fast-exits at `scheduler.py:807-814` when `rnumel` is not a static power of
two -- which is the common case for the *shapes this feature targets*, so the exit does not fire
where it matters).

**Concrete change.** Replace the `scheduler.py:8761-8766` condition with
`NestedReduction._is_enabled_for(node1, node2)` plus the two `is_reduction` shape tests, and
make `simd.py:2737-2741` call the same helper. If `_is_gpu_triton_backend` is judged too strong
for `CUDACombinedScheduling`, at minimum add `not V.graph.cpp_wrapper` and hoist all three into
one classmethod so they cannot drift again.

**Est. LOC removed.** ~4, but the point is compile time and a gate that stops having three
different meanings.

**Risk.** Low-medium. Adding `_is_gpu_triton_backend` could disable sub-parent fusion under a
non-triton `cuda_backend`; run the full sweep. Adding `not V.graph.cpp_wrapper` is a pure
narrowing and is what the feature's own TODO at `scheduler.py:1548-1549` already asks for.

**Verdict.** yes

---

### 2. Count the plan builds across both modules, not just `simd.py`

**Summary.** Recommendation item 4.3 counts three builds inside `simd.py` plus two at codegen.
The real number for one `Scheduler.can_fuse(node1, node2)` is up to eight, because the scheduler
independently builds the same plan on the same combined node list before the backend is ever
called, and because the pre-existing arg-swap recursion re-runs the whole leaf check a second
time.

**Where.**
- `scheduler.py:8768` -- build #1, `sub_parent_epilogue_plan([*node1.get_nodes(), *node2.get_nodes()], numel, rnumel)`.
- `scheduler.py:8772` and `scheduler.py:8780` -- builds #2 and #3 on the reindex retry path.
- `simd.py:2668` and `simd.py:2673` -- builds #4 and #5 inside `_sub_parent_epilogue_leaf_violation`.
- `simd.py:2615` -- build #6, inside `_can_fuse_sub_parent_reduction_epilogue`.
- `simd.py:2592` -- `return self.can_fuse_horizontal(node2, node1)`, and
  `can_fuse_horizontal = can_fuse` (`simd.py:2595`), so control re-enters `can_fuse` and runs
  `_sub_parent_epilogue_leaf_violation(node2, node1)` at `simd.py:2421` again: builds #7 and #8
  over the same combined node set.

Two sub-points that item 4.3 does not make:

(a) `scheduler.py:8767-8792` and `simd.py:2613-2618` compute the *same predicate* -- build the
plan over `node1.get_nodes() + node2.get_nodes()` at the reduction's `(numel, rnumel)`, then
require every node of `node2` to be in `plan.epilogue_nodes`. One is in the scheduler and feeds
scoring plus relaxed dep matching; the other is in the backend and is the actual gate. Whatever
"build the plan once per pair" ends up looking like, it has to span both files, otherwise
half the duplication survives.

(b) The docstring of `_sub_parent_epilogue_leaf_violation` (`simd.py:2653-2655`) explicitly
says it "checks both arg orders via the combined node set". If that is true, the re-entry at
`simd.py:2592` is guaranteed to reach the same answer, so builds #7/#8 are pure waste and the
recursion should skip the check (e.g. an `_already_checked` flag, or hoisting the check out of
`can_fuse` into the two real callers).

**Concrete change.** Fold this into item 4.3's scope: one plan per `(node1, node2)` pair
threaded from `Scheduler._can_fuse` through to the backend gate, and suppress the leaf check on
the swapped re-entry.

**Est. LOC removed.** ~10, on top of item 4.3's ~16. The real payoff is 8 planner runs to 1.

**Risk.** Medium -- same risk profile as 4.3. Measure a large benchmark; this is the item most
likely to show up in a compile-time profile.

**Verdict.** yes-if (do it as part of 4.3, not separately)

---

### 3. Stop declaring the reduced-output family's constants on the sub-parent tree

**Summary.** Commit 1 gave the sub-parent derived tree the *grouped reduction's* two named
constants. Commit 2 then had to add a per-kernel dedup map to `TritonKernel` because two
families now emit identical constant lines. Removing the copy removes the reason for the map,
and also removes two dead `tl.constexpr` lines from every standalone sub-parent kernel.

**Where.**
- `simd.py:1780-1790` -- `_grouped_axis_named_constants`, returning
  `(nested_<AXIS>_LOCAL_REDUCTION_SIZE, ...)` and `(nested_<AXIS>_REDUCED_BLOCK, ...)`.
- `simd.py:1883-1891` -- `make_reduced_output_family`: the reduced tree's `block_size` *is*
  `reduced_block_sym` and its `block_offset` uses `local_reduction_size_sym`, so it genuinely
  needs both constants defined.
- `simd.py:1949-1963` -- `make_sub_parent_family`: `block_size=FloorDiv(group_tree.block_size(), factor)`
  and `block_offset=FloorDiv(group_tree.block_offset(), factor)`. It references neither symbol,
  yet passes `named_constants=self._grouped_axis_named_constants(self.group_tree)` at
  `simd.py:1958`.
- `triton.py:3297-3299` + `triton.py:7823-7838` -- `_named_constants` and
  `_codegen_named_constant`, added by commit 2 to dedup those two emissions.

Verified: those are the *only* two `named_constants=` producers in the file (`grep -n
"DerivedIterationRangesRoot(\|named_constants" simd.py`), and both pass
`_grouped_axis_named_constants(self.group_tree)` on the same layout -- in
`make_reduced_output_family.build` the early `if tree is not self.group_tree: return tree` at
`simd.py:1878-1882` means `tree` is always the group tree. The two emitted lines are therefore
always byte-identical, which makes the `existing != line` conflict branch at
`triton.py:7831-7835` unreachable.

The sub-parent family's only consumer of those symbols is
`materialize_value_at_sub_parent_resolution`'s reduced-broadcast arm (`simd.py:2047-2056`,
`num_groups_str` and `local_reduction_size_sym`). In the append path the reduced-output family
already emitted them first -- `_GroupedReductionOpsHandler.reduction` calls
`self._family.ensure_headers(k)` at `simd.py:2233`, and `_codegen_grouped_reduction`
(`simd.py:3299-3312`) always runs before `_codegen_sub_parent_pointwise` (`simd.py:3313-3326`).
In the standalone path that arm is unreachable: it requires a CSE value whose parent dim string
is `nested_R0_REDUCED_BLOCK`, and the only producer of such values is `emit_reduce(...,
layout.output_shape)` in `_GroupedReductionOpsHandler`, which the standalone path never
instantiates. So the standalone kernel emits two constexpr lines nothing reads -- one of them
`nested_R0_REDUCED_BLOCK = R0_BLOCK // <rnumel>`, which is `0` whenever the kernel is looped.

**Concrete change.** Drop `named_constants=` from `make_sub_parent_family` (`simd.py:1958`).
Then delete `TritonKernel._named_constants` (`triton.py:3297-3299`) and
`_codegen_named_constant` (`triton.py:7823-7838`), restoring the base
`code.writeline(...)` loop in `iteration_ranges_codegen_header` -- except keep commit 2's
"constants go to `self.body`, not the loop-local buffer" behaviour, which is the load-bearing
half of that hunk. Safe fallback if the ordering argument is not convincing: keep emission where
it is but make `_named_constants` an `OrderedSet[str]` of emitted names and drop the unreachable
conflict assert (7 lines, zero risk).

**Est. LOC removed.** ~22 (full version), ~7 (fallback), plus two dead lines in every generated
standalone sub-parent kernel.

**Risk.** Medium for the full version -- it rests on "reduced-output headers always precede
sub-parent headers in the append path". Low for the fallback.

**Verdict.** yes-if

---

### 4. Dispatch the sub-parent path from `codegen_node`, not `_codegen_nodes`

**Summary.** `codegen_node` builds a plan whose only effect is to decide not to compute a
`coalesce_analysis` that the sub-parent path then discards anyway. Moving the dispatch one level
up deletes the branch from `_codegen_nodes` entirely and removes a build.

**Where.** `simd.py:3839-3851` (`has_sub_parent_epilogue` -> `coalesce_analysis = None`) and
`simd.py:3640-3645` (`_codegen_nodes` builds the plan again and dispatches).
`_codegen_reduction_with_sub_parent_epilogue` never receives `coalesce_analysis`; it passes
`None` explicitly to both `SIMDKernelFeatures` and `get_tiling_and_scores`
(`simd.py:3721-3727`).

**Concrete change.**

```python
# codegen_node, replacing lines 3839-3851
sub_parent = self._find_sub_parent_epilogue_plan(nodes)
if sub_parent is not None:
    return self._codegen_reduction_with_sub_parent_epilogue(nodes, *sub_parent)
...existing coalesce_analysis block, unchanged...
return self._codegen_nodes(nodes, coalesce_analysis)
```

and delete `simd.py:3640-3645`. The other `_codegen_nodes` caller
(`simd.py:3121`, the mix-order-reduction epilogue split) passes a pointwise-only list, and
`_find_sub_parent_epilogue_plan` only fires when the list contains a reduction
(`simd.py:2753-2755`), so it loses nothing.

**Est. LOC removed.** ~8, plus one planner run per codegen'd node.

**Risk.** Low.

**Verdict.** yes (fold into item 4.3's second paragraph, which asks for the same thing but does
not say the discarded value is the whole reason the first build exists)

---

### 5. Collapse the float8 bitcast plumbing

**Summary.** The "bitcast to uint8, reshape/split/broadcast, bitcast back" dance is spelled at
four sites, and it introduced an optional `value_expr` kwarg on `_reshape_expr` whose only
purpose is to let one of those sites pass a pre-bitcast string. Everything needed is already on
the `CSEVariable`.

**Where.**
- `triton.py:6195-6204` -- `_bitcast_reshape_expr(value, shape, dtype)`; every caller passes
  `dtype == value.dtype`: `simd.py` asserts it at `triton.py:6212-6214` and `6224-6226`, and
  `emit_broadcast_via_reshape`'s `dtype` argument is `value.dtype` at both callsites
  (`simd.py:2116` and `simd.py:2169`).
- `triton.py:6259-6271` -- `_reshape_expr`'s `value_expr: str | None = None`, used by exactly
  one of its two callers.
- `triton.py:6166`, `6185`, `6202`, `6250` -- four `dtype in TRITON_FLOAT8_DTYPES` tests.
- `triton.py:6231-6239` -- `emit_broadcast_via_reshape` also takes `out_shape`, which is
  `final_shape` at both callsites (`simd.py:2117` and `simd.py:2170`). This redundancy predates
  the stack (one callsite on `main`), but the stack made it a pattern by adding the second.

**Concrete change.** Give `_bitcast_reshape_expr` the signature `(value, shape)` and read
`value.dtype`; drop the `dtype` and `out_shape` parameters from `emit_broadcast_via_reshape`.
Keep `_reshape_expr`'s `value_expr` only if `_bitcast_reshape_expr` cannot be expressed without
it -- note that folding the bitcast *into* `_reshape_expr` is wrong, because `emit_reshape`
(`triton.py:6126-6138`) is on the grouped-reduction path and must not bitcast.

**Est. LOC removed.** ~12.

**Risk.** Low.

**Verdict.** yes-if

---

### 6. Trim the new `plan is None` arm (the piece nobody has reviewed)

**Summary.** The most recent amend added `_is_sub_parent_shaped` and a new arm to
`_sub_parent_epilogue_leaf_violation`. One of its two `isinstance` checks is unreachable, and
the other is a `FusedNestedReductions` carve-out sitting inside a predicate that is otherwise
purely about tile shapes -- it only exists because commit 2's `can_fuse_nested_reduction_append`
shares this function.

**Where.** `simd.py:2674-2686`.

`isinstance(node2, scheduler.FusedNestedReductions)` at `simd.py:2680-2682` cannot be true:
`Scheduler._can_fuse` returns `False` for a `FusedNestedReductions` `node2` at
`scheduler.py:9004-9005`, before any backend gate; the two backend entry points that reach
`SIMDScheduling.can_fuse` are `backend_can_fuse(node1, node2)` (`scheduler.py:9291`, `9312`) and
`can_fuse_horizontal(node1, node2)` (`scheduler.py:9319`), all downstream of that return; and
the arg-swapped re-entry at `simd.py:2592` passes the original `node2`, which by the same
argument is not one. The `node1` half *is* reachable, but only via
`can_fuse_nested_reduction_append` (`simd.py:2597-2606`), which is dispatched only when
`isinstance(node1, FusedNestedReductions)` (`scheduler.py:9279-9283`) -- so on that path the
whole `plan is None` arm is a constant `False`.

**Concrete change.** Delete the `node2` isinstance. Then, if item 4.4 lands (replace the backend
hook with a three-line early return in `SIMDScheduling.can_fuse`), move the `node1` carve-out to
that early return instead, so `_sub_parent_epilogue_leaf_violation` goes back to being about
plans and tile shapes.

**Est. LOC removed.** ~5.

**Risk.** Low; the deletion is a no-op by the argument above, the relocation depends on 4.4.

**Verdict.** yes-if (the `node2` deletion is unconditional yes)

---

### 7. `must_materialize_names` means two different things at its two callsites

**Summary.** One parameter, two definitions from two commits, and it doubles as the
`allow_reduced_broadcast` switch.

**Where.** `simd.py:2306`, `2318-2323`, `2340` (the handler) and the two callers:
- standalone (`simd.py:3769-3773`): every buffer written by any parent node.
- append (`simd.py:3325`): `node.sub_parent_broadcast_source_names`, which is
  reduced-domain outputs plus reduction outputs (`scheduler.py:3821-3832`).

At `simd.py:2340` the same set is forwarded as `allow_reduced_broadcast=name in
self._must_materialize_names`, so in the standalone path *every* parent buffer is granted
reduced-broadcast permission, and in the append path only genuinely reduced-shaped ones are.
The name says "must be materialized"; the second use says "may be reduced-shaped".

**Concrete change.** Do this with item 4.6. Once `allow_reduced_broadcast` is gone,
`must_materialize_names` has one meaning again and the standalone path's over-broad set stops
mattering. If 4.6 is rejected, split into two explicitly named parameters rather than reusing
one set for both roles.

**Est. LOC removed.** 0 (it is a naming/semantics fix that rides on 4.6).

**Risk.** Low.

**Verdict.** yes-if

---

### 8. (Observation, recommending against) Two planners for one job

`FusedNestedReductions._parent_half_source_layouts` (`scheduler.py:3923-3989`, 67 lines) is the
append path's re-derivation of what `NestedReduction.sub_parent_epilogue_plan`
(`scheduler.py:635-735`) does for the standalone path: it rebuilds `fused_buffer_names`,
`fullres_numel`, `reduction_reads` and `source_writes`, calls the same
`_sub_parent_epilogue_source_deps` and `_sub_parent_epilogue_source_loads_are_unambiguous`, then
adds its own `allowed_internal_names` check. Node selection is likewise duplicated: the
standalone path picks epilogue nodes by `_sub_parent_epilogue_rate` (`scheduler.py:817-843`),
the append path by the hardcoded factor-2 numel test in `_classify_grouped_pointwise_nodes`
(`scheduler.py:1422-1429`).

This is the largest structural artifact of the commit split, and it is what recommendation
batch 7's rename is papering over -- the two vocabularies exist because the two planners do.
I am still recommending **no**: the append path operates over a different node partition with
mutation renames applied and takes its factor from the domain classifier rather than from a rate
computation, so a merged planner needs enough parameters that it is unlikely to be shorter or
clearer. Worth one comment on `_parent_half_source_layouts` saying it is the append-path
counterpart of `sub_parent_epilogue_plan` and listing which of that function's guards it
deliberately omits.

**Verdict.** no

---

## Possible correctness concerns

Flagged in passing; not verified as bugs, and out of scope for this pass.

1. **The sub-parent path bypasses the cpp-wrapper exclusion.** `NestedReduction._is_enabled_for`
   (`scheduler.py:546-554`) excludes `V.graph.cpp_wrapper` because of the TODO at
   `scheduler.py:1548-1549`: "enable nested reduction with cpp wrapper after validating the
   additional autotuning meta (min_xblock / min_rblock)". The standalone sub-parent path sets
   exactly those (`simd.py:3749`, `simd.py:3750-3752`) and is gated only by
   `supports_sub_parent_epilogue and config.triton.nested_reduction` (`simd.py:2737-2741`), with
   no cpp-wrapper check anywhere. See item 1.

2. **`nested_R0_REDUCED_BLOCK` is degenerate in a looped standalone kernel.** It is defined as
   `FloorDiv(group_tree.block_size(), local_reduction_size_sym)` (`simd.py:1785-1789`) with
   `local_reduction_size = parent_rnumel` (`simd.py:3754-3758`), so for a non-persistent kernel
   `R0_BLOCK < parent_rnumel` and the constant evaluates to `0`. Harmless today only because I
   believe nothing in the standalone path reads it (see item 3); if the reduced-broadcast arm at
   `simd.py:2047-2056` ever becomes reachable there, it would divide by that value.

3. **The leaf check is not actually arg-order independent.** Its docstring
   (`simd.py:2653-2655`) claims it is, but `sub_parent_epilogue_plan` iterates `nodes` in the
   order given (`scheduler.py:659`, `752`) and rejects on
   `output_lanes != tuple(sorted(output_lanes))` (`scheduler.py:798-799`). Since
   `_sub_parent_epilogue_leaf_violation` builds `nodes` as `[*node1.get_nodes(),
   *node2.get_nodes()]` (`simd.py:2667`), the call at `simd.py:2421` and the swapped re-entry
   via `simd.py:2592` can in principle disagree.
