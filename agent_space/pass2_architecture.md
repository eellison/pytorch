# Sub-parent epilogue fusion: architecture review

Read against the **final state** of the stack tip `f020897c032` (`git show f020897c032:<path>`),
not the diff. All `file:line` refs are against those blobs. Base is
`56de3c7e2327b636c9f349ee926534a75ae18f3e` (merge-base with `origin/main`);
the stack is 4 commits, +3717/-538 over 10 files.

This is a shape review, not a line-level pass. `STACK_SIMPLIFICATION_REVIEW.md` and
`STACK_SIMPLIFICATION_RECOMMENDATIONS.md` already cover the line-level findings; where
this doc reaches the same conclusion by a different route I say so and do not re-derive it.

## Fusion traced end to end

**`test_producer_consumer_rmsnorm_chunk_swiglu` / `test_rmsnorm_chunk_swiglu_kernel_form`**
(`test/inductor/test_nested_reduction.py:1006`, `:2666`). B=32, D=1024, bf16:

```python
h = x + residual
variance = h.pow(2).mean(dim=-1, keepdim=True)
h = h * torch.rsqrt(variance + 1e-6) * weight
gate, up = h.chunk(2, dim=-1)
return F.silu(gate) * up
```

`h` is inlined (the FileCheck pins `{0: 1, 1: 1, 2: 2}` argument occurrences in the
persistent kernel: `x` and `residual` once each, `weight` twice), so the two graph inputs
are the sub-parent *sources*, not an intermediate buffer.

1. `Scheduler._can_fuse_impl` (scheduler.py:9223) calls `_nested_index_equivalent_dep_names`
   → `NestedReduction.sub_parent_epilogue_plan([reduction nodes..., swiglu], 32, 1024)`
   (scheduler.py:8768). This is **plan build #1**, and it happens purely so the memory-score
   heuristic does not reject the pair before legality runs.
2. `_sub_parent_epilogue_parent_rnumel(1024)` → 1024 (static power of two, scheduler.py:807).
3. `_sub_parent_epilogue_candidate_nodes` (scheduler.py:738): the swiglu node has
   `node_numel = 16384`, `full_numel = 32768`; `_sub_parent_epilogue_rate` returns
   `(factor=2, output_lanes=1)`; `SIMDKernel.is_compatible((32, 512), [[32, 512]])` passes.
4. `_sub_parent_epilogue_source_deps` (scheduler.py:865) walks the swiglu node's reads.
   For the `x` read at `1024*x + r`: `2 * product(dep.ranges) == full_numel` ✓, the unique
   full-resolution producer dep is the reduction's read at `1024*x0 + r0`.
   `lane = Mod(1024*x + r, 2)` is not statically 0 or 1, so the INTERLEAVED arm
   (scheduler.py:933) is skipped and the CONTIGUOUS arm (scheduler.py:949) runs:
   `_unique_trailing_sub_parent_dim` finds `lane_dim = 1` (1024 == 512*2),
   `lane_exprs = [r, r + 512]`, substitution gives `expected_by_lane = [1024x + r, 1024x + r + 512]`,
   `_matches_unique_sub_parent_lane` finds the unique match (lane 0) and cross-checks it against
   `_contiguous_sub_parent_epilogue_emitted_lane` = `FloorDiv(Mod(0, 1024), 512)` = 0. ✓
   The `up` read at `1024x + r + 512` resolves to lane 1 the same way. Both are recorded
   under one entry `(x, CONTIGUOUS)` — `add_source` (scheduler.py:883) keys by name, so the
   plan carries **one layout per source name, not one lane per read**. Same for `residual`.
   `weight` falls through the `not reduction_deps: continue` branch and is re-loaded from
   global at half resolution.
5. Plan = `epilogue_nodes=(swiglu,)`, `output_lanes=(1,)`, `factor=2`, `parent_rnumel=1024`,
   `source_layouts=((x, CONTIGUOUS), (residual, CONTIGUOUS))`.
6. `SIMDScheduling.can_fuse` runs `_sub_parent_epilogue_leaf_violation` **first**
   (simd.py:2421), which builds the plan **again** (simd.py:2668) and, if that had not covered
   the consumer, a third time with `check_leaves=False` (simd.py:2673). The final arm of
   `can_fuse` then calls `_can_fuse_sub_parent_reduction_epilogue`, **build #4** (simd.py:2615).
7. Codegen: `codegen_node` builds the plan a **fifth** time only to null out
   `coalesce_analysis` (simd.py:3839), and `_codegen_nodes` a **sixth** (simd.py:3640).
8. `_codegen_reduction_with_sub_parent_epilogue` (simd.py:3655) builds a
   `_GroupedReductionLayout` with `local_reduction_size = parent_rnumel = 1024`, so
   `num_groups == 1` — the standalone path reuses the *nested reduction* codegen substrate
   with a degenerate group. `make_sub_parent_family(2)` adds a derived R tree with
   `numel = rnumel/2`, `block_size = RBLOCK/2`, suffix `lane2`.
9. Persistent variant: `_SubParentSourceLoadMaterializer.load` (simd.py:2379) intercepts the
   parent-schedule loads of `x` and `residual` and calls
   `materialize_value_at_sub_parent_resolution` (simd.py:2028), which for CONTIGUOUS emits
   `emit_split_via_reshape_permute(value, (XBLOCK, 2, RBLOCK//2), (0, 2, 1), parts)`
   → `tl.split(tl.permute(tl.reshape(...)))`. Two sources × one split each = the
   `check_count("tl.split(", 2)` + `check("tl.permute(tl.reshape(")` the test pins.
10. `_SubParentPointwiseRemapHandler.load` → `_resolve_remapped_value` (simd.py:1533)
    re-derives the lane from the *codegen-side* index:
    `FloorDiv(Mod(offset, factor * child_extent), child_extent)` → `parts[1]` for the `up` read.
11. Looped variant: `internal_source_names` is empty (both sources are graph inputs), so
    `epilogue_source_layouts` is `{}` and the epilogue simply re-loads `x`/`residual` from
    global at half resolution — hence `check_not("tl.split(")`.

Everything below is grounded in this trace plus the MXFP6 path
(`_mxfp6_pack_four_to_three`, test file:118), which is the only place `output_lanes > 1`
is exercised.

---

## 1. Where does the legality/codegen split fall, and is it right?

### What the plan carries vs. what codegen re-derives

`SubParentEpiloguePlan` (scheduler.py:626) carries six fields: `epilogue_nodes`,
`epilogue_node_output_lanes`, `parent_nodes`, `sub_parent_factor`, `parent_rnumel`,
`source_layouts: tuple[(name, layout)]`.

Codegen re-derives, in `_codegen_reduction_with_sub_parent_epilogue` (simd.py:3655-3822):

| Re-derived | Could live in the plan? |
|---|---|
| `internal_source_names` (source names ∩ parent buffer names) | yes, trivially |
| the 36-line deferred-source-chain topological reorder (:3665-3702) | yes — it is a scheduling decision |
| `must_materialize_names` = all parent buffer names | yes |
| stage grouping (`zip` + `groupby` + `itemgetter`, :3790-3798) | yes — store stages |
| per-load contiguous lane (`_resolve_remapped_value`) | **no** — see below |
| `epilogue_source_layouts` filtered by `kernel.persistent_reduction` (:3760) | **no** — persistence is decided when the kernel is created |
| `min_xblock` / `min_rblock` (:3738-3752) | **no** — same reason |

So the split is *mostly* in a defensible place. Two things are genuinely
post-kernel-construction and must stay on the codegen side. The rest is drift.

### The per-load lane cannot be hoisted, and that is the crux

A single epilogue node reads **multiple lanes of the same source**: swiglu reads `x` at
lane 0 (`gate`) and lane 1 (`up`). `add_source` (scheduler.py:883) deliberately collapses
those into one `(name, layout)` entry. So there is no "the lane" for a source; the lane is a
property of each *load site* and must be recovered at codegen from the load index. That is
correct and there is no smaller alternative.

But it forces the contiguous lane formula to be written twice, on two different expression
objects:

```
scheduler.py:1257  FloorDiv(Mod(offset, parent_rnumel), child_extent)      # offset from dep.index
simd.py:1548       FloorDiv(Mod(candidate, factor * child_extent), child_extent)  # from codegen index
```

(`factor * child_extent == parent_rnumel` by construction at simd.py:2083.) They agree only
because both take the constant term of an affine index; nothing type-checks or asserts the
relationship. When they disagree, the failure mode is an `AssertionError` at *codegen* time
(simd.py:1556) rather than a fusion rejection. The interleaved lane is duplicated the same
way (`Mod(dep.index, factor)` at scheduler.py:929 vs `Mod(index, len(value))` at
simd.py:1563), with the second `AssertionError` at simd.py:1567.

### The contract nobody writes down

There is a **third** coupling that has no assert at all. The scheduler's matcher decides
"this read is lane *k*"; codegen returns `parts[k]`. What `parts[k]` physically contains is
defined only by the even/odd recursion in `TritonKernel._emit_recursive_split`
(triton.py:6157-6192) — `names[0::2]` from the even half, `names[1::2]` from the odd half.
Change that recursion (e.g. to a linear scan) and the scheduler still accepts every plan and
every kernel silently miscompiles. The FileCheck tests pin the *count* of `tl.split`, and
the numeric tests catch it end-to-end, but nothing states the contract.

This is the strongest argument for "the plan should carry more, codegen should be
mechanical": the lane↔elements correspondence is the entire feature, and it is currently an
implicit agreement between two modules.

### The recompute-at-codegen decision, and what it costs

The plan is **never stored**. It is rebuilt from a node list every time anyone needs it —
up to six times per candidate pair in the trace above. Contrast `FusedNestedReductions`
(scheduler.py:3755), which computes `parent_half_source_layouts` once in `__init__`
(scheduler.py:3841) and stores it. The stack therefore contains both patterns for the same
information.

The recompute choice is not free. Because codegen recomputes the plan over whatever node set
the scheduler ended up with, `_sub_parent_epilogue_leaf_violation` (simd.py:2642, 87 lines)
has to run as a **global precondition at the very top of `SIMDScheduling.can_fuse`**
(simd.py:2421), before any numel branch, for *every* reduction/pointwise pair in every
Triton compile. Its job is to guarantee that whatever the scheduler assembles, codegen's
recomputed plan will still cover it. If the plan were attached to the fused node, most of
that check collapses to "is this node in my plan".

Related, and worth fixing regardless of the above: `_nested_index_equivalent_dep_names`
(scheduler.py:8746) gates only on `config.triton.nested_reduction` (:8891) — unlike
`NestedReduction.is_candidate`, which goes through `_is_enabled_for` and checks
`_is_gpu_triton_backend` (scheduler.py:546). So on a CPU compile the sub-parent planner still
runs for every reduction→pointwise pair with a static power-of-two rnumel. The backend gate
exists only on the simd-side entry point (`supports_sub_parent_epilogue`, simd.py:2738), not
on the scheduler-side one.

### Verdict on Q1

The split is in roughly the right place — the two post-kernel facts justify a codegen side
that is not purely mechanical. But three specific things belong on the plan side:

1. **`lane_stride` instead of `SubParentSourceLayout`** (see Q2). This is what turns the
   lane formula from two hand-written expressions into one.
2. **Stages instead of parallel `epilogue_nodes` / `epilogue_node_output_lanes` tuples.**
   Already recommendation 3.2. The architectural point that doc does not make: the
   sortedness precondition at scheduler.py:798 exists *only* because codegen chose
   `itertools.groupby`, which groups adjacent keys. A plan is currently **rejected** for an
   ordering that codegen could handle by grouping adjacent runs instead of requiring global
   sortedness. Legality should not encode a codegen implementation detail. Storing stages
   fixes the drift *and* removes a rejection.
3. **Source `MemoryDep`s, not just names.** The plan stores `(name, layout)`;
   `_sub_parent_epilogue_leaf_violation` then reconstructs `planned_source_deps` as a
   *superset* (every full-resolution `MemoryDep` with a matching name, simd.py:2695-2705)
   versus the planner's one-per-name selection. That divergence was already noted as an
   unverified correctness concern; storing the deps removes the second derivation entirely.

Do **not** attach the plan to a new `FusedSubParentEpilogue(FusedSchedulerNode)` subclass
right now, even though `BaseScheduling.fuse` (scheduler.py:11379) has the hook and
`FusedNestedReductions` is the precedent. Estimated ~60 new lines, ~90 deleted, but it
changes which class the scheduler sees for these groups, and `estimate_runtime`, combo-kernel
filtering (scheduler.py:4340-4357), `codegen_node` dispatch and `benchmark_fused_nodes` all
key off node class. The win is compile time and conceptual tidiness, not line count.
Revisit if the compile-time measurement recommendation 4.3 asks for comes back bad.

---

## 2. Is INTERLEAVED/CONTIGUOUS the right abstraction?

**Short answer: the two-way distinction is right, the way it is materialized is not.** The
premise in the review brief — "two different codegen paths (`tl.split` recursion vs
reshape+permute)" — does not survive reading the code. Both paths call the *same*
`_emit_recursive_split`:

```
triton.py:6206  emit_split_via_reshape(value, reshape_shape, names)
                    reshaped = _bitcast_reshape_expr(...)
                    _emit_recursive_split(reshaped, names, reshape_shape, dtype)

triton.py:6217  emit_split_via_reshape_permute(value, reshape_shape, permute_dims, names)
                    reshaped = _bitcast_reshape_expr(...)
                    permuted = f"tl.permute({reshaped}, {permute_dims})"
                    _emit_recursive_split(permuted, names, permuted_shape, dtype)
```

The entire difference is one `tl.permute` line. There is one codegen strategy, not two.

### The unification that works

Both layouts are mixed-radix splits of the parent axis into two digits; they differ only in
digit order:

| | parent coord | lane extraction | reshape before split |
|---|---|---|---|
| INTERLEAVED | `factor*child + lane` | `Mod(idx, factor)` | `(..., child, factor)` |
| CONTIGUOUS | `child + lane*child_extent` | `FloorDiv(Mod(idx, factor*S), S)`, `S = child_extent` | `(..., factor, child)` + permute |

Parameterize by `lane_stride S ∈ {1, child_extent}` and every codegen-side expression
unifies:

- **lane extraction**: `FloorDiv(Mod(idx, factor*S), S)`. At `S = 1` this is `Mod(idx, factor)`.
  One formula replaces the two arms of `_resolve_remapped_value` (simd.py:1533-1573, 41
  lines), deletes `_ContiguousSubParentRemappedValue` (simd.py:1513) and the
  `RemappedRangeValue` three-way union, and removes the type-encodes-layout smell already
  flagged as 2.2.
- **materialization**: `materialize_value_at_sub_parent_resolution` (simd.py:2074-2090) loses
  its `if CONTIGUOUS / else INTERLEAVED` branch; one `emit_split_lanes(value, factor,
  lane_stride)` reshapes to `(..., outer, factor, S)`, permutes only when `S != 1`, and
  recurses. This subsumes recommendation 2.3 (`permute_dims=None`) and is strictly more
  general at the same size.

Combined: ~40-50 lines and two concepts (`_ContiguousSubParentRemappedValue`, the two emit
methods) removed. Low risk — it is a pure refactor of the emit path with numeric tests
covering both layouts. Re-test: the full `test_nested_reduction.py` sweep; the FileCheck
`tl.split` counts and the `tl.permute(tl.reshape(` check are exactly the pins that would
catch a mistake.

### The unification that does not work

Do **not** push `lane_stride` into the scheduler matchers. The unified parent-coordinate map
is `(child // S)*(S*factor) + lane*S + (child % S)`. At `S = 1` sympy cancels it to
`child*factor + lane` for free. At `S = child_extent` it cancels only if sympy knows
`child < child_extent`, which it does not — the matcher would have to supply a range
assumption or special-case anyway. The current per-layout substitution forms
(scheduler.py:1131 and scheduler.py:1206) are what make `statically_known_equals` provable.
This confirms recommendation 4.2's "keep the lane derivation per-layout" from a different
direction.

### The four matchers are two axes, and the second one is a precision ladder

`{INTERLEAVED, CONTIGUOUS}` × `{same-rank, flat}`. The second axis is not a second semantics
— it is a fallback for when the consumer's loop nest does not line up dimension-for-dimension
with the reduction's. The same-rank matcher substitutes `reduction_var_i → dep_var_i`
(exact, cheap for sympy); the flat matcher builds a flat index and substitutes
`reduction_var_i → FloorDiv(parent_index, stride) % size` (general, much harder for sympy to
cancel). So same-rank is a **precision optimization**, and the code never says so.

The asymmetry gives it away: interleaved *falls through* to flat when
`_unique_trailing_sub_parent_dim` returns `None` (scheduler.py:1127-1130), contiguous
*returns False* in the same situation (scheduler.py:1201). Whether that is intentional is
undocumented. Naming the axis (`..._exact` / `..._via_flat_index`) and stating the fallthrough
policy costs zero lines and removes most of the "why are there four of these" tax.

### Verdict on Q2

Keep the two-valued distinction; it is one bit and the enum is a fine way to carry it.
Replace the enum's *value* with a `lane_stride`, which makes the codegen side uniform without
touching the matchers. Do not attempt an index-mapping abstraction on the legality side.

---

## 3. How much of this is a consequence of doing it in the scheduler?

### The irreducible core

The feature needs one capability Inductor does not have: **cross-lane data movement within a
tile**. For interleaved packing, the value the epilogue needs at position *j* lives at
positions *2j* and *2j+1* of the producer's register tile; for chunked SwiGLU it lives at *j*
and *j+512*. Inductor's `OpsHandler` has exactly three ops that move data across lanes —
`reduction`, `scan`, `sort` — and none expresses "give me the even sub-tile". Every layer
below the scheduler inherits this: there is no IR-level spelling for the operation, so no
lowering-level or FX-level rewrite can produce one.

### Alternative A: FX / post-grad pattern rewrite

Recognize `reduction → slice-consumer` before scheduling and rewrite it into a fused custom
op or template. **Clearly worse for the stated use cases.** The epilogue set is open-ended:
`silu(gate)*up`, `silu(a)*b + tanh(c)*d`, alternating sums over 8 and 16 chunks,
`inline_asm_elementwise` NVFP4/MXFP4 packing, the MXFP6 4→3 bit-shuffle. A pattern-based
approach needs one pattern per epilogue and forfeits generality — which is the whole point
of doing it in Inductor rather than writing the CUDA kernel. It would also have to make the
profitability call (the `min_xblock` carve-out at simd.py:3738, the persistent-vs-looped
source filtering at simd.py:3760) before knowing the tile shape.

### Alternative B: make the consumer a normal full-resolution node

Give the epilogue the parent's `(numel, rnumel)` domain and mask the store. Two blockers.
First, `out[b, j] = f(h[b, j], h[b, j+512])` still needs a cross-lane read — the masking
buys nothing. Second, Inductor has no predicated-store-by-index mechanism, and you would
waste half the lanes. Infeasible without the same primitive.

### Alternative C: a first-class "derived iteration domain" node-schedule entry

The honest alternative. Teach `SIMDKernelFeatures` / `generate_node_schedule` /
`select_tiling` / `codegen_node_schedule` that a node's iteration space may be an affine
subdivision of the kernel's tile, declared by a map. This is precisely what
`_DerivedIterationFamily` + `_IterationSpace` + `sub_parent_iteration_values` already are,
but built as a private branch (`_codegen_nodes` at simd.py:3640 dispatches out of the normal
path entirely) rather than as a facility.

Doing it properly is **larger and riskier**, not smaller: it touches the node-schedule entry
type, feature analysis, tiling selection and the main codegen loop, all of which are shared
with every non-nested kernel. The right time to promote it is when there is a third consumer.
The current shape — own entry point, own plan, reuse of the existing
`_GroupedReductionLayout` substrate — is the correct size for a first feature.

Note how much the reuse buys: the standalone path builds
`_GroupedReductionLayout.from_kernel(kernel, Integer(parent_rnumel), local_reduction_in_r=True)`
(simd.py:3745), which makes `num_groups == 1` — a degenerate grouped reduction. It then reuses
`make_sub_parent_family`, `sub_parent_iteration_values`, `parent_dim`, `child_block` and the
whole `_PointwiseRemapHandler` stack unchanged. That is why the *codegen* side of a
standalone sub-parent kernel is only ~170 lines (`_codegen_reduction_with_sub_parent_epilogue`)
plus ~110 lines of new layout methods. The append path and the standalone path share
`sub_parent_iteration_values` verbatim (simd.py:3288 and simd.py:3800). Credit where due:
this is well factored.

### Bounding the scheduler-integration tax

Lines whose existence is attributable to Inductor's pairwise, stateless `can_fuse` API rather
than to the feature's semantics:

| | lines |
|---|---|
| `_sub_parent_epilogue_leaf_violation` (simd.py:2642) | 87 |
| `_is_sub_parent_shaped` (simd.py:2621) | 21 |
| `_nested_index_equivalent_dep_names` score bridge (scheduler.py:8746) | 49 |
| `_reindex_sub_parent_consumer` + `_reindexed_dep_order` (scheduler.py:8795) | 95 |
| `_can_fuse_nested_reduction_append` (scheduler.py:8890) | 40 |
| `can_fuse_nested_reduction_append` hook (BaseScheduling + simd + cuda + xpu) | 26 |
| `_sub_parent_epilogue_plan` + `_find_sub_parent_epilogue_plan` wrappers (simd.py:2729) | 33 |
| `_can_fuse_sub_parent_reduction_epilogue` (simd.py:2608) | 13 |
| **total** | **~364** |

Against roughly 2000 non-test added lines, that is ~18%. The legality semantics themselves
(~680 lines in `NestedReduction`) and the codegen capability (~470 lines in simd.py + ~70 in
triton.py) are intrinsic — any layer would pay them, possibly in a different notation.

A hypothetical Inductor with a "fusion group planner" (plan once per group, hand the plan to
codegen) would eliminate maybe 200 of the 364: the leaf gate, the wrapper pair, and the
duplicate `_can_fuse_sub_parent_reduction_epilogue`. The score bridge and the reindexing
would survive in some form. **So: the feature is not being made large by living in the
scheduler. It is being made ~10% larger.**

---

## 4. Interface surface added to shared abstractions

### `BaseScheduling.can_fuse_nested_reduction_append` — delete (already 4.4)

`BaseScheduling` (scheduler.py:11340) gains a virtual with a default that every future
backend inherits and will never implement. Two pure-delegation overrides exist
(`CUDACombinedScheduling` at cuda_combined_scheduling.py:110, `XPUCombinedScheduling` at
xpu_combined_scheduling.py:66) purely because those classes derive from `BaseScheduling`, not
`SIMDScheduling`. The dispatch (scheduler.py:9276-9282) selects it by
`isinstance(node1, FusedNestedReductions)`. Recommendation 4.4 already covers this; I confirm
the analysis and add: the composite backends make the cost 4 sites, not 2.

### `SIMDScheduling.supports_sub_parent_epilogue` — beyond 4.4

A second, parallel backend-capability mechanism (simd.py:2401 = `False`, triton.py:8057 =
`True`) alongside the existing `BackendFeature` enum + `get_backend_features(device)`
(`codegen/common.py:437`). Every future SIMD backend now has two places to look for "what can
this backend do".

Worth noting it does not even work as a gate on the scheduler side: the scheduler-level entry
point `_nested_index_equivalent_dep_names` bypasses it entirely (Q1). So the flag guards the
`cast("TritonKernel", ...)` in codegen but not the planning cost. Either fold it into
`BackendFeature` (and apply it at both entry points), or delete it and let the
`isinstance(..., TritonScheduling)` check that codegen implicitly relies on be explicit.
~10 lines either way; the value is not adding a second capability vocabulary.

### `FusedNestedReductions` — the divergent `can_fuse_with` signature

`FusedMixOrderReductions.can_fuse_with(self, other)` (scheduler.py:3713) and
`FusedNestedReductions.can_fuse_with(self, other, *, can_reorder)` (scheduler.py:3849) are
dispatched from the same `isinstance` ladder (scheduler.py:9002-9008) with different
signatures. Anyone adding a third fused-node kind has to notice. One keyword-only argument
with a default on the base would make the ladder uniform. Small, but this is a shared
extension point.

Separately, `FusedNestedReductions` now carries 8 public-looking attributes
(`grouped_reduction`, `group_size`, `grouped_axis`, `group_size_in_r`, `domain_context`,
`grouped_pointwise_domains`, `sub_parent_broadcast_source_names`, `parent_half_source_layouts`)
and an `__init__` that raises `AssertionError` on 3 paths (scheduler.py:3778, :3792, :3815).
Recommendation 3.4 covers trimming this. The architectural note is that this is what
"attach the plan to the node" costs in practice — which is the main reason I do not recommend
doing the same for the standalone path (Q1).

### `TritonKernel` — fine

This stack adds exactly five methods to `TritonKernel`: `_emit_recursive_split`,
`_bitcast_reshape_expr`, `emit_split_via_reshape`, `emit_split_via_reshape_permute`,
`_codegen_named_constant`. `emit_reshape`, `emit_reduce`, `emit_broadcast_via_reshape`,
`min_xblock`, `min_rblock`, `use_range_trees` and `DerivedIterationRangesRoot` all pre-exist
at the base commit. Two of the five should be one (Q2). The rest is proportionate — this is
a well-behaved addition to the kernel API and I would keep it as is.

Likewise `SchedulerNode.snapshot_loop_state` / `restore_loop_state` / `apply_loop_reindexing`
/ `apply_new_loop_order` / `_LoopMutationTracker`, all used by `_reindex_sub_parent_consumer`,
pre-exist. No new node-mutation surface. Good.

### `SubParentEpiloguePlan` — the dataclass is the right idea, under-populated

Six fields, of which two are parallel tuples that codegen re-zips (Q1 item 2) and one is a
name where a dep is wanted (Q1 item 3). The concept — an explicit plan object crossing the
scheduler/codegen boundary — is right and is the best thing about the design. Make it carry
the whole decision.

---

## 5. Naming and conceptual load

Distinct concepts a reader must hold to follow one fusion decision:

1. **parent** — the reduction owning the grid, `(numel, rnumel)`
2. **sub-parent** — a `1/factor` slice of the parent tile
3. **parent-half** — sub-parent at factor 2, in the append path (`PointwiseDomain.PARENT_HALF`)
4. **factor** — the input slicing ratio, 2/4/8/16
5. **lane (input)** — which of the `factor` slices a read takes
6. **lane (output)** — `output_lanes` / `output_lane`: how many sub-parent-resolution outputs
   one epilogue node produces, and which one this emission is
7. **layout** — INTERLEAVED / CONTIGUOUS
8. **domain** — `PointwiseDomain`: REDUCED / LOCAL_REDUCTION_INPUT / PARENT_FULL / PARENT_HALF
9. **domain context** — `PointwiseDomainContext`, the four domain extents
10. **family** — `_DerivedIterationFamily`: kernel-side range trees + index substitutions +
    materialized values
11. **iteration space** — `_IterationSpace`: source-side coordinate groups + values
12. **stage** — a run of epilogue nodes sharing an `output_lanes`, emitted `output_lanes` times
13. **source** — a buffer whose full-resolution tile must be split
14. **grouped reduction** — the second reduction of the append path, degenerate
    (`num_groups == 1`) in the standalone path

Fourteen. Three collapses:

**(a) "lane" is two unrelated concepts sharing a word — fix this first.** Input-slice index
(5) and output multiplicity (6) meet in the same dataclass (`sub_parent_factor` next to
`epilogue_node_output_lanes`, scheduler.py:626-632) and the same signature
(`sub_parent_iteration_values(family, factor, output_lanes, output_lane)`, simd.py:1965). They
are not related: MXFP6 has factor 4 (four input slices) and a stage with `output_lanes = 3`
(the `torch.stack((low, middle, high), -1)` node, which produces three trailing values per
child position). The derived tree is even named `lane{factor}` (simd.py:1957), i.e. sense (5).
Rename (6) to `output_repeats` / `pack_width` — zero lines, and it removes the single worst
comprehension tax in the stack. This is *not* the same finding as recommendation batch 7
(`parent_half` vs `sub_parent`), which is about (2)/(3).

**(b) parent-half and sub-parent-at-factor-2 are one concept.** Already batch 7. Confirmed:
`PARENT_HALF_FACTOR = 2` is even borrowed as a stand-in for the literal 2 in the standalone
interleave gate (scheduler.py:934).

**(c) family and factor are one thing.** `_DerivedIterationFamily` is built by
`make_sub_parent_family(factor)` and then `factor` is threaded alongside it into
`sub_parent_iteration_values(family, factor, ...)` (simd.py:1965),
`materialize_value_at_sub_parent_resolution(kernel, family, factor, ...)` (simd.py:2028), and
both handlers' `__init__` (simd.py:2296, :2355). The family already carries the derived tree
that encodes the factor. Store `factor` on the family and four signatures shrink. ~10 lines,
zero risk.

Two pairs that look collapsible and are **not**: `_DerivedIterationFamily` (target) vs
`_IterationSpace` (source) are genuinely different halves of a remap and should stay two
types — but the relationship deserves one sentence somewhere, because both docstrings
describe themselves and neither describes the pair. And `PointwiseDomain.PARENT_FULL` vs
"a parent node" are distinct (the former is a classification inside the append path, the
latter is plan partitioning).

Fourteen down to eleven, plus one word disambiguated. That is a real but bounded improvement;
this is not a design drowning in concepts — most of the fourteen earn their place.

---

## Ranked recommendations

Sized in lines-of-source delta; "sweep" = the three-file validation in
`STACK_SIMPLIFICATION_RECOMMENDATIONS.md`. Items marked ⊕ are additions to the existing plan;
the rest reframe or extend an item already there.

| # | Recommendation | Size | Risk | Re-test |
|---|---|---|---|---|
| 1 ⊕ | **Replace `SubParentSourceLayout` with `lane_stride`.** One lane formula (`FloorDiv(Mod(idx, factor*S), S)`), one emit helper (`emit_split_lanes(value, factor, lane_stride)`), delete `_ContiguousSubParentRemappedValue` and the `RemappedRangeValue` union. Subsumes 2.2's dataclass item and 2.3 entirely. Do **not** propagate `lane_stride` into the matchers. | −40/−50 | low | full sweep; the `tl.split` counts and `tl.permute(tl.reshape(` FileChecks are the pins |
| 2 ⊕ | **Rename output-lane vocabulary.** `output_lanes`/`output_lane`/`epilogue_node_output_lanes` → `output_repeats` etc. Land with batch 7, separately from the `parent_half` rename. | 0 | none | compile only |
| 3 | **Store stages on the plan** (recommendation 3.2) **and drop the sortedness rejection** at scheduler.py:798 in favour of grouping adjacent runs. The reframing: legality currently rejects a plan because codegen chose `groupby`. | −15, +1 accepted case | low-med | full sweep + an unsorted-lane-order test |
| 4 | **Store source `MemoryDep`s on the plan**, deleting the superset reconstruction at simd.py:2695-2705. Closes correctness concern #6 from the prior review as a side effect. | −10 | low-med | full sweep |
| 5 ⊕ | **Add the backend gate to the scheduler-side entry point.** `_nested_index_equivalent_dep_names` (scheduler.py:8761) checks only `config.triton.nested_reduction`; add `_is_gpu_triton_backend`, matching `_is_enabled_for`. Compile-time only, but it currently runs the planner on CPU compiles. | +3 | low | full sweep + a CPU compile-time spot check |
| 6 | **Delete the `can_fuse_nested_reduction_append` hook** (4.4). Confirmed; note it is 4 sites, not 2, because of the two composite backends. | −29 | low-med | full sweep |
| 7 ⊕ | **Fold `supports_sub_parent_epilogue` into `BackendFeature`, or delete it.** A second backend-capability vocabulary next to `get_backend_features`. | ~0 | low | full sweep |
| 8 ⊕ | **Put `factor` on `_DerivedIterationFamily`.** Removes a correlated parameter from four signatures. | −10 | none | compile + sweep |
| 9 ⊕ | **Name the precision axis in the matchers** (`_exact` / `_via_flat_index`) and say in one comment why interleaved falls through to flat and contiguous does not. Pairs with 4.2; do it even if 4.2 is skipped. | 0 | none | none |
| 10 ⊕ | **State the lane↔elements contract** where `_emit_recursive_split` defines it (triton.py:6157): "`parts[k]` holds parent positions ≡ k, matching `_contiguous_sub_parent_epilogue_emitted_lane` / the `Mod(index, factor)` interleaved lane." One comment; the alternative is a structural test. | 0 (or +15 for a test) | none | none |
| — | **Do not** introduce a `FusedSubParentEpilogue` node class to carry the plan. ~+60/−90, but it changes node-class dispatch across `estimate_runtime`, combo kernels and `codegen_node`. Reconsider only if 4.3's compile-time measurement is bad. | | | |
| — | **Do not** unify the four matchers on a single affine map. Sympy cannot cancel the general form for the contiguous case without a range assumption. | | | |

**Bottom line.** No materially smaller design exists. The feature needs a cross-lane
tile-movement capability that Inductor does not have at any layer, and the two alternatives
(FX pattern rewrite, full generalization of the node-schedule domain) are respectively
narrower and larger. About 18% of the added non-test code is scheduler-integration tax, of
which perhaps half is addressable. What is genuinely wrong is smaller and more specific than
the size suggests: the plan under-carries its own decision, so three facts get re-derived on
the codegen side — two of them as hand-written duplicates of a formula, one (the lane↔elements
contract) as no statement at all.

---

## Possible correctness concerns

Tripped over while tracing; not verified, not part of the recommendations.

1. **The deferred-source reorder can silently no-op** (simd.py:3665-3702). When a source is
   written inside the kernel and the reorder's guard fails
   (`if not any(node.ancestors & deferred_names for node in leading_nodes)`, simd.py:3702),
   `parent_nodes` is left in its original order. In a looped kernel the source would then be
   produced before the reduction loop and consumed by an epilogue emitted after it. The
   planner's guard at scheduler.py:703-709 only rejects sources that are *also read by the
   reduction*. Worth confirming the remaining case is unreachable.
2. **CONTIGUOUS with `output_lanes > 1`** (empty cell in the prior review's matrix, item 5.3).
   Reaching it, `sub_parent_iteration_values` sets `source_values[2] = pair_var*output_lanes +
   output_lane` (simd.py:1984), introducing a nonzero constant into the codegen index that
   `_resolve_remapped_value`'s offset extraction (simd.py:1543) would fold into the lane —
   whereas the scheduler's `_contiguous_sub_parent_epilogue_emitted_lane` never sees it. If
   that path is genuinely reachable, the two lane derivations diverge there. This is a second
   reason to prefer item 5.3's "reject it explicitly".
3. **`num_store` patched after the fact** (simd.py:3818, already flagged as concern #7 in the
   prior review). Reconfirmed while tracing: `kernel.num_store = max(kernel.num_store,
   len(kernel.store_buffer_names))` runs *after* the epilogue stages, so anything reading
   `num_store` during epilogue codegen sees the undercount.
