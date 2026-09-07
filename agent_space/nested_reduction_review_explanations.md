# Nested Reduction Special-Case Review Map

Scope: current nested-reduction source diff against
`d69ec4e17a07ec00e3f6944b1013cc2acd8f393f`, after the cleanup pass that
removed manual buffer removal, `max_xblock`, and stale nested-only caches.

Goal: identify every remaining special case, explain how it diverges from the
existing machinery, and state whether that divergence is justified or what the
alternative would be.

## Quick Take

There are still real irregularities. The most review-sensitive ones are:

1. Nested reduction has a custom `FusedNestedReductions` node type.
2. `SIMDScheduling.can_fuse` has a nested bypass because normal SIMD fusion
   requires compatible `(numel, rnumel)`.
3. `codegen_nested_reduction` manually orchestrates a two-resolution schedule.
4. `DisableReduction` / `EnableReduction` markers from node2 are reused only
   as ordering separators.
5. `min_xblock` / `min_rblock` flows into config generation and coordinate
   descent.
6. Template/user Triton name-based fusion scoring is broader than nested
   reduction and should be reviewed as a generic scoring change.

The main simplification already achieved is that nested no longer has custom
manual buffer removal or a custom broadcast-value cache; it now relies on normal
`store_cache`, CSE, kernel-local buffer removal, and normal split/reindex
helpers where possible.

## Pass 1: Inventory

### Fusion Selection and Scheduler

1. `_is_gpu_triton_backend`
2. `NestedReduction.can_fuse`
3. Grouped-reduction prefiltering inside `NestedReduction.can_fuse`
4. `NestedReduction._get_grouped_reduction_info`
5. Same-total-numel / mismatched-rnumel gate
6. Grouped-axis divisibility / quotient proof
7. `NestedReduction.is_group_size_in_r`
8. `_flattened_group_size_in_r`
9. Group-size bound and power-of-2 gate
10. `_is_profitable_group_size_in_x`
11. `_would_use_3d_tiling`
12. `_pointwise_nodes_match_grouped_resolution`
13. `_node2_pointwise_nodes_are_supported`
14. `_node1_fullres_epilogues_are_compatible`
15. Full-resolution epilogue restriction for group-size-in-X
16. `get_node_fullres_epilogues`
17. `get_shared_input_reads`
18. `FusedNestedReductions`
19. `FusedNestedReductions.can_fuse_with`
20. `Scheduler.can_fuse` special handling for `FusedNestedReductions`
21. `Scheduler.fuse` construction of `FusedNestedReductions`
22. Nested-scoped normalized-dependency score for score-0 pairs
23. `FusionScore.non_nested_score`
24. `SIMDScheduling.can_fuse` nested bypass
25. `init_group_node`: remove fused node's own operation names from ancestors
26. `_has_intermediate_dependencies`
27. `MemoryDep.normalize_without_broadcast` in `fusable_read_and_write`
28. `_try_reindex_pointwise_for_reduction(require_vertical_fusion=True)`
29. `score_fusion_memory` name-based fallback for templates/user Triton kernels

### SIMD Codegen

30. `IterationRangesRoot.block_size`, `block_offset`, `mask_name`,
    `owns_mask`, `full_range`
31. `DerivedIterationRangesRoot`
32. `SIMDKernel.active_range_trees`
33. `SIMDKernel.use_range_trees`
34. `SIMDKernel.is_broadcasted` using active root identity
35. `prepare_indexing` singleton-loop canonicalization
36. `SIMDKernel.iteration_ranges_codegen_header` abstract hook
37. `_split_iteration_ranges` support for splitting one flat range across
    three tiled groups
38. `_decompose_index_into_ranges`
39. `_DerivedIterationFamily`
40. `_GroupReductionLayout`
41. `group_prefix = "nested_*"`
42. `construct_group_reduction_vars`
43. `make_reduced_output_family`
44. `resolve_full_resolution_load`
45. `materialize_value_at_parent_resolution`
46. `_GroupedReductionOpsHandler`
47. `_PointwiseRemapHandler`
48. `codegen_nested_reduction`
49. `missing_node1_outputs` assertion after node1 codegen
50. Explicit `kernel.codegen_body()` flush before node2 codegen
51. `_finalize_nested_reduction_kernel`
52. `_launch_kernel_and_cleanup`
53. `_codegen_nested_node2_schedule`
54. Skipping node2 `DisableReduction` / `EnableReduction`
55. `_map_iteration_values_to_node_sizes`
56. `codegen` dispatch for `FusedNestedReductions`
57. Config patch collection over node1 + node2 schedules

### Triton Codegen and Runtime

58. `TritonSymbols.get_block_size/get_block_offset` delegate to roots
59. `_mask_name_for_symbol`
60. `_is_reduction_index_symbol`
61. `IndexingOptions.has_rmask` by owning active tree
62. Dense mask construction uses `tree.mask_name()` / `tree.block_size_str()`
63. `triton_shape_dims`, `triton_shape_str`, widened `triton_reshape`
64. `TritonKernel.emit_reshape`
65. `TritonKernel.emit_reduce`
66. `TritonKernel.emit_broadcast_via_reshape`
67. Derived-root header emission in `iteration_ranges_codegen_header`
68. Derived roots always keep explicit masks
69. `min_xblock` / `min_rblock` in `inductor_meta`
70. `triton_config_reduction` min-block enforcement
71. Persistent reduction config threading of min-block metadata
72. Coordinate-descent `value_too_small` min-block enforcement
73. Occupancy fallback respecting `min_rblock`

## Pass 2: Justification and Alternatives

### 1. `_is_gpu_triton_backend`

Existing mechanism: mix-order reduction already checked GPU + Triton backend
inline.

Divergence: factored into a helper and reused by nested reduction.

Judgment: justified. This is a simplification, not a nested special case.

Alternative: keep duplicate checks in both fusion classes. Worse for review.

### 2. `NestedReduction.can_fuse`

Existing mechanism: `Scheduler.can_fuse` decides generic dependency legality;
backend `can_fuse` checks compatible iteration spaces.

Divergence: nested adds shape/codegen legality: grouped axis, group size,
resolution, pointwise remapping, tiling shape.

Judgment: justified. These are not just dependency checks.

Alternative: introduce a generic "multi-resolution fusion plan" object that
generic fusion and SIMD can query. Cleaner long-term, too large for this PR.

### 3. Grouped-reduction prefiltering inside `NestedReduction.can_fuse`

Existing mechanism: the old version had a separate one-use
`NestedReduction.is_candidate()` helper.

Divergence: the prefilter now lives directly at the top of `can_fuse()`, since
the score-0 choices bypass no longer calls it separately.

Judgment: justified simplification. There is one nested legality entry point
instead of a second public-ish classmethod.

Alternative: split it back out if another callsite needs a cheap prefilter.

### 4. `_get_grouped_reduction_info`

Existing mechanism: normal reduction lowering supports many reduction types and
multiple reductions inside a fused node.

Divergence: nested accepts only one grouped reduction, one static group size,
<=2 iter dims, and direct scalar Triton reductions.

Judgment: justified as a codegen boundary. `_GroupedReductionOpsHandler` lowers
`reshape(value)` then `tl.<op>(..., axis)` for a single CSE value. Welford,
argmax/argmin, online softmax, and multiple reductions need tuple/state/index
lowering from the normal reduction path.

Alternative: call into the normal reduction lowering after introducing a nested
axis into its iteration domain. That would be much larger and would likely need
a real multi-dimensional range-tree plan.

### 5. Same-total-numel / mismatched-rnumel gate

Existing mechanism: regular fusion usually requires compatible `(numel,
rnumel)` or known product relationships.

Divergence: nested permits different `(numel, rnumel)` only when total logical
elements match and rnumel differs.

Judgment: justified. This PR covers local dependent reductions over the same
logical data. It intentionally does not own output-size-reducing nested
reductions like `256k -> 4k -> 1`.

Alternative: support true output-size-reducing nested reductions with separate
grid ownership. That is a different feature.

### 6. Grouped-axis divisibility / quotient proof

Existing mechanism: view/reshape guards prove validity at IR level, but generic
fusion does not expose "this consumer quotient came from this parent axis."

Divergence: nested explicitly proves either `parent_axis % group_size == 0` or
`FloorDiv(parent_axis, group_size) == consumer_groups`.

Judgment: justified. Codegen uses floor division to derive reduced indices; an
unproved split could silently mis-map groups.

Alternative: carry reshape/view provenance from scheduler deps into codegen.
Cleaner, but unavailable today.

### 7. `is_group_size_in_r`

Existing mechanism: normal reduction does not need to know whether a consumer's
small dim came from X or R; it owns one iteration space.

Divergence: nested must choose whether to reshape `XBLOCK` or `RBLOCK`.

Judgment: justified. This is the minimal geometry classification codegen needs.

Alternative: have scheduler construct and store a full nested layout plan rather
than a boolean. That may be better if NVFP4/half-resolution grows this.

### 8. `_flattened_group_size_in_r`

Existing mechanism: explicit loop ranges make axis identity obvious.

Divergence: flattened consumers lose axis names, so nested recovers axis
identity from shared-read stride and divisibility.

Judgment: justified but review-sensitive. It is one of the more heuristic-looking
pieces.

Alternative: normalize the dependency/index expressions into a provenance-aware
axis map. That would likely be a generic reindexing/dependency improvement.

### 9. Group-size bound and power-of-2 gate

Existing mechanism: Triton block sizes are power-of-2-oriented, but normal
reduction can handle non-power-of-2 logical numels through masks.

Divergence: nested requires the local group size itself to be static, bounded,
and power-of-2.

Judgment: justified for landing. The number of groups may be non-power-of-2;
only the local reduction width is constrained. Current targets are FP8/MXFP8
style power-of-2 local groups.

Alternative: allow arbitrary static group sizes if Triton reshape/reduce and
autotune floor handling are proven for them. Follow-up, not needed now.

### 10. `_is_profitable_group_size_in_x`

Existing mechanism: normal fusion scoring uses memory score and generic tiling
heuristics.

Divergence: nested adds a profitability guard for group-size-in-X because it
forces `min_xblock = group_size`, which can fight coalesced R access.

Judgment: keep, but classify as policy, not correctness. This is not required
for soundness.

Alternative: simplify to "inner reductions may group R, outer reductions may
group X" if benchmarks show that catches the same cases. The current code is
more general for non-standard stride/coalescing layouts.

### 11. `_would_use_3d_tiling`

Existing mechanism: SIMD can select x/y/z/r tilings.

Divergence: nested rejects when normal tiling would use y/z.

Judgment: justified. Current nested layout assumes exactly x and r parent
trees. Supporting y/z would require derived tree construction and remapping for
3D tiled reductions.

Alternative: generalize `_GroupReductionLayout` over an arbitrary list of range
trees. That is a meaningful follow-up, not a small cleanup.

### 12. `_pointwise_nodes_match_grouped_resolution`

Existing mechanism: generic fusion checks dependency legality and compatible
iteration spaces, not semantic resolution.

Divergence: nested explicitly recognizes reduced-output and full-resolution
pointwise nodes.

Judgment: justified. The same fused scheduler node can contain `[X, groups]`
and `[X, groups, G]` bodies.

Alternative: model resolution as a first-class field on scheduler nodes. That
would reduce nested-specific checks and help NVFP4 later.

### 13. `_node2_pointwise_nodes_are_supported`

Existing mechanism: a fused reduction schedule can contain prologue/epilogue
pointwise, but all bodies are interpreted in the same codegen plan.

Divergence: nested validates that each pointwise subnode is either a producer or
consumer of the single grouped reduction and has a supported resolution.

Judgment: justified. This prevents disconnected or ambiguous pointwise nodes
from entering a multi-resolution kernel.

Alternative: require node2 to be exactly one reduction and defer all pointwise
fusion. Simpler but would lose important fused reduction cases we now test.

### 14. `_node1_fullres_epilogues_are_compatible`

Existing mechanism: normal epilogues run after the reduction with regular
range mapping.

Divergence: nested peels node1 full-resolution epilogues and remaps them into
node2's full-resolution source coordinates.

Judgment: justified. Without the check, size-compatible but differently ordered
epilogues can be mis-mapped.

Alternative: rely only on generic loop reindexing. We tried to reuse it where
possible, but node1 full-res epilogues here are emitted inside custom nested
orchestration, not by normal schedule fusion.

### 15. Full-resolution epilogue restriction for group-size-in-X

Existing mechanism: normal loop reindexing can sometimes reorder pointwise into
reduction-compatible loops.

Divergence: nested disallows downstream full-resolution consumers for
group-size-in-X.

Judgment: justified as a correctness boundary. Shape compatibility alone can
confuse `[B, D, K]` with `[B, K, D]`.

Alternative: use dependency-aware loop mapping for this path, not size-only
`_split_iteration_ranges`.

### 16. `get_node_fullres_epilogues`

Existing mechanism: `generate_node_schedule` handles prologue/epilogue ordering
inside one schedule.

Divergence: nested extracts node1 full-resolution epilogues so node1 reduction
can run first and the epilogue can run after grouped values are available.

Judgment: justified. The existing schedule cannot express "node1 body first,
then node2 grouped reduction, then node1 full-resolution epilogue consuming
node2 result" in one flat schedule.

Alternative: extend schedule markers to support multi-stage dependency domains.

### 17. `get_shared_input_reads`

Existing mechanism: generic fusion scoring works from exact dep overlap.

Divergence: nested uses buffer-name overlap, excluding node1 outputs, for axis
classification.

Judgment: justified but narrow. It is not used as a correctness proof by
itself; it helps recover flattened-axis identity.

Alternative: derive axis identity from normalized dep equivalence instead of
read-name overlap. Better long-term.

### 18. `FusedNestedReductions`

Existing mechanism: regular `FusedSchedulerNode` flattens subnodes and emits
one codegen schedule.

Divergence: nested stores `node1`, `node2`, `node2_reduction`, `group_size`,
and `group_size_in_r` because node1 owns launch/grid while node2 is staged
inside node1's tile.

Judgment: justified. This is the largest architectural special case, but it
keeps the non-standard state explicit.

Alternative: add a generic staged fused-node abstraction. That may be the right
destination if more multi-resolution fusions appear.

### 19. `FusedNestedReductions.can_fuse_with`

Existing mechanism: generic fused nodes allow more fusion through
`Scheduler.can_fuse`.

Divergence: nested only allows downstream pointwise of node2 at known
resolutions.

Judgment: justified. Arbitrary prepending/appending could place a node in the
wrong resolution or before its producer.

Alternative: disable all post-fusion extension. Simpler but loses useful
reduced-output epilogues.

### 20. `Scheduler.can_fuse` handling for `FusedNestedReductions`

Existing mechanism: special fused nodes like mix-order already override
extension behavior.

Divergence: nested follows the same pattern: delegate to the custom fused node;
do not fuse before it.

Judgment: justified and idiomatic relative to `FusedMixOrderReductions`.

Alternative: make `BaseSchedulerNode.can_fuse_with` virtual. Larger refactor.

### 21. `Scheduler.fuse` construction of `FusedNestedReductions`

Existing mechanism: backend `fuse` normally creates `FusedSchedulerNode`.

Divergence: nested constructs its custom fused node when `NestedReduction.can_fuse`
is true.

Judgment: justified because generic fused-node codegen would emit the wrong
single-resolution schedule.

Alternative: teach `FusedSchedulerNode` to carry an optional codegen strategy.

### 22. Nested-scoped normalized-dependency score

Existing mechanism: `InductorChoices.can_fuse` rejects reduction fusions with
`shared_data_score == 0`.

Divergence: before the score-0 rejection, `score_fusion_memory()` now tries
existing `MemoryDep` normalizations only for pairs that satisfy
`NestedReduction.can_fuse()` and whose exact dep score is zero.

Judgment: justified for landing. It removes the nested-specific exception from
`InductorChoices.can_fuse()` but keeps the scoring behavior scoped to nested
reduction instead of changing generic reduction fusion ordering.

Alternative: make this a generic vertical normalized-dependency score later.
That broader version needs care because reduction->pointwise cases can need
loop-reindex repair before they are safe to score as shared-data matches.

### 23. `FusionScore.non_nested_score`

Existing mechanism: fusion order uses template/type/memory/proximity scores.

Divergence: nested gets a specific ordering bit so ordinary viable fusions win
first.

Judgment: justified. It prevents nested from stealing simpler same-space
fusions. The target quant pattern typically has only one meaningful producer,
so ranking nested later is safe.

Alternative: lower nested memory score instead. Less explicit and interacts
poorly with threshold logic.

### 24. `SIMDScheduling.can_fuse` nested bypass

Existing mechanism: SIMD backend rejects mismatched `(numel, rnumel)` unless
generic compatible product cases apply.

Divergence: nested has already proven its own cross-axis compatibility and
bypasses the normal SIMD check.

Judgment: justified but review-sensitive. It is the backend mirror of
`NestedReduction.can_fuse`.

Alternative: factor the normal SIMD compatibility check to accept an explicit
fusion plan. Better architecture, much larger.

### 25. `init_group_node`: subtract own operation names from ancestors

Existing mechanism: grouped/fused nodes merge ancestors from subnodes.

Divergence: this removes the group's own operation names from the merged
ancestor set.

Judgment: justified as a generic cleanup. A fused/grouped node should not list
itself as an ancestor. Nested made this visible because it relies on ancestor
checks to distinguish producers/consumers.

Alternative: add nested-specific ancestor filters. Worse.

### 26. `_has_intermediate_dependencies`

Existing mechanism: vertical fusion rejects unmet deps through direct
dependency matching.

Divergence: after allowing one nested-style mismatch, this helper preserves the
normal "no extra intermediate producer" safety check.

Judgment: justified. It prevents the score-0/nested path from hiding unrelated
unmet deps.

Alternative: only allow nested when all remaining deps are from node1 outputs.
That is stricter but would duplicate this logic inline.

### 27. `normalize_without_broadcast`

Existing mechanism: `fusable_read_and_write` compares exact or normalized
`MemoryDep`s.

Divergence: it now also compares after removing broadcast-only dimensions.

Judgment: justified as a generic dependency fix. This directly addresses
"reduced write satisfies broadcasted full-res read" without name-only matching.

Alternative: keep nested-specific dependency exceptions. Worse and less
correct.

### 28. Reindex rollback with `require_vertical_fusion`

Existing mechanism: loop reindexing can improve shared-data score for
pointwise/reduction fusion.

Divergence: this path snapshots and rolls back unless vertical legality also
succeeds.

Judgment: justified as a safety improvement to an existing generic mechanism.
Nested benefits, but the code is not nested-specific.

Alternative: let speculative reindexing persist after score improvement. That
was too weak when vertical dependency checks still failed.

### 29. Template/user Triton name-based scoring fallback

Existing mechanism: scoring mostly uses exact dep intersection.

Divergence: template/user Triton cases can score by matching buffer names for
`StarDep`/view-like deps.

Judgment: review-risk. This is broader than nested reduction. It seems useful
for template epilogue fusion, but should be reviewed as a generic scoring
change.

Alternative: move it to a narrower template epilogue-specific scorer or teach
deps to normalize template output views.

### 30. Root block/mask/full-range helpers

Existing mechanism: many Triton paths assumed block/mask names from prefixes
like `x`, `r0_`.

Divergence: roots now own `block_size`, `block_offset`, `mask_name`, and
`full_range`.

Judgment: justified. This is the core genericization that makes derived roots
possible without string special cases.

Alternative: hardcode derived-root cases at every Triton site. Worse.

### 31. `DerivedIterationRangesRoot`

Existing mechanism: each root owns a physical loop/grid axis.

Divergence: a derived root is a lens on a parent physical loop: different
logical numel/block/offset, same loop/grid placement.

Judgment: justified. A plain root would create the wrong physical loop or
program-id ownership.

Alternative: represent the reduced index as only an expression, not a root.
That falls apart for masks, block size, active tree lookup, and stores.

### 32. `active_range_trees`

Existing mechanism: code often iterated over `self.range_trees`.

Divergence: code can now ask for only the trees active in the current stage,
especially hiding reduction trees outside the reduction loop.

Judgment: justified. It is a general cleanup made necessary by family swapping.

Alternative: every mask/index callsite manually filters reduction and derived
trees. Worse.

### 33. `use_range_trees`

Existing mechanism: one kernel had one range-tree family.

Divergence: nested temporarily swaps parent/full/reduced families.

Judgment: justified. This lets normal load/store/indexing code run unchanged
under the right resolution.

Alternative: pass a family argument through all indexing APIs. Larger and more
invasive.

### 34. `is_broadcasted` by active root identity

Existing mechanism: broadcast detection used prefix/index positions.

Divergence: it now maps symbols to owning active root by identity because
derived roots can share prefixes with parents.

Judgment: justified. Prefix matching is wrong once `r0` and `reduced_r0` both
exist.

Alternative: encode unique prefixes for all derived roots. That would break
existing SymT prefix assumptions elsewhere.

### 35. Singleton-loop canonicalization

Existing mechanism: symbolic simplification did not necessarily replace active
size-1 loop vars with zero.

Divergence: `prepare_indexing` substitutes symbols from active singleton trees
with `0`.

Judgment: justified. Batch/group size 1 must CSE loads/stores correctly, and
the replacement is semantically valid for an active singleton loop.

Alternative: add this to global sizevar simplification. Broader and riskier.

### 36. `iteration_ranges_codegen_header` abstract hook

Existing mechanism: Triton emitted range headers directly from its concrete
kernel.

Divergence: SIMDKernel exposes the hook so derived-family code can ask the
backend to emit headers.

Judgment: justified. Family code is backend-agnostic; header syntax is Triton
specific.

Alternative: make `_DerivedIterationFamily` Triton-only. Worse layering.

### 37. `_split_iteration_ranges` three-way split

Existing mechanism: split/reindex supported simple flattening patterns.

Divergence: it can split one flat pointwise range across three tiled groups for
native bmm-style `(z, y, x, r=1)`.

Judgment: keep as a generic reindexing improvement, but it is not central to
nested reduction. It should be reviewed as "normal tiled fusion reuse" support.

Alternative: reject those mappings or write nested-specific decomposition.

### 38. `_decompose_index_into_ranges`

Existing mechanism: `map_kernel_groups_to_node_sizes` returns remapped
variables, not decomposition of already chosen source values.

Divergence: nested needs to decompose source expressions back into a pointwise
node's expected body variables.

Judgment: justified and small.

Alternative: extend `map_kernel_groups_to_node_sizes` to support value mapping
directly. Could be a cleanup later.

### 39. `_DerivedIterationFamily`

Existing mechanism: kernel range-tree state is global.

Divergence: this object scopes an alternate family plus body-index substitutions
and one-time header emission.

Judgment: justified. It makes mutable kernel state explicit and scoped.

Alternative: dynamic attributes on the kernel or ad hoc context variables.
Worse.

### 40. `_GroupReductionLayout`

Existing mechanism: normal reduction layout is implicit in `(numel, rnumel)`
and range trees.

Divergence: nested creates an explicit geometry object for grouped-axis
selection, reshape shapes, output shapes, derived tree creation, and broadcast
lifting.

Judgment: justified. This is where the multi-resolution complexity belongs.

Alternative: split this logic between scheduler and handlers. That was harder
to review and caused earlier bugs.

### 41. `group_prefix = "nested_*"`

Existing mechanism: block/mask symbols are classified by names like `X` and
`R0_`.

Divergence: nested constants use a prefix that intentionally avoids SymT
block-type matching.

Judgment: justified. Plain `X_GROUP_SIZE` was liable to be mistaken for an
X-block symbol and pick up the wrong mask.

Alternative: remove all string-prefix SymT classification. Desirable long-term,
too broad for this PR.

### 42. `construct_group_reduction_vars`

Existing mechanism: normal reduction bodies receive their own iter/reduce vars.

Divergence: nested maps node2's grouped body vars onto parent x/r tree vars,
including special handling for one-group degenerate cases.

Judgment: justified. This is the core coordinate transform for the grouped
reduction.

Alternative: force node2 to be generated from a separate LoopBody designed for
parent x/r coordinates. That would duplicate LoopBody logic.

### 43. `make_reduced_output_family`

Existing mechanism: stores use the current kernel root extents.

Divergence: grouped reduction output stores use a derived root whose grouped
axis is `parent / group_size`.

Judgment: justified. It lets normal store/mask code handle reduced-output
stores.

Alternative: custom reduced store index and mask at every store. Worse.

### 44. `resolve_full_resolution_load`

Existing mechanism: loads either read memory or CSE store-cache values at the
current resolution.

Divergence: nested load resolution materializes internal store-cache values at
the resolution required by the current stage.

Judgment: justified. This is the replacement for the old custom remapped-value
cache and is now lazy/CSE-backed.

Alternative: precompute all possible broadcasts. More code and more register
pressure.

### 45. `materialize_value_at_parent_resolution`

Existing mechanism: IR-level broadcast/expand happens before codegen.

Divergence: nested sometimes needs to broadcast a register CSE value created
inside the kernel, not an IR buffer.

Judgment: justified. IR-level expand cannot see this register value.

Alternative: spill the reduced value to memory and reload with normal IR
broadcast. Correct but defeats the fusion.

### 46. `_GroupedReductionOpsHandler`

Existing mechanism: `CSEProxy.reduction` lowers reductions through the normal
reduction path.

Divergence: grouped reduction intercepts reduction, emits reshape + direct
Triton reduce, and stores through the reduced family.

Judgment: justified for current supported reductions. This is the smallest
lowering for scalar local grouped reductions.

Alternative: integrate grouped-axis support into normal reduction lowering.
Cleaner but much larger.

### 47. `_PointwiseRemapHandler`

Existing mechanism: pointwise bodies run in the kernel's current iteration
space.

Divergence: nested remaps pointwise loads/stores into reduced or full-resolution
families and can resolve loads from internal store-cache values.

Judgment: justified. It reuses normal body code while changing only the
coordinate environment.

Alternative: clone/rewrite LoopBody nodes with new symbols before codegen.
Larger and less local.

### 48. `codegen_nested_reduction`

Existing mechanism: normal SIMD codegen emits one generated schedule for one
iteration space.

Divergence: nested runs node1 codegen normally, flushes, then interprets node2
schedule under reduced/full families, then finalizes one kernel.

Judgment: justified. This is the unavoidable feature-specific orchestration.

Alternative: make generic codegen support multiple staged iteration spaces.
That is the long-term abstraction if this pattern repeats.

### 49. `missing_node1_outputs` assertion

Existing mechanism: missing internal values usually fail later during load.

Divergence: nested asserts immediately that node2/node1 epilogues reading
node1 outputs can find them in `store_cache`.

Judgment: justified as a correctness guard and diagnostic. It is not a behavior
path.

Alternative: rely on the later full-resolution load assertion. Less clear.

### 50. Explicit `kernel.codegen_body()` flush before node2

Existing mechanism: schedule markers open/close reduction loops for a single
schedule.

Divergence: nested manually flushes node1 pending code so grouped reduction
runs after node1's full-resolution values exist.

Judgment: justified. In looped reductions this separates accumulation/post-loop
normalization from the grouped consumer.

Alternative: add a generic schedule marker for "finish this stage and keep the
same kernel." Larger.

### 51. `_finalize_nested_reduction_kernel`

Existing mechanism: finalization logic lived inline in normal codegen.

Divergence: nested uses a helper to define, mark-run, launch, and cleanup.

Judgment: justified refactor. It is shared with normal codegen after cleanup.

Alternative: duplicate launch/finalization code in nested. Worse.

### 52. `_launch_kernel_and_cleanup`

Existing mechanism: normal codegen manually emitted comments, profile guards,
kernel call, checks, and buffer cleanup.

Divergence: this is factored so nested and normal codegen use the same launch
path. `free_buffers=False` preserves normal intermediate-hook ordering.

Judgment: justified. This addressed a review concern and reduces duplication.

Alternative: keep two launch paths and risk ordering drift.

### 53. `_codegen_nested_node2_schedule`

Existing mechanism: `codegen_node_schedule_with_kernel` executes schedules
directly.

Divergence: nested reuses `generate_node_schedule` for ordering but interprets
each subnode with nested handlers.

Judgment: justified. This is exactly the reuse we wanted: normal scheduling
order, custom resolution/codegen environment.

Alternative: manually classify prologue/reduction/epilogue ordering without
`generate_node_schedule`. Worse.

### 54. Skipping `DisableReduction` / `EnableReduction`

Existing mechanism: these markers close and reopen a standalone reduction loop.

Divergence: nested treats them as ordering separators only.

Judgment: justified and important. The grouped reduction is emitted as
immediate reshape+reduce inside the parent loop. Honoring node2's standalone
loop markers would close the parent loop incorrectly.

Alternative: invent nested-specific schedule markers. That would be clearer
only if we make multi-stage schedules first-class.

### 55. `_map_iteration_values_to_node_sizes`

Existing mechanism: `SIMDKernel.map_kernel_groups_to_node_sizes` maps group
sizes to body variables.

Divergence: nested wraps it to map already-chosen source values into each
pointwise node's own ranges.

Judgment: justified and good reuse.

Alternative: custom nested splitter. Worse.

### 56. Codegen dispatch for `FusedNestedReductions`

Existing mechanism: `SIMDScheduling.codegen` emits generic fused nodes.

Divergence: nested fused node dispatches to `codegen_nested_reduction`.

Judgment: justified because generic codegen would use the wrong single
iteration space.

Alternative: register codegen methods per fused-node type. Pure organization.

### 57. Config patch collection over node1 + node2 schedules

Existing mechanism: config patches are collected over the schedule being
codegened.

Divergence: nested final kernel includes node1, node1 full-res epilogues, and
node2 schedule, so config patches are collected over that combined list.

Judgment: justified. Otherwise scoped config in node2 operations could be
missed.

Alternative: ban node2 ops that need config patches. Too restrictive.

### 58. Root-owned block size/offset in `TritonSymbols`

Existing mechanism: static maps from SymT to `XBLOCK`/`R0_BLOCK` and offsets.

Divergence: lookup delegates to the owning root.

Judgment: justified. Derived roots need expressions like `R0_BLOCK // G` and
`r0_offset // G`.

Alternative: special-case derived roots at each caller. Worse.

### 59. `_mask_name_for_symbol`

Existing mechanism: infer mask names from symbol prefixes.

Divergence: prefer the owning range-tree entry.

Judgment: justified. Derived symbols do not follow regular `x`/`r0_` mask
names.

Alternative: enforce derived symbol names that still match old prefix rules.
Fragile and conflicts with SymT parsing.

### 60. `_is_reduction_index_symbol`

Existing mechanism: check SymT reduction symbol types.

Divergence: derived symbols are classified by owning root.

Judgment: justified. It prevents non-persistent derived r-indexed loads from
being hoisted out of the reduction loop.

Alternative: encode derived symbols as SymT.R0_INDEX. That would collide with
parent symbols.

### 61. `IndexingOptions.has_rmask` by owning active tree

Existing mechanism: `str(mask).startswith("r")`.

Divergence: mask ownership is checked against active range trees.

Judgment: justified. Prefix checks are wrong with derived masks and outside
reduction loops.

Alternative: keep string prefixes and add more exceptions. Worse.

### 62. Dense mask construction by root methods

Existing mechanism: dense masks used `tree.prefix` and `tree.prefix.upper()`.

Divergence: dense masks use `tree.mask_name()` and `tree.block_size_str()`.

Judgment: justified genericization. Normal roots produce the same strings as
before.

Alternative: duplicate derived-root dense mask logic.

### 63. Shape formatting helpers

Existing mechanism: `triton_reshape` expected sympy shapes and called
`index_to_str` itself.

Divergence: helpers accept mixed sympy/int/pre-rendered string dimensions.

Judgment: justified. Nested shape constants like
`nested_R0_REDUCED_BLOCK` are already rendered Triton symbols.

Alternative: pass only sympy expressions for nested constants. That was harder
because some dimensions are Triton constexpr names by construction.

### 64. `emit_reshape`

Existing mechanism: codegen sites could emit raw Triton strings through CSE.

Divergence: TritonKernel exposes a structured reshape emitter.

Judgment: justified. It centralizes shape formatting and CSE shape metadata.

Alternative: raw strings in `_GroupedReductionOpsHandler`. Worse.

### 65. `emit_reduce`

Existing mechanism: normal reduction lowering owns reduction emission.

Divergence: grouped handler needs one local reduce after reshape.

Judgment: justified for simple reductions.

Alternative: route through normal reduction lowering. Larger, but worth
considering if supported reduction types expand.

### 66. `emit_broadcast_via_reshape`

Existing mechanism: Triton has no lazy IR-level broadcast for register CSE
values.

Divergence: helper emits reshape -> broadcast_to -> reshape and CSEs it.

Judgment: justified. It dedupes repeated broadcasts through existing CSE.

Alternative: custom nested broadcast cache. Already removed, worse.

### 67. Derived-root header emission

Existing mechanism: regular root headers emit index and mask based on prefix.

Divergence: derived headers emit nested group constants, reduced index, and
derived mask.

Judgment: justified. The header belongs where other range headers are emitted.

Alternative: emit derived headers from nested codegen next to the grouped
reduction. That made generated code order worse and duplicated backend logic.

### 68. Derived roots always keep explicit masks

Existing mechanism: constant-mask optimization removes masks when block divides
numel.

Divergence: derived roots skip constant-mask optimization.

Judgment: justified. Their block sizes are autotuned expressions like
`R0_BLOCK // G`, not fixed config keys.

Alternative: teach constant-mask analysis about derived block expressions.
Small follow-up if performance warrants it.

### 69. `min_xblock` / `min_rblock` metadata

Existing mechanism: kernel features flow ordinary inductor metadata to runtime.

Divergence: nested adds block-size floors.

Judgment: justified. The grouped axis must contain at least one whole group.

Alternative: filter choices only in codegen. Insufficient because runtime
autotune and coordinate descent can still mutate configs.

### 70. `triton_config_reduction` min-block enforcement

Existing mechanism: config generation clamps to size hints and target block
products.

Divergence: it also applies explicit minimum X/R block constraints.

Judgment: justified correctness plumbing for grouped reshape legality.

Alternative: generate only one fixed config for nested. Worse perf and still
needs tuner guards.

### 71. Persistent config threading

Existing mechanism: persistent configs call `triton_config_reduction` from
several branches.

Divergence: each branch passes min-block metadata.

Judgment: justified. Missing one path makes autotuning select illegal configs.

Alternative: apply min-block filtering after config generation. Possible
cleanup, but current threading is explicit.

### 72. Coordinate-descent min-block enforcement

Existing mechanism: tuner already rejects too-small blocks for native matmul.

Divergence: tuner now also rejects nested min X/R blocks.

Judgment: justified. Coordinate descent mutates configs after generation.

Alternative: disable coordinate descent for nested kernels. Too restrictive.

### 73. Occupancy fallback respecting `min_rblock`

Existing mechanism: autotuner may shrink the largest R block for occupancy.

Divergence: it refuses to shrink `R0_BLOCK` below `min_rblock`.

Judgment: justified. Otherwise a legal generated config can become illegal
after occupancy fallback.

Alternative: rerun the min-block validator after every config transform. More
generic, possible cleanup.

## Remaining Items That Could Become Follow-Ups

1. Broaden the nested-scoped normalized score into generic dep equivalence for
   reshape/broadcast-related vertical deps without skipping needed reindex
   repairs.
2. A first-class multi-resolution or staged-fusion plan. This could reduce the
   custom fused node and SIMD backend bypass.
3. Dependency-aware pointwise resolution mapping. This could enable
   group-size-in-X full-resolution epilogues.
4. A generic block-size-floor abstraction in autotuning. This could make
   `min_xblock` / `min_rblock` feel less nested-driven.
5. General 3D/y/z nested layout support. This would remove `_would_use_3d_tiling`.
6. Shape/provenance-aware grouped-axis classification. This could replace the
   stride-based flattened fallback.

## Things Already Simplified

1. Manual nested internal buffer removal was deleted. Nested now uses normal
   store bookkeeping and `Kernel.remove_kernel_local_buffers()`.
2. `kernel.post_loop_combine.clear()` and `kernel.post_loop_store.clear()` were
   removed from nested codegen.
3. `max_xblock` and the `1048576` cap were removed.
4. `supports_constant_mask()` was removed; Triton directly treats derived roots
   as requiring explicit masks.
5. The custom remapped-values cache was removed; CSE/store_cache handle reuse.
6. Full-resolution pointwise remapping now reuses
   `SIMDKernel.map_kernel_groups_to_node_sizes` rather than a nested splitter.
7. The nested-specific score-0 bypass in `InductorChoices.can_fuse()` was
   replaced with nested-scoped normalized scoring in `score_fusion_memory()`.
8. One-use `NestedReduction.is_candidate()` was inlined into `can_fuse()`.
