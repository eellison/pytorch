# Sub-parent epilogue stack - current follow-ups

Last reconciled: 2026-08-27, against
`agent_space/followup_indexed_rebase_wt`, with the F2a compatibility prototype
in `agent_space/f2a_option_c_wt`.

## Current open items

### 1. General delayed group arithmetic and CSE

Exact per-read fusion proof and the measured NVFP4 cast-before-broadcast case
are complete. Codegen collapses the exhaustive final plan to a validated
per-name replay contract. The current narrow rule keeps only an unguarded, lane-free
group-width source through `to_dtype`, then immediately materializes it at
child resolution. It deliberately keeps masked sources and every other
operation eager.

Generalize only for a concrete workload that benefits from keeping additional
arithmetic, comparison, selection, or inline-asm results at group resolution.
Each extension must be domain-aware, fail closed through ordinary
`CSEVariable.shape`, reuse the exact access relation, and carry a source-form
test plus paired performance evidence. Do not restore a broad allowlist or
masked callback wrapper speculatively.

Parent-width divide-before-split was also prototyped and rejected for now. It
improves a forced-persistent `4096x4096` kernel by `1.273x`, but fresh default
`4096x512` and `4096x1024` kernels are unchanged because their `XBLOCK=1/2`
launches already have zero spills. The prototype costs net `+177` production
lines and its main helper is CC 26. Revisit only when a default-selected
persistent workload gains at least 5%, starting from the exact final-scale
relation rather than reviving general group-expression handling. See
`agent_space/parent_divide_split_complexity_audit_20260827.md`.

Final narrow design and measurements:
`agent_space/lazy_cse_f2a_results_20260826.md`.

### 2. Strict parent-to-grouped plan records

Replace the inherited `_fusable_read_after_index_equivalence` branch only with
a source-anchored X/R-frame proof that can be rebuilt after loop merging. A
first attempt based on generic normalized/broadcast equality was reverted: it
could erase axis identity and disabling the legacy path lost more than 20
existing fusions.

The replacement must record raw temporal `MemoryDepMatch` pairs, reject
transposed/equal-size crossed axes and ambiguous or out-of-order writers, and
preserve the existing nested fusion corpus before deleting the legacy helper
and ownership branch. See
`agent_space/indexed_relation_adversarial_review_20260826.md`.

### 3. Dynamic padded scale preshuffle

Aligned scale preshuffle is complete in #191775: it is a parent-stage permuted
store and fuses with the `(4,3)` pack. Dynamic padded layouts still require an
explicit scatter/padding model similar to #193599; do not assume that PR lands.

The follow-up must keep auxiliary padding writes separate from sub-parent
epilogue mutation checks, retain masked tail stores, and prove the logical to
physical padded mapping. A generic partial inverse returning
`(logical_index, is_valid)` remains an alternative only if it produces equally
good one-kernel code.

Current DCN analysis: `agent_space/mxfp6_43_dcn_followup.md` and
`agent_space/mxfp6_native_pack2_final_validation_20260826.md`.

### 4. Generalize proven domains

- Add sub-parent rates beyond `(2, 1)`, `(4, 1)`, and `(4, 3)` only with
  end-to-end legality, codegen, and kernel-form coverage.
- Support dynamic MXFP6 feature extents once consumer reindexing can retain the
  required read/write dimensions.
- Support dynamic or non-power-of-two contiguous reduction extents only after
  defining their mapping onto padded persistent blocks.
- Retain dimension provenance through dependency normalization so additional
  shared broadcasts and multiple equivalent parent accesses can be proved
  without name-only matching.

### 5. Shared temporal dependency analysis

`_order_sub_parent_parent_nodes` currently rebuilds writer maps, real reads,
and transitive producer/consumer closure locally. Reuse a shared temporal
producer/user DAG when scheduler planning exposes one. WeakDeps must continue
to provide ordering without being treated as values.

### 6. Rollout and targeted coverage

- Before enabling nested reduction by default, profile repeated plan
  construction inside the fusion loop. The flag-off path must remain cheap.
- Keep both fuzzing modes: replay-based differential fuzzing protects the
  established kernel corpus, while shape-generating fuzzing explores new graph
  structures and fallback paths.
- Consider a checked-in SM100 native E2M3 plus scale-swizzle integration test;
  current scheduler behavior is pinned generically and the native composition
  is validated in scratch.
- Retain broad-GPU FP8 split/broadcast coverage as hardware becomes available.
- Consider explicit multi-output pointwise IR if one-instruction tuple outputs
  begin losing CSE when their consumers land in different derived passes.

### 7. Consider broader access identity only for a concrete need

`SubParentAccessRelation.source_accesses` is plural only for parent-lane
relations. Target kernels genuinely produce multiple raw witnesses for one
logical source: a flat `[B, R]` access and a grouped `[B, G, L]` access can map
to the same explicit parent-frame index. The planner retains both so raw-frame
fusion and X/R-boundary checks do not discard either representation. Internal
and reduced-broadcast relations are always singleton.

F1 deliberately does not choose a canonical runtime access. Fusion retains and
checks the exact per-read relations. After codegen rebuilds that exhaustive
plan, the specialized resolver validates that each buffer has one coherent
source set, liveness role, and direct-or-lane behavior, then captures values by
name. It derives and validates the actual lane from the replay load index.

There are two possible longer-term designs:

1. Let codegen independently match indexed accesses, following the ordinary
   CSE model. A completed prototype did this correctly, but reconstructing
   scheduler `MemoryDep` identities from the active FX interpreter was fragile
   and leaked scheduler representation details into codegen.
2. Have fused scheduler nodes retain the exact read/write matches established
   during fusion, then derive forwarding from those internal edges. Today
   `FusedSchedulerNode` merges external dependency summaries and discards
   satisfied internal edges; `FusedStagedReduction` adds no match state, and
   nested append planning rebuilds rather than retains the prior plan.

Do not pursue either alternative without a concrete case that the contained
name-keyed replay contract cannot represent. If option 2 is pursued, preserve
raw planner witnesses on the fused node and define their lifetime across plan
rebuilds explicitly.

## Recently closed

- Fusion now requires a planned group-width broadcast relation to remain valid
  after the dependency normalization performed by `merge_loops()`. A masked or
  padded relation whose frames diverge declines early to the ordinary
  two-kernel fallback instead of failing during codegen replanning.
- Exact per-read access relations now authorize sub-parent fusion. The old
  `SubParentSourceLayout` policy and name-based `source_layouts`,
  `broadcast_source_names`, and `internal_dependency_names` views are gone.
  Codegen validates and collapses their consequences per source name, derives
  replay lanes from actual load indices, reloads optional external sources on a
  CSE miss, and fails loudly for missing in-kernel sources.
- NVFP4 now converts the group-width FP8 scale to FP32 before broadcasting it.
  The rule is intentionally narrow: unguarded lane-free source, typed
  `to_dtype`, then immediate target-shape materialization. It improves the
  12-row NVFP4 matrix by `1.301x` geometric mean while leaving MXFP4 and the
  protected MXFP6/source corpus unchanged.
- Exact dependency records replaced broad name-level fusion authorization.
- Derived-family mask propagation covers direct, split, broadcast, and
  reshaped values, including indirect indexing and masked tails.
- Looped internal sources are scheduled into the final reduction pass through
  `generate_node_schedule`.
- MXFP6 `(4,3)` packing, native `pack=2` conversion, and aligned DCN scale
  preshuffle produce one competitive kernel.
- The standalone staged append requires an already formed
  `FusedStagedReduction`, with exact/shifted regressions.
- The unreachable reverse reduction/pointwise retry and its invalid-order mock
  test were removed.

## Historical detail

The remaining sections preserve earlier investigation context. Their status
claims are superseded by the checklist above; consult the dated design/review
documents before reviving an older proposal.

## Structural

### Planner entry points remain separate

Standalone planning starts with one reduction plus pointwise consumers. Nested
planning starts with a dependent reduction pair. They share the stage types,
pointwise compatibility helpers, source-layout proof, lane algebra, and
codegen. Combining the two remaining entry funnels is worthwhile only if it
deletes a real legality boundary; a larger generic hierarchy would not help.

### Source matching is intentionally partial

Both layouts use `_try_get_sub_parent_source_layouts` and
`MemoryDep.normalize_with_ranges`. The latter cannot recover the positions of
nontrivial dimensions dropped during dependency normalization, so shared
broadcast sources conservatively decline. The direct TODO is to retain that
dimension provenance during normalization, not to add another scheduler-side
matcher.

### Final plans are rebuilt after fusion

This is intentional. `merge_loops` can change ranges, so codegen derives the
final `StagedReductionPlan` from post-fusion nodes. The stable contract across
phases is `FusedStagedReduction` identity, not a cached pre-merge plan.

## Contracts and diagnostics

### Output-group ordering rejects silently

Candidate output-lane counts must already be sorted before `groupby` creates
`SubParentEpilogueStage.output_groups`. Reordering could move stores, so the
planner declines instead. The rejection has no dedicated test or log.

### Multiple parent accesses remain conservative

The plan currently requires one normalized parent access per shared source
name. Generalizing this should reuse the structural index cache work from
#188180; name-only matching would be unsafe. This is a lost-fusion boundary,
not a correctness risk.

The completed forwarding layer keeps exact load identity in planning, not in
runtime codegen lookup. Planning retains each proved source-to-consumer access
relation; codegen rebuilds the final plan, validates one coherent replay
contract per name, and derives the actual lane from the replay index. It:

- retains every exact source access that normalizes to the one proved parent
  index, while distinct parent indices still decline fusion;
- removes layout-policy buckets and uses latest-writer semantics for values
  produced during replay; and
- gives standalone and nested sub-parent stages the same source-resolution path.

Ordinary CSE invalidation remains authoritative. A planned external relation
may fall back to the normal derived-index load when its live source is absent;
a required in-kernel relation fails loudly. Masked sources are not captured,
and a masked consumer with a concrete fill uses the physical load path.

### Give inlined values explicit cross-domain identity

The completed narrow follow-up keeps an exact unguarded group-width source live
through one `to_dtype`, then immediately broadcasts the converted value. This
removes the common NVFP4 FP8 bitcast/broadcast round trip without introducing a
second value-identity or projection cache. Masked sources and all other
operations remain eager.

If a measured workload needs a longer group-width chain, extend the typed
operation boundary one operation at a time. Do not use FX origins or force a
memory realization solely to connect unnamed values. The single-FP8-conversion
kernel-form test remains the tripwire for lost reuse.

## Test gaps

- The nominal rank-1 contiguous permute path is not covered. A 1-D input and
  `B=1` both retain a rank-2 `[1, R0_BLOCK]` kernel tile, so reachability is
  unclear.
- There is no direct negative test for output-group ordering.
- Broad-GPU fp8 split/broadcast coverage remains limited; the strongest cases
  are in the nested-reduction kernel-form tests.
- MXFP6's repeating flagship input is weaker than random or group-asymmetric
  data against group off-by-one errors, although exact eager comparison and
  preshuffled tests cover the implemented path.
- Dynamic MXFP6 feature extent falls back because consumer reindexing requires
  static read/write dimensions. Dynamic batch is covered and fuses.
- Dynamic or non-power-of-two CONTIGUOUS reduction extent falls back. Supporting
  it requires defining how logical lanes map onto a padded persistent block;
  external looped reloads alone do not settle the persistent case.

## Performance deferred

The sub-parent kernels no longer impose a minimum XBLOCK. The autotuner chooses
the row count; casts do not justify a legality floor. Cooperative reduction is
important for the largest NVFP4 shapes.

Before enabling `nested_reduction` by default, profile the repeated standalone
and append planning in the fusion loop. The current flag-off path exits before
symbolic work, so this is not a landing blocker.

The internal-CONTIGUOUS persistence heuristic is consulted again when codegen
rebuilds the post-`merge_loops` plan. No phase-dependent flip is known; if one
is found, retain the approved persistence requirement on the staged node rather
than duplicating the heuristic inputs.

### Generalize index inversion to padded scale swizzles

Ordinary equal-size scale swizzles can fuse through loop index inversion. A
padded blocked layout such as D115935267 is not bijective: physical padding
positions have no logical scale input, and the physical output is larger than
the reduced scale domain. The current inversion path therefore cannot replace
the explicit padded scatter.

Investigate a partial inverse that returns `(logical_index, is_valid)` and a
derived output domain that can emit the logical value for valid positions and
a constant for padding. If that generic path produces equivalent one-kernel
code, remove the dedicated padded-swizzle implementation. This is not a blocker
for landing the explicit implementation first.

### Canonicalize mixed complex/component loads

The abandoned local prototype `e70e8f88fe5` combined full-lane interleaved
reads only after planning proved that they used the same logical source buffer.
A kernel that accesses the same storage both as complex values and through real
components may instead present different dtypes, views, names, or byte offsets
and miss that proof.

Investigate canonicalizing those accesses to one aligned parent-resolution
load, then serving complex and real/imag consumers from register splits. The
proof must operate on the underlying byte range, preserve storage offsets,
masks, mutation versions, and loop-local CSE lifetime, and retain narrower
loads when not all components are consumed. Confirm the actual
post-decomposition IR and PTX/SASS before choosing an implementation. This is
a separate load coalescing optimization, not a blocker for the staged-reduction
stack.

Upstream realization can still duplicate elementwise work. A five-`exp` versus
one-`exp` experiment measured no change at `512x1024` and about 2.4% at
`8192x4096`. That is a global realization-policy question, not staged-codegen
logic.

## Closed by this stack

- Dynamic and non-power-of-two interleaved reduction extents.
- Real runtime dynamic-shape tests, not only `mark_dynamic` on one input.
- Looped and persistent standalone, nested, contiguous, and MXFP6 codegen.
- Mandatory scheduler gates for nested append fusions.
- Producer-first ordering for reverse-discovered nested pairs.
- StarDep and WeakDep preservation under index-equivalent relaxation, with
  dense/injective producer-write and MemoryDep-only scoring checks.
- Dependency-safe internal-source reordering plus a positive full-resolution
  fork in looped and persistent modes, including a three-pass looped case with
  a reduction in the deferred source chain.
- One kernel context keeps CSE invalidation and store accounting aligned with
  actual staged emission; no post-hoc store-counter repair remains.
- Derived epilogue accesses participate in index-width analysis without
  affecting the parent tiling schedule.
- Common planner/codegen contiguous-lane formula.
- Lane-asymmetric factor-8/16 CONTIGUOUS numerics and a factor-8 INTERLEAVED
  rejection.
- Multi-output CONTIGUOUS and mixed per-buffer source layouts in both kernel
  forms.
- Symbolic CONTIGUOUS reduction extents decline through one dynamic two-kernel
  fallback graph across multiple runtime sizes.
- Kernel-lifetime named constants emitted in the prologue.
- Standalone stages no longer force broad parent-buffer materialization;
  nested stages retain only explicit reduced/broadcast source names.
- Nested stage planning preserves node-local temporal buffer versions instead
  of collapsing them through the graph-final mutation map.
- Nested append relaxation proves every ordinary grouped-output read against
  its producer write; a matching buffer name or derived-domain classification
  is not sufficient.
- The grouped-axis boundary is retained in nested sub-parent domains.
- Grouped-axis X is directly tested to have no sub-parent domain.
- Recognized ghstack trailers on every commit.
## Represent multi-output pointwise results explicitly

`inline_asm_elementwise` lowers tuple outputs to separate Pointwise nodes and
reunifies the underlying asm through Triton kernel CSE. This is correct, but
sharing can be lost when scheduling places the outputs in different derived
passes. Consider a native multi-output pointwise/shared-result IR node so the
single-invocation contract does not depend on realization and CSE lifetime.

## Derived-domain projection design record

Successor designs for the sub-parent projection/forwarding machinery
(delayed projection, divide-then-split, structured domains, indexed
forwarding, unnamed-value identity) are consolidated with measurements and
three review iterations in
`agent_space/derived_domain_projection_proposals.md`. Read its status table
before starting any related follow-up.

### Make staged plans authoritative for parent-to-grouped dependencies

During the indexed-forwarding rebase (F1), record the inherited
parent-to-grouped nested-reduction relations as exact planner-owned source and
consumer accesses. Then remove the grouped-stage
`_fusable_read_after_index_equivalence` exception from
`_prove_staged_fusion_dependencies`, leaving strict matching or exact plan
membership as the only staged-fusion paths.

Reuse the existing parent/grouped equivalence proof inside planning rather than
re-deriving it in fusion legality. Require mutation-style coverage for each
relation family: removing a required record must fail a focused one-kernel test,
not silently reduce fusion. The complete alternatives and acceptance criteria
are in `agent_space/pr191775_nested_equivalence_paths.md`.
