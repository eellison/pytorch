# Indexed forwarding and lazy projection acceptance matrix

Date: 2026-08-26

This is a release gate for the follow-ups after the frozen #191775 tree at
`agent_space/pr191775_layered_split_wt`. It is intentionally adversarial. A
green happy-path test is not sufficient if a missing relation, stale value, or
same-name access can still be accepted.

The frozen baseline is the layered worktree content, not clean `HEAD` alone.
At measurement time its `HEAD` was `6b8ef64bd373`; the staged binary diff hash
was `8b841b64605acd39e0cd3e3520b4842dccd197ded41db206b3437202bc29fb4f`
and the unstaged binary diff hash was
`8a7f5eca32cfa865f4ea784b2b23858fd359bf7e66d72c2f5ea6b4e0c1e451c0`.
Recompute those hashes before an A/B; a mismatch means recapture the baseline.

No GitHub action is part of this plan. All work remains local and uncommitted
until a human has reviewed the exact diff and any text intended for GitHub.

## Required layering

1. **F1: exact indexed forwarding.** Change the identity and lookup mechanism,
   but preserve accepted graphs and generated kernels. Exact per-read records
   become the sole authority for sub-parent forwarding and codegen. The
   inherited parent-to-grouped scheduler proof remains a scoped exception.
2. **F2a: group-only lazy broadcast/CSE.** Starting only from an F1-resolved
   group-width value with no lane selection, keep scalar scale work at group
   width and widen at the first lane-varying operation or barrier. This is the
   narrow replacement for the former `_GroupInvariantBroadcast` behavior.
3. **F2b: optional persistent factor-2 parent projection.** Move a proved
   parent/lane operation before one split only if a separate implementation and
   benchmark review earns it. F2b must not be bundled into F2a.

**P2G: source-anchored parent-to-grouped plan ownership** is a named later
follow-up. It is not part of F1 or a prerequisite for F2a.

F1 fails review if the new indexed mechanism coexists with the old
layout/name-based forwarding scaffolding. F2a fails review if it grows back
into the historical general `_DerivedDomainProjection` design without meeting
the separate F2b gates.

## Core invariants

- Planner records use the temporal dependency names present in the node-local
  `MemoryDep`s. `mutation_renames` is applied only when comparing a producer
  write with a consumer read at fusion time.
- A relation identifies an exact consumer access and its exact permitted source
  access or equivalent source alternatives. A bare buffer name never selects a
  projection.
- The plan is rebuilt after fusion and loop merging. No relation object or
  normalized index from an earlier loop state is reused.
- Every same-name sub-parent consumer read must have an exact plan relation. A
  missing consumer relation is a planner decline or compiler error, never an
  implicit reload. For an existing planned relation, a missing in-kernel
  source is a compiler error; only an unavailable external source may use an
  ordinary physical load at the remapped consumer index, bypassing name-only
  store forwarding.
- Synchronizing stores, TMP/indirect dependencies, non-injective writes, and
  ambiguous reaching writes remain fail closed.
- Source lookup owns mask/fill semantics. Projection materialization owns the
  target shape and must assign masks from the target iteration family.
- CSE liveness remains authoritative. No staged cache or lazy state crosses a
  `codegen_body()` or `cse.invalidate()` boundary.
- Correctness may not depend on expression CSE. F2 performance does depend on
  CSE finding the repeated group-width scalar chain, so generated-form tests
  must make loss of that reuse visible.

## Permanent-test budget

The checked-in suite should remain small. Reuse existing scheduler and nested
reduction fixtures, parameterize adversaries that exercise one contract, and
keep generated kernels as assertions on existing integration tests. The target
is at most four new test methods across the two files:

1. one parameterized exact-access resolver test covering distinct indices,
   external planned-source reload, required in-kernel miss, stale CSE, and a
   poisoned same-name `store_cache`;
2. one parameterized guard test covering exact guard, the single allowed
   unguarded-to-guarded transition, and mask/fill mismatches;
3. one ownership-scope test proving that sub-parent reads require exact plan
   membership while only inherited grouped-stage reads may reach the retained
   parent-to-grouped helper; and
4. one F2a allowlist shape-contract/barrier test covering every allowed op and
   the unknown/default barrier.

Extend the existing unsafe-write test for atomic/TMP/multiwrite cases, and add
conversion, reciprocal, masks, splits, loads, stores, and loop-count assertions
to the existing functional tests. The large differential matrix, semantic
fuzz, local mutants, kernel-hash corpus, and performance sweeps below are
scratch-only acceptance artifacts; they are not a reason to add one permanent
test per matrix row.

## F1 correctness matrix

| ID | Bug class / adversary | Required fixture | Passing result |
| --- | --- | --- | --- |
| F1-01 | Temporal name is canonicalized too early | Unit plan with distinct temporal producer/read names plus a `mutation_renames` mapping | Stored relation retains both original names. A copied/renamed pair matches only inside fusion proof; the plan and codegen keys are unchanged. |
| F1-02 | Mutation rename collision | Two temporal versions that map to one final name | Multiple reaching writes decline. Neither version can overwrite the other's exact source-cache entry. |
| F1-03 | User-visible mutation | Run both `case_mutated_scale_reader` and `case_source_mutated_between` patterns, plus existing `test_*rejects*_mutation` cases | Exact outputs and mutated inputs; current mutation-bearing staged candidates still decline. F1 does not silently broaden mutation support. |
| F1-04 | Atomic/synchronizing write forwarded as its post-store value | Parameterize every currently supported non-`None` store mode, including `atomic_add` | Planner relation is rejected and the codegen recorder does not record the update operand. `store_reduction` with ordinary mode remains recordable. |
| F1-05 | TMP, indirect, StarDep, or WeakDep treated as a value relation | Extend `test_planned_dependency_matches_reject_unsafe_write` and the non-MemoryDep fixtures | No relaxed match. Ordering-only `WeakDep` remains ordered; `StarDep` is not converted into a value relation; a `MemoryDep` classified as TMP/symbolically indirect declines. This does not reject a proved staged value merely because a later pointwise op performs an indirect gather. |
| F1-06 | Ambiguous same-name writes | One exact write plus a second shifted write of the same temporal name | Fusion declines even if one write would otherwise match. Do not choose first/latest by iteration order. |
| F1-07 | Same name, different consumer indices | One node performs two reads of one name at distinct normalized indices; first plan both, then remove one relation in a local mutant | With both relations, each read resolves only its exact source. With one omitted, planning declines or compilation raises for the unaccounted staged read, whether the buffer is external or internal. It never reloads merely because its relation is missing, and never receives the other read's value. |
| F1-08 | Consumer relation collision | Construct two incompatible relations for the same normalized consumer access | Plan/store construction raises; last-writer-wins dictionary behavior is forbidden. |
| F1-09 | Equivalent source alternatives | Two parent accesses normalize to the same proved parent index, then repeat with one shifted index | Equivalent alternatives are retained and the live one may resolve. A distinct normalized parent index declines planning. No Cartesian product may authorize an unproved source/consumer pair. |
| F1-10 | Planner/codegen frame drift | Sweep nested/standalone x persistent/looped and compare each emitted `load`/`store` logical access with its planned access | Every expected access is observed and equal after the one documented normalization. Missing, duplicate, wrong-op, wrong-name, wrong-mode, or wrong-index extraction raises. |
| F1-11 | P1 transpose or X/R-boundary erasure | Existing nested and standalone transposed-frame tests plus the original P1 attack harness | The adversarial graph declines or leaves the unsafe consumer outside the staged kernel. Positive row-major controls still fuse. |
| F1-12 | Planned external source unavailable | Retain the exact consumer relation to an external source, force only its live source CSE entry absent, and seed generic `store_cache` with a wrong same-name value | Codegen calls the physical load path at the planned remapped index, increments load accounting, and does not consult the same-name store-cache entry. This fallback is unavailable when the consumer relation itself is missing. Numerics match eager. |
| F1-13 | Required in-kernel exact-source miss | Retain a planned relation to a kernel-written value and delete, expire, or overwrite the recorded exact source | Compilation raises a targeted `lost required source` error containing the consumer/source access. It must not reload a removed buffer or fall through to generic CSE. |
| F1-14 | Stale CSE value | Record a source, invalidate it with the same boundary used by `codegen_body()`, then resolve | Lookup misses. Any materialized memo tied to the old source is cleared or unusable. Reusing the stale variable is a hard failure. |
| F1-15 | Mask/fill collision | Matrix: exact guarded source; unguarded source to guarded consumer; guarded to unguarded; different mask; different fill; `fill=None`; boolean fill; `+0.0` versus `-0.0` | Exact guard matches resolve. Unguarded to guarded is allowed only when an explicit consumer `where(mask, value, fill)` is emitted. All other mismatches miss. Fill identity must be typed/bit-preserving, not Python equality alone. |
| F1-16 | Source mask copied onto a differently shaped result | Materialize direct, broadcast, and lane-split values with parent and tail masks | Each result's `mask_vars` is assigned from `family.mask_vars_for_shape(result.shape)`. No source-mask union/copy remains. |
| F1-17 | Parent-to-grouped exception escapes its scope | Instrument the existing inherited grouped-stage positives and a sub-parent read with the same normalized shape | `_fusable_read_after_index_equivalence` is reachable only for `nested_stage` reads after strict and sub-parent exact checks. It never authorizes a sub-parent forwarding relation and is never consulted by codegen. Preserve the existing positive fusions and safety gates. |
| F1-18 | Ambiguous stage ownership | Rework `test_nested_dependency_matches_scope_index_equivalence` for `nested`, `both`, and `none` ownership | `nested` may use the scoped inherited helper; `both` takes the stricter sub-parent exact-plan path; `none` declines. |
| F1-19 | Looped pass-boundary suppression is too broad | Existing two-pass and three-pass internal-source tests plus a reduced/IDENTITY-only and BROADCAST-only case | Only an internal source with a lane selection may keep the parent pass open. Direct/group values do not suppress the flush. Loop counts remain 2, 2, 2, and 3 for the four existing regressions. |
| F1-20 | Required source produced in the wrong pass | Move a lane source before the reduction or make its producer chain feed a later reduction | Planner reorders the closed dependency chain into the final pass or declines. It never forwards a value from an earlier RBLOCK iteration. |
| F1-21 | Standalone/nested divergence | Run the same external, parent-written, reduced-broadcast, and internal-identity patterns through both entry points | Both paths use the same exact resolver and miss policy. Differences are limited to their already-proved schedules, not lookup semantics. |
| F1-22 | Degenerate shape classification | `B=1`, `G=2`, rank-1-looking values, and scalar side inputs | Direct, group-width, and lane-selecting relations remain unambiguous. Unknown/shape-less internal values fail closed. |

### F1 structural deletion gate

The following must be absent from the staged codegen contract after F1:

- `NestedReduction.SubParentSourceLayout`;
- `SubParentEpilogueStage.source_layouts`;
- `SubParentEpilogueStage.broadcast_source_names`;
- `SubParentEpilogueStage.internal_dependency_names`;
- `_SubParentSourceLoadResolver`;
- `_DerivedIterationFamily.remapped_values` and its name-based `resolve_load`;
- `forwarded_store_names` and `masked_forward_names` plumbing;
- a staged codegen fallback to `kernel.cse.store_cache[name]`.

Local name sets used only to discover candidates are allowed. No name-derived
classification may cross the planner/codegen boundary. A grep-clean result is
necessary but not sufficient; the sub-parent relation-removal mutations below
must also fail. The scheduler may retain one explicitly scoped call to
`_fusable_read_after_index_equivalence` for inherited `nested_stage` reads.
That exception is not forwarding scaffolding and may not gain a codegen caller.

### Deferred P2G follow-up

Deleting the parent-to-grouped helper is not an F1 gate. Disabling it lost more
than 20 existing fusions, while the first attempted relocation accepted
transpose/equal-size axis swaps and could not reconstruct its relation after
post-fusion loop merging.

P2G must instead build a source-anchored frame proof from the final producer
write and grouped consumer read after plan reconstruction. It must explicitly
prove X/R axis correspondence, record the exact relation in the rebuilt plan,
preserve all existing positive fusions, and only then delete
`_fusable_read_after_index_equivalence` and its broadcast helper. Its mutation
suite must remove reshape and broadcast records, swap equal-size axes,
transpose/regroup indices, reuse a pre-merge record, and inject TMP,
synchronizing, and multiwrite dependencies. Every unsafe mutant must decline;
every removed required record must lose fusion or fail loudly.

## F1 functional and kernel-form matrix

Every row runs in persistent and forced-looped mode unless marked otherwise.

| Relation/topology | Existing anchor | Required result |
| --- | --- | --- |
| Nested INTERLEAVED | `test_producer_consumer_rmsnorm_interleaved_pair_epilogue`, NVFP4/MXFP4 tests | Exact numerics, one staged kernel, unchanged input load/store counts and normalized kernel hash. |
| Standalone INTERLEAVED | `test_standalone_sub_parent_epilogue`, `test_standalone_nvfp4_inline_asm` | Exact numerics, one staged kernel; persistent has one factor-2 split and looped has no split where currently pinned. |
| Nested BROADCAST | `test_producer_consumer_broadcasts_outer_reduction_output`, shifted reduced-source negative | Exact read fuses; shifted group read does not. |
| Standalone BROADCAST | `test_standalone_sub_parent_allows_reduced_sibling_source`, shifted reduction-output negative | Exact reduced sibling resolves through the indexed path in both modes; shifted access declines. |
| IDENTITY/internal | `test_producer_consumer_sub_parent_intermediate`, MXFP6 internal source | Forward exact store/read access without a load; shifted, transposed, consumer-before-writer, and duplicate-writer cases decline. |
| External reload | `test_independent_sub_parent_source` and nested equivalent | A dead/missing parent value is reloaded at the derived index with no output-buffer reload and no same-name store-cache hit. |
| Dynamic | `test_dynamic_sub_parent_epilogue`, `test_dynamic_standalone_sub_parent_epilogue` | One compiled graph covers every runtime shape; exact relation matching survives symbolic extents. |
| Masks | both indirect-index-mask tests and mismatched-pad-fill test | Exact numerics; gathers and stores retain `lane*_index_mask & xmask`; mismatched external fills reload rather than forward. |
| Output grouping | factor-4/three-output MXFP6 tests | Lane order and stores are unchanged; each consumer uses its own proved lane. |

F1 is behavior preserving. Capture and hash normalized generated kernels before
the change and require byte identity afterward for at least:

- nested NVFP4 and MXFP4;
- standalone NVFP4;
- nested and standalone reduced BROADCAST;
- looped internal source with two and three parent passes;
- MXFP6 4:3, internal-source 4:3, aligned preshuffle, and scale swizzle;
- indirect masked gather; and
- dynamic-R factor 2.

Any F1 kernel-text difference is a blocking review item, not something a
runtime benchmark can waive.

## F2a group-only lazy broadcast/CSE matrix

F2a consumes only the exact live source returned by F1. It does not inspect a
buffer name, reopen the plan maps, infer ownership from a name, or introduce a
general projected-value algebra. A deferred group value is an ordinary live
`CSEVariable`; its existing `shape` is the only domain state. There is no
deferred wrapper or projection cache.

| ID | Behavior | Passing result |
| --- | --- | --- |
| F2A-01 | Group source recognition | Only a resolved relation with no lane selection, no pending consumer guard, and a CSE shape equal to the grouped domain is eligible. `B=1`, `G=2`, scalar, unknown-shape, and parent-shaped values have dedicated positive/negative tests. |
| F2A-02 | Scalar-chain deferral | Pure scalar operations with only scalar/group operands remain group-shaped. The scale divide, clamp, conversion, cast-back, and reciprocal are emitted once and CSE with the reduced-stage chain. |
| F2A-03 | First lane-varying join | When any tensor operand is lane-varying, group values are widened with the existing group-to-child materializer before the operation. No parent-width/lane projection is delayed in F2a. |
| F2A-04 | Store barriers | `store` and `store_reduction` always receive concrete target-domain values. No group-width value reaches a lane-shaped store. |
| F2A-05 | Callback and stateful barriers | `masked`, `load_seed`, `bucketize`, random, reduction, scan, sort, indirect indexing, and callback/subgraph operations materialize first. A closure-captured group value in `masked` is covered directly. |
| F2A-06 | Unknown operation | The operation classifier is fail closed. Only the reviewed positive allowlist may preserve group shape; every other `ops_handler.OP_NAMES` member and an unclassified synthetic op widens/materializes without needing a second exhaustive barrier list. |
| F2A-07 | Inline assembly | Pure `inline_asm_elementwise(pack=1)` may preserve a group value. Packed or impure inline assembly is a barrier. |
| F2A-08 | Target masks | Group intermediates keep their natural group mask. Every widened value gets an assignment from `family.mask_vars_for_shape(target.shape)`, not copied/unioned source masks. Odd-tail indirect gather and store tests must inspect the emitted predicate. |
| F2A-09 | Flush lifetime | F2a creates no lazy cache or state that survives `codegen_body()`. Looped parent lane sources remain eager/reloaded; group-only deferral begins only after a live F1 resolution in the current pass. |
| F2A-10 | CSE loss | Perturb or disable the expected group-chain CSE hit in a test build. Correctness remains exact, but the one-conversion/one-reciprocal kernel-form test fails, proving CSE is a performance tripwire only. |
| F2A-11 | Unaffected forms | Factor-4/MXFP6, parent-lane projection, and unrelated nested reductions remain byte-identical to F1. No staged launch heuristic change is included. |

The initial positive allowlist is exactly `to_dtype`, `abs`, `neg`, `add`,
`sub`, `mul`, `truediv`, `minimum`, `maximum`, `eq`, `ne`, `lt`, `le`, `gt`,
`ge`, `where`, and pure `inline_asm_elementwise(pack=1)`. Each entry must prove
through current shape propagation that scalar-compatible inputs return the
exact group shape. `shape=None`, wrong-rank, or tuple/multiple results are
barriers unless that operation has a small explicit result-shape rule.

Required generated form for affected NVFP4/MXFP4 scale chains:

- exactly one numeric FP8/E8M0 conversion;
- exactly one reciprocal sequence for MXFP4;
- no `broadcast_to(...).to(float8...)` form;
- group-only scale arithmetic precedes the broadcast;
- no output-buffer reload and no additional kernel; and
- masked full block, fixed oversized block, static tail, and dynamic-R forms
  all satisfy the same rule.

## Optional F2b gate

F2b is a separate change. It may proceed only after F2a is accepted and a
fresh fixed-config B200 run reproduces a useful parent divide-before-split win.

- Eligible values come from an exact F1 relation with a proved lane. No
  name/layout inference is allowed.
- Only persistent factor-2 is enabled initially. Looped and factor-4/MXFP6
  remain on the F2a/eager parent path and are byte-identical.
- Parent-only replay remains flat so it can hit ordinary CSE. A single
  structured parent/group join is created, followed by one all-lane split.
- The materialization cache is keyed by live CSE value, projection, target
  shape, and active family; it caches the complete lane tuple, never one entry
  per lane, and cannot cross a flush.
- Float8 reshapes use the dtype-preserving bitcast helper. Bounds and target
  masks are retained.
- Kernel form has one parent operation before one split, no eager scale
  expansion, no intermediate global traffic, at most two relevant TTGIR
  `convert_layout` operations, at most 2048 bytes shared memory for the
  audited 4096x4096 configuration, and zero local-memory spills.
- Performance must improve the persistent 4096x4096 flagship by at least 10%
  versus F2a in at least two of three paired runs. No protected matrix row may
  regress by more than 2% and 0.25 us. Otherwise F2b does not land.

The historical full F2 implementation is not the default target. It added 453
net production lines, four types, and about twenty class members, and put
`_PointwiseRemapHandler.load` at nesting depth 6. That implementation is useful
as a semantic and benchmark reference, not as the complexity budget.

## Swizzle, DCN, and MXFP6 gates

| Case | Required form and result |
| --- | --- |
| Generic scale swizzle | `test_rmsnorm_block_scale_swizzle_kernel_form` stays one kernel and keeps the direct swizzled store index. |
| MXFP6 4:3 | `test_mxfp6_four_to_three_pack_kernel_form` keeps one staged kernel, four stores, the existing load counts, and exactly three recursive `tl.split` statements. Packed bytes and scale are exact. |
| MXFP6 internal source | `test_mxfp6_internal_source_kernel_form` keeps one staged kernel, five stores, no output reload, and the existing split count. |
| Aligned scale preshuffle | Unshifted `test_producer_consumer_mxfp6_pack_scale_swizzle` fuses; shifted access remains a multi-kernel fallback. The scale and all three packed lanes store directly. |
| Already-staged DCN append | Exact scale append remains one kernel. Rolled/shifted scale and a consumer of a sub-parent output remain two kernels. Full plan reconstruction, not an initial-formation shortcut, makes the decision. |
| Native E2M3 `pack=2` composition | On SM100+, one kernel, one inline-asm site, four stores, bit-exact to the compiled software graph. At `rows=7,width=96`, all four stores are masked. `pack=4` is forbidden because HOP element grouping is unspecified. |
| Dynamic padded swizzle | Out of scope unless the explicit scatter/padding work is present. Without it, decline safely. Do not weaken the X/R frame proof or fold auxiliary padding writes into sub-parent mutation handling. |

For the B200 `2048x3072` aligned DCN/native case, compare in the same process
against the frozen source and configuration. The candidate must remain within
2% and 0.25 us of the frozen median, use at most 32 registers/thread, use zero
spills, and emit one kernel. Historical anchors are 8.50-8.70 us for native
`pack=2`, 15.46 us for standalone software 4:3, and 16.28 us for DCN software
4:3. The paired ratio, not an absolute historical number, is authoritative.

## Fuzz and mutation gates

### Differential fuzz

Run the existing 62-case/142-invocation matrix twice:

1. capture the frozen F1 baseline;
2. compare F1 against it; then capture F1 and compare F2a against F1.

Required summary for both comparisons:

- 62 compile cases and 142 runtime invocations;
- one compiled graph for every dynamic case;
- all integer, packed, and float8 tensors bit-exact;
- all floating tensors bit-exact to the immediately preceding layer
  (`max_abs_error == 0` for the recorded matrix);
- expected fallback cases remain fallback: odd `R=4607`, shifted source,
  mismatched masked-fill 8-bit cases;
- generated conversion count is one wherever the matrix declares one.

Also run the three-seed semantic matrix: 33 compile cases, 99 invocations, no
crash or mismatch. Extend the matrix with same-name/different-index reads,
`B=1`, `G=2`, `+0.0/-0.0` fills, and one masked indirect gather per factor.

### Mutation score

Create local throwaway mutants; do not commit them. Every mutant below must be
killed by a focused test or the full suite. Required mutation score: 100%.

1. Remove one INTERLEAVED/lane relation.
2. Remove one BROADCAST/group relation.
3. Remove one IDENTITY/internal relation.
4. Route a sub-parent read through `_fusable_read_after_index_equivalence`.
5. Broaden that helper to a read not owned by `nested_stage`.
6. Replace an exact consumer index with a same-name shifted index.
7. Drop mask from the source key.
8. Drop fill from the source key.
9. Apply mutation renames while constructing the plan instead of match time.
10. Permit a second same-name reaching write.
11. Record a non-`None` store mode.
12. Reuse a source/materialization across `codegen_body()` invalidation.
13. Let IDENTITY or BROADCAST suppress the looped pre-epilogue flush.
14. Remove the `store` barrier in F2a.
15. Remove the `store_reduction` barrier in F2a.
16. Remove the masked-callback barrier in F2a.
17. Treat an unknown op as projection-safe.
18. Copy/union source masks instead of assigning target-family masks.
19. Broaden optional F2b to looped or factor-4 without an explicit gate.

For relation-removal mutants, the expected kill is lost fusion or a targeted
compile error. A numerically correct one-kernel result through name forwarding
is a surviving mutant and blocks acceptance.

## Performance protocol and thresholds

Use one B200, a fixed clock/idle state, identical inputs and configs, CUDA graph
replay, 100 warmups, 500 repetitions, and at least 21 randomized/interleaved
paired rounds. Report median and p10/p90. Disable coordinate descent for the
mechanism comparison; measure it separately if a heuristic change is proposed.

| Layer/case | Gate |
| --- | --- |
| F1 runtime | Generated source is identical. If any reviewed source change remains, no row may exceed `1.02x` baseline and `+0.25 us`. |
| F1 compile time, feature disabled | Median at most `1.02x` frozen and p90 at most `1.05x`; staged plan/access extraction counters remain zero. |
| F1 compile time, feature enabled | Median at most `1.05x` frozen and p90 at most `1.10x` on standalone, nested, and MXFP6 graphs. |
| F2a protected runtime matrix | No row exceeds `1.02x` F1 and `+0.25 us`; geometric-mean ratio at most `1.00x`. Structural one-conversion/one-reciprocal checks are mandatory even when latency is tied. |
| F2b flagship | Persistent `4096x4096` at least 10% faster than F2a in two of three paired runs; no protected row exceeds `1.02x` and `+0.25 us`. |
| MXFP6/DCN | No row exceeds `1.02x` frozen and `+0.25 us`; native aligned DCN retains one kernel, at most 32 registers, and zero spills. |

Protected runtime rows are NVFP4 and MXFP4 at `128x4096`, `4096x4096`, and
`4096x8192`; looped tail `4096x4608`; MXFP6 4:3 at `128x384`; and aligned DCN
at `2048x3072`. Add `4096x6144` and `4096x7168` when comparing to FlashInfer.
For that external comparison, no row may be more than 10% slower and the
geometric mean must be at least parity. FlashInfer is never the correctness
oracle.

The historical full-projection anchors are 42.880 to 32.640 us at 4096x4096
and 552.832 to 528.224 us at 4096x8192. They establish opportunity, not a gate
for F2a. The isolated fair F2b opportunity was about 18-20%; the 47% number
combined parent-value reuse with split ordering and must not be cited.

## Complexity audit

Metrics below were produced with Python's `ast` module. Branches count `if`,
loops, conditional expressions, exception/else/finally arms, match arms,
boolean edges, and comprehension generators/filters. Cyclomatic complexity is
`branches + 1`; nesting counts nested control constructs. LOC is inclusive
function LOC. The selected-surface aggregate is useful only within the same
source generation; the historical artifacts have an older scheduler base.

### Current frozen baseline

| Function | LOC | Branches | CC | Max nesting | Required simplification |
| --- | ---: | ---: | ---: | ---: | --- |
| `NestedReduction._order_sub_parent_parent_nodes` | 90 | 40 | 41 | 4 | Do not grow it in F1. Move writer/read closure to a shared temporal DAG when available. |
| `Scheduler._prove_staged_fusion_dependencies` | 135 | 35 | 36 | 4 | Keep the inherited grouped-stage exception visibly disjoint from sub-parent exact matches; F1 must not grow CC or nesting. P2G later replaces it with an exact lookup. |
| `NestedReduction.sub_parent_epilogue_plan` | 123 | 29 | 30 | 2 | Delete cross-boundary name views; keep candidate-name sets local and convert once to exact relations. No CC increase. |
| `NestedReduction._plan_nested_sub_parent_stage` | 118 | 20 | 21 | 2 | Share relation construction with standalone rather than adding a second lookup policy. No CC increase. |
| `NestedReduction._sub_parent_internal_projections` | 56 | 16 | 17 | 5 | Flatten validation through indexed writer/read records; target nesting <= 4. |
| `NestedReduction._try_get_sub_parent_source_projections` | 141 | 15 | 16 | 3 | Separate normalization from per-consumer lane proof; do not add another nested scan. |
| `NestedReduction._sub_parent_broadcast_projections` | 42 | 14 | 15 | 3 | Reuse one relation constructor and one frame predicate; do not duplicate the grouped-stage broadcast proof. |
| `SIMDScheduling._codegen_reduction_with_sub_parent_epilogue` | 113 | 10 | 11 | 2 | Use the same exact-store setup/emission helper as nested codegen. |
| `_SubParentSourceLoadResolver.materialize` | 34 | 10 | 11 | 2 | Delete with F1. |
| `_PointwiseRemapHandler.load` | 17 | 6 | 7 | 2 | Keep lookup tri-state shallow; after F1/F2a target CC <= 12 and nesting <= 4. |

Current selected surface: 22 functions, 1016 function LOC, 227 branches,
maximum CC 41, maximum nesting 5.

The `6b8ef64bd373` commit's production diff against its parent is `+688/-228`
physical lines (net `+460`) across `scheduler.py`, `simd.py`,
`simd_kernel_features.py`, and `triton.py`. The staged follow-ups use the
layered snapshot above as zero; they do not get to hide new mechanism cost
inside that earlier delta. The relevant frozen mechanism has no public API, and the
compatibility inventory contains two type-level mechanisms
(`SubParentSourceLayout`, `_SubParentSourceLoadResolver`), three stage name/layout
views, one family name cache, two codegen name sets, and one generic
`store_cache[name]` fallback. Every one is scheduled for deletion in F1.

### Historical delta reference

| Artifact | Production delta | Added mechanism | Complexity result | Disposition |
| --- | ---: | --- | --- | --- |
| Historical F1 package | `+567/-240`, net `+327` | 6 types, 2 module helpers, 13 indexed-store methods; removed the old resolver/views | Selected surface: 26 functions, 947 LOC, 208 branches, max CC 33, max nesting 4 | Semantic reference only; the current F1 budget is `<= +150` net. |
| Historical full F2 | `+457/-4`, net `+453` | 4 types and about 20 class members | `_DerivedDomainProjection.apply`: 63 LOC/CC 22; remap `load`: 49 LOC/CC 16/nesting 6 | Reject as the default F2a shape. |
| Historical narrow group adapter | class 77 LOC, 6 methods | One group-broadcast helper | `apply`: 19 LOC/CC 9/nesting 2 | Starting complexity reference for F2a; add exact-source and fail-closed barriers without building a general domain algebra. |

### Proposed mechanism budget

| Layer | Production LOC delta from prior layer | Private type delta | Module-helper delta | Method budget | Public API delta |
| --- | ---: | --- | ---: | ---: | ---: |
| F1 | target `<= +150` net | At most 6 additions; delete the 2 frozen compatibility types | At most 2 | At most 13 indexed-store methods | 0 |
| F2a | `<= +150` net | Prefer 1 adapter; hard maximum 2 | At most 1 | At most 8 | 0 |
| F2b, optional | F2a+F2b combined `< +300` net | At most 1 pending-lane value type | At most 1 | At most 8 | 0 |

F1's six-type ceiling is not a target. It allows the historical exact-access
records, guard/key records, resolved-source record, and indexed store only if
each remains materially useful after deleting the old resolver. A wrapper used
only to rename another wrapper must be folded away.

### Duplicate-proof and compatibility audit

| Duplicate surface | Acceptance action |
| --- | --- |
| Exact sub-parent plan membership plus inherited `_fusable_read_after_index_equivalence` | F1 keeps the paths disjoint by stage ownership. P2G later adds a source-anchored frame record, proves post-merge reconstruction, and then deletes both legacy helpers. |
| `ProjectedSourceAccess` plus `SubParentSourceLayout` and the three stage name/layout views | Carry the proved consumer lane on the exact relation; derive direct/group/lane materialization from that relation and live value shape; delete the enum and views. |
| Indexed access store plus `_SubParentSourceLoadResolver`, `remapped_values`, and `store_cache[name]` | Make the indexed store the sole resolver; delete all three name-based paths in the same F1 diff. |
| Guard-aware source keys plus `masked_forward_names` | Key mask/fill per access and delete the name set. Keep only the explicit unguarded-source to guarded-consumer transition. |
| Nested and standalone resolver setup | Extract or inline one common setup/emission path; do not maintain two miss policies or two materialization loops. |
| Narrow group adapter plus historical `_DerivedDomainProjection` | Land only one. F2a must delete/avoid general domain wrappers, parent structured views, view/split caches, and parent defer flags. |
| F2b exact parent replay plus a staged launch heuristic | Keep the heuristic out of F2b. It requires its own A/B evidence and change. |

### Complexity gates

- F1 production net delta, tests excluded: target `<= +150` lines. Exceeding it
  requires a written mechanism inventory and explicit approval. The indexed
  store plus any retained old resolver/layout/name path is an automatic fail,
  regardless of LOC.
- F1 adds no public API. New private functions must have CC `<= 12` and nesting
  `<= 3`; orchestration methods may reach nesting 4. Existing hotspots may not
  increase. `_prove_staged_fusion_dependencies` remains at most CC 36 and
  nesting 4 until P2G removes the scoped legacy branch.
- P2G is measured independently. Its accepted end state cannot retain both the
  new parent/grouped records and either legacy helper, and
  `_prove_staged_fusion_dependencies` must be simpler than the F1 result.
- F2a production net delta, tests excluded: `<= +150` lines, at most two new
  private types and eight methods, with its main dispatcher CC `<= 10` and
  nesting `<= 2`.
- Optional F2b must keep F2a+F2b below `+300` net production lines, add at most
  one pending-lane value type and one stage-scoped split cache, and add at most
  eight methods. No new function may exceed CC 15 or nesting 3. Missing the
  performance gate means delete F2b rather than relax this budget.
- Tests do not count toward the production-LOC budget. Comments, type aliases,
  and fields do count in the mechanism inventory even when AST metrics do not.

Post-implementation measurement template:

| Layer / SHA | Production `+/-` and net | Type delta / compatibility types | Module helpers | Functions/methods | Public API | Scoped LOC / branches | Max CC / nesting | Compatibility scan | Verdict |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Frozen layered snapshot | `6b8ef64bd373` itself is `+688/-228`, net `+460`; layered content is follow-up zero | 2 legacy compatibility types | 1 retained lane helper | 22 scoped functions | 0 | `1016 / 227` | `41 / 5` | old resolver/views present | baseline |
| F1 checkpoint | `+498/-362`, net `+136`; remeasure final | 6 added, 2 codegen compatibility types deleted | 1 added, 2 relevant total | measure final | 0 | fill | fill | zero layout/name codegen paths; scoped P2G helper retained | within LOC budget; AST pending |
| P2G, later | measure independently | no layout/name type | measure | both legacy methods deleted; measure additions | 0 | fill | lower than F1 | exact rebuilt records only | fill |
| F2a | measure; net `<= +150` | measure; prefer 1, max 2 | measure; `<= 1` | measure; `<= 8` | 0 | fill | fill | no name lookup or deferred-domain cache | fill |
| F2b, if pursued | measure; combined F2 `< +300` | measure; `<= 1` pending-lane type | measure; `<= 1` | measure; `<= 8` | 0 | fill | fill | persistent factor-2 only | fill |

## Required commands

Use the existing worktree overlay because the editable install points at the
main checkout. Run full files, not only selectors:

```bash
cd "$WT"
PYTORCH_WORKTREE="$PWD" \
TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
conda run --no-capture-output -n pytorch-3.12 \
python ../indexed_projection_work/run_worktree_tests.py \
test/inductor/test_inductor_scheduler.py -q

PYTORCH_WORKTREE="$PWD" \
TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
conda run --no-capture-output -n pytorch-3.12 \
python ../indexed_projection_work/run_worktree_tests.py \
test/inductor/test_nested_reduction.py -q
```

F2 also runs the full `TestTritonHeuristics` class only if it changes launch
metadata or heuristics. The preferred F2a has no such change.

Static checks:

```bash
conda run --no-capture-output -n pytorch-3.12 \
python -m py_compile torch/_inductor/scheduler.py \
torch/_inductor/codegen/common.py torch/_inductor/codegen/simd.py \
torch/_inductor/codegen/triton.py test/inductor/test_inductor_scheduler.py \
test/inductor/test_nested_reduction.py

spin quicklint
git diff --check HEAD
```

Before capturing any baseline from the layered worktree, verify its identity:

```bash
git diff --cached --binary | sha256sum
git diff --binary | sha256sum
```

Baseline suite floor from the frozen tree is 108 scheduler passes with 6
environment skips and 395 nested-reduction passes with 8 environment skips.
New tests increase the pass count. Acceptance allows no new skip, xfail,
deselection, or failure. Historical `kernel_num_gb` overlay failures are not a
reason to exclude tests from the current full-file run.

## Missing or contradictory requirements resolved here

1. **Mask equality versus guarded fallback.** "Exact mask/fill identity" and
   "unguarded source may serve guarded consumer" are compatible only with an
   explicit consumer-side `where`. This document permits exactly that one
   transition and no other implication proof.
2. **Fill identity was underspecified.** Plain Python equality aliases values
   such as `+0.0` and `-0.0`. The cache key must preserve the typed emitted
   constant semantics; the adversarial fill matrix is now required.
3. **Mutation wording could imply new support.** Applying mutation renames at
   match time is required infrastructure, not permission to fuse a
   mutation-bearing sub-parent stage. Existing alias/mutation rejection stays.
4. **Parent-to-grouped ownership is deferred.** The desired end state is still
   an exact plan record, but deleting the helper now loses more than 20 existing
   fusions. The first relocation also admitted transpose/equal-size axis swaps
   and failed after post-merge replanning. F1 therefore retains the helper only
   for inherited grouped-stage fusion. P2G must supply the source-anchored frame
   proof, reconstruction, and mutations before taking ownership and deleting
   it.
5. **Old F2 scope conflicts with the current goal.** Older records require a
   general deferred-domain wrapper and parent divide-before-split. The immediate
   F2a is group-only. Parent divide-before-split is optional F2b and must earn
   its complexity with the separate 10% performance gate.
6. **Old operation policy was fail open.** A deny-list of barriers contradicts
   the requirement that unknown/stateful operations cannot see lazy values.
   F2a uses one small positive allowlist with shape-contract tests; everything
   else, including newly added `OP_NAMES`, defaults to materialization. It must
   not recreate the historical exhaustive 133-op partition.
7. **Looped behavior descriptions conflict.** F2a may keep group-only scalar
   work deferred in looped kernels, including tails. Parent lane projection
   remains eager/reloaded; no parent value or projection cache crosses a pass.
8. **Mask ownership changed after the historical F2.** Historical `_copy_masks`
   evidence is not acceptance evidence. The current gate is target-shape mask
   assignment through `mask_vars_for_shape`.
9. **`must_forward` ownership is underspecified.** Deriving it from a bare name
   is valid only while one temporal version can be written in-kernel. The gate
   is the exact reaching-writer relation; any future relaxation of mutation or
   multiwrite rejection requires a real version token before name-derived
   `store_buffer_names` membership is sufficient.
10. **Dynamic padded swizzle is separate.** F1/F2 must neither claim nor obtain
    it accidentally. It requires the explicit padding/scatter model and its
    auxiliary-write rules.
11. **Old validation relied on focused selectors.** The swizzle regression was
    missed that way. Full scheduler and nested-reduction files are mandatory at
    every accepted layer.
12. **Consumer miss and source miss were conflated.** A same-name sub-parent
    read absent from the plan indicates an incomplete proof and must decline or
    raise. Physical reload is permitted only after that consumer relation was
    found and its planned external source has no live forwarded CSE value.

## Final signoff checklist

- F1 has one exact resolver and zero old layout/name codegen mechanisms. Exact
  per-read records are the sole authority for sub-parent forwarding; the one
  documented inherited grouped-stage scheduler exception remains scoped.
- Every F1 negative fails at the intended layer: planner decline, physical
  external reload, or loud required-source error.
- F1 generated kernels are byte-identical to frozen captures.
- F2a is group-only, fail closed for operations, target-mask correct, and
  structurally emits one shared scale chain.
- F2b is absent unless its correctness, resource, complexity, and performance
  gates all pass independently.
- Full suites, differential fuzz, relation/barrier mutation score, static
  checks, and paired B200 measurements are recorded against exact SHAs.
