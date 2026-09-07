# Lazy projection rebase plan (2026-08-26)

## Decision

Split the follow-up after indexed forwarding:

1. **F2a: lazy group broadcast/CSE.** Land the small, high-confidence part:
   keep exact `lane=None` group-width values at group resolution through the
   scale-side scalar chain, then broadcast only at the first lane-varying use,
   store, or unsupported operation.
2. **F2b: parent-width divide-before-split.** Treat this as an optional second
   slice. Rebuild it only on exact `lane!=None` relations, and only if a fresh
   F1-vs-F2a-vs-F2b comparison reproduces a material persistent-factor-2 win
   with substantially less machinery than the historical F2.

Do not replay the canonical F2 patch wholesale. Its useful behavior is real,
but its general parent/group domain wrapper is unnecessary for F2a.

## Assumed F1 endpoint

F2 should consume the final per-read contract directly:

- `ProjectedConsumerAccess(access: MemoryDep, lane: int | None)`;
- `ProjectedSourceAccess(sources, consumers, must_forward)`;
- exact emitted-load matching through normalized LoopBody accesses;
- source identity keyed by exact `MemoryDep` plus mask/fill guard;
- `lane != None` means a proved parent-lane selection;
- `lane == None` is resolved from the live CSE shape as direct forwarding or a
  group-width broadcast; and
- an internal miss/liveness loss is loud, while an external miss reloads via
  the ordinary derived-index path without name-based store forwarding.

F2 needs a narrow resolve-before-materialize seam from F1. It must not reopen
F1's source/consumer maps, infer treatment from a buffer name, or add another
projection enum. Concretely, consume `_IndexedProjectedValueStore.resolve_source`
before its eager `_materialize` step; the resolved object already carries the
live exact value and lane witness.

## F2a: Group-Only Deferral

Use only the handler-interception idea from the old
`_GroupInvariantBroadcast`, not that name as a new value abstraction and not
the full `_DerivedDomainProjection` implementation. The delayed F2a value is
an ordinary `CSEVariable`; its existing `shape` is the only domain state.

At each remapped load:

- resolve the exact consumer relation once;
- preserve F1's required-source and guard/fill decisions;
- eagerly use F1 materialization for `lane != None` in F2a;
- return a live group-width CSE value directly when `lane == None`, its grouped
  axis equals `num_groups`, and either its guard matches or the only pending
  guard is the enclosing `masked` callback case below; and
- leave direct child-width/scalar values concrete.

No deferred wrapper is needed for group values. For operations routed through
`CSEProxy._default`, it already asks `ShapePropagationOpsHandler` for the result
shape and records it on the ordinary CSE variable. `_PointwiseRemapHandler`
only needs a small pre-operation operand check: recognize group-shaped operands
and widen them when:

- another tensor operand is lane-varying;
- the result is stored;
- the operation is memory-, position-, state-, collective-, or subgraph-based;
  or
- the operation is not in a small positive allowlist of proven scalar ops.

The initial positive allowlist should be no broader than the emitted scale
chains: `to_dtype`; unary `abs`/`neg`; `add`, `sub`, `mul`, `truediv`;
`minimum`/`maximum` (the clamp lowering); `eq`, `ne`, `lt`, `le`, `gt`, `ge`
when needed by `where`; `where`; and pure
`inline_asm_elementwise(pack=1)` when exercised. Their shape contract is
concrete in the current code:

- `ShapePropagationOpsHandler.to_dtype` returns `value.shape` exactly;
- the listed scalar elementwise operations use `broadcast_shapes_for_args`, so
  group plus scalar-compatible operands produce the group shape, while the
  pre-operation check first widens a group operand if another operand is
  child-shaped; and
- pack-1 inline assembly passes only its positional tensor inputs to that same
  broadcast calculation and returns one elementwise result.

Do not infer this merely from an op's scalar semantics. Add a focused shape
contract test for every allowlisted name and inspect every returned CSE leaf.
If an op returns `shape=None`, a different rank/domain, or multiple results, it
is a barrier unless that op gets a tiny, local, justified result-shape update.
Tuple-valued ops are initially barriers. Unknown operations materialize, so an
`OpsHandler` extension cannot silently become incorrect. This is simpler and
safer than the historical 133-op exhaustive partition.

`masked` is a callback boundary, not an allowlisted scalar op. Wrap its callback
locally, let ordinary group-shape propagation run inside it, and widen every
group-shaped returned CSE leaf before control returns to `masked`; the existing
masked implementation then reapplies its predicate. This needs no
`_MaterializedProjectionCallback` class. Packed or impure inline assembly is
also a barrier.

This gives the intended form without a second CSE system:

```text
exact group source
  -> cast/clamp/divide/reciprocal at group width
  -> ordinary kernel CSE reunifies repeated lane-body replays
  -> one final group-to-child broadcast
  -> first lane-varying multiply/divide/pack
```

Correctness must not depend on the CSE hit. A miss may duplicate group work but
must remain numerically correct; kernel-form tests make the hit a performance
contract.

## F2b: Optional Parent Deferral

The historical parent-width divide-before-split result remains worth testing,
but is not part of minimum F2a.

If remeasurement justifies F2b, use only exact `lane!=None` relations:

- represent a pending lane selection with one small value record containing
  the live parent CSE value, lane, and whether it has reached structured parent
  shape;
- replay parent-only scalar work on flat `[X, R]` values so it can CSE with the
  parent stage;
- at the first parent/group join, view the parent as `[X, groups, G]` and the
  group operand as `[X, groups, 1]`, use native broadcasting, and retain the
  lane selection as the only residual projection;
- materialize on mixed lanes, stores, barriers, or unsupported operations; and
- cache the complete split tuple by source CSE/factor/target family, never by
  selected lane.

F2b should initially be persistent factor 2 only. Looped parent values remain
subject to ordinary CSE invalidation, and factor-4/MXFP6 parent projection stays
on F1's eager path. If same-domain replay no longer produces one shared parent
expression, stop: add recorded inline provenance later rather than an FX-origin
or structural-expression matcher.

## Reuse And Delete

Useful historical pieces:

- the shape predicates and pre-operation widening policy demonstrated by
  `_GroupInvariantBroadcast` in `nvfp4_group_projection_draft_20260818`, folded
  into the remap handler rather than ported as a projection object;
- `_PointwiseRemapHandler._default` and store interception;
- the callback-disabled materialization pattern for `masked`;
- the existing group-to-child broadcast emitter, including FP8 bitcast safety;
- `_DerivedIterationFamily.mask_vars_for_shape` / `set_value_masks`;
- the exact F1 resolve/materialize split and CSE-liveness check;
- for F2b only, dtype-safe structured views and an all-lanes split cache; and
- the historical fuzz, kernel-form, resource, and benchmark harnesses.

Do not port into F2a:

- `_GroupInvariantBroadcast` as a standalone object;
- `_DerivedValueDomain`, `_DeferredDerivedValue`,
  `_MaterializedProjectionCallback`, parent-flat/structured state, projection
  factoring, `_view_cache`, or `_split_cache`;
- `parent_value`, `_parent_shape`, `_group_shape`, `_view`, `_split`, or
  `defer_interleaved_projection`;
- `emit_reshape_preserving_dtype` or any Triton change;
- layout-enum branches, name buckets, resolver adapters, `remapped_values`, or
  duplicated source/materialization caches;
- the old scheduler-side rebuilt-chain/backstop analysis;
- the 8-bit materialization frontier; or
- the Blackwell staged-reduction heuristic without a fresh benchmark proving it
  is still needed.

For F2b, retain only the parent-lane record, structured join, dtype-safe view,
and one all-lanes split cache. Ordinary CSE should replace the historical view
cache. F2b must not reintroduce layout enums or scheduler-side expression
matching.

## Exact Mask Policy

Access guards and Triton range masks are separate concerns.

1. F1 remains authoritative for source/consumer mask and fill compatibility.
   An unguarded source may remain deferred for a guarded consumer only inside
   the existing `ops.masked` callback that reapplies that consumer predicate;
   the callback barrier must materialize it before returning. Any other pending
   guard/fill application takes F1's eager concrete path.
2. Group-width scalar operations retain their naturally propagated group-shape
   masks while they stay group-width.
3. Every group-to-child broadcast, direct child materialization, and parent
   split output gets a fresh assignment from
   `family.mask_vars_for_shape(kernel, result.shape)`, preferably through
   `family.set_value_masks(...)`.
4. Assignment replaces the prior set. Never union/copy source masks onto a
   differently shaped value. In particular, parent `r0_mask` or reduced masks
   must not appear on lane-shaped values; those receive the active
   `lane{factor}_..._mask` (and compatible passthrough masks such as `xmask`).
5. F2b structured rank-3 views carry no copied mask metadata. Any operation
   that consumes masks or indices is a barrier, and the eventual lane
   materialization receives target-family masks.
6. Run mask derivation with the derived family active. Reuse
   `set_value_masks`, whose `ensure_active` handling also covers pre-flush
   internal materialization.

This replaces historical `_copy_masks`; that helper must not return.

## Pass And Lifetime Boundaries

- F2 changes no scheduling or fusion proof.
- Pre-epilogue flush suppression remains restricted to internal relations whose
  consumer has `lane is not None`. A `lane=None` direct/group relation must not
  hold a reduction loop open; this is the prior replay P0 boundary.
- A lazy CSE value or projection cache entry lives only within one unflushed
  codegen pass. In nested emission that starts after the required outer-body
  flush; in standalone lane forwarding it may start in the parent pass only
  when the approved `lane is not None` suppression keeps that same pass open.
- The F1 exact-access registry may remain allocated across a `codegen_body()`
  call so it can record later sources, but every old CSE entry is stale:
  resolution must keep using `cse.contains_value`, and no F2 result/cache may
  cross `codegen_body()`/`cse.invalidate()`.
- F2a owns no projection cache. Each use relies on the live CSE value selected
  by F1 and ordinary expression CSE.
- F2b's split cache is stage-scoped, omits lane from its key, checks source CSE
  liveness on every hit, and is cleared/destroyed at any body flush.
- Do not turn persistent liveness into a planner promise. If F1 cannot resolve a
  live external source, reload eagerly; if a required internal source is gone,
  fail loudly.

## Acceptance Gates

Correctness and regression:

- Run the complete `test_nested_reduction.py` and
  `test_inductor_scheduler.py`, plus `test_triton_heuristics.py` only if launch
  policy changes.
- Keep the F1 exact-relation mutation battery green for lane, group/direct,
  temporal writer, mutation rename, mask/fill, required miss, and external
  reload behavior.
- Add direct tests for group-only propagation, lane-mix/store/`store_reduction`
  materialization, masked callback materialization, packed/impure inline-asm
  barriers, and an unknown-op fail-closed path.
- Add an allowlist contract test that runs each allowed op through current
  shape propagation: group-only/scalar inputs must return the exact group
  shape, and a child operand must trigger pre-op widening and a child-shaped
  result. Reject `None`, wrong-rank, and tuple output shapes unless handled by
  an explicit per-op rule.
- Pin target mask replacement on persistent and looped indirect/tail cases,
  including `G=2`; generated gathers/asserts must use the lane-family mask and
  must not retain the parent/reduced mask.
- Re-run the historical semantic and adversarial differential matrices:
  factor 2/4, BF16/FP16/FP32, full/tail/dynamic shapes, mask-fill mismatch,
  internal/external sources, and group chains with FP8/uint8/int8/no narrowing.

Kernel form:

- NVFP4 row and swizzled-scale forms: one FP8 conversion, scale-side cast and
  reciprocal/division before the group-to-lane broadcast, one fused kernel, and
  no output-buffer reload or intermediate global traffic.
- MXFP4: one scale conversion/reciprocal sequence and one pack sequence.
- Looped full block, oversized fixed block, `D=4608` tail, and dynamic R retain
  delayed group work.
- MXFP6 `(4,3)`, internal-source, preshuffled, and aligned DCN scale-swizzle
  forms retain numerics, store/split counts, and one-kernel behavior. Dynamic
  padded scatter remains out of scope.
- F2b additionally requires one parent-width scale application before one
  split, both lanes selecting that split, and no eager scale expansion.

Performance and resources:

- Compare accepted F1, F2a, and (if built) F2b under identical fixed configs,
  then under the default heuristic, on at least `128x4096`, `4096x4096`,
  `4096x4608`, and `4096x8192` for NVFP4 and MXFP4; include MXFP6 as a
  non-regression workload.
- F2a must remove duplicate scale-side work and have no new spills or more than
  a 2% repeatable regression in any enabled configuration. Regressing configs
  use eager broadcast; do not revive a dtype-specific 8-bit frontier.
- F2b must reproduce at least a 10% fixed-config win on the profitable
  persistent `4096x4096` case, show no more than a 2% geomean regression across
  the matrix, introduce no spills, and match or improve the historical physical
  target (two layout conversions and about 2 KiB shared memory). The old 1.31x
  headline is evidence, not a result to assume after rebase.
- Reintroduce `staged_reduction` launch metadata/heuristics only as a separate,
  measured tuning slice. If needed, keep it CUDA-SM100/sub-parent-specific;
  ordinary nested kernels, HIP/XPU, dynamic R, and tail configs must remain
  unchanged.

Complexity:

- Historical full F2 added 457 and removed 4 production lines (`simd.py`,
  `triton.py`, and reduction heuristics). F2a should stay in `simd.py`, add at
  most one small helper, no deferred-value type or cache, and target no more
  than 150 net production lines. The earlier group-only patch was 109 additions
  and 6 deletions in production code, so this is a realistic ceiling.
- F2a+F2b together must remain below 300 net production lines, with at most one
  small pending-lane value type and one split cache. Exceeding that budget, or
  requiring scheduler/name/layout scaffolding, is a stop-and-redesign signal.
- Treat absence as a review gate: production `rg` after the rebase should find
  no `_GroupInvariantBroadcast`, `_DerivedValueDomain`, `_DeferredDerivedValue`,
  `_MaterializedProjectionCallback`, `_DerivedDomainProjection`, `_copy_masks`,
  layout enum, or name-keyed projection map. Any surviving compatibility
  scaffold must be deleted before evaluating the LOC budget or performance.

## Implementation Sequence

1. Freeze and review F1; record its final resolve/materialize and guard API.
2. Add F2a's exact-load opt-in and small fail-closed group-broadcast helper;
   assign target masks on widening and add focused unit/kernel-form tests.
3. Run full suites, mutation tests, fuzzing, source checks, and fixed/default
   performance/resource measurements. Land no heuristic with this step unless
   the data requires it.
4. Prototype F2b only on persistent factor-2 exact lane relations. Keep the
   parent flat until a group join, use one structured view path and one complete
   split cache, then run its structural/TTGIR/performance gates.
5. Keep F2b only if it clears both the performance and complexity thresholds;
   otherwise ship F2a and leave parent divide-before-split as a measured future
   optimization.
