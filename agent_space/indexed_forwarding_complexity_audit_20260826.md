# Indexed forwarding complexity audit

Date: 2026-08-26

The final flattened-snapshot assessment is at the end of this document and
supersedes the earlier recommendation to flatten the relation.

Worktree: `agent_space/followup_indexed_rebase_wt`

Snapshot: unstaged diff hash
`a633c31b5c4cdf1b78c579d63aa1c65ba0d8c3f47e4c3db2102ab7087da1df15`.
The frozen #191775 content is staged, so all numbers below compare the working
tree with the index.

## Measurements

Production diff:

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `torch/_inductor/codegen/simd.py` | 396 | 210 | +186 |
| `torch/_inductor/scheduler.py` | 75 | 111 | -36 |
| **Total** | **471** | **321** | **+150** |

AST metrics count `if`, loops, boolean terms, conditional expressions, and
comprehensions, while excluding nested function bodies from their parent.
Across only functions changed or replaced by F1:

| Metric | Frozen base | F1 | Delta |
| --- | ---: | ---: | ---: |
| Functions/methods | 24 | 37 | +13 |
| Function-body LOC | 1,157 | 1,288 | +131 |
| Cyclomatic complexity | 208 | 254 | +46 |
| Branch points | 184 | 217 | +33 |
| Maximum nesting | 5 | 4 | -1 |

The increase is concentrated in the exact runtime resolver. F1 introduces no
new function above CC 12. The new hotspots are:

| Symbol | LOC | CC | Nesting |
| --- | ---: | ---: | ---: |
| `_logical_memory_access` | 32 | 12 | 2 |
| `_IndexedProjectedValueStore.materialize_projections` | 26 | 11 | 3 |
| `_IndexedProjectedValueStore.resolve_sources` | 21 | 6 | 3 |
| `_IndexedProjectedValueStore._apply_consumer_guard` | 28 | 6 | 1 |

Existing planner hotspots are not materially worsened:

| Symbol | Frozen base | F1 | Assessment |
| --- | ---: | ---: | --- |
| `_try_get_sub_parent_source_projections` | CC 16 | CC 17 | The extra branch derives exact `must_forward` state. |
| `_sub_parent_broadcast_projections` | CC 15 | CC 16 | The extra branch wraps the exact consumer record. |
| `_sub_parent_internal_projections` | CC 17, nesting 5 | CC 17, nesting 4 | Readability improved. |
| `_prove_staged_fusion_dependencies` | CC 36 | CC 36 | Unchanged complexity; P2G owns later removal of the legacy branch. |

Do not extract one-use helpers merely to lower these scores. The new runtime
checks are mostly flat, fail-closed contract validation.

Tests add 286 lines and delete 15. There are exactly four new test methods plus
one parameterized extension, matching the permanent-test budget. The 98-line
`test_projected_access_identity_and_cache_key` is dense but branch-free; do not
grow another test family around each resolver row.

## Structural deletion gate

The old name/layout forwarding contract is gone. A current-tree grep finds no:

- `NestedReduction.SubParentSourceLayout`;
- `SubParentEpilogueStage.source_layouts`;
- `SubParentEpilogueStage.broadcast_source_names`;
- `SubParentEpilogueStage.internal_dependency_names`;
- `_SubParentSourceLoadResolver`;
- `_DerivedIterationFamily.remapped_values`;
- `forwarded_store_names` or `masked_forward_names`.

The remaining `broadcast_source_names` variables in scheduler planning are
local candidate partitions. They do not cross the planner/codegen boundary and
are not the deleted name-keyed codegen policy.

## Recommended simplification

### Flatten the relation at the planner boundary

The remaining avoidable complexity is a double representation of the same
per-read fact:

1. `ProjectedSourceAccess` groups `sources` with a tuple of
   `ProjectedConsumerAccess` values.
2. `_IndexedProjectedValueStore.__init__` immediately flattens each consumer
   into a separate `_ProjectedLoad`.

Make the plan per-read directly:

```python
@dataclasses.dataclass(frozen=True)
class SubParentAccessRelation:
    sources: tuple[MemoryDep, ...]
    consumer: MemoryDep
    lane: int | None
    must_forward: bool
```

Keep `sources` plural because equivalent live source alternatives are a real
requirement. Emit one relation per exact consumer read. Codegen can normalize a
relation with `dataclasses.replace` and use the same record as its lookup value.
When eager lane materialization sees several consumers with the same source
set, deduplicate that source set or rely on the existing materialized-value
cache so the flattening does not duplicate emitted reshapes/splits.

This deletes `ProjectedConsumerAccess`, `_ProjectedLoad`, `has_lane`, and the
consumer Cartesian-product loop in `projected_access_pairs`. It also makes the
stage field naturally `access_relations` instead of `source_projections` and
lets codegen use `relation.lane is not None` directly. Expected impact is
roughly 25-40 fewer production lines, two fewer record types, lower nesting in
the plan-pair and lane-proof loops, and no behavioral change.

As part of the same rewrite, key the runtime lookup by name then exact access:
`dict[str, dict[MemoryDep, SubParentAccessRelation]]`. That removes the parallel
`_planned_consumer_names` state while preserving the cheap name prefilter before
`_logical_memory_access` reconstruction. This is worthwhile as part of the
flattening, not as a standalone refactor.

### Remove projection-era vocabulary after flattening

The remaining records describe exact access relations, not layout policy.
Rename the codegen surface accordingly:

- `_IndexedProjectedValueStore` -> `_SubParentValueResolver`;
- `_ResolvedProjectedSource` -> `_ResolvedSubParentSource`;
- `ProjectedRangeValue` -> `_SubParentMaterializedValue`;
- `project_value_to_sub_parent_resolution` ->
  `materialize_value_at_sub_parent_resolution`;
- `materialize_projections` -> `materialize_sources`;
- `projected_values` -> `value_resolver`.

This is close to LOC-neutral, but it makes the requested deletion visible: the
only remaining concept is an exact per-read relation and its runtime resolver.

## Keep

- `_AccessGuard` and `_SourceAccessKey`: mask/fill identity is part of source
  identity; a plain `(name, index)` cache would reintroduce silent collisions.
- `_logical_memory_access`: CC 12 comes from explicit validation of operation,
  name, index node, store mode, and scheduler-node context. It is flat and no
  equivalent helper exists in the codebase.
- `_PointwiseRemapHandler._load_without_store_forwarding`: the external-source
  fallback must bypass the generic name-only `store_cache`; sharing the normal
  load helper would undo that guarantee.
- `_codegen_sub_parent_output_groups`: this is a useful unification. It removes
  duplicated nested/standalone replay and keeps resolver scope identical.
- `_ResolvedProjectedSource.value` only if F2a immediately consumes the raw
  live group-width CSE before materialization. It is unused by F1 itself; delete
  it if F2a no longer uses that seam.

## Verdict

F1 passes the old-scaffolding deletion gate and its new control flow is flatter
than the frozen implementation. The raw `+150` is defensible for exact access,
guard, liveness, fallback, and loud-miss semantics, but the grouped planner
record plus per-read runtime record is redundant. Flattening that boundary is
the one simplification worth doing before F2a; further helper extraction would
mostly game metrics or move complexity rather than remove it.

## Final flattened snapshot

This section supersedes the pre-flatten snapshot and recommendation above.

Snapshot: unstaged diff hash
`ce766480d4bed7972584fbbc146608dc4a3d6c50c3e5a075f6b98c8e65f1c16e`.

### Final inventory

Production diff:

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `torch/_inductor/codegen/simd.py` | 406 | 208 | +198 |
| `torch/_inductor/scheduler.py` | 123 | 177 | -54 |
| **Total** | **529** | **385** | **+144** |

The file-level AST inventory is the most stable comparison because the
flattening intentionally renamed several planner functions:

| Metric | Frozen base | Flattened F1 | Delta |
| --- | ---: | ---: | ---: |
| Functions/methods | 651 | 663 | +12 |
| Class definitions | 60 | 62 | +2 |
| Cyclomatic complexity | 3,553 | 3,591 | +38 |
| Maximum nesting | 8 | 8 | 0 |

Across functions changed, replaced, added, or deleted by the final diff:

| Metric | Frozen base | Flattened F1 | Delta |
| --- | ---: | ---: | ---: |
| Functions/methods | 27 | 39 | +12 |
| Function-body LOC | 1,442 | 1,585 | +143 |
| Cyclomatic complexity | 274 | 312 | +38 |
| Branch points | 247 | 273 | +26 |
| Maximum nesting | 5 | 4 | -1 |

Relative to the prior F1 snapshot, flattening reduced production net LOC from
`+150` to `+144`, aggregate CC growth from `+46` to `+38`, net function growth
from `+13` to `+12`, and net class/type growth from `+4` to `+2`. Raw LOC moved
only slightly because the explicit relation names and validation remain, but
the ownership model is materially smaller.

The final hotspots are:

| Symbol | Frozen base | Flattened F1 | Assessment |
| --- | ---: | ---: | --- |
| `_logical_memory_access` | absent | CC 12, nesting 2 | Flat validation; keep. |
| `_SubParentValueResolver.materialize_sources` | absent | CC 11, nesting 3 | Guard/source alternatives are explicit; keep. |
| `_try_get_sub_parent_access_relations` | CC 16 predecessor | CC 17, nesting 3 | One added requiredness fact; keep. |
| `_sub_parent_broadcast_access_relations` | CC 15 predecessor | CC 16, nesting 2 | One relation-construction branch; keep. |
| `_sub_parent_internal_access_relations` | CC 17, nesting 5 predecessor | CC 16, nesting 4 | Improved. |
| `_prove_staged_fusion_dependencies` | CC 36, nesting 3 | CC 35, nesting 3 | Improved; P2G owns the remaining legacy branch. |

No new F1 function exceeds CC 12. Aggregate changed-surface nesting falls from
5 to 4. `_PointwiseRemapHandler.load` alone rises from nesting 2 to 3 because
the exact-relation miss chooses between an ordinary remapped load and the
store-cache-bypassing external reload; that branch is the required behavior,
not extraction debt.

Tests now add 293 lines and delete 35, net `+258`. The apparent extra new test
names include two mechanical projection-to-relation renames; the actual F1
budget remains four new test methods plus extensions to existing tests.

### API and vocabulary result

The planner now emits one exact per-read record:

```python
SubParentAccessRelation(
    source_accesses=...,
    consumer_access=...,
    parent_lane=...,
    requires_live_source=...,
)
```

Codegen consumes that same record directly. The flattening removed:

- `ProjectedConsumerAccess`;
- `_ProjectedLoad`;
- `ProjectedSourceAccess.has_lane`;
- the grouped-consumer-to-runtime-load conversion;
- `_planned_consumer_names`, replacing it with one name-then-exact-access map;
- the extra consumer loop in the plan's exact-access iterator.

The old projection/layout vocabulary and compatibility scaffolding are
grep-clean. The only remaining uses of the word "projection" describe the
physical lane operation in comments or an error message. The remaining local
`broadcast_source_names` sets classify planner candidates and do not cross the
planner/codegen boundary.

### Remaining simplification assessment

No further structural simplification is justified inside F1:

- `_AccessGuard`, `_SourceAccessKey`, and `_ResolvedSubParentSource` represent
  independent facts: exact guard identity, cache identity, and a live source
  selected for one consumer.
- `_SubParentValueResolver` is one class because recording and resolving must
  share exact source keys and CSE liveness. Splitting it would add state transfer.
- `resolve_sources` and `materialize_source` remain separate for F2a, which
  needs the raw live group-width value before eager materialization and guard
  application. If F2a abandons that path, remove the currently F1-unused
  `_ResolvedSubParentSource.value` field and reconsider the split.
- `materialize_sources` may revisit equivalent source sets for several
  consumers, but `_materialized` prevents duplicate emitted reshapes/splits.
  Adding a second deduplication index would increase permanent state for a tiny
  planning-time saving.
- The local `.normalize()` calls are the explicit scheduler-to-codegen frame
  boundary. Removing them to save lines would weaken the key contract.
- The inherited parent-to-grouped equivalence branch is the only remaining
  dual proof source, and it is already isolated as P2G. Reworking it in F1 would
  reopen the architectural scope the user explicitly paused.

Final verdict: the flattened F1 is within its `+150` production budget, has no
new high-complexity function, and has removed the old projection-era API rather
than layering indexed forwarding beside it. Further changes now would mostly
rename or redistribute the irreducible exact-access checks; proceed to F2a on
this structure after correctness verification completes.
