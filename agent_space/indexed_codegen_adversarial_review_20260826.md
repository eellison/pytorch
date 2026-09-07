# Indexed codegen adversarial review

Date: 2026-08-26

Reviewed worktree: `agent_space/followup_indexed_rebase_wt`

Unstaged diff hash at final review: `a633c31b5c4cdf1b78c579d63aa1c65ba0d8c3f47e4c3db2102ab7087da1df15`

## Verdict

No remaining concrete correctness bug was found in exact access recovery,
guard/fill handling, CSE liveness, store recording, external fallback,
shape-derived materialization, or loop-boundary handling.

Two completion gates remain:

1. The exact final snapshot still needs the full nested-reduction file and the
   protected normalized-kernel corpus rerun. The full file passed on the
   behavioral-fix snapshot, and focused final-snapshot coverage is green, but
   the acceptance document requires the full commands on the exact result.
2. Two scheduler helpers are each one cyclomatic point above their frozen
   baselines. The increases encode exact requiredness and wrapped consumer
   records rather than avoidable nesting. They should be explicitly accepted
   as semantic cost, or replaced only by a genuine relation-construction
   simplification. Replacing comprehensions with `map` only to change the AST
   score would make the code worse.

## Formal Complexity Audit

Production delta, tests excluded:

- `simd.py`: `+396/-210`
- `scheduler.py`: `+75/-111`
- total: `+471/-321`, net `+150`

The delta is at the documented F1 limit and is justified by removing the old
resolver/layout/name contract while adding exact access identity, guard keys,
liveness checks, physical-load fallback, and one shared nested/standalone
replay path.

Mechanism inventory:

- six private types added and two compatibility types deleted;
- two module helpers added;
- 13 `_IndexedProjectedValueStore` methods;
- no public API;
- no retained layout enum, stage name views, family name cache, old resolver,
  forwarded/masked name plumbing, or `store_cache[name]` fallback.

Measured gates that now pass:

- `_logical_memory_access`: CC 12, nesting 2;
- `_IndexedProjectedValueStore.materialize_projections`: CC 11, nesting 3;
- `_sub_parent_internal_projections`: nesting reduced from 5 to 4;
- `_prove_staged_fusion_dependencies`: CC 36, unchanged;
- `_PointwiseRemapHandler.load`: CC reduced from 7 to 6;
- standalone sub-parent orchestration: CC reduced from 11 to 8.

Remaining measured increases:

- `_try_get_sub_parent_source_projections`: CC 16 to 17. The added branch cost
  is the exact temporal-write requiredness calculation and lane-bearing
  consumer record.
- `_sub_parent_broadcast_projections`: CC 15 to 16. The added cost is wrapping
  each exact consumer access in its relation record.

## Resolved Adversarial Findings

### Resolver ownership escaped the sub-parent stage

An earlier snapshot enabled `_IndexedProjectedValueStore.resolve_load()` in
parent-full and grouped replay. A graph where `x` was shared by an outer stage
and the sub-parent output failed with:

`sub-parent stage has no planned projection for MemoryDep('arg0_1', ...)`

The resolver is now enabled only by `_codegen_sub_parent_output_groups`.
Earlier stages retain the wrapper only to record exact sources and stores. The
regression is folded into the existing parent-full-source integration test via
the `shared_external_source` parameter.

### Guarded source loads were not recorded

An earlier snapshot recorded only unguarded loads. The current `load()` records
every exact access under `_AccessGuard(mask, fill)`. The permanent matrix now
covers exact guarded lookup, unguarded-to-guarded lookup with explicit
`where`, `fill=None`, guarded-to-unguarded rejection, different-mask rejection,
different-fill rejection, boolean fill coercion, typed fill identity, and
signed zero identity.

### Source alternatives stopped at the first live but unusable value

The resolver previously chose one live source before attempting shape
materialization. A required relation failed if that source had an unsupported
shape even when a later equivalent source was projectable. `resolve_load()` now
tries all live alternatives until one materializes.

### Eager lifetime materialization emitted every equivalent alternative

The first alternative fix initially materialized all matching sources. That
could emit unused duplicate splits/broadcasts before the first consumer.
`materialize_projections()` now stops at the first usable source per guard, and
an unguarded success covers guarded consumers.

## Codegen Path Assessment

- `_logical_memory_access` rebuilds the node-local temporal `MemoryDep` and
  validates operation, name, index node, store mode, and current scheduler node
  before normalization.
- A same-name planned consumer without its exact normalized access raises. A
  required source miss reports both source alternatives and the consumer.
- Source liveness is checked before consulting a materialized memo. The stale
  CSE test includes an existing memo, so reversing that order is detected.
- Ordinary loads, stores, grouped `store_reduction`, and sub-parent internal
  stores are recorded through the same wrapper. Every non-`None` store mode is
  excluded; tests cover add atomic, non-add atomic, and TMA representatives.
- External planned-source misses call `kernel.load`/`indirect_load` directly,
  preserving load accounting and trace behavior while bypassing generic
  name-only store forwarding.
- Live-value shape is authoritative: scalar/singleton and child-width values
  are direct, group-width values broadcast, parent-width values split, and
  unknown shapes fail closed. A lane is required only when materialization
  actually returns a tuple.
- Only required lane projections suppress the standalone pre-epilogue flush.
  Direct and group-width values retain the normal boundary. Nested and
  standalone output replay share one handler-lifetime helper.

## Projection Surface

`ProjectedSourceAccess`, `ProjectedConsumerAccess`, and `source_projections`
are not the deleted layout-policy machinery. They carry the essential exact
planner-to-codegen relation: equivalent source accesses, exact consumer read,
per-consumer lane witness, and requiredness. Removing them would reintroduce
name or shape inference at the boundary.

Renaming them to `SubParentAccessRelation`, `SubParentConsumerAccess`, and
`access_relations` could make that distinction clearer, but it is naming churn,
not a code-size or correctness improvement.

`_ResolvedProjectedSource.value` is unused by F1. Remove it if F1 is reviewed as
a standalone endpoint. Keeping it is reasonable only as an explicit immediate
F2a handoff, because F2a must inspect a live lane-free source before eager
projection and guard application. The single-caller `can_defer_projection`
method has the same justification; otherwise it can be inlined.

## Verification

- exact final snapshot: `test_inductor_scheduler.py`: 124 passed, 6 skipped;
- exact final snapshot focused nested source/broadcast/internal/mask set:
  12 passed, 385 deselected;
- behavioral-fix snapshot full `test_nested_reduction.py`: 389 passed,
  8 skipped;
- shared-external-source scope regression: 4 passed;
- protected pre/post corpus: 16 persistent/looped kernel files were byte
  identical, including nested/standalone NVFP4, MXFP4, MXFP6, reduced broadcast,
  and internal-source cases;
- `py_compile` passed for both production and test files;
- `git diff --check` passed.

## Recommendation

**Needs discussion**, not a correctness rewrite. Accept the two one-point
hotspot increases explicitly, retain the resolved-source value only if F2a is
immediate, and run the exact final full nested and kernel-hash gates before
declaring F1 complete.
