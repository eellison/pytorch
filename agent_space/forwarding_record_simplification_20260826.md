# Indexed forwarding record simplification

Date: 2026-08-26

Reviewed worktree: `agent_space/followup_indexed_rebase_wt`

Reviewed unstaged diff hash:
`a633c31b5c4cdf1b78c579d63aa1c65ba0d8c3f47e4c3db2102ab7087da1df15`

## Verdict

The current planner-to-codegen data is represented three times:

1. `ProjectedSourceAccess` groups equivalent source accesses and consumers.
2. `ProjectedConsumerAccess` adds a lane witness to each consumer.
3. `_IndexedProjectedValueStore.__init__` immediately flattens both into one
   `_ProjectedLoad` per exact consumer read.

That double planner/runtime representation is unnecessary. Replace all three
records with one per-read relation used directly by planning, fusion proof, and
codegen:

```python
@dataclasses.dataclass(frozen=True)
class SubParentAccessRelation:
    """Source accesses that may satisfy one exact sub-parent read."""

    source_accesses: tuple[MemoryDep, ...]
    consumer_access: MemoryDep
    parent_lane: int | None
    requires_live_source: bool
```

This is the minimum sound boundary record. Each field is load-bearing:

- `source_accesses` must remain plural because equivalent parent accesses may
  have different CSE lifetimes or materializable shapes. The current unit test
  proves that the first live alternative can be unusable while a later one is
  valid.
- `consumer_access` is the exact per-read authorization. A buffer name is not
  sufficient.
- `parent_lane` is the planner's witness for the non-bijective parent split. It
  cannot be recovered from value shape alone.
- `requires_live_source` distinguishes an external source that may be reloaded
  from an in-kernel-written value whose miss must be a compiler error.

The stage field should become `access_relations`, and codegen should normalize
and index the same relation directly. A name-indexed outer dictionary retains
the cheap prefilter without a parallel name set:

```text
dict[temporal_name, dict[normalized_consumer_access, relation]]
```

This removes `ProjectedConsumerAccess`, `_ProjectedLoad`, `has_lane`,
`_planned_consumer_names`, the constructor's planner-to-runtime flattening, and
one nesting level from access-pair iteration.

## Why `MemoryDepMatch` Is Not The Plan Record

`MemoryDepMatch` should remain the scheduler's fusion result, not the codegen
contract:

- It is binary, while one consumer needs a set of equivalent source
  alternatives.
- Its first field is specifically a producer write. Sub-parent sources may be
  external loads as well as writes.
- It carries no parent-lane witness or reload/liveness contract.
- Fusion returns mutation-renamed matches, while the codegen plan must retain
  raw node-local temporal names and apply renames only when matching fusion
  dependencies.
- It covers producer-output residuals for vertical legality, not every exact
  external, broadcast, or same-stage relation codegen must resolve.

The plan may expose an iterator that converts each relation to
`MemoryDepMatch(source, consumer)` for fusion membership. Persisting only those
binary matches would require a second lane map and regrouping by consumer,
which recreates the removed scaffolding in a less explicit form.

## Requiredness

Do not derive `requires_live_source` from `kernel.store_buffer_names` or from
stores observed while emitting code:

- A bare name loses temporal/version identity.
- If a planned producer is accidentally omitted, observation-based derivation
  would classify its consumer as external and permit an unsafe memory reload.
- The flag is needed before emission to decide whether a lane relation may keep
  the standalone parent pass open.

It is technically possible to recompute the fact at codegen from raw planned
writes, but that is not a simplification. The correct writer inventory spans
standalone parent nodes and, for nested codegen, parent nodes, grouped nodes,
outer local-reduction-input nodes stored only in `pointwise_domains`, and prior
sub-parent output groups. That topology walk must happen before normalization
or mutation renaming.

The planner already has the exact provenance at relation construction. Keep
the derived boolean on the relation, rename it from `must_forward` to
`requires_live_source`, and compute it as:

```text
any(raw source access is a raw node-local write emitted by this kernel)
```

The aggregation is relation-wide. If any equivalent source is the in-kernel
writer, a total miss may not fall back to memory.

## Eager Materialization And Flush Boundaries

Flattening consumers must preserve the current materialization behavior:

1. Try all equivalent source alternatives until one materializes.
2. Group candidates by access guard, prefer an unguarded source, and stop after
   the first usable source for that guard.
3. Do not eagerly materialize every alternative.
4. Repeated per-consumer relations may revisit one source set, but the existing
   `_materialized` cache makes successful emission idempotent. Optional dedup
   must key the exact normalized source tuple, never the buffer name.

The two lane filters remain intentionally different:

- Nested emission eagerly materializes every relation with
  `parent_lane is not None` before sub-parent replay.
- Standalone emission suppresses the pre-epilogue flush only for
  `requires_live_source and parent_lane is not None`.

Combining those predicates would recreate the earlier bug where a direct or
broadcast in-kernel source unnecessarily kept the reduction loop open.

## API Cleanup

Suggested mechanical names after flattening:

- `source_projections` -> `access_relations`
- `_try_get_sub_parent_source_projections` ->
  `_try_get_sub_parent_access_relations`
- `_sub_parent_internal_projections` ->
  `_sub_parent_internal_access_relations`
- `_sub_parent_broadcast_projections` ->
  `_sub_parent_broadcast_access_relations`
- `_IndexedProjectedValueStore` -> `_SubParentValueResolver`
- `_ResolvedProjectedSource` -> `_ResolvedSubParentSource`
- `get_projected_load` -> `get_relation`
- `materialize_projections` -> `materialize_sources`

Keep the three planner proof helpers separate. Interleaved lane selection,
trailing broadcast, and temporal same-stage forwarding prove different index
relations; merging their proofs would hide correctness conditions rather than
remove them.

## Estimated Effect

Against the reviewed `+150` production-line F1 snapshot, this should remove
approximately 25-40 production lines and leave F1 around `+110` to `+125` net:

- two fewer record types overall;
- one fewer method (`has_lane`);
- one less nested loop in resolver initialization;
- one less Cartesian nesting level in plan-to-fusion conversion; and
- no planner/runtime adapter record.

The maximum module nesting is unlikely to change, but the resolver constructor
and fusion-pair path become locally flatter. Further helper extraction solely
to lower AST scores would move complexity rather than remove it.

`_ResolvedSubParentSource` may remain for the immediate lazy-broadcast follow-up,
which needs the raw live CSE value before eager materialization. If that work is
not immediate, F1 can inline it, but deleting and immediately recreating the
same seam is not useful churn.
