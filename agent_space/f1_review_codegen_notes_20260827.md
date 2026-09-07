# F1 codegen review: exact sub-parent access resolution

## Scope

This guide covers the codegen half of commit `fc09fb38287`, including the
mask-ownership simplification, relative to `d4167dba5b8`, in:

`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

The change replaces buffer-name plus layout-category forwarding with exact
source-to-consumer `MemoryDep` relations. The current codegen delta is
concentrated in `torch/_inductor/codegen/simd.py` (`+316/-208`). Scheduler code
is included here only where it defines the contract consumed by codegen.

Useful commands:

```bash
cd /data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt
git diff HEAD^ -- torch/_inductor/codegen/simd.py
git diff HEAD^ -- torch/_inductor/scheduler.py
```

## The change in one sentence

The old implementation said "a load of buffer X uses layout Y." The new
implementation says "this exact consumer access may use one of these exact
source accesses, selecting this parent lane if needed, and the source either
must still be live or may fall back to memory."

## Recommended review order

1. Read the planner/codegen record:
   [`SubParentAccessRelation`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095>).
2. Read how codegen reconstructs the same logical access:
   [`_logical_memory_access`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527>).
3. Read resolver construction and recording:
   [`_SubParentValueResolver.__init__`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2369>) through
   [`store_reduction`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2439>).
4. Read liveness, lookup, and resolution:
   [`_materialize`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2444>) through
   [`resolve_load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2531>).
5. Read the conservative mask boundary in
   [`resolve_sources`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492>) beside existing
   [`TritonKernelOverrides.masked`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532>).
6. Read shape materialization and mask assignment:
   [`materialize_value_at_sub_parent_resolution`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074>) and
   [`set_value_masks`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1635>).
7. Finish with nested and standalone integration:
   [nested emission](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3361>) and
   [standalone emission](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3841>).

## Before and after

| Parent implementation | F1 implementation | Review significance |
| --- | --- | --- |
| `ProjectedSourceAccess` | `SubParentAccessRelation` | One record now describes one exact consumer access, its source alternatives, optional lane, and liveness requirement. |
| `SubParentSourceLayout.{INTERLEAVED,BROADCAST,IDENTITY}` | No layout enum | Codegen derives split, broadcast, or direct use from the live CSE value's shape. |
| `source_projections` | `access_relations` | The planner/codegen boundary carries access facts, not name categories. |
| `_SubParentSourceLoadResolver` | `_SubParentValueResolver` | The resolver records loads, stores, and reduction stores, then resolves exact reads. |
| `family.remapped_values[name]` | `_values[MemoryDep]` | Unguarded values are keyed by normalized access, not only buffer name. |
| `family.resolve_load(name, index)` | `get_relation` plus `resolve_load` | A name hit is insufficient; the exact consumer `MemoryDep` must match. |
| `source_layouts`, `broadcast_source_names`, `internal_dependency_names` | Deleted | The stage no longer exposes three name-based compatibility views. |
| `forwarded_store_names`, `masked_forward_names` | Deleted | Exact relations replace the name sets; existing masked-load codegen retains mask/fill ownership. |

The old `_DerivedIterationFamily.remapped_values` state and its name/index lane
lookup are deleted. `_DerivedIterationFamily` now only represents an iteration
family and its index/mask mechanics:
[`_DerivedIterationFamily`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1579>).

## End-to-end flow

```text
scheduler proves SubParentAccessRelation records
                       |
                       v
resolver normalizes source and consumer MemoryDeps
                       |
                       v
parent/grouped emission runs through _SubParentValueResolver
  load/store/store_reduction -> reconstruct logical MemoryDep
  -> record unguarded CSE value under exact source access
                       |
                       v
materialize required parent-lane values before a CSE flush
                       |
                       v
sub-parent replay runs through _PointwiseRemapHandler
  -> reconstruct exact consumer MemoryDep
  -> find its relation
  -> forward only when existing masked codegen does not require a physical fill
  -> find a live unguarded source
  -> split, broadcast, or directly use by value shape
  -> select the planned lane
  -> existing ops.masked applies any outer predicate/fill
                       |
          +------------+-------------+
          |                          |
 required in-kernel miss       optional external miss
          |                          |
          v                          v
  compiler assertion       ordinary derived-domain reload
                           without store-cache forwarding
```

## 1. The record codegen consumes

[`SubParentAccessRelation`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095>) contains:

- `source_accesses`: one or more equivalent source-side `MemoryDep` alternatives.
- `consumer_access`: the one exact load authorized to reuse those sources.
- `parent_lane`: a constant lane for a parent-width split, or `None` when shape
  alone determines direct use or group-width broadcast.
- `requires_live_source`: `True` when the source is written in this kernel and
  therefore cannot safely fall back to a memory reload.

The record validates only that it is nonempty and all accesses name the same
buffer. The actual index relation is proved by the planner and checked again by
fusion legality. The main parent-to-child builder records the constant lane and
derives requiredness from whether any source access is an in-kernel write:
[`_try_get_sub_parent_access_relations`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094>), especially
[the relation construction](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1192>).

Internal epilogue values and reduced/group-width values are always in-kernel
sources, so their relation builders set `requires_live_source=True`:
[`_sub_parent_internal_access_relations`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1266>) and
[`_sub_parent_broadcast_access_relations`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373>).

The stage stores the records directly, with no derived name sets:
[`SubParentEpilogueStage`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2157>).

## 2. Reconstructing the exact access during codegen

[`_logical_memory_access`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527>) is the bridge from an interpreted load/store back to the scheduler's dependency vocabulary.

It reads the current FX operation, validates the operation kind and buffer name,
finds the named `get_index` expression in the current `SchedulerNode`'s
`LoopBody`, rebuilds a `MemoryDep` from that body's original variables/ranges,
checks store mode, and normalizes the result.

The important detail is that this uses the node's logical loop-body frame, not
the already-remapped Triton index passed to the handler. That makes the lookup
comparable to the scheduler record even though codegen is currently executing
under a derived iteration family. It also keeps temporal buffer names; mutation
renames are used by fusion legality, not by codegen's value cache.

Every mismatch is an assertion. This helper is intentionally not a best-effort
parser because an incorrect access identity would turn into silent wrong
register forwarding.

## 3. Resolver initialization

[`_SubParentValueResolver.__init__`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2369>) normalizes both sides of every relation after codegen's loop merging has occurred.

It builds two indexes:

- `_source_accesses`: all normalized accesses worth recording while the parent
  and grouped stages execute.
- `_relations_by_name[name][consumer_access]`: the exact consumer lookup table.

The outer name dictionary is only a cheap first filter. The inner normalized
`MemoryDep` lookup is the authorization decision. If two records assign
different meanings to the same exact consumer access, construction raises at
[the conflict check](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2399>).

Runtime state is separate:

- `_values` maps a normalized source `MemoryDep` to its unguarded emitted CSE value.
- `_materialized` caches the corresponding direct, broadcast, or split result.

## 4. Recording loads and stores

The source stages execute with the resolver installed as the active ops handler.
Its methods first delegate to ordinary codegen, then record the resulting value
only if the reconstructed access appears in `_source_accesses` and no load mask
is active:

- [`_record`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2407>) ignores guarded executions, normalizes the access, records only planned sources, and drops any stale materialized form for that access.
- [`load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2419>) records the CSE result of an ordinary unguarded load.
- [`store`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2425>) records the value written by a non-atomic, unguarded store after emitting the store.
- [`store_reduction`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2439>) records the completed unguarded reduction result.

Atomic/TMA store modes are not recorded. Their operand is not necessarily the
value subsequently present in memory, so treating it as the produced source
would be unsound.

Guarded source values are intentionally absent from the resolver cache. This
keeps source identity in the same normalized `MemoryDep` vocabulary as the
planner and leaves predicate/fill semantics in the existing masked-load path.

## 5. Exact consumer lookup

[`_PointwiseRemapHandler.load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2321>) first asks the resolver for the current load's relation.

[`get_relation`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2479>) has three outcomes:

- The buffer name has no planned relation: return `None`; this is an ordinary
  derived-domain load.
- The name and exact normalized consumer access match: return that relation.
- The name is known but the exact consumer access is absent: raise. This is the
  loud-on-unplanned-read property that replaces the old name-only forwarding.

That final branch is important. Falling back merely because the index differs
would hide a planner/codegen disagreement and could also accidentally reuse a
same-name store-cache value at the wrong position.

## 6. Existing masked-load ownership and source alternatives

[`resolve_sources`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492>) considers only unguarded live values whose exact access is listed by the relation. It declines immediately when `_load_other` is concrete.

This follows existing
[`TritonKernelOverrides.masked`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532>):

- a direct external load receives a concrete `_load_other`, so the physical
  `tl.load` owns its predicate and fill and the resolver does not intercept it;
- a body that may return an in-kernel/store-cache value receives
  `_load_other=None`, and the callback emits an outer `where` around the result;
- an unmasked consumer also has `_load_other=None`.

The resolver returns its unguarded value unchanged in the latter two cases. It
does not implement a parallel guard-matching or fill-reconstruction system.
The only demonstrated capability lost is reusing an earlier unguarded external
load when a later direct masked load needs a concrete fill; that optional source
reloads instead, without losing staged fusion or correctness.

Each candidate must also still satisfy `kernel.cse.contains_value`. That method
checks ordinary CSE values, store-cache values, and reduction-cache values:
[`CSE.contains_value`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/common.py:2138>). A value dropped by a loop/body flush is therefore not treated as live merely because the resolver still has its Python object.

## 7. Shape-driven materialization

[`_materialize`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2444>) first enforces liveness, then delegates to
[`materialize_value_at_sub_parent_resolution`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074>). The result is cached per exact source access.

The layout method chooses from the live CSE shape:

| Live value shape on the grouped axis | Result |
| --- | --- |
| Scalar `()` | Reuse directly. |
| Singleton `1` | Reuse directly; Triton broadcasting supplies the invariant axis. |
| Child width `parent_block / factor` | Reuse directly and attach child-family masks. |
| Group width `num_groups` | Reshape/broadcast each group value across `G / factor` child positions, then attach masks. |
| Full parent width `parent_block` | Reshape to `[..., child, factor]`, split into `factor` CSE values, then attach masks to every part. |
| Anything else or unknown shape | Return `None`; caller either tries another source, reloads an optional external value, or fails a required source. |

The child-width check deliberately precedes group-width broadcasting. When
`group_size == factor`, the two symbolic widths coincide, but one child element
already represents one group and must remain a direct value.

For a parent-width example with factor 4, a live `[B, R]` value becomes
`[B, R/4, 4]`, then four `[B, R/4]` CSE values. `parent_lane=2` selects the third
one. The lane is planner-proven; codegen no longer recovers it from the emitted
index with `% factor`.

For a group-width scale example, `[B, R/G]` becomes
`[B, R/G, G/4]` and then `[B, R/4]`. No lane is selected because the scale is
constant within each group.

## 8. Masks

Every nonscalar value split, broadcast, or directly carried at child resolution
receives masks from the active sub-parent iteration family through
[`set_value_masks`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1635>).

[`mask_vars_for_shape`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1609>) left-pads lower-rank shapes as Triton broadcasting does, keeps only family masks whose non-singleton dimensions fit the value, and lets `kernel.filter_masks` remove masks that are statically unnecessary. The planner's exact divisibility proof is what makes the derived-family mask an exact projection of the parent tail mask.

These are derived-family validity masks used by indirect indexing. They are
separate from an `ops.masked` consumer predicate. Consumer predicates and fills
remain entirely owned by `TritonKernelOverrides.masked`.

## 9. Required-source failure versus external fallback

[`resolve_load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2531>) tries every planned source alternative. If none remains live/materializable:

- `requires_live_source=True`: raise a compiler assertion containing the source
  and consumer accesses. An in-kernel-written value cannot silently reload:
  the buffer may be removed, and a cross-thread read-after-write may not be
  ordered.
- `requires_live_source=False`: return `None`, allowing an external buffer to be
  reloaded at the consumer's derived index.

The optional fallback uses
[`_load_without_store_forwarding`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2338>), not the normal wrapper load. This intentionally bypasses name-keyed store forwarding while retaining normal Triton indexing, indirect-load handling, load accounting, and operation tracing.

If a prior in-kernel store was invalidated, the fallback marks the buffer as
required so kernel-local buffer elimination cannot remove it:
[`must_keep_buffers`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/common.py:2530>). The underlying Triton load also emits the existing read-after-write barrier for an invalidated store:
[`TritonKernel.load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:5070>).

## 10. Early materialization and liveness

[`materialize_sources`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2462>) exists only for relations that need a parent-lane split before a codegen flush can invalidate the parent-width CSE.

It tries the relation's normalized source alternatives and accepts the first
materializable live source. Missing optional sources are harmless. Missing
required sources raise immediately.

Relations with `parent_lane=None` are left lazy. Direct child-width values,
group-width scale broadcasts, scalars, and singletons are materialized on the
first actual consumer load. This keeps the pre-epilogue eager boundary limited
to the one operation that cannot be recovered after losing the parent-width
register: splitting it into lanes.

## 11. Nested-pipeline integration

The nested path constructs one resolver from the stage's access relations at
[lines 3470-3484](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3470>).

The resolver wraps both source-producing regions:

- The outer schedule at [lines 3484-3491](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3484>).
- Parent-full pointwise work plus the grouped schedule at
  [lines 3510-3529](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3510>).

The grouped reduction handler wraps the current ops handler, so its ordinary
loads/stores still flow through the resolver:
[`_codegen_grouped_reduction`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3697>).

Before replay, every relation with a parent lane is offered to
`materialize_sources`; required relations must succeed. The shared helper then
replays each output group/lane with indexed forwarding active:
[`_codegen_sub_parent_output_groups`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3795>).

## 12. Standalone integration

The standalone path begins at
[`_codegen_reduction_with_sub_parent_epilogue`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3841>).

It extracts only required lane relations at
[lines 3858-3862](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3858>). The parent schedule then runs under the resolver at
[lines 3917-3927](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3917>).

The flush decision is the load-bearing part:

- With no required lane relation, flush the parent body before the epilogue.
  Reduced cached values may remain live, and optional external values may reload.
- With a required lane relation, materialize it before the flush and keep the
  parent and derived replay in the same pending body.

The output groups are then replayed through the same helper as the nested path,
followed by the final `kernel.codegen_body()`.

## 13. Concrete load traces

### Parent-width in-kernel source

Example: an NVFP4/MXFP6 pack reads fixed lanes of a full-resolution value.

1. The planner records the full parent access, the exact child consumer access,
   `parent_lane=0..factor-1`, and `requires_live_source=True`.
2. Parent emission records the live unguarded `[XBLOCK, RBLOCK]` CSE under its
   exact access.
3. Before a destructive flush, `materialize_sources` reshapes and splits it.
4. Replay looks up the exact consumer relation and selects the recorded lane.
5. A stale or absent value raises; memory fallback is forbidden.

### Reduced/group-width scale

Example: `[B, D/G]` scale consumed by a `[B, D/factor]` packing epilogue.

1. The relation has `parent_lane=None` and `requires_live_source=True`.
2. The reduction store records a group-width value.
3. First use sees `parent_dim == num_groups` and broadcasts each scale across
   `G/factor` child positions.
4. The materialized broadcast is cached for later output-lane replays.

### Shared external source

Example: a graph input is read in both the parent and sub-parent stages.

1. The relation has `requires_live_source=False`.
2. If the parent load is still live, codegen may split and forward it.
3. If it is no longer live, resolution returns `None` and the handler emits a
   real derived-domain load without consulting store forwarding.

### Masked consumers

For a masked body whose result may come from an in-kernel/store-cache value,
`TritonKernelOverrides.masked` sets `_load_other=None`. The resolver may return
an unguarded exact source, and the existing callback applies
`where(mask, result, fill)` around the body.

For a direct external masked load, `_load_other` is concrete. The resolver
declines forwarding, and the ordinary physical `tl.load` carries the predicate
and fill. Guarded source executions are never cached.

### Internal epilogue value

Example: the first output group stores an intermediate read by a later group.

1. The planner records the exact store and later read with
   `requires_live_source=True`, `parent_lane=None`.
2. The resolver records the store value.
3. The later replay resolves that exact read and derives direct use or trailing
   broadcast from the stored CSE shape.

## High-scrutiny sections

1. **Logical access reconstruction**:
   [`_logical_memory_access`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527>).
   Confirm that every wrapped load/store form supplies the same operation,
   temporal name, index expression, ranges, and mode that planning recorded.
2. **Exact lookup and loud mismatch**:
   [`get_relation`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2479>).
   A same-name, different-index read must assert rather than inherit another
   relation or silently use name-based forwarding.
3. **Masked-load ownership**:
   [`_record`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2407>),
   [`resolve_sources`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492>), and
   [`TritonKernelOverrides.masked`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532>).
   Verify that guarded sources are ignored, concrete-fill consumers use a
   physical load, and the existing outer-`where` path handles forwarded values.
4. **Liveness and failure policy**:
   [`_materialize`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2444>) and
   [`resolve_load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2531>).
   Required in-kernel sources must never fall back; optional external sources
   must be allowed to reload.
5. **Store-cache bypass on fallback**:
   [`_load_without_store_forwarding`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2338>).
   This is what prevents a different-index load from receiving a same-name
   store-cache value.
6. **Shape dispatch**:
   [`materialize_value_at_sub_parent_resolution`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074>).
   Check the child/group collision ordering and the full-parent reshape/split
   axis convention.
7. **Mask ownership**:
   [`set_value_masks`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1635>) for derived-family tail/index validity, separately from
   [`TritonKernelOverrides.masked`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532>) for consumer predicates and fills.
8. **Flush boundary**:
   [standalone lines 3926-3939](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3926>) and
   [nested lines 3530-3546](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3530>).
   A required parent-width source must be split while its CSE is still live.

## Tests to keep open while reading

- Exact access identity, cache invalidation, required misses, and source
  alternatives:
  [`test_sub_parent_access_identity_and_source_cache`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:434>).
- Existing masked-load ownership for guarded sources, outer-`where` consumers,
  and concrete-fill fallback:
  [`test_sub_parent_resolver_uses_masked_load_ownership`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:518>).
- External fallback and atomic-store exclusion:
  [`test_sub_parent_external_fallback_and_atomic_store`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:552>).
- Child-width/group-width collision:
  [`test_group_width_equal_to_child_width_is_direct`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:582>).
- In-kernel source followed by mutation, which pins temporal-name behavior:
  [`test_producer_consumer_sub_parent_source_mutated_later`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1243>).
- Live in-kernel parent source versus shared external source fallback:
  [`test_producer_consumer_inlined_parent_full_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1402>).

## Lower-priority mechanical changes

These are worth confirming after the high-scrutiny path, but they do not define
new behavior independently:

- `RemappedRangeValue` renamed to `MaterializedSubParentValue`.
- `_SubParentSourceLoadResolver` renamed and expanded to
  `_SubParentValueResolver`.
- Planner helpers renamed from `*_projections` to `*_access_relations`.
- `projected_access_pairs` renamed to `sub_parent_access_pairs`.
- Old layout enum, stage name-view properties, forwarding-name parameters, and
  `family.remapped_values` plumbing deleted.
- The repeated output-group replay loop extracted into
  `_codegen_sub_parent_output_groups`.

## Current size and validation

Relative to `HEAD^`, production changes are `+447/-389`, net `+58` lines:
the scheduler is net `-50` and `codegen/simd.py` is net `+108`. Across the two
production files, aggregate AST cyclomatic complexity moves from 3,552 to
3,568. `_logical_memory_access` remains the only new function above CC 6; the
simplified `materialize_sources` is CC 6.

The current worktree passed the full scheduler suite (118 tests, 6 skipped),
the focused cat/MXFP6 set (4 tests), the scheduler `-k sub_parent` set (18
tests), `py_compile`, and `git diff --check`. Its full nested run completed 399
tests successfully with 8 skipped. An earlier AOT checksum failure was traced
to selecting a broken system `openssl` under the Conda library path.

The separately instrumented conservative policy passed the complete
nested-reduction suite (399 tests, 8 skipped) and preserved all ten protected
generated-kernel hashes. The reachability audit observed 943 successful
forwards, all from unguarded sources. Masked consumers that forwarded all used
`_load_other=None`; consumers with a concrete fill followed the ordinary
physical-load fallback. See
[`f1_access_guard_reachability_20260827.md`](</data/users/eellison/pytorch/agent_space/f1_access_guard_reachability_20260827.md>).

## Review endpoint

After this change, every forwarded sub-parent load should be explainable by one
`SubParentAccessRelation`, one live unguarded value keyed by its normalized
source `MemoryDep`, and one explicit shape materialization. If a review path
still depends only on a buffer name, an implicit layout category, or a separate
resolver-owned mask/fill algebra, it is outside the intended F1 design.
