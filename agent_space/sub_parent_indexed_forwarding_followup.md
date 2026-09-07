# Sub-parent indexed-forwarding follow-up

## Decision

Keep the proven name-keyed compatibility layer in #191775. Replace it in a
dedicated follow-up that owns the planner-to-codegen access contract.

Passing `source_projections` into the resolver is not independently mechanical:
nested codegen activates reduced `BROADCAST` projections while standalone
codegen intentionally uses ordinary CSE/store-cache broadcasting, and
`IDENTITY` epilogue writes are distinct from parent-written `INTERLEAVED`
sources whose liveness controls the final loop boundary. Moving the existing
three views without preserving those distinctions changes behavior without
fixing index-blind forwarding.

## Goal

Resolve a derived-stage load from its planner-proven `ProjectedSourceAccess`,
including the consumer access, instead of selecting a treatment from the
buffer name alone.

The source relationship is the information an ordinary memory load would have
carried in its index:

```text
parent-width value  -> split into child lanes  (INTERLEAVED)
group-width value   -> repeat across child     (BROADCAST)
child-width value   -> forward unchanged      (IDENTITY)
```

Two reads of one buffer name may therefore use different mappings. Each read
must match its own planned consumer access; an unplanned read must fail loudly
rather than inheriting the name's treatment.

## Main design problem

Planner records are expressed in scheduler-node loop frames. Codegen receives
loads after loop merging and kernel-tree binding. The follow-up must define one
normalization into the emitted source frame so a codegen `(name, index)` can be
matched to the corresponding planned consumer without broad name or normalized
index equivalence.

Fusion and codegen may rebuild the plan, as they do today. The contract is the
relation and its normalization rule, not object identity across the fusion
boundary.

## Required properties

1. Use temporal dependency names to identify writers. Apply mutation renames
   only when matching producer and consumer accesses.
2. Preserve loud-on-miss behavior for in-kernel-written values. An external
   value may fall back to a normal derived-index memory load.
3. Preserve target-family masks when materializing or forwarding a value.
   Masked-context loads must not enter the ordinary forwarding cache unless
   their predicate is represented in the relation.
4. Preserve the standalone and nested scheduling contracts while removing
   their name-derived compatibility views.
5. Record inherited parent-to-grouped relations so staged fusion can delete
   `_fusable_read_after_index_equivalence` rather than maintaining two provers.

## Acceptance tests

- The original P1 transpose/X-R-boundary adversary declines.
- An unowned read declines; a read owned by both stages takes the stricter
  planned-relation path.
- Multiple same-name writes decline without exact reaching-write identity.
- Mutation-renamed producer/read matching succeeds only at the matching step.
- Atomic, TMP, StarDep, WeakDep, indirect, and synchronization cases retain
  their current fail-closed behavior.
- Persistent and looped INTERLEAVED, BROADCAST, and IDENTITY cases remain
  numerically exact.
- Masked odd-tail gather/store cases retain the target-family mask.
- Mutating away each required relation causes a focused test to lose fusion or
  fail compilation, rather than silently forwarding by name.
- Existing nested broadcast, standalone reduced broadcast, looped internal
  source, NVFP4/MXFP4, and MXFP6 kernel forms remain byte-identical unless a
  reviewed codegen improvement intentionally changes them.

## Deletion endpoint

Remove these name-based compatibility views from `SubParentEpilogueStage`:

- `source_layouts`
- `broadcast_source_names`
- `internal_dependency_names`

Then remove the resolver's per-name treatment and the staged-fusion legacy
equivalence branch once every accepted relation is planner-owned.
