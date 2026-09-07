# Indexed parent-to-grouped relation review (2026-08-26)

## Verdict

The attempted `_nested_stage_source_projections` implementation must not land.
It converted the legacy permissive normalization into a plan record without
adding the missing stage-coordinate proof. During this review it was reverted.

The current post-revert tree is safer and preserves the frozen behavior, but it
does **not** complete the requested F1 endpoint: parent-to-grouped reads still
use `_fusable_read_after_index_equivalence`, while only sub-parent reads require
an exact planned pair. F1 therefore still has two authorities for non-exact
staged dependencies, and removing a parent-to-grouped relation cannot kill
fusion.

## Blocking Findings

### 1. Stride-order normalization is not an exact value relation

The attempted builder accepted:

```python
same_index
or _deps_match_normalized(read, write)
or _consumer_broadcasts_source(read, write)
```

`_deps_match_normalized` uses `normalize_with_stride_order()` when ranks match.
That helper deliberately reorders loops by memory stride. It can prove that two
accesses cover the same dense address set, but not that the value produced at a
given staged coordinate is the value consumed at that coordinate.

For example, a row-major write `4*w0 + w1` over `(2, 4)` and a transposed read
`d0 + 4*d1` over `(4, 2)` can canonicalize to the same stride-ordered traversal.
They are not interchangeable for name/store-cache forwarding. This is the P1
class of X/R-boundary error that exact planning is meant to close.

`_deps_match_normalized` may remain narrowly local to the existing speculative
loop-reindexing profitability check. It must not authorize a staged relation,
and its docstring should not call stride-reordered equality the same row-major
mapping.

### 2. The broadcast helper erases the axis identity needed by nested codegen

`_consumer_broadcasts_source` has two context-free relaxations:

- it removes every consumer variable absent from the address; and
- it allows any same-rank dimension to expand by any integral factor whose
  synthetic tail simplifies out of the address.

Neither rule knows `grouped_axis`, `group_size`, or the consumer's classified
`PointwiseDomain`. Equal extents make this observably ambiguous. For example,
`write=w0` over `(16,)` and `read=d1` over `(16, 16)` compare equal after the
unused `d0` is removed, even though `d1` may be the wrong logical X/R axis.
Likewise, the quotient rule can accept regrouping on an axis other than the one
nested codegen splits.

The helper is acceptable only as the inherited, explicitly scoped legacy
behavior. It is not suitable for creating an "exact" relation. A new planner
proof must allow only the broadcast axes and exact factor implied by the
current `PointwiseDomainContext`.

### 3. The current fallback still violates single-authority F1

After the revert, `_prove_staged_fusion_dependencies` still builds
`nested_stage_reads` and `sub_parent_reads`, gives sub-parent ownership the
strict planned-pair path, and sends nested-only reads to
`_fusable_read_after_index_equivalence`. The existing
`test_nested_dependency_matches_scope_index_equivalence` explicitly expects a
nested read with an empty plan relation set to succeed.

That is a valid compatibility checkpoint, but it fails the stated F1 gates:

- the staged plan is not authoritative for parent-to-grouped edges;
- `_fusable_read_after_index_equivalence` and its broadcast proof remain;
- relation-removal mutation tests cannot cover this relation family; and
- future changes can drift between the planner's domain model and the generic
  scheduler equivalence logic.

Do not describe the current tree as complete indexed forwarding. Either label
it an interim Path-A state, or finish the explicit relation work before F1 is
reviewed as the final layer.

### 4. The attempted reaching-write scan was incomplete

The reverted builder collected only same-name writes whose tuple position was
less than the reader's position. This has four problems:

- one earlier write plus a second later same-name write was accepted rather
  than treated as an ambiguous multiwrite;
- a consumer before an in-stage writer was treated as an external read instead
  of forcing the stage to decline;
- tuple position was used as the reaching-write proof even though generated
  schedules are rebuilt and may contain reduction enable/disable boundaries;
  and
- non-`MemoryDep` writes with the same name were ignored when counting
  ambiguity.

Fusion's later `len(writes) == 1` check covers only writes visible on the
current producer argument after mutation renaming. It does not repair an
incorrect full-stage relation inventory. Planning must first inventory every
raw temporal write in the candidate, reject more than one writer for a name,
and prove that the unique writer precedes every recorded reader in the emitted
stage order or dependency DAG.

### 5. Mutation handling is only half specified

The correct boundary is to store raw node-local `MemoryDep`s and apply
`mutation_renames` only while matching the current producer and consumer.
However, the current fusion proof first renames every planned pair into a set.
Two temporal versions can collapse to the same renamed `MemoryDepMatch`, losing
the version identity that made the plan exact. In addition,
`ProjectedSourceAccess.__post_init__` requires every source and consumer to
already have one raw name, so it cannot represent a genuinely cross-version
relation.

The smallest safe F1 policy is to keep mutation-bearing staged candidates
unsupported and make that rejection cover the complete nested topology. If
cross-version forwarding is intentionally supported later, membership must be
checked against the raw `(source, consumer)` pair first; renaming is then used
only to locate the producer write for ordinary fusion legality. Do not collapse
the plan itself through `mutation_renames`.

### 6. Safety is split between planning and fusion

The attempted builder could record TMP, synchronizing, or non-injective writes,
then rely on `_memory_dep_supports_projection` to reject them later. A rebuilt
codegen plan therefore did not carry the same invariant as an accepted fusion
plan. The helper name is also too broad: it checks source admissibility, not the
source/consumer projection relation.

Apply source safety while constructing every non-exact relation, and retain a
small defensive check at fusion only if desired. Use `MemoryDep.is_indirect()`
rather than only testing `SymT.TMP`. For injectivity, an unused statically-unit
write axis is harmless; the current all-vars-present condition should not make
`B=1` a lost fusion. All non-unit axes must participate, and the normalized
write must still be provably dense.

## Smallest Robust Design

1. Leave `fusable_read_and_write` as the first, ordinary exact-match path. Do
   not record those pairs in the staged plan.
2. Add one parent/grouped relation builder that receives the existing
   `PointwiseDomainContext` and pointwise-domain classification. It records raw
   one-to-one `MemoryDepMatch` values only for otherwise non-exact edges.
3. Give that builder exactly two proof forms:
   - **in-order reshape:** reindex both accesses into the same explicit
     parent/grouped frame with `MemoryDep.normalize_with_ranges` and require
     equality; never use stride-order normalization;
   - **known-axis broadcast:** allow only the local/group axis identified by
     `grouped_axis` to be absent or expanded, require the expansion factor to be
     exactly the planned `group_size`, substitute that one axis, and require the
     tail variable to disappear.
4. Inventory writes by raw temporal name over the whole candidate. Require one
   value-producing writer, no same-name `StarDep`/synchronizing write, and a
   proved writer-before-reader order. A missing writer is external only when no
   candidate-stage writer of that temporal name exists anywhere.
5. Store nested fusion-only pairs directly on `NestedReductionStage` as
   `tuple[MemoryDepMatch, ...]`. Do not wrap them in
   `ProjectedSourceAccess`: lane, source alternatives, and `must_forward` are
   codegen concerns needed by sub-parent forwarding but not by this proof.
6. In `_prove_staged_fusion_dependencies`, build the raw planned-pair set once.
   For each remaining producer-output read, require one raw producer write and
   raw pair membership, then return the renamed pair for
   `_can_fuse_vertical_impl`. Delete `nested_stage_reads`, `sub_parent_reads`,
   the ownership branch, and `_fusable_read_after_index_equivalence`.
7. Delete `_consumer_broadcasts_source` with the legacy branch. Keep the
   normalization helper only near its unrelated loop-reindexing caller, with a
   name that states that weaker purpose.

This keeps geometric proof in planning and leaves fusion with only:

```text
ordinary exact match
or
unique raw producer/read pair present in the current staged plan
```

It should be near LOC-neutral: the explicit relation predicate replaces the
legacy broadcast helper, while the ownership sets and legacy branch disappear.
A design that adds the new builder but retains the old branch is not simpler
and does not satisfy the mutation tests.

## Focused Tests

Keep this to two parameterized scheduler tests plus existing integration tests.

1. **Parent/grouped relation construction.** Cover grouped-axis R and X
   positives for in-order reshape, pure broadcast, and exact quotient broadcast.
   In the same table reject shifted, transposed, equal-size crossed-axis,
   wrong-factor regrouping, TMP/indirect, atomic, non-injective, an earlier plus
   later same-name write, and consumer-before-writer cases. Include `B=1` and a
   statically-unit write axis.
2. **Plan authority.** Rewrite
   `test_nested_dependency_matches_scope_index_equivalence` so the exact raw
   pair is required for `nested` and `both`; deleting or changing that pair must
   return `None`. Add a mutation-rename collision row and verify raw pair
   membership is checked before renamed matching.

Existing grouped-axis X/R, parent-output broadcast, shifted-parent-output, and
P1 transpose integration tests remain the end-to-end gate. Their normalized
kernel text must remain byte-identical for accepted cases. The present F1 tree
has no direct test for parent/grouped relation construction because that
builder was reverted.

## Current Cleanup Gate

Before calling F1 complete:

- no staged caller of `_fusable_read_after_index_equivalence`;
- no context-free `_consumer_broadcasts_source` in staged legality;
- no stride-order-normalized equality in a planner relation;
- no ownership split in `_prove_staged_fusion_dependencies`;
- no relation set normalized through `mutation_renames` before raw membership;
- no accepted plan with multiple or out-of-order same-name writers; and
- no extra `ProjectedSourceAccess` field on `NestedReductionStage` unless
  nested codegen actually consumes that richer type.

The current post-revert tree intentionally does not clear the first four gates.
