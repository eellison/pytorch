# Parent-to-grouped dependency paths

This note describes the two reasonable ways to handle non-exact dependencies
inside staged nested-reduction fusion.

## Common requirement

Normal fusion remains the first check. If a producer write and consumer read
already match through `fusable_read_and_write`, no staged exception is needed.

The question is how to authorize a remaining non-exact dependency.

## Path A: preserve the existing grouped-stage proof

This is the current prerequisite implementation.

- Reads owned by a sub-parent epilogue must appear as exact source/consumer
  pairs in `ProjectedSourceAccess`.
- Reads owned only by `NestedReductionStage.grouped_nodes` may use the existing
  `_fusable_read_after_index_equivalence` proof from the original nested-
  reduction implementation.
- A read present in both sets is treated as a sub-parent read first and must
  therefore be explicitly recorded by the plan.
- Reads outside those two categories are rejected.

The ownership classification is the load-bearing boundary. An unowned read
must decline, a read owned by both stages must take the stricter sub-parent
branch, and no sub-parent-owned read may reach the generic equivalence helper.
`test_nested_dependency_matches_scope_index_equivalence` directly pins those
three cases.

Conceptually:

```text
strict dependency match
    or
exact plan record for a sub-parent relation
    or
existing equivalence proof for an inherited parent-to-grouped relation
```

### Advantages

- Preserves the already-supported #190594 parent-to-grouped behavior.
- Does not require re-specifying every existing nested-reduction relation.
- Makes the new #191775 behavior fail closed: the broad equivalence helper
  cannot authorize a new sub-parent relation.
- Keeps the prerequisite smaller and easier to compare against #190594.

### Costs

- Fusion legality still has two sources of proof for non-exact dependencies.
- Correctness depends on classifying read ownership before applying the older
  equivalence helper.
- A future change to the grouped-stage codegen convention could drift from the
  generic equivalence proof.

### What would later be deleted

Only the grouped-stage ownership set and its generic-equivalence branch in
`_prove_staged_fusion_dependencies`. The exact projection records, residual
dependency walk, safety checks, scoring, and sub-parent tests remain useful.

## Path B: make the staged plan the sole authority

Extend planning so parent-to-grouped relations are also represented by exact
source/consumer access records. Then staged fusion accepts a non-exact
dependency only when the current plan contains that exact pair.

Conceptually:

```text
strict dependency match
    or
exact relation recorded by the current staged plan
```

`_prove_staged_fusion_dependencies` would no longer call
`_fusable_read_after_index_equivalence` for staged fusion.

### Required implementation

1. During nested-stage planning, identify each parent write/read consumed by
   `grouped_nodes`.
2. Prove its relation in the explicit parent/grouped coordinate frame used by
   nested codegen. Do not authorize it through buffer names or unconstrained
   flattened `MemoryDep.normalize()` equality.
3. Store the raw source and consumer `MemoryDep`s in the same planner-owned
   relation format used by sub-parent projections.
4. Rebuild those records whenever fusion or codegen replans from current nodes;
   do not cache records across loop mutation or `merge_loops`.
5. Require every non-exact staged dependency to be a member of the resulting
   relation set, after the existing synchronization, indirect-index, and
   producer-injectivity checks.

### Advantages

- The plan becomes the single source of truth for every non-exact staged edge.
- Fusion legality and staged codegen share one vocabulary.
- Adding another derived domain requires adding a planner proof, rather than
  widening a generic scheduler exception.
- The ownership distinction disappears from fusion legality.

### Costs

- This relocates part of the already-reviewed #190594 proof from scheduler
  legality into staged planning.
- The parent/grouped relation needs an explicit frame proof for every existing
  nested layout, including grouped-axis X and R cases.
- It needs broader regression coverage because a missing record can disable
  existing nested fusion even when codegen remains valid.
- It makes the prerequisite larger before adding any MXFP6 behavior.

### Required tests

- Existing grouped-axis X and R nested reductions remain one staged kernel.
- Parent-to-grouped reshape and broadcast cases remain accepted.
- Shifted, transposed, regrouped, indirect, multiwrite, and synchronized cases
  decline.
- Empty and nonempty staged match sets preserve scoring and suppress later loop
  rewrites.
- Persistent and looped NVFP4 kernel forms remain unchanged.
- For each supported relation family, removing its planned record must make a
  focused test fail. This mutation-style check prevents missing records from
  appearing only as silent lost fusion.

## Recommendation

Use Path A for the current #191775 review split. It is a strict containment
boundary: inherited #190594 relations retain their existing proof, while every
new sub-parent relation requires an exact plan record.

Treat Path B as the cleaner convergence target and implement it as part of the
indexed-forwarding rebase (F1). F1 needs planner records as an immediate codegen
input, so that work can relocate the parent-to-grouped proof and exercise its
records in the same change. A standalone proof-relocation PR would add review
cost without an immediate consumer. Adopting Path B later does not discard the
current exact-record work; it removes one remaining exception and extends the
same model to the older stage.
