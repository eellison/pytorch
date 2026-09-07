# Nested Reduction Fusion Split Issues

This note separates the scheduler/fusion irregularities that currently sit near
nested reduction. They are related, but not all the same kind of change.

## Summary

There are four separable fusion issues:

1. **Broadcast write/read dep equivalence**
   Generic vertical fusion should understand that a reduced write can satisfy a
   read that broadcasts that value across a larger loop domain.

2. **Loop reindex retry after vertical failure**
   Generic pointwise/reduction fusion sometimes needs to try loop reindexing
   after `can_fuse_vertical()` fails, not only when memory score is low.

3. **Nested score survival**
   The initial nested reduction pair can score zero under exact `MemoryDep`
   matching, so it needs a nested-scoped score bridge to survive the
   "no shared data" gate.

4. **Extra intermediate-dependency safety**
   Once any dep matching is relaxed, vertical fusion must still reject extra
   unmet deps from producers that are not part of the fused pair.

## 1. Broadcast Dep Equivalence

Current code:

`Scheduler.fusable_read_and_write()` falls back to:

```python
read = orig_read.normalize_without_broadcast()
write = orig_write.normalize_without_broadcast()
return read.index == write.index and read.size == write.size
```

This is not nested-specific. It broadens generic vertical fusion.

Why nested wants it:

Full-resolution epilogues can read a grouped output at full resolution:

```text
write: buf2[32*d0 + d1]             ranges [B, groups]
read:  buf2[32*d0 + (d1 // 128)]    ranges [B, D]
```

After removing broadcast-only loop dimensions, these describe the same value.

Why it is separable:

This is a generic `MemoryDep` equivalence rule. It can be reviewed and tested
without any nested reduction codegen. A standalone PR/commit should include
tests where a producer writes `[B, groups]` and a consumer reads the same buffer
broadcast over `[B, groups, G]`, plus negative tests for offset/index mismatch.

Landing recommendation:

This should either be a preparatory commit before nested reduction, or scoped
back to nested if we want the first nested PR to avoid generic vertical-fusion
changes. The current broad form is defensible, but it is a real generic change.

## 2. Loop Reindex Retry

Current code:

When generic vertical fusion fails, `Scheduler.can_fuse()` may call:

```python
self._try_reindex_pointwise_for_reduction(
    node1,
    node2,
    require_vertical_fusion=True,
)
```

Why nested wants it:

After the initial nested pair is fused, a full-resolution pointwise consumer may
need to be reindexed into the grouped reduction's loop space before generic
vertical matching can see the write/read relationship.

Why it is separable:

This is also generic pointwise/reduction fusion machinery. It reuses the
existing loop reindexing path; nested only made the missing retry obvious.

Review risk:

The helper mutates loop state. It snapshots and rolls back if reindexing does
not improve deps or if `can_fuse_vertical()` still fails. A stricter version
could roll back unless the full vertical decision succeeds, including
`V.choices.can_fuse_vertical()` and backend `can_fuse_vertical()`.

Landing recommendation:

This can be a preparatory fusion-fix commit with loop-ordering tests. It does
not need nested codegen in the same diff.

## 3. Nested Score Survival

Current code:

`InductorChoices.can_fuse()` no longer has a nested-specific score-0 bypass.
Instead, `Scheduler.score_fusion_memory()` calls a nested-scoped helper when
exact dep scoring returns zero:

```python
if score == 0:
    score = self._score_fusion_memory_by_nested_reduction_deps(...)
```

The helper first checks `NestedReduction.can_fuse(node1, node2)`.

Why this should stay nested-scoped for now:

A broader generic normalized score was too risky. In particular,
reduction->pointwise pairs may need loop reindexing before they are safe to
score as shared-data matches. Giving them a nonzero score too early can skip
that repair.

Landing recommendation:

Keep this in the nested scheduler commit. Add a TODO for a future generic
normalized vertical-dep score.

## 4. Extra Intermediate-Dependency Safety

Current code:

`Scheduler.can_fuse_vertical()` computes remaining deps after matching node1
writes against node2 reads. It then rejects if any remaining dependency is
produced by an intermediate fused node that depends on node1:

```python
if self._has_intermediate_dependencies(node1.get_operation_names(), remaining_deps):
    return False
```

Why nested wants it:

Nested needs to tolerate one specific write/read mismatch, but it must not
accidentally hide unrelated unmet dependencies.

Why it is separable:

This is a generic safety check for vertical fusion after dependency matching is
relaxed. It can travel with the broadcast-dep or loop-reindex commit.

Landing recommendation:

Keep it with the generic fusion fixes, not with the core nested codegen.

## Suggested Stack Split

1. **Generic vertical fusion prep**
   Broadcast dep equivalence, loop reindex retry, intermediate-dep safety, and
   loop-ordering/fusion tests.

2. **Nested scheduler detection**
   `NestedReduction.can_fuse`, `FusedNestedReductions`,
   nested-scoped score survival, nested fusion ordering.

3. **SIMD/Triton range infra**
   Root-owned block/mask helpers, derived roots, active range trees, reshape /
   reduce / broadcast helpers.

4. **Core nested codegen**
   Group layout, reduced-output family, grouped reduction handler, basic
   reduced-output stores/epilogues.

5. **Full-resolution pointwise support**
   Pointwise remapping, schedule reuse, lazy full-resolution load resolution,
   broadcast-back epilogues.

6. **Autotune and coverage**
   `min_xblock` / `min_rblock` metadata, coordinate descent guard, dynamic/B=1
   / mask / non-persistent / rejection coverage.

## What a Fully Generic Fix Would Look Like

The generic scheduler currently mixes three concepts into one integer
`shared_data_score`:

1. Do these nodes share enough semantic data to consider fusion?
2. How should this candidate be ordered relative to other candidates?
3. Has the loop/index relationship already been repaired enough for vertical
   legality?

Nested reduction hit all three at once. A cleaner generic design would separate
them.

### A. Add Semantic Dep Equivalence

Add a helper such as:

```python
deps_equivalent_for_vertical_fusion(read: Dep, write: Dep) -> bool
```

It would handle:

- exact `MemoryDep` equality,
- normalized loop-order equality,
- broadcast-stripped equality,
- no indirect `TMP` symbols,
- same mutation mode / no synchronization-requiring writes.

`Scheduler.fusable_read_and_write()` and scoring could both call this instead
of each having their own partial matching rules.

For broadcast specifically, the proof should be explicit:

```text
write index after dropping broadcast-only vars == read index after dropping
broadcast-only vars
and write logical size == read logical size after dropping those vars
```

This is generic and should have standalone tests independent of nested
reduction.

### B. Separate "Candidate Has Shared Data" from "Fusion Ordering Score"

Today a zero memory score can kill a candidate before vertical legality runs.
If generic normalized deps are added directly to `score_fusion_memory()`, they
also affect fusion ordering. That is risky.

Better shape:

```python
memory_score = score_fusion_memory_exact(...)
has_shared_data = memory_score > 0 or has_semantic_vertical_dep_match(...)
```

Then `InductorChoices.can_fuse()` can avoid the "no shared data" rejection
without inflating the ordering score.

This would replace the nested-scoped score bridge cleanly:

- nested pair has exact score 0,
- semantic dep match says "yes, keep candidate alive",
- ordering can still prefer ordinary fusions through `FusionScore`.

### C. Make Loop Reindexing Transactional Across the Full Vertical Decision

Current reindex repair snapshots state, mutates loop bodies, then rolls back if
`can_fuse_vertical()` still fails. But the full decision also includes:

- `V.choices.can_fuse_vertical(...)`,
- backend `can_fuse_vertical(...)`,
- recomputed score after reindex.

Generic fix:

```python
with try_reindex_pointwise_for_reduction(...) as changed:
    if changed and full_vertical_decision_succeeds():
        commit()
    else:
        rollback()
```

or pass a verifier callback into the reindex helper:

```python
self._try_reindex_pointwise_for_reduction(
    node1,
    node2,
    verify=lambda: full_vertical_decision_after_reindex(...),
)
```

That would make the mutation safety story crisp and not nested-specific.

### D. Use the Same Semantic Dep Helper for Intermediate-Dep Safety

Once dep matching becomes more permissive, remaining unmatched deps must be
classified carefully:

- matched by semantic read/write equivalence: ok,
- weak deps / empty tensor special cases: existing behavior,
- produced by an intermediate node that depends on node1: reject,
- unrelated external input: ok.

The current `_has_intermediate_dependencies()` is a small safety patch. A
generic version would be part of a "vertical dependency resolution" helper that
returns:

```python
VerticalDepResolution(
    matched_deps=...,
    external_deps=...,
    blocking_intermediate_deps=...,
)
```

Then both ordinary vertical fusion and nested fusion would use the same result.

### E. Optional: Cache Nested/Fusion Analysis

`NestedReduction.can_fuse()` is now called from scoring, backend checks, and
fused-node creation. If this becomes a compile-time concern, the generic fix is
not to hand-cache random callsites, but to introduce a small per-pair fusion
analysis cache:

```python
scheduler.get_fusion_analysis(node1, node2)
```

That can memoize:

- nested legality,
- grouped reduction info,
- group axis,
- tiling/coalescing result,
- semantic dep resolution.

This is not required for correctness, but it is the clean answer to repeated
analysis.
