# Nested Reduction Dependency Equivalence

## Summary

Nested reduction fusion needs a scheduler-level dependency match that is
slightly more flexible than ordinary exact `MemoryDep` equality.

The motivating cases are nested reductions where one scheduler node writes a
buffer in one iteration domain, while a dependent pointwise or reduction reads
the same buffer through a broadcasted or refactored domain. The addresses are
equivalent, but the dependency objects are not syntactically equal.

The current design keeps this relaxation local to scheduler vertical fusion:

- Ordinary exact read/write matches remain unchanged.
- Relaxed matching is enabled only through `allow_index_equivalence`.
- The relaxed path rejects producer-side broadcasts and non-dense producer
  writes before accepting normalized or broadcast-equivalent reads.
- The old generic `MemoryDep.normalize_without_broadcast()` helper is removed,
  because dropping broadcast dimensions is not a generally valid dependency
  normalization rule.

The main difficulty is that scheduler dependencies are address expressions, not
semantic tensor views. We need to prove that two address expressions identify
the same producer elements without accidentally accepting producer-side aliasing
or changing ordinary fusion legality.

## Background

The scheduler uses `MemoryDep` objects to represent reads and writes:

```text
MemoryDep(name, index, var_names, size)
```

For vertical fusion, `Scheduler.can_fuse_vertical()` tries to prove each unmet
consumer dependency is satisfied by a producer write. The common fast path is
exact:

```text
read.index == write.index
read.size has write.size as a prefix
same buffer name
```

There are already some normalizations for loop-order mismatches, but ordinary
fusion mostly expects producer and consumer dependencies to line up exactly.

Nested reduction breaks that assumption. The fused computation has multiple
logical domains:

1. `REDUCED`: the grouped reduction output domain, for example `[B, D / G]`.
2. `LOCAL_REDUCTION_INPUT`: the grouped reduction input domain before reducing
   the local lane, for example `[B, D / G, G]`.
3. `PARENT_FULL`: the outer reduction parent tile after broadcast-back, for
   example `[B, D]`.

A single pointwise node may need to be compatible with more than one of those
domains. The usual strategy of rewriting the pointwise loop body so dependencies
match one producer can make it stop matching another producer. For nested
fusion, it is cleaner to keep the scheduler legality check capable of proving a
small set of access-equivalent dependency forms.

## Concrete Mismatches

### Pure Broadcast Dimension

The first important shape is an extra consumer loop dimension that is absent
from the read address:

```text
read:  d1, {d0: 1024, d1: 16}
write: d0, {d0: 16}
```

The read repeats each producer element across `d0`. If we drop the unused read
dimension and normalize, both sides refer to the same producer element.

This appears in small-dim-in-X nested reductions, where a consumer is expressed
over a full parent tile but reads a reduced output.

### Quotient Broadcast Dimension

The second important shape is a same-rank expanded dimension where the consumer
uses only the quotient:

```text
read:  32*d0 + FloorDiv(d1, 128), {d0: 128, d1: 4096}
write: 32*d0 + d1,                {d0: 128, d1: 32}
```

Here `d1` on the read side can be split as:

```text
d1 = write_d1 * 128 + tail
```

Substituting into the read index gives:

```text
32*d0 + FloorDiv(write_d1 * 128 + tail, 128)
```

With `tail in [0, 128)`, range simplification proves this is:

```text
32*d0 + write_d1
```

That matches the producer write. If the tail remains in the simplified address,
the helper rejects the match.

## Why The Obvious Fix Was Too Broad

The initial fix introduced:

```python
MemoryDep.normalize_without_broadcast()
```

It dropped any range var absent from the read index and then normalized the
dependency. That made the pure broadcast case work, but it was too broad as a
general `MemoryDep` API.

The core problem is that removing broadcast dimensions is only valid for the
consumer side of this proof. It is not a symmetric dependency normalization.

For example, these are not equivalent in general:

```text
read-like:  d0, {d0: 8, d1: 4}
write-like: w1, {w0: 8, w1: 4}
```

Both have an unused dimension, but the unused dimension is on different logical
axes. Dropping dimensions without more context can make crossed broadcasts look
compatible.

Another unsafe producer shape is:

```text
write: w0, {w0: 16, w1: 4}
```

This writes the same address for all `w1`. If the relaxed path lets this match a
consumer read by normalization, the scheduler may fuse a consumer before all
producer iterations have produced a well-defined value.

A third unsafe producer shape is all-vars-present but still aliased:

```text
write: w0 + w1, {w0: 2, w1: 2}
```

Both vars appear in the expression, but `(w0=0, w1=1)` and `(w0=1, w1=0)` write
the same address. The "all write vars appear" check is necessary but not
sufficient.

This is why the helper belongs in scheduler legality, not on `MemoryDep` as a
general normalization primitive.

## Current Design

### Entry Point

`Scheduler.fusable_read_and_write()` now has a flag:

```python
allow_index_equivalence: bool = False
```

The default path preserves ordinary behavior. It accepts exact read/write
matches before considering relaxed equivalence.

When `allow_index_equivalence=True`, the relaxed path can additionally accept:

- existing normalized equivalence via `deps_match_normalized()`
- the nested broadcast forms handled by `_fusable_read_after_broadcast()`

### Exact Matches Come First

Both exact-match paths are checked before relaxed-only guards:

1. The original read/write dependency objects match directly.
2. If `loop_ordering_after_fusion` is enabled and the ranks differ, the
   dependencies match after the existing loop-normalization path.

This is important. Some exact producer/consumer pairs are legal even if their
write index is gapped or otherwise not dense:

```text
read == write == 33*d0 + d1, {d0: 128, d1: 32}
```

`allow_index_equivalence=True` should not make an exact match less legal than
ordinary fusion. The relaxed guards only apply after both exact-match paths have
failed and we are asking for a non-exact dependency proof.

### Producer-Side Guard

For non-exact relaxed matches, the producer write must pass two checks:

1. Every producer loop var appears in the producer write index.
2. The producer write can be proven dense by either normal loop normalization or
   stride-order normalization.

In code, the second check is conceptually:

```python
write.normalize().is_contiguous()
or write.normalize_with_stride_order().is_contiguous()
```

The two alternatives cover different dense cases:

- `normalize().is_contiguous()` handles normal symbolic row-major dense writes,
  for example `s1*w0 + w1` over `{w0: s0, w1: s1}`.
- `normalize_with_stride_order().is_contiguous()` handles static dense layouts
  in a different stride order, including channels-last-style access patterns.

Together they reject:

- producer-side broadcast writes like `w0` over `{w0: 16, w1: 4}`
- aliased writes like `w0 + w1` over `{w0: 2, w1: 2}`
- gapped non-exact writes like `33*w0 + w1` over `{w0: 128, w1: 32}`

This guard is intentionally conservative. Rejecting a hard-to-prove dense write
means we miss a fusion; accepting an aliased producer write would be a
correctness bug.

### Broadcast Helper

There is a single helper:

```python
_fusable_read_after_broadcast(read, write)
```

It handles the two consumer-side broadcast forms:

1. Pure broadcast dimensions absent from `read.index`.
2. Same-rank expanded dimensions used through an exact quotient.

The helper does not try to prove producer safety. That is done once in
`fusable_read_and_write()` before both `deps_match_normalized()` and the
broadcast helper, so normalized equivalence cannot bypass the producer guard.

## Why One Helper

There were briefly two helpers:

- one for pruning unused broadcast dimensions
- one for splitting expanded dimensions

That was technically fine but made the scheduler API feel bigger than the
concept. The current single helper keeps the external shape simple:

```python
deps_match_normalized(...) or _fusable_read_after_broadcast(...)
```

The helper docstring lists the two cases with examples. This keeps the proof
local without implying there are multiple independent scheduler concepts.

## Interaction With Scoring

Fusion scoring happens before final vertical legality. If exact dep scoring is
zero, a valid nested pair can be rejected before `can_fuse_vertical()` has a
chance to prove relaxed equivalence.

The scheduler therefore has a secondary scoring path:

```python
_score_fusion_memory_by_fusable_read_write()
```

It uses the same `fusable_read_and_write(..., allow_index_equivalence=True)`
logic for nested candidates, so the scoring and legality definitions are not
silently different.

This is important for cases where the producer output is read through a
broadcasted or refactored access. Without this score bridge, the dependency is
legal but the heuristic score can remain zero.

The relaxed scoring is not intended as a general replacement for exact scoring.
It is gated by `allow_index_equivalence` or by the cheap
`NestedReduction.is_candidate()` filter.

## Mutation Rename Interaction

Earlier versions tried to handle mutation renames inside
`fusable_read_and_write()` itself. That was too surprising because other
dependency comparisons do not generally mutate names internally.

The cleaner boundary is:

- `can_fuse_vertical()` applies `rd.rename(self.mutation_renames)` before
  calling `fusable_read_and_write()`.
- `fusable_read_and_write()` compares the concrete dependency objects it is
  given.

This keeps mutation versioning at the scheduler boundary and avoids hidden name
rewrites inside the equivalence helper.

## Tests

The focused scheduler test covers these cases:

| Case | Default | Relaxed | Purpose |
| --- | --- | --- | --- |
| quotient broadcast | reject | accept | Needed for expanded same-rank read dims |
| quotient tail remains | reject | reject | Ensures the split tail cannot affect address |
| pure broadcast | reject | accept | Needed for unused read broadcast dims |
| dynamic dense | reject | accept | Ensures symbolic row-major dense writes are not over-rejected |
| exact gapped | accept | accept | Ensures relaxed mode does not reject exact matches |
| producer broadcast | reject | reject | Prevents producer-side unused dimensions |
| producer alias | reject | reject | Prevents all-vars-present aliasing |

The test lives in `test/inductor/test_inductor_scheduler.py` because this is a
scheduler dependency proof, not a codegen behavior. End-to-end nested reduction
tests still validate that the actual nested fusions are selected and lowered
once the later codegen commits are replayed.

## Review Findings Addressed

Two review passes found real issues:

1. The producer-side guard was inside the broadcast helper, but
   `deps_match_normalized()` ran before the helper. That meant normalized
   equivalence could bypass the guard. The guard is now in
   `fusable_read_and_write()` before both relaxed checks.

2. Checking that every write var appears in the write index did not prove
   non-aliasing. `w0 + w1` over `(2, 2)` is the counterexample. The guard now
   additionally requires the producer write to normalize to a contiguous dense
   access by normal or stride-order normalization.

Additional local investigation found:

3. `normalize_with_stride_order().is_contiguous()` alone rejects symbolic
   row-major dense writes because stride hints fall back for symbolic strides.
   The guard now also accepts `normalize().is_contiguous()`.

4. Running the relaxed guard before exact matches changed exact-match behavior.
   Both the direct exact match and the loop-normalized exact match now come
   before relaxed-only checks.

## Remaining Limitations

The design is conservative by intent.

Known missed-fusion risks:

- Dynamic channels-last or other symbolic stride-permuted dense writes may fail
  the dense-write proof if neither normalization can prove contiguity.
- Non-affine or more complex quotient patterns are rejected unless
  `simplify_with_ranges()` can prove the tail disappears.
- The helper only handles consumer-side broadcasts. It is not a general
  dependency equivalence engine.

These are acceptable because the failure mode is to skip nested fusion and fall
back to separate kernels. The main requirement is not to accept a producer write
whose address pattern aliases or broadcasts across producer iterations.

## Landability Checklist

Before landing this scheduler commit:

- Run `python test/inductor/test_inductor_scheduler.py -k fusable_read`.
- Run `python test/inductor/test_inductor_scheduler.py -k nested_reduction`.
- After replaying the codegen/test commits, run
  `python test/inductor/test_nested_reduction.py`.
- Check that ordinary exact matches still work with
  `allow_index_equivalence=True`.
- Keep the relaxed dependency logic scheduler-local unless there is a broader
  audited use case for general dependency equivalence.

## Recommendation

Keep the current shape:

- no `MemoryDep.normalize_without_broadcast()`
- one scheduler-local `_fusable_read_after_broadcast()` helper
- exact match before relaxed guards
- relaxed guards shared by both `deps_match_normalized()` and the broadcast
  helper
- tests for both accepted nested patterns and rejected producer-side hazards

This is still more scheduler logic than ideal, but it is scoped to the actual
nested-reduction problem and it avoids turning a local legality exception into a
global dependency-normalization rule.
