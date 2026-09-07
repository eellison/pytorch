# Nested Reduction Design Difficulties

This note summarizes the main design issues we ran into while cleaning up the
nested reduction scheduler/codegen path.

The short version: nested reduction fusion is hard because scheduler dependency
matching proves memory access equivalence, but codegen needs axis/loop-body
coordinates. Those are related, but they are not the same thing.

## Context

The motivating pattern is a dependent pair of reductions over different logical
axes, for example:

```python
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
s = (w[:, :, None] * x_normed).sum(dim=1)
out = x_normed + s[:, None, :]
```

There are multiple logical domains in play:

```text
outer reduction input/output:     [B * K, D] or [B, K, D]
grouped reduction input domain:   [B, K, D] or [B, D, K]
grouped reduction output domain:  [B, D]
full-resolution epilogue domain:  [B, K, D]
```

The scheduler mostly sees `MemoryDep` objects and loop ranges. Codegen needs to
call each loop body with the right loop values. That distinction is the source
of most of the complexity.

## MemoryDep Is Not An Axis Model

A `MemoryDep` describes an access:

```text
buffer name
index expression
loop variables
loop ranges
mode
```

It can prove that two loop nests access the same storage locations. It does not
tell us which logical axis is the grouped axis, the parent full axis, or the
local reduction lane.

For example:

```text
write: buf2, index=16384*d0 + 1024*d1 + d2, size=(16, 16, 1024)
read:  buf2, index=16384*d0 + d1 + 1024*d2, size=(16, 1024, 16)
```

These two deps access the same physical storage with different loop order:

```text
write logical order: [B, K, D]
read logical order:  [B, D, K]
```

`normalize_with_stride_order()` can prove they are access-equivalent. But that
does not tell codegen what `iter_vars` to pass into the pointwise body. The
pointwise body still expects its own loop ranges.

That is why scheduler legality and codegen remapping must remain separate.

## Why Exact Dependency Matching Fails

Generic vertical fusion works well when a producer write and consumer read have
the same access expression and compatible loop sizes:

```text
write: buf, index=d0, size=(N,)
read:  buf, index=d0, size=(N,)
```

Nested full-resolution consumers often read producer outputs through broadcasted
or refactorized domains, so exact matching is too strict.

### Broadcasted grouped output

For:

```python
out = x_normed + s[:, None, :]
```

the grouped reduction writes:

```text
write: buf1, index=d0, size=(1024,)
```

The full-resolution epilogue reads:

```text
read:  buf1, index=d1, size=(16, 1024)
```

The `16` dimension is a broadcast dimension. The read loop has `[K, D]`, but the
index ignores `K`.

Exact matching fails because:

```text
read.index != write.index
read.size != write.size
```

But this is a valid dependency. The relevant proof is:

```python
read.normalize_without_broadcast() == write.normalize()
```

This removes loop dimensions that are not used by the read index, then compares
the normalized access.

### Broadcast split

Another shape we saw:

```text
write: 32*d0 + d1,                 {d0: 128, d1: 32}
read:  32*d0 + FloorDiv(d1, 128),  {d0: 128, d1: 4096}
```

The read has a larger dimension that is really:

```text
d1_read = d1_write * 128 + tail
```

After substituting:

```text
FloorDiv(d1_write * 128 + tail, 128) -> d1_write
```

assuming:

```text
tail in [0, 128)
```

The read and write are equivalent, and any remaining `tail` in the simplified
index would cause the match to fail. This is conservative because only exact
divisible splits are accepted.

## Why The Normal Pointwise Reindex Flow Is Not Enough

Inductor already has machinery that tries to reindex a pointwise node so it
matches a reduction. That works when there is one dominant reduction domain.

Nested reduction has multiple relevant reduction domains:

```text
outer reduction domain
grouped reduction input domain
grouped reduction output domain
parent full-resolution domain
```

A pointwise can be related to more than one of them. For example:

```python
y = full_res_pointwise(outer_reduction_result)
s = (w * y).sum(dim=K)
out = y + s[:, None, :]
```

The same full-resolution value `y` may need to:

```text
feed the grouped reduction input domain
remain compatible with the outer reduction result
feed a later full-resolution consumer
```

If we globally mutate/reindex the pointwise node to match one reduction, we may
break its relation to another reduction or to another consumer.

So the better model is per-use remapping:

```text
scheduler: classify which nested domain a pointwise belongs to
scheduler: prove producer/consumer deps are access-equivalent
codegen: choose loop values for this pointwise emission site
codegen: call the pointwise body with values shaped for that body
```

This is why nested codegen needs explicit source selection instead of relying
only on the normal "make pointwise match reduction" path.

## Why `allow_index_equivalence` Exists

`allow_index_equivalence` is intentionally narrower than "do anything that
normalizes."

It is used when a fused-node hook has already validated that a pointwise is in a
domain codegen knows how to remap. In that case, vertical dependency matching may
accept access-equivalent reads in addition to exact reads.

It covers cases like:

```text
1. broadcasted read:
   write [D]
   read  [K, D], index ignores K

2. pure loop-order change:
   write [B, K, D]
   read  [B, D, K]

3. exact divisible broadcast split:
   write [B, D/G]
   read  [B, D] through FloorDiv(D, G)
```

It does not replace the remaining scheduler checks. `can_fuse_vertical()` still
checks intermediate dependencies and rejects cycles or loads before producers.

This is important because access equivalence only answers:

```text
"Can this producer write satisfy this consumer read?"
```

It does not answer:

```text
"Can this whole fused graph be scheduled legally?"
"Can codegen pass the right loop vars to this body?"
```

## Why Scoring Needed A Fallback

Fusion scoring runs before full legality. For reductions, a zero memory score
can prevent a candidate from reaching `can_fuse_vertical()`.

Exact memory scoring is based on set intersection of deps. It misses:

```text
write [D] vs read [K, D] broadcast
write [B, K, D] vs read [B, D, K] loop-order equivalent
```

So `_score_fusion_memory_by_fusable_read_write()` gives a score to vertical
producer-output deps that `fusable_read_and_write()` accepts.

The important cleanup is that this is no longer a nested-specific dependency
walker. It is phrased as:

```text
for producer writes:
  find consumer reads that the normal vertical matcher accepts
  give the pair a memory score
```

That keeps scoring and legality aligned.

## Why Codegen Needs Full-Resolution Source Selection

Once the scheduler selects a fusion, codegen has to emit each pointwise node by
calling:

```python
sn._body(iter_vars)
```

Those `iter_vars` must be shaped like `sn.get_ranges()`.

For full-resolution pointwise nodes in nested codegen, the right source may be:

```text
parent-full source:       [B, K, D] or [B*K, D]
local-reduction source:   [B, D, K]
flattened full source:    [B*K*D]
```

We hit this failure:

```text
CantSplit: 1024 not divisible by 16384
```

The dependency equivalence was real, but codegen had selected the wrong source
iteration space for the pointwise body. It tried to split values shaped like one
domain into ranges expected by another.

The fix is `_select_full_resolution_pointwise_source()`:

```text
1. exact parent-full match
2. exact local-reduction match
3. compatible parent-full split
4. compatible local-reduction split
5. exact flattened parent-full match
6. exact flattened local-reduction match
```

The flattened source is only selected when the pointwise body is actually flat.
It preserves row-major value order:

```text
flat = ((v0 * size1) + v1) * size2 + v2
```

This keeps the axis decision in codegen, where the loop-body contract is known.

## Why Grouped Axis Classification Was Simplified

The original grouped-axis classification tried to infer axis identity from
loop-body memory coefficients. That was fragile because `MemoryDep`/index
expressions are about access patterns, not semantic axes.

The simpler boundary is shape-based:

```text
group-in-R:
  iter ranges   [outer_x, outer_r / G]
  reduce ranges [G]

group-in-X:
  iter ranges   [outer_x / G, outer_r]
  reduce ranges [G]
```

With singleton squeezed forms:

```text
group-in-R, outer_x == 1:
  iter ranges   [outer_r / G]
  reduce ranges [G]

group-in-X, outer_x / G == 1:
  iter ranges   [outer_r]
  reduce ranges [G]
```

This rejects ambiguous flattened or higher-rank grouped reductions unless there
is explicit axis provenance. That is conservative, but much easier to reason
about.

## Why The B=64 Full-Resolution Epilogue Expectation Changed

The test originally expected the B=64 full-resolution epilogue to stay outside
the nested fused grouped node.

After allowing parent-full index equivalence, the epilogue can fuse:

```text
grouped output write:
  buf1, index=4096*d0 + d1, size=(64, 4096)

full-res epilogue read:
  buf1, index=4096*d0 + d2, size=(64, 16, 4096)
```

This is a broadcast over `K=16`.

Numerics were checked:

```text
max_abs_diff 5.72e-06
allclose True
codegen_nested_reduction 1
generated_kernel_count 1
```

So the old expectation was a conservative implementation boundary, not a
correctness requirement.

## The Main Invariant

The core invariant should be:

```text
Scheduler may accept index-equivalent deps only when the caller has already
validated that codegen can remap the consumer's loop body.
```

This is why `allow_index_equivalence` is not a global default.

The scheduler proves:

```text
the read/write access is legal
there are no extra unmet intermediate deps
the pointwise belongs to a supported nested domain
```

Codegen proves:

```text
the pointwise body can be called with values from a supported source domain
the source can be split or flattened to match that body's ranges
```

Both are required.

## Practical Review Checklist

When reviewing this area, check these separately:

1. Does dependency matching prove access equivalence, not axis equivalence?
2. Does the caller enabling `allow_index_equivalence` also validate domain/codegen support?
3. Does scoring use the same matcher as legality, so candidates are not dropped too early?
4. Does codegen choose a source iteration space that matches the pointwise body's ranges?
5. Are unsupported forms rejected before codegen, rather than accepted and failing in remap?
6. Are intermediate deps still checked by the normal vertical fusion path?

## Summary

The special cases are needed because nested reduction fusion sits at the
intersection of:

```text
memory dependency equivalence
logical nested domains
loop-body coordinate remapping
fusion scoring heuristics
vertical dependency legality
```

Generic fusion usually only needs a subset of those at once. Nested fusion needs
all of them to line up.

The current design is intentionally split:

```text
MemoryDep equivalence: fusable_read_and_write(..., allow_index_equivalence=True)
Scoring bridge:        _score_fusion_memory_by_fusable_read_write()
Domain classification: NestedReduction.PointwiseDomain
Codegen source choice: _select_full_resolution_pointwise_source()
```

That split is the important part. It keeps memory access reasoning out of axis
classification, and keeps codegen loop-body remapping explicit instead of
implicitly relying on dependency normalization.
