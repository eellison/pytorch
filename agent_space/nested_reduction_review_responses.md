# Nested Reduction Review Responses

## Lines 7187-7197: avoid mucking up `can_fuse`

Agreed. The nested-specific control flow was too visible in the vertical
fusion branch. I factored the selection into
`_nested_index_equivalent_dep_names(...)`.

The intent is:

- `can_fuse` computes the nested index-equivalent dep names once, after the
  usual basic gates.
- That computation requires full `NestedReduction.can_fuse(...)`; it is not a
  cheap candidate-only escape hatch.
- The same dep-name set is then used for the local score bridge and vertical
  legality, and it is still narrowed to producer outputs that the consumer
  actually reads.

So the generic memory scorer stays generic. The nested-specific score bridge
now lives in the `can_fuse` flow, where the nested dep-name set is already
available.

## Line 7369: add a comment for the first exact match

This is the normal exact-dependency path and should run before any normalization
or relaxed matching. I added a comment to make that ordering explicit.

The reason this matters is that exact matching is the existing scheduler
behavior. `allow_index_equivalence=True` should only add cases after exact
matching fails; it must not make an exact match less legal.

## Line 7384: difference between the two exact-match checks

They are intentionally the same predicate on different dependency objects.

The first check compares `original_read` and `original_write`. This preserves
the raw dependency semantics, including layouts that are exact but not dense or
contiguous.

The second check compares `read` and `write` after optional
`config.loop_ordering_after_fusion` normalization. That config can merge loop
variables before fusion, so deps that were not syntactically equal can become an
ordinary exact match after `normalize()`.

I factored this predicate into `_same_index_with_prefix_size(...)` so the two
sites are visibly the same check. I also added a comment before the second site
explaining that it is only the post-normalization re-check.

## Lines 7411-7415: why contiguity matters

The relaxed matcher is only meant to accept consumer-side reshapes and
broadcasts. Once read and write indices differ, we need a producer-side safety
condition.

The producer write must be dense and injective over its logical domain. If the
producer write has gaps or aliases, a consumer-side broadcast can hide that fact
and make two different producer iterations look equivalent to one consumer
read. That would turn a missed-fusion problem into possible incorrect codegen.

The two guards cover the producer side:

- `OrderedSet(write.var_names) <= write.index.free_symbols` rejects producer
  broadcasts, where some producer loop variable does not affect the address.
- `write.normalize().is_contiguous()` or
  `write.normalize_with_stride_order().is_contiguous()` rejects non-dense or
  aliased producer writes after normalizing the logical access.

The stride-order variant is there for dense layouts that are contiguous in a
different loop order. The point is not that the final memory must be row-major;
it is that the producer writes a dense, one-to-one logical region before we
allow the consumer to read it through a different view.

## Line 7443: comment on the first broadcast strategy

This is the pure consumer-broadcast strategy.

Example:

```
read:  d1, {d0: 1024, d1: 16}
write: d0, {d0: 16}
```

The read has an extra loop var, `d0`, that does not affect the address. We drop
unused read vars, rebuild the `MemoryDep` over the smaller read domain, and
then compare normalized deps. This only relaxes consumer-side broadcast because
producer-side broadcast was rejected before entering this helper.

I added a code comment before this block.

## Line 7458: comment on the second broadcast strategy

This is the quotient-broadcast strategy.

Example:

```
read:  32*d0 + FloorDiv(d1, 128), {d0: 128, d1: 4096}
write: 32*d0 + d1,                {d0: 128, d1: 32}
```

The read axis is larger than the write axis. We split the read variable into:

```
read_var = write_var * factor + tail_var
```

Then we simplify the read index with `tail_var` ranged over `[0, factor)`. The
match only succeeds if the tail disappears from the simplified index and the
result equals the producer write index. If the consumer reads partial groups or
uses the expanded dimension in a way that still depends on the tail, the match
fails.

I added a code comment before this block too.
