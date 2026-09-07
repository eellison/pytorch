# Nested Reduction XBLOCK Codegen Fix

## Short version

XBLOCK nested reductions exposed that "full resolution" is not one domain.

For a grouped reduction in X, the grouped reduction body and the final full
output have different logical loop orders:

- Producer/prologue full domain: node2 grouped-reduction body domain.
- Consumer/epilogue full domain: node1 parent SIMD tile domain.

The bug was treating these as the same. That could compile but map logical
coordinates incorrectly, so pointwise prologues/epilogues read or wrote the
right buffers at the wrong logical indices.

## Concrete example

Take:

```python
B, K, D = 16, 16, 1024
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
s = (w[:, :, None] * x_normed).sum(dim=1)
out = x_normed + s[:, None, :]
```

The parent RMSNorm reduction runs over:

```text
parent x/r tile: [B * K, D]
x0 in [0, 256), r0 in [0, 1024)
b = x0 // K
k = x0 % K
d = r0
```

The nested grouped reduction computes `s[b, d]` by reducing over `k`.
Its body naturally sees:

```text
node2 grouped body full domain: [B, D, K]
iter = [b, d]
reduce = [k]
```

The full-resolution consumer `out[b, k, d]` does not want that body domain.
It wants the parent tile:

```text
consumer full domain: [B * K, D]
or logically [b, k, d] through x0/r0
```

## What was wrong

Earlier codegen used one "full-resolution" source mapping for all pointwise
nodes around node2. That mapping came from the grouped reduction body:

```text
producer_full = [B, D, K]
```

That is correct for a prologue feeding the grouped reduction. For example:

```python
y = relu((x_flat / rms).reshape(B, K, D) + bias[:, None, :])
(w[:, :, None] * y).sum(dim=1)
```

The reduction body consumes `y[b, k, d]` while looping as `[b, d, k]`, so the
prologue must be remapped into the grouped body coordinates.

But that same mapping is wrong for a full-resolution consumer after the grouped
reduction:

```python
out = x_normed + s[:, None, :]
```

The consumer is not feeding the grouped reduction. It is producing the final
full output on the parent tile. If it is emitted using `[B, D, K]`, codegen can
assign `D` and `K` in the wrong logical positions, or fail to split the source
groups into the consumer's `[B, K, D]` shape. That is the numerics issue: the
kernel can still run, but values are associated with the wrong `(b, k, d)`
coordinate.

The reverse is also true. If we use only the parent `[B * K, D]` domain for all
full-resolution pointwise, full-resolution prologues can become wrong because
the grouped reduction consumes them in `[B, D, K]` order. The
`test_fullres_prologue_small_dim_in_x_loop_order` case exists for this.

## Fix

We made the domain choice explicit.

Scheduler legality now classifies each pointwise node around the grouped
reduction as one of:

```text
reduced        - runs at grouped output resolution, e.g. [B, D]
producer_full  - full-resolution producer feeding node2's grouped reduction body
consumer_full  - full-resolution consumer after node2's grouped reduction
```

The domains are:

```text
producer_full_domain = (*node2_iter_ranges, *node2_reduce_ranges)
consumer_full_domain = (node1_numel, node1_rnumel)
```

For the XBLOCK example:

```text
producer_full_domain = [B, D, K]
consumer_full_domain = [B * K, D]
```

SIMD codegen now computes both source mappings:

```python
producer_full_groups, producer_full_values = (
    self._full_resolution_iteration_values(
        node2_reduction_body,
        iter_remapped,
        reduce_remapped,
    )
)
consumer_full_groups, consumer_full_values = (
    layout.full_resolution_iteration_values()
)
```

Then `_codegen_nested_node2_schedule()` chooses the mapping based on dependency
direction:

```text
prologue: reduction node has this pointwise as an ancestor
epilogue: this pointwise has the reduction node as an ancestor
```

Prologues use `producer_full_*`; consumers use `consumer_full_*`.

## Dependency cleanup needed for full consumers

After the codegen domain split, scheduler legality still needed to understand
that a reduced write can satisfy a broadcasted full-resolution read.

For example:

```text
write scale: scale[b, group]
read scale:  scale[b, d // G]
```

or:

```text
write s: s[b, d]
read s:  s[b, k, d]
```

These are valid vertical dependencies. We moved that understanding into generic
write->read matching instead of keeping a nested-only dependency bypass.

The generic matcher now canonicalizes broadcast reads by:

- dropping loop variables that do not appear in the read index,
- folding `FloorDiv(var, group_size)` into a reduced-size variable when the
  original var only appears through that grouped expression,
- comparing the canonicalized read with the write through normal dependency
  normalization.

We did not make this a global fusion scoring rule, because that changed
unrelated fusion ordering. Nested follow-up fusion supplies a local score, then
still goes through normal `can_fuse_vertical()` legality and backend checks.

## Tests that cover this

- `test_fullres_epilogue_small_dim_in_x`
  - Verifies XBLOCK full-resolution consumers fuse.
  - Includes `B=1`, because batch-1 flattening must not hide a correctness gap.

- `test_fullres_prologue_small_dim_in_x_loop_order`
  - Verifies full-resolution prologues still use the grouped reduction body
    coordinate order.

- `test_fullres_x_epilogue_rejects_intermediate_dependency`
  - Verifies full consumers do not fuse past an extra intermediate producer.

- `ImplDetailTest.test_broadcast_write_read_dep_matching`
  - Covers generic broadcast write->read dependency matching outside nested
    codegen.

## Mental model

The grouped reduction has two useful full-resolution views:

```text
producer view: grouped body coordinates
consumer view: parent kernel coordinates
```

Using the wrong view is not just a scheduling/profitability issue. It changes
which logical element a pointwise operation sees, so the failure mode is wrong
numerics.
