# Nested Reduction Store-Cache Vertical Dependencies

## The Core Irregularity

Normal vertical fusion proves that a consumer read is satisfied by a producer
write by matching `MemoryDep`s:

```text
producer writes: MemoryDep("buf0", d0, {d0: B})
consumer reads:  MemoryDep("buf0", d0, {d0: B})
```

If those deps do not match, `Scheduler.can_fuse_vertical()` normally rejects the
fusion with "memory deps did not match." That is the right default: it prevents
cases like reading `x + 1` from a producer that wrote `x`.

Nested reduction has one intentional exception. The grouped reduction may read
the outer reduction's output in a different logical iteration space, but codegen
does not satisfy that read by emitting a memory load. The outer reduction value
is still in `kernel.cse.store_cache`, so the grouped stage reads the cached
register value and remaps/broadcasts it inside the fused kernel.

So the dependency is real:

```text
grouped reduction depends on outer reduction output
```

but the generic memory-dep proof can fail:

```text
outer writes buf0 in outer coordinates
grouped reads buf0 in grouped-body coordinates
```

The first landing stack should not carry a broad generic fix for this. It keeps
the exception nested-scoped: `NestedReduction.can_fuse()` owns legality, and a
small nested safety check preserves the normal intermediate-dependency
rejection before the pair can bypass exact vertical `MemoryDep` matching.

## Example 1: Normal Vertical Fusion

```python
a = x + 1
out = a * 2
```

The producer writes `a`, and the consumer reads `a` with the same index. Generic
vertical fusion matches the write/read deps and fuses.

No nested-specific behavior is needed.

## Example 2: RBLOCK Nested Reduction

```python
x_normed = rmsnorm(x)                       # [B, D]
amax = x_normed.reshape(B, D // G, G).abs().amax(dim=-1)
```

Conceptually:

```text
outer reduction:
  tile: [B, D]
  output: inv_rms, shape [B, 1] or [B]

grouped reduction:
  body: [B, D // G, G]
  reduced output: [B, D // G]
```

The grouped body may read values produced by the outer reduction, such as
`inv_rms`. Generic `MemoryDep` matching can see different expressions because
the grouped body is expressed in `[B, groups, G]` while the outer reduction was
emitted in `[B, D]`.

Codegen handles this by:

1. Emitting the outer reduction.
2. Leaving its internal output in `kernel.cse.store_cache`.
3. Emitting the grouped body.
4. Letting normal `inner.load(name, remapped_index)` hit `store_cache`.
5. Applying the parent-full load transform if the cached value needs a
   broadcast.

The scheduler therefore must allow the unmatched `node1` output dep, but only
after `NestedReduction.can_fuse()` has proved this is the supported nested
shape.

## Example 3: XBLOCK Nested Reduction

```python
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)  # local reduction is K
```

The same input can be read through differently factorized domains:

```text
outer reads x as:   [B * K, D]
grouped reads x as: [B, D, K] or [B, K, D] depending on body order
```

There may be no exact shared `MemoryDep`, so initial fusion scoring can say
"no shared data" and never call nested legality. The X-axis path uses a
nested-scoped `same_nested_iteration_footprint` scoring bridge:

```text
same buffer name
same product of ranges
different factorization
```

That bridge is not a legality proof. It only gives the pair enough score for
`NestedReduction.can_fuse()` and the nested vertical safety check to run. A
follow-up generic dep-normalization/reindexing change could remove this
nested-only scoring bridge.

Concrete deps from `B=32, K=16, D=1024`:

```text
node1 = outer RMSNorm reduction
node2 = weighted reduce-K grouped reduction

node1 buffers:
  OrderedSet(['buf0'])

node1 writes:
  MemoryDep('buf0', d0, {d0: 512})

node2 unmet:
  MemoryDep('buf0', 16*d0 + d2, {d0: 32, d1: 1024, d2: 16})
```

These refer to the same logical value:

```text
node1 row index: d0 in [0, B*K)
node2 row index: 16*b + k, carried inside body coords [B, D, K]
```

The `d1: 1024` dimension is the passthrough `D` lane. The RMSNorm statistic is
broadcast over that dimension, so the read index does not use `d1`.

Generic vertical fusion sees:

```text
write: buf0[d0] over {d0: 512}
read:  buf0[16*d0 + d2] over {d0: 32, d1: 1024, d2: 16}
```

and cannot prove this read is satisfied by that write. Nested codegen can
satisfy it because the value is in `store_cache` and the grouped body remaps
coordinates explicitly.

## Example 4: Why Strided Slice Must Fall Back

```python
x = x[:, ::2]
rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
x_norm = x / rms
out = x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)
```

The scheduler saw three reductions:

```text
op0: partial/earlier reduction from the mean
op1: another reduction needed to finish the RMS statistic
op2: grouped amax after x_norm
```

The bad intermediate version fused `op0 + op2` as nested. But `op2` still had
an unmet dependency on `op1`, and `op1` depended on `op0`. That produced a
cycle:

```text
op1 -> op0_op2 -> op1
```

This is why the nested path still needs an intermediate-dependency safety check.
It can bypass exact matching for `node1`'s own outputs, because codegen
satisfies those through `store_cache`, but unrelated dependencies such as
`op1` must remain visible and reject the fusion.

The bypass is intentionally limited to remaining `MemoryDep` reads of buffers
produced by `node1`. It should not accept weak deps, star deps, mutation deps,
or arbitrary same-name dependencies; those continue to use the normal vertical
fusion rules.

Concrete deps from that rejected candidate:

```text
node1 = op0
node2 = op2

node1 buffers:
  OrderedSet(['buf0'])

node1 writes:
  MemoryDep('buf0', 16*d0 + d1, {d0: 32, d1: 16})

node2 unmet:
  MemoryDep('buf1', d0, {d0: 32})
```

There is no unmatched `node1` output dep here. The remaining dep is `buf1`,
which is produced by the intermediate reduction `op1`, so generic vertical
fusion must reject it. This is the case that showed why the exception cannot be
a separate nested-only "looks safe" helper; it needs to stay inside the normal
vertical dep resolver so intermediate deps are still checked.

## Current Invariant

Scheduler may admit nested fusion only if:

1. `NestedReduction.can_fuse(node1, node2)` proves the pair is a supported
   nested reduction.
2. The nested vertical safety check rejects extra intermediate dependencies.
3. X-axis local reductions may fuse, but only through nested legality/codegen,
   not through a broad generic dep-equivalence rule.
4. All admitted nested fusions must still pass the ordinary generic safety
   checks around weak deps, mutation, and intermediate dependencies.

This is deliberately narrower than a generic "broadcast deps are equivalent"
rule.

## Better Long-Term Shape

The nested-scoped bridge is still a sign that scheduler dependency matching does
not model in-kernel value availability across iteration domains. A cleaner
future design would make this relation explicit, for example:

```text
producer writes logical value V
consumer reads V at compatible resolution
codegen satisfies V through store_cache/register remap
```

Then nested reduction would not need to say "ignore this node1 dep mismatch" or
carry a nested-specific score bridge. Vertical dependency resolution would
understand that this specific producer/consumer value is satisfied in-kernel,
while all unrelated deps remain normal memory deps.
