# Nested Reduction: Full-Resolution Prologue Loop-Order Issue

This note documents the correctness bug found after the loop-local prologue
fix. It is a separate issue: the value lifetime was correct, but the
full-resolution pointwise body was run with the wrong logical coordinates for
small-dim-in-x.

## Short Version

The failing pattern was:

```python
def f(x, w, bias):
    # x: [B, K, D]
    x_flat = x.reshape(B * K, D)
    rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
    y = torch.ops._inductor_test.realize(
        torch.relu((x_flat / rms).reshape(B, K, D) + bias[:, None, :])
    )
    out = (w[:, :, None] * y).sum(dim=1)
    return y, out
```

For `B=16, K=16, D=1024`, nested reduction fused but produced wrong numerics.
The key point is that the grouped reduction body is logically `[B, D, K]`:

```text
node1: rms over D
  x_flat: [B*K, D]

node2: weighted sum over K
  iter vars: [B, D]
  reduce vars: [K]
```

The physical parent tile is still `[X, R] == [B*K, D]`, but the body that
computes `y` wants logical variables `[B, D, K]`, not `[B, K, D]`.

## What Went Wrong

The old full-resolution pointwise emitter did:

```text
flat_index = x_index * rnumel + r_index
pointwise_iter_vars = decompose(flat_index, pointwise_body_ranges)
```

For small-dim-in-r, this happened to line up with the grouped body order.
For small-dim-in-x, it does not.

With `x_index = b * K + k` and `r_index = d`, the old flat expression was:

```text
flat = (b * K + k) * D + d
```

Decomposing this flat index into `[B, D, K]` gives:

```text
b' = flat // (D * K)
d' = (flat // K) % D
k' = flat % K
```

Substituting the original expression:

```text
b' = b
d' = (k * D + d) // K  mod D
k' = d % K
```

So `D` and `K` are effectively mixed. The generated code showed exactly that:
loads and stores used complex `% 16` / `// 16` expressions derived from the
flat `[B*K, D]` order rather than the grouped body's `[B, D, K]` order.

## Why CSEVariable Shape Is Not Enough

The existing `CSEVariable.shape` metadata is useful, but it answers a different
question:

```text
Does this register value exist at full resolution or reduced-output resolution?
If a full-resolution consumer reads it, do we need to broadcast it back up?
```

That is what `resolve_full_resolution_load()` uses it for. It can see a value
with shape `[X, groups]` and materialize:

```text
[X, groups] -> [X, groups, 1] -> [X, groups, G] -> [X, R]
```

The bug here is not only the shape of a value. It is the binding of a pointwise
body's symbols to the current tile coordinates. A value can have the right
shape and still be computed at the wrong logical element if we call the
pointwise body with variables in the wrong order.

For the failing case, the emitter needs to call the full-resolution prologue
with:

```text
body vars: [B, D, K]
values:    [non_group_var, other_var, group_var]
```

The CSE value shape does not tell us that mapping.

## The Fix

The fix is to stop emitting nested pointwise work by flattening the parent
`[X, R]` tile and decomposing it again. Instead, pointwise emission now starts
from the grouped reduction body's own logical coordinates.

For the grouped reduction, codegen already constructs:

```text
iter_remapped   # values for node2 body iter vars
reduce_remapped # values for node2 body reduce vars
```

For small-dim-in-x:

```text
node2 body vars: [B, D] + [K]
source groups:   [B, D, K]
source values:   [group_index, d_index, lane_index]
```

For small-dim-in-r:

```text
node2 body vars: [B, groups] + [G]
source groups:   [B, groups, G]
source values:   [b_index, group_index, lane_index]
```

Full-resolution pointwise prologues and epilogues are now mapped from those
source groups/source values into each pointwise node's own body ranges using
the same splitting logic as normal tiled fusion. That gives the prologue the
same logical coordinates the grouped reduction will use when it consumes the
value.

Reduced-output pointwise emission was also moved to the same coordinate-mapping
path. That removes the old nested pointwise shortcut:

```text
flatten body vars -> decompose into pointwise vars
```

and makes reduced-output and full-resolution pointwise emission use one model:

```text
current logical groups + current logical values -> target pointwise body vars
```

## Why This Is the Right Scope

This does not try to make arbitrary loop-order fusion work. It relies on the
same compatibility condition as existing tiled fusion: the target pointwise
body ranges must be splittable from the current logical groups. If they are
not, the existing `_split_iteration_ranges` path rejects the mapping.

That is the right failure mode for this PR. The pattern we need for nested
reduction is in scope because the full-resolution prologue is just the grouped
reduction's logical iteration space with a pointwise body around it. The fix
makes that explicit instead of relying on the parent `[X, R]` flattening order.

## Test Added

The positive test is:

```text
test_fullres_prologue_small_dim_in_x_loop_order
```

It returns both:

```text
y   # the realized full-resolution prologue output
out # the grouped reduction output
```

Returning `y` is important because it verifies the prologue's external store
indexing too, not just the later grouped reduction result.

