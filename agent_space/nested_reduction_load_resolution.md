# Nested Reduction Load Resolution

This note describes the current nested-reduction load design. Older notes may
call this a "load resolver"; after the latest cleanup, the code is more
accurately a post-load resolution transform.

## Principle

Nested codegen should not invent a second load cache.

Loads still go through the normal Inductor path:

```text
ops.load(name, index)
  -> CSEProxy.load(name, index)
     -> if name is in kernel.cse.store_cache: return store_cache[name]
     -> otherwise emit a real memory load at index
```

The nested path only adds one step after the normal load returns:

```text
normal loaded CSEVariable
  -> maybe_broadcast_value_to_parent_resolution(...)
  -> value at the resolution required by this nested stage
```

So the nested handler is not responsible for deciding whether a load is a
store-cache hit, an actual memory load, or an indirect load. That remains owned
by `CSEProxy.load`.

## Store Cache Semantics

`kernel.cse.store_cache` is keyed by buffer name, not by `(name, index)`.

That is intentional in the generic codegen path. A store in the current kernel
records the whole tile/register value for that buffer:

```text
store buf, index, value
  -> store_cache["buf"] = value
```

A later load of that same buffer name can return the stored tile value directly.
The index only matters when the value is not in `store_cache` and codegen must
emit a real memory load.

This means nested reduction should not try to answer "load buffer X at index I"
from a custom cache. If it did, it would either duplicate generic CSE behavior
or accidentally ignore index semantics for real memory loads.

## What The Nested Transform Does

Nested reduction has multiple stage resolutions:

- parent-full: the original parent tile, e.g. `[XBLOCK, RBLOCK]`
- reduced-output: one value per local group, e.g. `[XBLOCK, RBLOCK / G]`

The grouped reduction stores reduced-output values into the normal store cache.
For example:

```text
scale: [XBLOCK, RBLOCK / G]
```

A later parent-full pointwise consumer may load `scale` while running at:

```text
[XBLOCK, RBLOCK]
```

The normal load returns the cached reduced-output CSE variable. The nested
post-load transform then lifts it:

```text
[XBLOCK, RBLOCK / G]
  -> reshape [XBLOCK, RBLOCK / G, 1]
  -> broadcast [XBLOCK, RBLOCK / G, G]
  -> reshape [XBLOCK, RBLOCK]
```

If the value is already parent-full, or if it is a scalar/broadcastable value,
the transform returns it unchanged.

## Why Not Key By Index?

For current nested reduction stages, the values that need resolution conversion
are kernel-local register tiles produced by previous stores in the same kernel.
They are already represented as `CSEVariable`s with shapes. The important
question is their resolution, not their memory index.

If a value is not in `store_cache`, the generic load path must handle the index.
That is why the current handler calls the normal load first and transforms the
returned value afterward.

## Safety Check

The generic CSE load path has one removed-buffer assertion:

```text
if a buffer was removed/internalized,
then a nested stage that loads it must find it in store_cache
```

This catches schedule/codegen bugs where we removed a memory store but failed to
keep the producer value available in the same kernel. It is a consistency check,
not a replacement load path.

This check belongs in `CSEProxy.load`, not in nested codegen. A load from a
removed buffer is invalid regardless of which fusion created the internal value.

## Reduced-Resolution Prologues

A reduced-resolution pointwise producer may feed the grouped reduction body. For
example:

```text
group_extra: [XBLOCK, num_groups]
x:           [XBLOCK, num_groups, local_reduction_size]
amax(abs(x + group_extra[:, :, None]), axis=-1)
```

In that case the grouped body loads `group_extra` while running at parent-full
resolution. The normal load returns the cached reduced-resolution value, and the
post-load transform broadcasts it across the local reduction lanes.

This is semantically correct even for non-idempotent reductions: the source
program contains a broadcast before the reduction, so `sum(group_extra[:, :,
None])` really is `local_reduction_size * group_extra`. The scheduler still
must classify the pointwise node's domain correctly; the load transform only
materializes the already-admitted resolution change.

## Code Shape

The current structure is:

```text
_PointwiseRemapHandler.load(...)
  -> remap index into the active iteration family
  -> call inner.load(name, remapped_index)
  -> apply optional load_transform(value)

_GroupedReductionOpsHandler.load(...)
  -> call inner.load(name, index)
  -> apply optional load_transform(value)
```

For parent-full nested stages, `load_transform` is represented by:

```text
_ParentFullLoadTransform(kernel, layout)
```

It applies `_GroupedReductionLayout.maybe_broadcast_value_to_parent_resolution`
to the value returned by the normal load path.

This keeps coordinate mapping, CSE, store-cache behavior, load counting, and
real memory loads in the generic path. Nested code only owns the resolution
conversion that generic CSE does not know how to infer.

## Future Direction

NVFP4/half-resolution should extend this same model:

```text
normal load -> value-resolution transform
```

The transform may become more general than "broadcast to parent", but it should
still operate on a `CSEVariable` returned by the normal load path rather than
implementing a separate name/index cache.
