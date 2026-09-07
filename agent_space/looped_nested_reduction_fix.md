# Looped Nested Reduction: Full Fix Design

## Problem

The current nested-reduction codegen REQUIRES persistent reduction for the
outer loop. It forces `override_persistent_reduction=True`. If the backend
rejects this (e.g. dynamic shapes, very large R), nested reduction can't fire.

The fundamental issue: when the outer reduction is looped (non-persistent),
node1's output is computed per-chunk inside the loop. After the loop, only
the reduction accumulator (e.g. `sum(x*x)`) survives in registers — the
full post-normalization tile is gone. The grouped reduction needs that tile.

## Key Insight

The grouped "reduction" (reshape + max2/sum over G) is NOT a cross-iteration
accumulation. It's a **per-chunk local computation**. Each chunk of RBLOCK
elements contains RBLOCK/G complete groups (enforced by
`nested_reduction_min_rblock >= group_size`). The local reduce within each
group is independent across chunks.

So in looped mode, the kernel needs TWO loops:
- Loop 1: outer reduction (standard)
- Post-loop: finalize (rsqrt, etc.)
- Loop 2: reload, normalize, reshape, local-reduce, store

## Concrete Kernel Structure

### Persistent (current, unchanged)

```python
# One pass — full tile in registers
x = tl.load(in_ptr0 + ...)                  # full [XBLOCK, RBLOCK]
acc = tl.sum(x * x, axis=1)                 # outer reduction
rsqrt_val = libdevice.rsqrt(acc / R + eps)
x_norm = x * rsqrt_val * weight             # normalize (x still in regs)
reshaped = tl.reshape(x_norm, [XBLOCK, RBLOCK//G, G])
group_max = max2(reshaped, axis=2)           # local reduce
tl.store(out_ptr + ..., group_max)
```

### Looped (new)

```python
# Loop 1: outer reduction
for r_offset in range(0, R, RBLOCK):
    x_chunk = tl.load(in_ptr0 + r_offset + ...)
    acc += x_chunk * x_chunk
# Post-loop
rsqrt_val = libdevice.rsqrt(acc / R + eps)

# Loop 2: normalize + grouped stage
for r_offset in range(0, R, RBLOCK):
    x_chunk = tl.load(in_ptr0 + r_offset + ...)   # RELOAD
    w_chunk = tl.load(in_ptr1 + r_offset + ...)    # RELOAD
    x_norm = x_chunk * rsqrt_val * w_chunk
    reshaped = tl.reshape(x_norm, [XBLOCK, RBLOCK//G, G])
    group_max = max2(reshaped, axis=2)
    tl.store(out_ptr + r_offset//G + ..., group_max)  # different offset per iteration
```

Loop 2 reloads x from global memory. This is slower than persistent (one
extra read of x) but correct. The grouped reduce is purely local within
each chunk.

## What Changes in the Code

### 1. Don't internalize node1 outputs when non-persistent

File: `torch/_inductor/codegen/simd.py`, `codegen_nested_reduction`

Current (line ~2582):
```python
internal_node1_outputs: set[str] = set()
for buf_name in node1.get_buffer_names():
    buf = self.scheduler.name_to_buf.get(buf_name)
    if buf is not None and buf.has_only_internal_users(fused_names):
        V.graph.removed_buffers.add(buf_name)
        internal_node1_outputs.add(buf_name)
```

Change: gate on `kernel.persistent_reduction`:
```python
internal_node1_outputs: set[str] = set()
if kernel.persistent_reduction:
    for buf_name in node1.get_buffer_names():
        buf = self.scheduler.name_to_buf.get(buf_name)
        if buf is not None and buf.has_only_internal_users(fused_names):
            V.graph.removed_buffers.add(buf_name)
            internal_node1_outputs.add(buf_name)
# else: node1 stores to memory, loop 2 reloads
```

BUT: `kernel.persistent_reduction` isn't known until after
`create_kernel_choices`. The internalization happens BEFORE kernel creation
(to influence buffer allocation). So either:
- Move internalization after kernel creation, or
- Predict persistence at the point of internalization (using
  `override_persistent_reduction` from kernel_kwargs)

The prediction approach: if we set `override_persistent_reduction=True` in
kernel_kwargs, we KNOW the kernel will be persistent. If we DON'T set it
(or can't because of dynamic shapes), we should NOT internalize.

```python
will_be_persistent = kernel_kwargs.get("override_persistent_reduction", False)
internal_node1_outputs: set[str] = set()
if will_be_persistent:
    for buf_name in node1.get_buffer_names():
        ...
```

### 2. Clear store_cache between loops

File: `torch/_inductor/codegen/simd.py`, `codegen_nested_reduction`

After the first `kernel.codegen_body()` (which emits loop 1), clear
store_cache entries for node1 outputs. This ensures the grouped reduction
body reloads from memory instead of reading stale register variables.

```python
with kernel:
    kernel.codegen_body()  # loop 1 + post-loop

    if not kernel.persistent_reduction:
        # Clear stale per-chunk register values so the grouped body
        # reloads from memory in loop 2.
        for name in node1.get_buffer_names():
            kernel.cse.store_cache.pop(name, None)
    
    # ... grouped reduction body ...
    # ... epilogues ...
    
    kernel.codegen_body()  # loop 2 (looped) or finalize (persistent)
```

### 3. The grouped reduction body runs inside loop 2

This is where the magic happens. After loop 1's `codegen_body()` completes
and the `disable_reduction()` context exits, `inside_reduction` is restored
to True. So any new code the grouped body emits goes into the kernel's
pending compute/store buffers for loop 2.

When `kernel.codegen_body()` is called again at the end, it emits loop 2
containing all the grouped-reduction code.

For persistent: `inside_reduction` may behave differently (no loop), but the
code still emits into the kernel body. This is the current behavior — no
change needed.

### 4. _GroupedReductionOpsHandler changes for looped

The handler's `store_reduction` currently uses `family.store` which calls
`kernel.store` directly. For looped mode, this store needs to happen INSIDE
the loop iteration (at a per-chunk offset). Since `inside_reduction=True` at
this point, `kernel.store` should naturally emit inside the loop body.

BUT: the store index needs to account for the chunk offset. The derived
range tree's `block_offset` carries this (via `r0_offset // G`). The
`is_loop=parent.is_loop` fix ensures the derived tree's offset is
loop-local. So the indexing should be correct.

**Key invariant**: `DerivedIterationRangesRoot.is_loop = parent.is_loop`
ensures the derived tree's block_offset is recomputed per iteration, not
stale from a previous stage. This was the dynamic-shapes bug fix and it's
the same invariant that makes loop 2 correct.

### 5. Epilogues run inside loop 2 alongside the grouped reduction

ALL epilogues (reduced-output AND full-resolution) run inside loop 2.
Everything is per-chunk:

```python
# Inside one iteration of loop 2:
x_chunk = tl.load(in_ptr0 + r_offset)           # reload x
w_chunk = tl.load(in_ptr1 + r_offset)           # reload weight
x_norm = x_chunk * rsqrt_val * w_chunk           # normalize

# Grouped reduction (per-chunk, local)
reshaped = tl.reshape(x_norm, [XBLOCK, RBLOCK//G, G])
group_max = max2(reshaped, axis=2)               # [XBLOCK, RBLOCK//G]

# Reduced-output epilogue (per-chunk)
scale = clamp(group_max / fp8_max, min=1e-12)

# Full-res epilogue (per-chunk — broadcast is WITHIN the chunk)
scale_bc = broadcast(scale, [XBLOCK, RBLOCK])    # RBLOCK >= G, so this is local
x_fp8 = (x_norm / scale_bc).to(fp8)

# Stores at chunk-specific offsets
tl.store(scale_out + r_offset//G, scale)
tl.store(fp8_out + r_offset, x_fp8)
```

Why this works:
- The broadcast from `[XBLOCK, RBLOCK//G]` to `[XBLOCK, RBLOCK]` is
  within-chunk because `RBLOCK >= G` (enforced by
  `nested_reduction_min_rblock`).
- The full-res family activates with the outer `(x_tree, r_tree)` range
  trees. In looped mode, `r_tree` IS the loop variable, so
  `r0_offset + tl.arange(0, RBLOCK)` naturally gives per-chunk offsets.
- The `_DerivedIterationFamily`'s `remapped_values` (broadcast CSE vars)
  are per-chunk since they're computed from per-chunk grouped outputs.

No epilogue type needs to be rejected — the per-chunk structure handles
both reduced-output and full-resolution epilogues identically to the
persistent case, just operating on RBLOCK elements per iteration instead
of all R elements at once.

### 6. Scheduler changes

`NestedReduction.can_fuse` currently requires static `rnumel2` (line 501).
For dynamic shapes: use `optimization_hint` to get the hinted value (the
other agent already applied this).

The scheduler doesn't need to know about persistent vs looped — that's a
codegen decision. But it DOES need to not reject dynamic shapes.

### 7. Remove the assert in create_kernel_choices

File: `torch/_inductor/codegen/triton.py`, line ~6878

Current:
```python
if not TritonKernel.has_persistent_RBLOCK(kernel_features.reduction_numel):
    assert not kernel_kwargs.get("override_persistent_reduction")
    kernel_kwargs["override_persistent_reduction"] = False
```

For dynamic shapes: if we DON'T set `override_persistent_reduction` (because
we can't guarantee persistent), this assert never fires. The kernel falls
back to loop reduction. We just need to NOT set the override for dynamic
shapes.

Change in `codegen_nested_reduction`:
```python
rnumel_hint = V.graph.sizevars.optimization_hint(rnumel1, fallback=0)
can_be_persistent = isinstance(rnumel1, (int, sympy.Integer)) or (
    rnumel_hint > 0 and TritonKernel.has_persistent_RBLOCK(rnumel1)
)
if can_be_persistent and (is_producer_consumer or rnumel_hint <= 8192):
    kernel_kwargs["override_persistent_reduction"] = True
```

If `can_be_persistent` is False (dynamic, no bound), we don't set the
override, and the kernel uses loop reduction. The rest of codegen handles
both paths.

## Summary of Changes

| # | File | Change | Lines |
|---|------|--------|-------|
| 1 | `simd.py` | Gate internalization on `will_be_persistent` | ~5 |
| 2 | `simd.py` | Clear store_cache for node1 between loops when looped | ~5 |
| 3 | `simd.py` | No handler changes (loop 2 emits naturally) | 0 |
| 4 | `simd.py` | Epilogues run inside loop 2 (no rejection needed) | 0 |
| 5 | `simd.py` | Use `can_be_persistent` instead of unconditional override | ~5 |
| 6 | `scheduler.py` | Accept hinted rnumel2 (already done by other agent) | ~3 |
| Total | | | ~20 LOC |

## What This Enables

- Dynamic shapes with nested reduction (looped mode, correct but slower)
- Very large R dimensions that don't fit in persistent RBLOCK
- The `is_loop=parent.is_loop` fix becomes load-bearing (not just defensive)

## What This Doesn't Change

- Persistent mode is unchanged (the common case)
- `_DerivedIterationFamily` abstraction is unchanged
- `_GroupedReductionOpsHandler` is unchanged (its output naturally goes
  into loop 2 when `inside_reduction=True`)
- No new handler or stage type

## Testing

New tests needed:
- Force non-persistent via large R or dynamic shapes
- Verify numeric correctness for looped nested reduction
- Verify the kernel has two loops (two `for roffset` blocks)
- Verify node1 output is stored to memory (not internalized)

Existing tests to update:
- Dynamic-shapes form checks: expect nested reduction to fire (once the
  scheduler accepts hinted rnumel2)
- Possibly add a `test_large_R_looped_nested_reduction` that uses
  R > max_persistent_RBLOCK

## Risk Assessment

- **Low risk**: persistent path is untouched
- **Medium risk**: store_cache clearing between loops — need to verify no
  other code reads stale entries
- **Medium risk**: loop 2 emit — need to verify `codegen_body()` emits a
  clean second loop without interfering with loop 1's state
- **Low risk**: internalization gating — straightforward conditional

The dominant risk is the loop 2 emit: does the kernel's state machine
correctly produce a second reduction loop after the first? The
`disable_reduction` mechanism suggests yes, but it hasn't been exercised
in this exact pattern (two loops, second one is a different body).
