# Nested Reduction Class Writeup

This writeup explains the new classes introduced by the nested-reduction
implementation. It is written for review: what each class owns, why it exists,
what concrete shape/index transformations it represents, and what would go
wrong if we tried to collapse it into existing machinery.

## Mental Model

Nested reduction fuses two dependent reductions that traverse the same logical
data at different resolutions.

There are two main patterns.

### Small Dim In R

Example: RMSNorm followed by block amax for FP8 scale.

```python
x = [4 rows, 512 cols]
group_size = 128
```

The outer reduction computes one normalization statistic per row:

```python
parent tile: [X, R] = [4 rows, 512 cols]
```

The grouped consumer views the full-resolution register tile as:

```python
[4 rows, 4 groups, 128 lanes]
```

It reduces over the 128 lanes and produces:

```python
[4 rows, 4 groups]
```

So the parent R index ranges over columns `0..511`, but the grouped output R
index ranges over groups `0..3`.

### Small Dim In X

Example: RMSNorm over `[B, K, D]`, then weighted sum over `K`.

```python
x = [4 batches, 16 K, 512 D]
group_size = 16
```

The outer reduction flattens `B*K` into the parent X axis:

```python
parent tile: [X, R] = [64 rows, 512 cols]
```

The grouped consumer views the full-resolution register tile as:

```python
[4 batches, 16 K lanes, 512 cols]
```

It reduces over the 16 K lanes and produces:

```python
[4 batches, 512 cols]
```

The B=1 case is important and should still fuse:

```python
x = [1 batch, 16 K, 512 D]
parent X = 16
group_size = 16
groups = parent X // group_size = 1
```

This is a single group, not `group_size == 1`. The grouped view is:

```python
[1 batch group, 16 K lanes, 512 cols]
```

and the output is:

```python
[1 batch group, 512 cols]
```

The code needs to preserve this as a real nested reduction path; otherwise B=1
workloads lose the fusion guarantee.

## Scheduler-Side Classes

### `NestedReduction`

`NestedReduction` is the scheduler-side recognizer and scoring helper. It is
not a fused node itself. It answers: can `node1` and `node2` legally become a
nested-reduction kernel?

It checks:

- the feature flag and Triton backend support;
- `node1` is a reduction and `node2` depends on it;
- `node2` has exactly one small reduction;
- that reduction size is exact, power-of-two, and at most
  `MAX_SMALL_REDUCTION`;
- the two nodes cover the same total element count;
- the nodes either share an input read or form a producer-consumer pattern;
- the small dimension can be classified as living in parent X or parent R.

Concrete small-dim-in-r example:

```python
x = [4, 512]
group_size = 128

node1: RMSNorm statistics
  group = (numel1=4, rnumel1=512)

node2: amax over groups
  view x as [4, 4, 128]
  group = (numel2=16, rnumel2=128)

total1 = 4 * 512 = 2048
total2 = 16 * 128 = 2048
group_size = 128
small_dim_in_r = True
```

Concrete small-dim-in-x example:

```python
x = [4, 16, 512]
group_size = 16

node1: RMSNorm over D after flattening B*K
  group = (numel1=64, rnumel1=512)

node2: weighted sum over K
  view normalized x as [4, 16, 512]
  group = (numel2=2048, rnumel2=16)

total1 = 64 * 512 = 32768
total2 = 2048 * 16 = 32768
group_size = 16
small_dim_in_r = False
```

The B=1 small-dim-in-x case is where classification gets easy to get wrong:

```python
x = [1, 16, 512]

node1:
  group = (numel1=16, rnumel1=512)

node2:
  group = (numel2=512, rnumel2=16)

total1 = 8192
total2 = 8192
```

Here `numel1 == group_size`, so there is exactly one group in X. The
implementation uses shared-read stride information in the flattened case so it
does not accidentally classify this as small-dim-in-r just because divisibility
checks are ambiguous.

Why this is a separate class:

- fusion legality belongs in the scheduler, not codegen;
- scoring nested candidates needs the same pattern knowledge as legality;
- codegen should receive an already validated fused node, not rediscover the
  pattern from scratch.

### `FusedNestedReductions`

`FusedNestedReductions` is the scheduler node created after
`NestedReduction.can_fuse()` succeeds. It records the two pieces of the fused
pair:

```python
node1 = outer/producer reduction
node2 = grouped consumer reduction plus any fused epilogues
```

It also computes and stores two key facts once:

```python
group_size
small_dim_in_r
```

This is important because scheduler and codegen must agree. If the scheduler
allowed a full-resolution epilogue because it thought the small dimension was
in R, but codegen reclassified it as X, the generated kernel would be wrong.

It also controls which downstream pointwise nodes may fuse into `node2`.

Reduced-output epilogue example:

```python
x = [4, 512]
amax = grouped_amax(x)        # [4, 4]
scale = clamp(amax / 448.0)   # [4, 4]
```

`scale` has the same `numel` as the grouped output, so it can run in the
reduced-output family.

Full-resolution epilogue example:

```python
x_norm = rms_norm(x)                         # [4, 512]
amax = grouped_amax(x_norm.view(4, 4, 128))  # [4, 4]
scale = clamp(amax / fp8_max)                # [4, 4]
x_fp8 = (x_norm.view(4, 4, 128) / scale[..., None]).to(fp8)
```

`x_fp8` is full-resolution: `[4, 512]`. `FusedNestedReductions.can_fuse_with()`
allows this only when `small_dim_in_r` is true, because the current full-res
epilogue path only knows how to lift `[X, groups]` values back across parent R.

Blocked full-resolution small-dim-in-x example:

```python
x = [4, 16, 512]
s = weighted_sum_over_K(x)       # [4, 512]
out = x_normed + s[:, None, :]   # [4, 16, 512]
```

This full-resolution epilogue would need to lift across parent X. The current
codegen does not support that case, so it must remain a separate kernel.

Why this class exists:

- a normal `FusedSchedulerNode` does not know there are two incompatible
  iteration spaces inside the fused group;
- it needs to preserve `node1` and `node2` separately for codegen;
- it owns the one-time classification metadata used by both scheduler and
  codegen;
- it gives downstream epilogue fusion a narrow policy instead of reusing the
  generic scheduler checks blindly.

## Codegen-Side Classes

### `DerivedIterationRangesRoot`

`DerivedIterationRangesRoot` is a range-tree root for the grouped output axis.
It is "derived" because it shares the parent tree's physical loop and launch
placement, but has reduced logical indexing.

Take the small-dim-in-r FP8 example:

```python
x = [4 rows, 512 cols]
group_size = 128
```

Parent R tree:

```python
numel        = 512
block_size   = R0_BLOCK
block_offset = r0_offset
index        = r0_offset + tl.arange(0, R0_BLOCK)
```

Derived R tree:

```python
numel        = 512 // 128 = 4
block_size   = R0_BLOCK // 128
block_offset = r0_offset // 128
index        = r0_offset // 128 + tl.arange(0, R0_BLOCK // 128)
mask         = index < 4
```

Persistent example:

```python
R0_BLOCK = 512
r0_offset = 0

parent r0_index:
  [0, 1, 2, ..., 511]

derived reduced_r0_index:
  [0, 1, 2, 3]
```

Looped example:

```python
R0_BLOCK = 256

loop 1: r0_offset = 0
  parent r0_index = [0, 1, ..., 255]
  derived index   = [0, 1]

loop 2: r0_offset = 256
  parent r0_index = [256, 257, ..., 511]
  derived index   = [2, 3]
```

Small-dim-in-x example:

```python
x = [4 batches, 16 K, 512 D]
parent X numel = 64
group_size = 16

parent X:
  numel        = 64
  block_size   = XBLOCK
  block_offset = xoffset

derived X:
  numel        = 64 // 16 = 4
  block_size   = XBLOCK // 16
  block_offset = xoffset // 16
```

B=1 small-dim-in-x example:

```python
x = [1 batch, 16 K, 512 D]
parent X numel = 16
group_size = 16
groups = 1

derived X:
  numel = 1
  block_size = XBLOCK // 16
  block_offset = xoffset // 16
```

That derived X root is the output batch axis. The 16 K values are the lanes
being reduced, not 16 output groups.

Why a plain `IterationRangesRoot` is not enough:

- it would use `R0_BLOCK` / `XBLOCK` instead of the reduced block size;
- it would use `r0_offset` / `xoffset` instead of `offset // group_size`;
- it would use parent mask names like `r0_mask`, colliding with parent masks;
- it would not know loop-local derived headers must be emitted inside a looped
  parent reduction;
- it would not carry readable named constants like
  `nested_R0_GROUP_SIZE`, `nested_R0_REDUCED_BLOCK`, and
  `nested_R0_REDUCED_NUMEL`.

The subclass is intentionally thin: it reuses normal range-tree entry creation,
but overrides the few pieces of geometry that differ from the parent.

### `_DerivedIterationFamily`

`_DerivedIterationFamily` is the temporary "active iteration space" for a
consumer stage. It does not describe one axis; it describes the whole set of
range trees that should be active while codegen runs a grouped reduction or
epilogue.

There are two modes.

#### Reduced-Output Family

Used for the grouped reduction output and reduced-resolution pointwise
epilogues.

Small-dim-in-r example:

```python
parent trees:
  X: rows,   numel = 4
  R: cols,   numel = 512

reduced-output family:
  X: parent X,       numel = 4
  R: derived R root, numel = 4 groups
```

The grouped store is now indexed as `[row, group]`, not `[row, column]`:

```python
tl.store(out + reduced_r0_index + 4 * x0, amax, reduced_r0_index_mask & xmask)
```

Small-dim-in-x example:

```python
parent trees:
  X: B*K, numel = 64
  R: D,   numel = 512

reduced-output family:
  X: derived X root, numel = 4 batches
  R: parent R,       numel = 512
```

The weighted-sum output is indexed as `[batch, D]`, not `[B*K, D]`.

`index_subs` remaps the loop-body symbols from the original consumer body into
the currently active reduced family. That lets normal `load()` and `store()`
codegen work without every caller learning about nested reductions.

#### Full-Resolution Family

Used for full-resolution epilogues after a small-dim-in-r grouped reduction.
It reuses the parent trees:

```python
full-resolution family:
  X: parent X, numel = 4
  R: parent R, numel = 512
```

The special part is `remapped_values`. If an epilogue reads `scale`, and
`scale` was produced at `[4 rows, 4 groups]`, the family stores a
broadcast-lifted register value:

```python
scale: [4, 4] -> [4, 4, 1] -> [4, 4, 128] -> [4, 512]
```

Then the full-resolution epilogue can read `scale` by name and get the register
value instead of trying to load a memory buffer.

Why this family abstraction exists:

- `kernel.range_trees` controls masks, dense shapes, indexing, and stores;
- reduced-output and full-resolution consumers need different active trees;
- derived headers must be emitted exactly once, and looped headers must be
  emitted in the loop body;
- pointwise epilogues should still use normal Inductor body codegen.

### `_GroupReductionLayout`

`_GroupReductionLayout` is the shape plan for the grouped consumer reduction.
It answers questions like:

- which parent tree is grouped?
- which tree passes through unchanged?
- what reshape should be emitted?
- which axis should Triton reduce?
- what is the reduced output shape?
- how do we build reduced-output and full-resolution families?

Small-dim-in-r shape plan:

```python
parent tile:
  [XBLOCK, R0_BLOCK]

grouped view:
  [XBLOCK, R0_BLOCK // G, G]

reduce axis:
  2

output shape:
  [XBLOCK, R0_BLOCK // G]
```

For `x = [4, 512]`, `G = 128`, persistent `R0_BLOCK = 512`:

```python
tmp = tl.reshape(value, [XBLOCK, 4, 128])
amax = tl.max(tmp, 2)
```

Small-dim-in-x shape plan:

```python
parent tile:
  [XBLOCK, R0_BLOCK]

grouped view:
  [XBLOCK // G, G, R0_BLOCK]

reduce axis:
  1

output shape:
  [XBLOCK // G, R0_BLOCK]
```

For `x = [4, 16, 512]`, `G = 16`, `XBLOCK = 64`:

```python
tmp = tl.reshape(value, [4, 16, R0_BLOCK])
sum = tl.sum(tmp, 1)
```

B=1 small-dim-in-x:

```python
parent X = 16
G = 16
groups = 1

tmp = tl.reshape(value, [1, 16, R0_BLOCK])
sum = tl.sum(tmp, 1)
output shape = [1, R0_BLOCK]
```

`construct_group_reduction_vars()` also handles this degenerate grouped axis:

```python
if groups == 1:
    non_group_var = 0
    group_var = parent_full_range_symbol
```

That means the output batch/group coordinate is fixed at zero, while the
reduction still iterates over the full parent X lane. This is the B=1 fusion
case we need to keep supported.

The layout also owns the full-resolution register lift shapes. For the FP8
scale example:

```python
scale reduced output:
  [X, groups] = [4, 4]

lift:
  [4, 4] -> [4, 4, 1] -> [4, 4, 128] -> [4, 512]
```

`CSEVariable.shape` tells us the old register shape. The layout supplies the
semantic mapping from grouped axis back to parent axis.

### `_GroupedReductionOpsHandler`

`_GroupedReductionOpsHandler` is the ops wrapper used only while running
`node2`'s reduction body. It intercepts two operations:

- `reduction()`
- `store_reduction()`

The original consumer body looks like an ordinary reduction over a small
dimension. For example:

```python
x_normed.view(4, 4, 128).abs().amax(dim=-1)
```

But inside the nested kernel, we do not want to generate a separate reduction
loop for `node2`. We already have a full-resolution register tile from the
outer reduction stage. The handler turns the normal reduction op into:

```python
reshaped = tl.reshape(value, [XBLOCK, groups_per_block, group_size])
reduced = tl.max(reshaped, axis=2)
```

Small-dim-in-r generated shape:

```python
tmp15 = tl.reshape(tmp14, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_GROUP_SIZE])
tmp20 = tl.max(tmp15, 2)
```

Small-dim-in-x generated shape:

```python
tmp14 = tl.reshape(tmp13, [nested_X_REDUCED_BLOCK, nested_X_GROUP_SIZE, R0_BLOCK])
tmp15 = tl.sum(tmp14, 1)
```

`store_reduction()` updates `store_cache` so later epilogues can read the
grouped output as a register value. It then stores through the reduced-output
family, so the index and masks are reduced-resolution.

Why it does not just call normal `store_reduction()`:

- normal reduction stores are post-loop stores for the outer reduction;
- grouped reduction stores happen per parent reduction iteration in the second
  pass;
- normal local-buffer cleanup can remove stores marked as internal, but the
  grouped store may still be the final output or feed epilogues.

### `_PointwiseRemapHandler`

`_PointwiseRemapHandler` runs pointwise epilogues under a chosen
`_DerivedIterationFamily`.

It handles two cases.

#### Reduced-Resolution Epilogue

Example:

```python
amax = grouped_amax(x)        # [4, 4]
scale = clamp(amax / 448.0)   # [4, 4]
```

The handler activates the reduced-output family. Loads of `amax` hit
`store_cache`; stores of `scale` use reduced indices:

```python
[row, group]
```

The epilogue body itself can stay normal pointwise code.

#### Full-Resolution Epilogue

Example:

```python
scale = [4, 4]
x_norm = [4, 512]
x_fp8 = (x_norm.view(4, 4, 128) / scale[..., None]).to(fp8)
```

The handler activates the full-resolution family. If the epilogue reads
`scale`, `load()` first checks `family.remapped_values` and returns the
broadcast-lifted register value:

```python
scale_fullres = [4, 512]
```

So the generated full-resolution code can look like ordinary pointwise code:

```python
tmp21 = tl.reshape(
    tl.broadcast_to(tmp20[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_GROUP_SIZE]),
    [XBLOCK, R0_BLOCK],
)
tmp24 = tmp14 / tmp21
tl.store(out + fullres_index, tmp24, xmask)
```

Load priority is:

```python
family.remapped_values
store_cache
remapped memory load
```

Why the handler exists:

- regular pointwise codegen would use the epilogue's original index space;
- internal buffers may have been removed from memory and exist only in
  `store_cache`;
- full-resolution epilogues need register values lifted from grouped output to
  parent tile shape before the body reads them.

## End-To-End Flow

The classes line up like this:

```python
NestedReduction.can_fuse(node1, node2)
  -> Scheduler.fuse()
  -> FusedNestedReductions(node1, node2)
  -> SIMDScheduling.codegen_nested_reduction()
```

Inside codegen:

```python
1. Generate node1 normally.
2. Keep node1 outputs in store_cache instead of global memory when internal.
3. Flush node1's first pass.
4. Build _GroupReductionLayout from the active parent X/R trees.
5. Build reduced-output _DerivedIterationFamily.
6. Run node2's reduction body under _GroupedReductionOpsHandler.
7. Split epilogues into reduced-output vs full-resolution.
8. Run reduced-output epilogues under _PointwiseRemapHandler + reduced family.
9. Run full-resolution epilogues under _PointwiseRemapHandler + fullres family.
10. Finalize and launch the single kernel.
```

The key design point is that only the active iteration family changes between
stages. The bulk of load/store/index code stays the existing Inductor machinery.

## Review Notes

The most important invariants to review are:

- `FusedNestedReductions.small_dim_in_r` is computed once and trusted by
  codegen.
- A derived root is only used for the grouped output axis; the other axis
  reuses the parent root.
- `groups == 1` is valid and important for B=1 small-dim-in-x fusion.
- `group_size == 1` is a different case from `groups == 1`.
- Full-resolution epilogues currently only fuse for small-dim-in-r.
- Full-resolution register lift is manual because the value is a CSE register,
  not an IR buffer that can be expanded before codegen.
