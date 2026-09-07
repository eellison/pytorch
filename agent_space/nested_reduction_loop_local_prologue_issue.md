# Nested Reduction: Loop-Local Full-Resolution Prologue Issue

This note documents the non-persistent nested-reduction failure found while
adding exotic indexing and prologue/epilogue coverage.

## Short Version

The concerning test was not just "fusion skipped in a hard case". It exposed a
real codegen lifetime bug.

The failing pattern was:

```python
def f(x, weight, bias, scale):
    x = F.rms_norm(x, (D,), weight)
    x_scaled = torch.ops._inductor_test.realize(torch.relu(x * scale + bias))
    amax = x_scaled.view(B, D // G, G).abs().amax(dim=-1)
    return torch.clamp(torch.log1p(amax), min=0.0, max=10.0)
```

Persistent outer reduction worked. Non-persistent outer reduction failed at
compile time with:

```text
AssertionError: buf1
```

The root cause was that `buf1` was a full-resolution value produced after the
outer reduction, but in the looped/non-persistent kernel that value is
tile-local. The original code emitted it in the outer reduction pass, flushed
the reduction loop, invalidated loop-local CSE state, and then the grouped
reduction tried to read `buf1` later as if it were still available.

The fix was to treat full-resolution outer-reduction epilogues that feed the
grouped reduction as grouped-pass prologues. They now run in the same loop
scope where the grouped reduction consumes them.

## Why This Was Alarming

There are legitimate reasons not to fuse:

- the layout is non-contiguous enough that the tiling/coalescing analysis says
  the fused kernel is unlikely to be profitable
- the grouped dimension cannot be identified safely
- the second reduction is not the small exact grouped reduction this feature is
  designed for

This was different. The shape and operation pattern are exactly in scope:

- RMSNorm or LayerNorm over the large axis
- pointwise work at full resolution
- grouped amax over `G`
- reduced-output pointwise epilogue

If this does not work in non-persistent mode, then the feature cannot be relied
on for normal fused scheduler nodes. Skipping the test would have left a hole in
the main contract.

## The Fused Graph Shape

For the failing non-persistent case, the nested fused node looked like:

```text
FusedNestedReductions(op0, op1, op2, op3)

node1:
  op0 reduction      writes buf0    group (B, D)
  op1 pointwise      writes buf1    group (B * D, 1)

node2:
  op2 reduction      reads buf1     group (B * D/G, G)
  op3 pointwise      reads buf2     group (B * D/G, 1)
```

`op0` computes the RMSNorm reduction result. `op1` computes the full-resolution
normalized/scaled/relu value. `op2` is the grouped amax. `op3` is the
reduced-output `log1p`/`clamp` epilogue.

The important point is that `op1` logically belongs between the outer
reduction and the grouped reduction. It is a consumer of the outer reduction,
but it is also a producer for the grouped reduction.

## What Went Wrong

Nested reduction codegen has two physical phases:

1. Generate `node1` using the outer reduction iteration space.
2. Generate the grouped reduction and its pointwise prologues/epilogues using
   the nested layout.

For persistent reductions, the outer reduction has no loop over `R`. Values
produced after the reduction are still live in the kernel body, so the bug did
not show up.

For non-persistent reductions, the outer reduction is emitted as:

```text
for r0_offset in tl.range(...):
    compute loop-local values
    accumulate reduction

post-loop:
    compute reduction result
```

Then the nested grouped pass opens another loop over the same physical tile. A
full-resolution value like `buf1` cannot be carried from the first loop body to
the second loop body through ordinary `store_cache`; it is loop-local.

The old schedule effectively did this:

```text
outer pass:
  emit op0
  emit op1 full-resolution pointwise
  flush codegen_body()
  invalidate loop-local store_cache entries

grouped pass:
  emit op2
  op2 loads buf1
```

Because `buf1` had only internal users, it had been marked removed. When the
grouped body could not find it in `store_cache`, it fell through to a normal
load path. Normal load of a removed buffer is illegal, so codegen failed with:

```text
AssertionError: buf1
```

## Why Keeping `store_cache` Longer Is Wrong

A tempting fix would be to stop invalidating `buf1`, or to special-case the
load path so it sees the stale store-cache entry.

That is not safe. In the non-persistent case, `buf1` depends on the current
`r0_offset` tile. Keeping it live across the loop boundary would mean using a
value computed in one loop context from another loop context. The failure is a
scope mismatch, not just over-eager cache invalidation.

The correct fix is to emit the full-resolution pointwise work inside the loop
scope where it is consumed.

## The Fix

Codegen now splits `node1` into:

- `node1_codegen_nodes`: reductions and non-full-resolution work that should
  remain in the outer pass
- `node1_fullres_epilogues`: full-resolution pointwise nodes that depend on a
  `node1` reduction and feed later nested work

Only `node1_codegen_nodes` are passed to the ordinary outer schedule:

```python
combined_schedule = self.generate_node_schedule(
    node1_codegen_nodes, numel1, rnumel1
)
```

Then the deferred full-resolution outer epilogues are prepended to the grouped
pass prologue list:

```python
self._codegen_group_reduction_prologue(
    kernel,
    [*node1_fullres_epilogues, *node2_prologues],
    node2_reduction_body,
    layout,
    iter_remapped,
    reduced_output_family,
    fullres_family,
)
```

So the physical order becomes:

```text
outer pass:
  emit op0
  flush outer reduction loop

grouped pass:
  emit op1 full-resolution pointwise prologue
  emit op2 grouped reduction
  emit op3 reduced-output epilogue
```

Now `buf1` is written into `store_cache` immediately before the grouped
reduction reads it, in the correct loop scope.

## Resolver Assertion Cleanup

The full-resolution resolver previously asserted when any internal buffer was
not available in `store_cache`.

That was too broad. Some internal buffers may legitimately be allocated memory
loads if they were not removed. The hard error should only apply when the
buffer has been removed from global memory and therefore must be materialized
in registers.

The assertion now checks:

```python
if name in V.graph.removed_buffers or name in kernel.removed_buffers:
    raise AssertionError(...)
```

If the buffer was not removed, falling back to a normal load remains legal.

## Normal Reduction Fusion Comparison

Regular reduction fusion already handles pointwise prologues because it picks a
single reduction iteration space for the fused node. A reduced prologue usually
looks like `[X, 1]` relative to the reduction tile, and Triton broadcasting can
handle that naturally.

Nested reduction is different. The parent tile is still `[X, R]`, and the
grouped reduction later interprets `R` as `[groups, G]`. Full-resolution
pointwise values are not just ordinary epilogues; if they feed the grouped
reduction, they must be generated in the grouped pass's physical loop.

This is another reason the resolution/role distinction should become
first-class:

```text
role:       outer epilogue, grouped prologue, grouped epilogue
resolution: full, reduced-output, later half
lifetime:   outer-loop value, grouped-loop value, post-group value
```

The failure came from classifying by graph position only: `op1` was an epilogue
of `node1`, but physically it needed to be a prologue of `node2`.

## Why The Current Fix Is PR-Sized

The larger cleanup is a `NestedReductionPlan` that explicitly records every
subnode's role and resolution. That is still the right architecture follow-up,
but it is bigger than a pre-land fix.

The current fix is intentionally smaller:

- it does not add a new planning object
- it does not change the grouped reduction layout model
- it reuses the existing `_codegen_fullres_pointwise` path
- it keeps resolution materialization centralized in `_GroupReductionLayout`

It fixes the actual lifetime bug without adding a second ad hoc broadcast path.

## Test Coverage

The test that used to be skipped now runs in both persistent and
non-persistent modes:

```text
NestedReductionTest.test_multi_op_prologue_and_epilogue
NestedReductionNonPersistentTest.test_multi_op_prologue_and_epilogue
```

The broader exotic indexing group also covers:

- transposed input: numerics correct, fusion may be rejected by coalescing
- strided-slice input: numerics correct, fusion may be rejected by coalescing
- multi-op prologue and epilogue: fuses in persistent and non-persistent modes
- full-resolution epilogue with multiple outputs
- grouped reduction with elementwise weight multiply

Full suite verification after the fix:

```text
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py
  Ran 180 tests, OK (skipped=60)

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction_internals.py
  Ran 39 tests, OK (skipped=13)
```

## Remaining Design Takeaway

The big lesson is that nested reduction cannot treat "pointwise around a
reduction" as a single category. The same pointwise node can be:

- a graph epilogue of the outer reduction
- a physical prologue of the grouped reduction
- full-resolution, reduced-output, or eventually half-resolution
- loop-local or post-loop depending on persistent vs non-persistent codegen

For the current PR, the explicit deferral of full-resolution outer epilogues is
enough. For the next cleanup, `FusedNestedReductions` should probably carry a
small plan that classifies each subnode once, so scheduler checks, buffer
internalization, and codegen all use the same role/resolution/lifetime model.
