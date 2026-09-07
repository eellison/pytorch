# Nested Reduction Pointwise Resolution Handoff

This note captures the issue found while trying to cover nested reduction with
the same pointwise/reduction resolution cases that regular fusion supports.

## Trigger

Regular pointwise + reduction fusion accepts pointwise nodes at two resolutions:

- full resolution: `pointwise_numel == red_numel * red_rnumel`
- reduced-output resolution: `pointwise_numel == red_numel`

Nested reduction has the same apparent surface area, but the role of the
pointwise node matters:

- prologue: pointwise producer that feeds the grouped reduction
- epilogue: pointwise consumer after the grouped reduction

Those are not symmetric in the current nested codegen.

## Current PR State

The current PR intentionally supports:

- full-resolution prologues before the grouped reduction
- reduced-output prologues before the grouped reduction
- reduced-output epilogues after the grouped reduction
- full-resolution epilogues after the grouped reduction, only for
  `small_dim_in_r`

The guard is in `NestedReduction._node2_pointwise_nodes_are_supported(...)` in
`torch/_inductor/scheduler.py`. It permits node2 pointwise prologues at either
full or reduced-output resolution. It permits epilogues at reduced-output
resolution, and full resolution only when the grouped axis is in R.

The tests that encode this boundary are:

- `test_reduction_fusion_pointwise_prologue_epilogue`
  - verifies a `FusedNestedReductions` node structurally contains full-res
    pointwise prologue work plus reduced/full epilogue work
  - parameterized over full, row-broadcast, and column-broadcast pointwise
    inputs
- `test_reduced_resolution_pointwise_prologue`
  - verifies a reduced-output prologue shape fuses, remains in node2, and
    matches eager numerics

## The Reduced-Resolution Prologue Failure

This pattern exposed the problem:

```python
def f(x, group_extra, epilogue_extra):
    sums = (x * x).sum(dim=-1, keepdim=True)
    inv = torch.rsqrt(sums / D + 1e-6)
    group_extra = torch.ops._inductor_test.realize(group_extra + sums)
    x = (x * inv).view(B, D // G, G)
    out = (x + group_extra[:, :, None]).abs().amax(dim=-1)
    return out + epilogue_extra
```

`group_extra` is a prologue node with shape `[B, D // G]`, while the grouped
reduction consumes it as `[B, D // G, G]`. Before the fix, scheduler could
fuse this into nested reduction and codegen produced wrong numerics.

The reason is that the grouped reduction body runs at full parent-tile
resolution. A reduced-output prologue value must be explicitly lifted from
`[X, groups]` to `[X, groups, G]` and then flattened back to `[X, RBLOCK]`
before it can be used inside the reduction over `G`.

## Normal Fusion Comparison

The closest ordinary fusion case is:

```python
def f(x, y):
    p = torch.ops._inductor_test.realize(y + 1.0)
    return (x.view(B, NG, G) + p[:, :, None]).sum(dim=-1)
```

The fused scheduler node has:

- prologue pointwise group: `(numel=B*NG, rnumel=1)`
- reduction group: `(numel=B*NG, rnumel=G)`

Generated Triton looks like:

```python
tmp2 = tmp0 + 1.0
tmp4 = tmp3 + tmp2
tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
tmp7 = tl.sum(tmp5, 1)[:, None]
```

Regular fusion can rely on normal Triton broadcasting because it has already
chosen the reduction's iteration space, where the reduced producer is `[X, 1]`.

Nested codegen is different: the parent kernel tile is still `[B, D]`, and the
grouped reduction decomposes `D -> [groups, G]` only inside the grouped
reduction. The reduced prologue is `[B, groups]`, not `[B*groups, 1]`, so normal
broadcasting does not land in the right parent tile without an explicit nested
layout transform.

## Current Implementation Shape

The landed fix follows the resolution-aware model:

- `_GroupedReductionOpsHandler` now has a `load_resolver`, like
  `_PointwiseRemapHandler`.
- The grouped reduction stage passes a full-resolution resolver from
  `_GroupReductionLayout.resolve_full_resolution_load(...)`.
- Reduced prologues are emitted under the reduced-output family.
- When the grouped reduction body loads a reduced prologue from `store_cache`,
  the layout materializes it at parent resolution:
  `[X, groups] -> [X, groups, 1] -> [X, groups, G] -> [X, RBLOCK]`.
- Full-resolution prologues and node1 reduction outputs go through the same
  resolver and are identity materializations.

This avoids a handler-local "if shape looks reduced, broadcast it" rule. The
handler declares that the grouped body needs parent-resolution values; the
layout owns the transform.

## What Would Have Been A Smell

The rejected quick fix was to teach `_GroupedReductionOpsHandler.load()` to look
in `kernel.cse.store_cache`, notice that a value has reduced-output shape, and
broadcast it directly in the handler.

That would have fixed the repro locally, but it was not a good landing shape:

- it inferred value resolution from `CSEVariable.shape` at a late load site
- it added another implicit broadcast path separate from the full-resolution
  epilogue resolver
- it made prologue legality and codegen behavior diverge again
- it did not scale cleanly to half-resolution / NVFP4, where the transform is
  not just broadcast

## Better Follow-Up Direction

The clean follow-up is the same one suggested by the broader resolution plan:
make pointwise resolution first-class.

Concretely:

1. Add a scheduler-owned `NestedReductionPlan`.
2. Classify each node2 pointwise subnode once:
   - role: `prologue` or `epilogue`
   - resolution: `full`, `reduced`, later `half`
3. Make codegen consume that plan instead of rediscovering roles and
   resolutions from ancestors and body numels.
4. Introduce an explicit value materialization API, roughly:

```python
materialize_value(name, source_resolution, target_resolution)
```

or equivalently a family method that makes the target resolution explicit.

Reduced-output prologues now use an intentional `reduced -> full`
materialization path. The same model should extend to NVFP4 by adding
`full/reduced -> half` materialization without another handler-local special
case.

## Open Questions

- Should reduced-output prologues be represented as explicit IR broadcast before
  nested fusion, or should nested codegen own the lift?
- Should the next cleanup be a minimal `NestedReductionPlan` for current
  full/reduced cases before any NVFP4 work?
- Can the full-resolution epilogue resolver and any future prologue resolver
  share one materialization API without making `_DerivedIterationFamily` too
  abstract?

## Verification From This Pass

Commands run after adding reduced-resolution prologue support and tests:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py -k reduction_fusion_pointwise
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py -k reduced_resolution_pointwise_prologue
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction_internals.py
git diff --check -- torch/_inductor/scheduler.py torch/_inductor/codegen/simd.py test/inductor/test_nested_reduction.py test/inductor/test_nested_reduction_internals.py
```

All passed.
