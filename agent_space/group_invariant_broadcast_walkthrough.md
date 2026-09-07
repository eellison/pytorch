# Group-Invariant Broadcast Walkthrough

This note explains why #190595 adds `_GroupInvariantBroadcast`, how it fits
with the existing nested-reduction handlers, and which limitation it works
around.

Current review tree:

```text
/data/users/eellison/pytorch/agent_space/pr190594_ci_fix
```

Primary code:

- [`_plan_nested_sub_parent_stage`](./pr190594_ci_fix/torch/_inductor/scheduler.py#L1208)
- [`materialize_value_at_sub_parent_resolution`](./pr190594_ci_fix/torch/_inductor/codegen/simd.py#L2006)
- [`_GroupInvariantBroadcast`](./pr190594_ci_fix/torch/_inductor/codegen/simd.py#L2248)
- [`_PointwiseRemapHandler`](./pr190594_ci_fix/torch/_inductor/codegen/simd.py#L2327)
- [`_SubParentSourceLoadResolver`](./pr190594_ci_fix/torch/_inductor/codegen/simd.py#L2391)
- [`_codegen_nested_reduction`](./pr190594_ci_fix/torch/_inductor/codegen/simd.py#L3214)

## Example

For an NVFP4-like operation with block size `G=16`:

```python
y = rms_norm(x)                              # [B, D]
groups = y.view(B, D // G, G)                # [B, D/G, G]
amax = groups.abs().amax(dim=-1)             # [B, D/G]
scale = (amax / 6).clamp(...).to(float8)      # [B, D/G]
pairs = groups.view(B, D // G, G // 2, 2)    # [B, D/G, G/2, 2]
packed = pack(
    pairs[..., 0] / scale.float()[..., None],
    pairs[..., 1] / scale.float()[..., None],
)
```

There are three relevant resolutions:

```text
parent:      [B, D]
reduced:     [B, D/G]
sub-parent:  [B, D/G, G/2]
```

`scale` is constant across the `G/2` pair positions within one group. It should
therefore remain at reduced resolution until it is combined with `pairs`.

## Captured Inductor IR

The following is the relevant graph fragment from a captured persistent NVFP4
compile. Buffer and temporary names are compiler-generated, but the shapes and
dependencies are the important part. The complete captured wrapper is in
[`nvfp4_persistent_wrapper.py`](./nvfp4_persistent_wrapper.py).

```text
%arg0_1 : bf16[128, 4096]                         # input
%arg1_1 : bf16[4096]                              # RMS weight

# Parent reduction and parent-full normalization
%convert_element_type : f32[128, 4096] = to_f32(%arg0_1)
%pow_1                : f32[128, 4096] = pow(%convert_element_type, 2)
%mean                 : f32[128, 1]    = mean(%pow_1, dim=1)
%add                  : f32[128, 1]    = add(%mean, epsilon)
%rsqrt                : f32[128, 1]    = rsqrt(%add)
%mul                  : f32[128, 4096] = mul(%convert_element_type, %rsqrt)
%mul_1                : f32[128, 4096] = mul(%mul, %arg1_1)
%normalized           : bf16[128, 4096] = to_bf16(%mul_1)

# Block-local reduction
%groups : bf16[128, 256, 16] = reshape(%normalized)
%abs_1  : bf16[128, 256, 16] = abs(%groups)
%amax   : bf16[128, 256]     = amax(%abs_1, dim=2)

# Reduced-domain scale chain
%div_scale : bf16[128, 256]     = div(%amax, scale_denominator)
%scale_f32 : f32[128, 256]      = to_f32(%div_scale)
%clamped   : f32[128, 256]      = clamp_min(%scale_f32, minimum)
%scale     : f8e4m3[128, 256]   = to_fp8(%clamped)
%scale_back: f32[128, 256]      = to_f32(%scale)
%scale_view: f32[128, 256, 1]   = unsqueeze(%scale_back)

# Sub-parent packing domain
%pairs : bf16[128, 256, 8, 2] = reshape(%groups)
%even  : bf16[128, 256, 8]    = select(%pairs, lane=0)
%odd   : bf16[128, 256, 8]    = select(%pairs, lane=1)
%q0    : f32[128, 256, 8]     = div(to_f32(%even), %scale_view)
%q1    : f32[128, 256, 8]     = div(to_f32(%odd), %scale_view)
%packed: i32[128, 256, 8]     = inline_asm_elementwise(%q0, %q1)
%bytes : u8[128, 256, 8]      = to_uint8(%packed)

return %scale, %bytes
```

At scheduler level, this becomes one typed fused node with several emission
domains:

```text
FusedNestedReductions
|
|-- parent schedule, group=(128, 4096)
|   |-- convert -> square
|   |-- mean reduction over D
|   `-- rsqrt/multiply -> normalized parent tile
|
|-- nested block-local stage, logical shape=[128, 256, 16]
|   `-- abs -> amax over G=16
|
|-- reduced pointwise stage, shape=[128, 256]
|   `-- divide -> clamp -> FP8 scale
|
`-- sub-parent stage, logical shape=[128, 256, 8]
    `-- even/odd projection -> scale -> inline-assembly pack
```

The original FX/Inductor graph has only one `%scale` node, but scheduler fusion
places its users in different iteration domains. The LoopBody for the packing
consumer may therefore contain an inlined copy of the scale expression rather
than a load of a separately materialized `%scale` buffer. This is the specific
gap that `_GroupInvariantBroadcast` handles.

The stage plan represents the same partition approximately as:

```text
StagedReductionPlan(
    parent_nodes=(parent reduction and parent-domain pointwise nodes),
    nested_stage=NestedReductionStage(
        grouped_reduction=amax,
        pointwise_domains=(scale nodes at REDUCED, ...),
    ),
    sub_parent_stages=(
        SubParentEpilogueStage(
            factor=2,
            source_layouts=((normalized, INTERLEAVED),),
            broadcast_source_names=(amax, scale-related named buffers),
            epilogue_nodes=(even/odd scaling and packing nodes),
        ),
    ),
)
```

This is descriptive notation rather than the Python `repr`: it shows which
pieces of the original Inductor graph each plan field owns.

## What Existing Handlers Do

### `_GroupedReductionOpsHandler`

This handler changes the meaning of a reduction operation. It reshapes a
parent tile into groups and reduces the local `G` axis:

```text
[B, D] -> [B, D/G, G] -> reduce G -> [B, D/G]
```

It does not control arbitrary pointwise operations after that reduction.

### `_PointwiseRemapHandler`

This handler runs a pointwise body in a derived iteration space. It remaps the
body's indices and asks an optional load resolver whether an earlier value can
be forwarded.

It historically treated every ordinary operation identically. Once a load had
been projected to lane width, all following arithmetic also ran at lane width.

### `_SubParentSourceLoadResolver`

This handler forwards values identified by buffer name. For a parent-width
value it may split the resident tile into interleaved lanes. For a name listed
in `broadcast_source_names`, it can instead forward the group-width value.

Its limitation is that only loads and stores have buffer names. Intermediate
expressions such as `clamp(amax / 6)` or the FP8 conversion do not.

### `_ParentFullLoadTransform`

This handles the opposite transition: it expands a reduced or singleton value
to the full parent tile for the grouped-reduction stage. It does not decide how
long a value should remain group-invariant inside the sub-parent stage.

## The Missing Behavior

Suppose `amax` is the only named value shared by the reduced and sub-parent
consumers. The scale expression is inlined into both consumer bodies.

Without delayed projection, the sub-parent body effectively does:

```text
amax_lane  = broadcast(amax_g)          # [B, D/G, G/2]
scale_lane = fp8(clamp(amax_lane / 6))  # repeated at lane width
packed     = pack(x_even / scale_lane, x_odd / scale_lane)
```

The reduced stage already emitted:

```text
scale_g = fp8(clamp(amax_g / 6))        # [B, D/G]
```

Ordinary kernel CSE cannot merge these expressions because one operates on a
group-width value and the other on a lane-width broadcast.

`_GroupInvariantBroadcast` changes when projection occurs:

```text
scale_g    = fp8(clamp(amax_g / 6))     # same shape; ordinary CSE reuses it
scale_lane = broadcast(scale_g)         # project only at lane interaction
packed     = pack(x_even / scale_lane, x_odd / scale_lane)
```

It does not introduce a second CSE system. It preserves the shapes required
for the existing kernel CSE to recognize the repeated expression.

## How It Works

The sub-parent body executes through `_PointwiseRemapHandler`. Its `_default`
method passes each operation through `_GroupInvariantBroadcast.apply`.

For each operation:

1. If none of its operands are group-invariant, nothing changes.
2. If all tensor operands remain scalar or group-invariant and the operation is
   ordinary scalar math, it executes at group resolution.
3. If a lane-varying operand is present, group-invariant operands are widened
   before the operation.
4. Operations with positional, stateful, reduction, or subgraph semantics are
   projection barriers and receive widened operands.
5. A store widens a still-group-invariant result because the output belongs to
   the sub-parent iteration domain.

For the example, division by `6`, clamp, FP8 conversion, conversion back to
FP32, and reciprocal stay at `[B, D/G]`. The division combining the reciprocal
with `x_even` or `x_odd` sees a lane-varying operand and triggers the broadcast.

## Why the Parent Schedule Must Use the Resolver

The resolver must observe the parent and grouped stages while they emit. That
is how it records the CSE-live value associated with each planned source name.
The nested emitter therefore owns the kernel context and runs the ordinary
schedule body underneath the resolver:

```python
with kernel:
    resolver = _SubParentSourceLoadResolver(...)
    with V.set_ops_handler(resolver):
        self._codegen_node_schedule_body(parent_schedule, kernel)

    kernel.codegen_body()
    # Emit grouped and sub-parent stages using the recorded values.
```

`codegen_node_schedule_with_kernel()` only wraps the body in `with kernel:`.
Calling it without installing the resolver would emit correct parent code but
would not capture the values needed by the later derived stage.

## Persistent And Looped Kernels

The projection policy is the same in both modes.

- Persistent kernels may retain parent loads in CSE and split them in
  registers.
- Looped kernels invalidate loop-local input loads, so parent-resolution data
  may be loaded again at derived indices.
- Reduced values produced after the parent reduction loop can remain available
  to the grouped and sub-parent stages in either mode.

`_GroupInvariantBroadcast` acts on the value that the resolver successfully
provides. It does not decide whether a parent load remains live.

## Correctness And Performance

Under the current Triton version, eagerly projecting the scale chain is
semantically valid. The purpose of delayed projection is to avoid repeating
the FP8 conversion and subsequent scalar work for every lane.

The important structural test requires exactly one FP8 conversion in the
generated kernel:

- [`test_nvfp4_inline_asm_kernel_form`](./pr190594_ci_fix/test/inductor/test_nested_reduction.py#L2885)
- [`test_nvfp4_swizzled_scale_kernel_form`](./pr190594_ci_fix/test/inductor/test_nested_reduction.py#L2908)

The numerical tests remain semantic guardrails:

- [`test_producer_consumer_rmsnorm_nvfp4_swizzled_scale`](./pr190594_ci_fix/test/inductor/test_nested_reduction.py#L1036)

If group-width CSE stops matching, correctness should remain intact, but the
kernel-form tests should fail because the generated kernel contains duplicate
FP8 conversion work.

## Why This Is Still Temporary

The planner currently divides sources into categories by buffer name:

```text
broadcast_source_names  -> forward unchanged at group resolution
source_layouts          -> split/project a parent-resolution value
everything else         -> ordinary derived-index load
```

That works under the planner's current uniqueness restrictions, but buffer
name is not a complete load identity. The longer-term mechanism should key
forwarding by the buffer version, normalized index and domain, mask, and fill
value. Such an indexed projection cache could represent the projection itself
and make much of the explicit source-name classification unnecessary.

Until then, `_GroupInvariantBroadcast` is the narrow codegen adapter that lets
ordinary CSE work across this particular domain boundary without introducing
origin tracking, forced realization, or a second expression graph.

## Review Questions

1. Which values are parent-width, reduced-width, and sub-parent-width?
2. Which named load does `_SubParentSourceLoadResolver` forward?
3. Why does broadcasting before the FP8 conversion prevent the CSE hit?
4. Which operation first combines the group-invariant scale with lane-varying
   data?
5. Why must stores and packed inline assembly be projection barriers?
6. What would full indexed, projection-aware forwarding replace here?
