# Nested Reduction Consumer Resolution Plan

This is a follow-up architecture note, not a pre-land cleanup. The current
branch supports:

- reduced-output consumers, shaped like the grouped reduction result
- full-resolution consumers, shaped like the original parent tile

NVFP4 packing needs a third thing:

- half-resolution consumers, shaped like the original parent tile divided by 2
  along the packed axis

The important correction is that half-resolution is not "broadcast back". It is
not a full-resolution epilogue with a smaller final store. It has its own
iteration space and pair-lane semantics.


## Current Branch

Today the code has two effective consumer spaces.

For small-dim-in-r with `x.shape == [B, D]` and `group_size == G`:

```text
parent tile:         [X, R]       == [B tile, D tile]
grouped view:        [X, R/G, G]
reduced output:      [X, R/G]
full resolution:     [X, R]
```

Reduced-output consumers run in a derived family:

```text
parent R numel       = D
derived R numel      = D / G
parent R block       = R0_BLOCK
derived R block      = R0_BLOCK / G
parent R offset      = r0_offset
derived R offset     = r0_offset / G
```

Full-resolution consumers reuse the parent family and materialize reduced
values at the parent extent:

```text
scale: [X, R/G]
reshape to [X, R/G, 1]
broadcast to [X, R/G, G]
reshape to [X, R]
```

That manual broadcast is currently necessary because the reduced value is a
register CSE value, not an IR buffer. There is no real buffer load that an
IR-level `expand` can attach to.


## Why Shape Alone Is Not Enough

`CSEVariable` already carries a shape, and we should use that information. But
shape by itself does not encode enough semantic information to remove the
explicit materialization logic.

The missing facts are:

- which logical axis is the grouped axis
- whether a value is reduced-per-group, full-resolution, or pair-resolution
- how a reduced value maps back to the parent tile
- for half-resolution, how the output index maps to two adjacent full-resolution
  lanes
- whether the value is a register-only value or a real memory buffer

For full resolution, `[X, R/G]` to `[X, R]` is not just "make shapes match".
It is specifically "insert a singleton lane axis after groups, broadcast by
`G`, then flatten groups and lanes back to `R`".

For half resolution, the problem is even less like ordinary broadcasting. The
consumer wants one output per pair of lanes, not one output per original lane.


## Resolution Vocabulary

Make consumer resolution first-class. The scheduler should classify each fused
consumer once, and codegen should consume that classification rather than
re-deriving it from body shapes.

Suggested initial vocabulary:

```python
class NestedConsumerResolution(enum.Enum):
    REDUCED_OUTPUT = "reduced_output"
    FULL = "full"
    HALF = "half"
```

For the current supported small-dim-in-r path:

```text
REDUCED_OUTPUT  [X, R/G]
FULL            [X, R]
HALF            [X, R/2]
```

For small-dim-in-x, the analogous shapes are:

```text
REDUCED_OUTPUT  [X/G, R]
FULL            [X, R]       # not currently supported by codegen
HALF            [X/2, R]     # possible future packing shape, not needed first
```

The first half-resolution implementation should stay limited to small-dim-in-r,
because that is the NVFP4 shape we actually need.


## Concrete NVFP4 Example

Use the pass-3 pattern from `agent_space/nested_reduction_fuzz.py`:

```python
x = F.rms_norm(x, (D,), weight)
x = x.view(B, D // G, G)
amax = x.abs().amax(dim=-1)
scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
xg = x.view(B, D // G, G // 2, 2)
scale_f = scale.float().unsqueeze(-1)
even = xg[..., 0].float() / scale_f
odd = xg[..., 1].float() / scale_f
packed = inline_asm_elementwise(even, odd, ...)
return packed.to(torch.uint8).view(B, D // 2), scale
```

Take `B=4`, `D=512`, `G=128`.

```text
parent x:             [4, 512]
grouped x:            [4, 4, 128]
amax/scale:           [4, 4]
pair grouped x:       [4, 4, 64, 2]
even lane:            [4, 4, 64]
odd lane:             [4, 4, 64]
packed output:        [4, 256]
```

This has three live resolutions at once:

```text
scale store:          [4, 4]       reduced output
normalized x source:  [4, 512]     full parent tile
packed output:        [4, 256]     half resolution
```

The half-resolution output index is not the group index:

```text
half_r = 0..255
group  = half_r // 64
pair   = half_r % 64
lane0  = group * 128 + pair * 2
lane1  = lane0 + 1
```

The reduced scale can be materialized for the half-resolution body as:

```text
scale: [X, groups]
reshape to [X, groups, 1]
broadcast to [X, groups, G/2]
reshape to [X, R/2]
```

That is not broadcasting back to the parent tile. It is materializing the
per-group scale at pair resolution so the pair pack body can read it.


## Half-Resolution Derived Root

For small-dim-in-r, half-resolution needs a derived R root:

```text
parent R numel       = D
half R numel         = D / 2
parent R block       = R0_BLOCK
half R block         = R0_BLOCK / 2
parent R offset      = r0_offset
half R offset        = r0_offset / 2
```

With `D=512` and `R0_BLOCK=512`:

```python
r0_index = 0 + tl.arange(0, 512)          # full lanes 0..511
half_r0_index = 0 + tl.arange(0, 256)     # pairs 0..255
```

With looped `R0_BLOCK=256`:

```text
iteration 0:
  parent r0_index      = 0..255
  half_r0_index        = 0..127

iteration 1:
  parent r0_index      = 256..511
  half_r0_index        = 128..255
```

The half family shares the parent loop and launch placement, just like the
reduced-output family. It is another lens on the same physical reduction loop.


## Value Materialization

The simplification should be to centralize value materialization by target
resolution. Instead of a full-resolution-only broadcast helper, the layout or
plan should expose:

```python
materialize_value(name, value, target_resolution)
```

Conceptually:

```text
value at REDUCED_OUTPUT, target REDUCED_OUTPUT:
  return value

value at REDUCED_OUTPUT, target FULL:
  [X, groups] -> [X, groups, 1] -> [X, groups, G] -> [X, R]

value at REDUCED_OUTPUT, target HALF:
  [X, groups] -> [X, groups, 1] -> [X, groups, G/2] -> [X, R/2]

value at FULL, target HALF:
  reshape [X, R] -> [X, R/2, 2]
  split lane 0 and lane 1 for pair consumers
```

This is where the existing shape metadata helps: it can tell whether a CSE is
already at the target tile shape. But the materializer still needs layout
metadata to know how to reshape, broadcast, split, and flatten.


## Plan Object

The follow-up cleanup should introduce a `NestedReductionPlan` carried by
`FusedNestedReductions`.

Sketch:

```python
@dataclasses.dataclass(frozen=True)
class NestedReductionConsumer:
    node: BaseSchedulerNode
    resolution: NestedConsumerResolution

@dataclasses.dataclass(frozen=True)
class NestedReductionPlan:
    group_size: sympy.Integer
    small_dim_in_r: bool
    reduced_output_consumers: tuple[NestedReductionConsumer, ...]
    full_consumers: tuple[NestedReductionConsumer, ...]
    half_consumers: tuple[NestedReductionConsumer, ...]
```

The exact structure can be flatter, but the ownership is the key:

- scheduler owns pattern legality
- scheduler owns consumer resolution classification
- codegen owns concrete range trees, reshapes, broadcasts, splits, and stores

Codegen should not rediscover whether an epilogue is reduced/full/half by
recomputing numels. It should read the plan.


## Scheduler Classification

Today `FusedNestedReductions.can_fuse_with()` accepts downstream pointwise nodes
when their `other_numel` equals either:

```text
node2 output numel       -> reduced output
node2 numel * rnumel2    -> full resolution
```

The half-resolution follow-up needs a third classified case:

```text
(node2 numel * rnumel2) / 2 -> half resolution
```

Initial legality should be narrow:

- only `small_dim_in_r`
- `group_size % 2 == 0`
- parent grouped axis divisible by 2
- half-resolution consumer depends on the grouped reduction result and the
  full-resolution parent value
- lane access is statically pair-shaped, e.g. equivalent to
  `[X, groups, G/2, 2]` followed by lane 0 and lane 1 reads

The scheduler does not need to know how to emit the split. It only needs to
record that this consumer is a legal half-resolution consumer.


## Codegen Shape

Codegen can then become a loop over planned consumer stages:

```text
1. emit outer reduction
2. emit grouped reduction into reduced-output family
3. emit REDUCED_OUTPUT consumers in reduced-output family
4. emit FULL consumers in full family
5. emit HALF consumers in half family
```

`_PointwiseRemapHandler` should stay generic. The family should supply:

- range trees
- index remapping
- flat index expression
- remapped/materialized values

The special logic belongs in family construction and value materialization, not
inside each consumer body.


## Bigger Landable Simplification

The larger simplification that still helps NVFP4 is to collapse the current
reduced-output and full-resolution epilogue emitters into one generic
pointwise-stage emitter.

For the current PR, this does not need to introduce the whole stage object or
plan. The lower-risk version is just a shared emitter helper:

```python
def _codegen_pointwise_epilogues(
    self,
    kernel,
    epilogue_nodes,
    family,
    flat_index,
    *,
    output_name_from_node=False,
) -> None:
    with family.activate(kernel):
        for ep_sn in epilogue_nodes:
            ep_iter = _decompose_flat_index(ep_sn._body, flat_index)
            output_name = (
                next(d.name for d in ep_sn.read_writes.writes)
                if output_name_from_node
                else None
            )
            handler = _PointwiseRemapHandler(
                V.get_ops_handler(),
                kernel=kernel,
                family=family,
                output_name=output_name,
            )
            with V.set_ops_handler(handler), kernel.set_current_node(ep_sn):
                ep_sn._body(ep_iter)
```

Then current code becomes:

```text
reduced:
  flat_index = flatten(node2 body iter vars through iter_remapped)
  _codegen_pointwise_epilogues(..., reduced_output_family, flat_index,
                              output_name_from_node=True)

full:
  fullres_family = layout.make_full_resolution_family(...)
  _codegen_pointwise_epilogues(..., fullres_family,
                              fullres_family.flat_index())
```

That version is compatible with NVFP4 because half-resolution would be another
call to the same helper with a half-resolution family and half flat index.

Today these two functions are mostly the same operation:

```text
_codegen_reduced_resolution_epilogue:
  build flat index
  activate reduced family
  for each node:
    decompose flat index
    run body with _PointwiseRemapHandler

_codegen_fullres_epilogue:
  build full family
  activate full family
  for each node:
    decompose flat index
    run body with _PointwiseRemapHandler
```

The differences are stage construction details, not emit details. So introduce a
small codegen-local stage object:

```python
@dataclasses.dataclass
class _NestedPointwiseStage:
    resolution: NestedConsumerResolution
    nodes: list[scheduler.SchedulerNode]
    family: _DerivedIterationFamily
    flat_index: sympy.Expr
```

Then codegen has one emitter:

```python
def _codegen_pointwise_stage(kernel, stage):
    with stage.family.activate(kernel):
        for ep_sn in stage.nodes:
            ep_iter = _decompose_flat_index(ep_sn._body, stage.flat_index)
            handler = _PointwiseRemapHandler(
                V.get_ops_handler(),
                kernel=kernel,
                family=stage.family,
            )
            with V.set_ops_handler(handler), kernel.set_current_node(ep_sn):
                ep_sn._body(ep_iter)
```

Current behavior becomes:

```text
REDUCED_OUTPUT stage:
  family = reduced_output_family
  flat_index = flatten(node2 body iter vars through iter_remapped)

FULL stage:
  family = layout.make_full_resolution_family(...)
  flat_index = family.flat_index()
```

NVFP4 becomes:

```text
HALF stage:
  family = layout.make_half_resolution_family(...)
  flat_index = family.flat_index()
```

This is more substantial than the enum-only cleanup, but it is still a real
simplification rather than new architecture for its own sake. It removes the
current shape where every new consumer resolution wants another bespoke
`_codegen_*_epilogue()` function.

The risk is that the reduced-output path has a small output-name/store-cache
quirk today. That can either stay as a field on `_NestedPointwiseStage`, or be
removed separately if it turns out to be redundant. That detail is why this is a
follow-up refactor, not the smallest pre-land cleanup.


## Batch Size 1

B=1 must remain a supported case. The important distinction is:

```text
groups == 1        # one group in the grouped axis
group_size == 1    # one lane per group
```

The B=1 case usually means `groups == 1`, not `group_size == 1`.

For example, small-dim-in-x can have:

```text
B = 1
G = 128
numel1 = B * G = 128
groups = numel1 / G = 1
```

That path needs explicit plan/layout representation so future refactors do not
accidentally reject it as degenerate. For half-resolution small-dim-in-r, B=1 is
not special in the packed axis; it is just:

```text
parent: [1, D]
scale:  [1, D/G]
packed: [1, D/2]
```


## Migration Plan

1. Add `NestedConsumerResolution` without changing generated code.

   This is the smallest landable simplification that also points directly at
   NVFP4. It does not need the whole plan object yet. Start with:

   ```python
   class NestedConsumerResolution(enum.Enum):
       REDUCED_OUTPUT = "reduced_output"
       FULL = "full"
   ```

   Then replace the current open-coded reduced/full classification with a named
   helper:

   ```python
   def classify_consumer_resolution(
       other_numel,
       reduced_output_numel,
       full_numel,
       *,
       small_dim_in_r,
   ) -> NestedConsumerResolution | None:
       ...
   ```

   Today it returns only `REDUCED_OUTPUT`, `FULL`, or `None`. The NVFP4 patch
   adds `HALF` and the narrow half-resolution legality checks in one obvious
   place.

   This is useful even before a full `NestedReductionPlan` because it makes the
   current code stop speaking in the binary language of "reduced else fullres".
   It also gives both scheduler fusion and codegen epilogue partitioning the
   same vocabulary, even if codegen still computes the partition locally for one
   patch.

2. Add `NestedReductionPlan` without changing generated code.

   The first patch can just move today's reduced/full classification out of
   `_codegen_group_reduction_epilogue()` and into `FusedNestedReductions`.

3. Rename the current full-resolution broadcast path into a generic value
   materialization API.

   Keep only reduced/full wired initially:

   ```text
   reduced value -> reduced output
   reduced value -> full resolution
   ```

4. Add half-resolution family construction for small-dim-in-r.

   This is a sibling of `make_reduced_output_family()`, not a variant of
   `make_full_resolution_family()`.

5. Add half-resolution value materialization.

   Required pieces:

   ```text
   reduced value -> half resolution
   full value -> split pair lanes for half resolution
   ```

6. Enable NVFP4 pass-3 fusion behind the narrow legality checks.

   The output-code test should assert:

   ```text
   scale store shape is reduced: [X, groups]
   packed store shape is half:   [X, R/2]
   pair split is present
   no global reload of normalized x between the grouped reduction and pack
   ```

7. After that lands, collapse the reduced/full/half epilogue emitters into one
   pointwise-family emitter if the code naturally falls out that way.


## What This Simplifies

This removes the binary mental model of:

```text
reduced output vs broadcast back
```

and replaces it with:

```text
consumer runs at an explicit resolution
values are materialized into that resolution
```

That matches the actual feature set:

- FP8 scale epilogue: reduced output
- FP8 quantize: full resolution
- NVFP4 pack: half resolution

It also gives us a better place to put the manual reshape/broadcast/split
logic. The logic does not disappear, but it becomes an implementation detail of
materializing register values into a named iteration family.


## Open Questions

- Should `HALF` be hard-coded as factor 2, or should the plan store
  `resolution_factor=2` so future packed formats can reuse the same path?
- Should half-resolution consumers be discovered only after a successful
  reduced/full nested fusion, or should scheduler build the whole plan in one
  pass?
- How much lane-pattern validation belongs in scheduler versus codegen?
- Do we need a named value-source object now, or can `_DerivedIterationFamily`
  keep `remapped_values` until half-resolution proves the abstraction boundary?
