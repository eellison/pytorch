# MXFP6 AITER field-order fusion diagnosis

Date: 2026-08-26

## Result

The three-kernel AITER-field-order graph is not missing the `(4, 3)` packing
plan. The complete graph has a valid structural plan. It fails for two later,
independent reasons:

1. The normal fusion rounds first form separate full-resolution encode and
   reduced-resolution pack groups. The final loop-reordering round then fuses
   the reduction with the encoder, but fusion does not run to a fixpoint after
   that round.
2. A diagnostic retry reaches the staged candidate, but the exact dependency
   proof rejects the pack read at the raw X/R boundary check. The producer has
   ranges `(rows * groups, 32)`, while the flattened consumer has ranges
   `(rows, groups * 8)`. The latter is codegen-compatible by splitting its
   second axis into `(groups, 8)`, but the current proof allows the X/R boundary
   only between raw axes.

The normalized access relation itself is correct. In the child frame,
`x = 4 * row + group` and `child_r = packed_quartet`; the read becomes
`32 * x + 4 * child_r + lane`, exactly the parent write with
`parent_r = 4 * child_r + lane`.

## Smallest safe solution

Preserve the existing logical group axis after applying AITER's field order:

```python
fields = torch.stack((scaled[..., :16], scaled[..., 16:]), dim=-1)
fields = fields.reshape(rows, width // 32, 32)
```

This is a metadata-only source rewrite. It does not move the permutation before
the reduction and does not change the output bytes. It gives the planner the
same group boundary used by codegen, so the reviewed `(4, 3)` path emits one
nested kernel without a new scheduler or projection rule.

On B200, CUDA-graph replay medians were:

| Shape | Flattened graph | Group-preserving graph | AITER |
|---|---:|---:|---:|
| 8192x128 | 8.561 us / 3 kernels | 3.759 us / 1 kernel | 4.709 us |
| 32768x128 | 20.837 us / 3 kernels | 10.843 us / 1 kernel | 12.891 us |
| 1024x1024 | 8.600 us / 3 kernels | 3.748 us / 1 kernel | 4.363 us |
| 128x12288 | 10.608 us / 3 kernels | 5.160 us / 1 kernel | 5.899 us |

All outputs were bitwise equal to the AITER reference.

Moving field ordering before the reduction also makes one kernel, but is slower
because the reduction loads in the permuted order: 4.957, 16.496, 4.956, and
6.902 us for the same shapes.

## Compiler alternative

The exact flattened graph can fit the current `(4, 3)` model, but supporting it
in the compiler requires both:

1. reconsidering fusion after the reorder round creates the reduction/encoder
   group; and
2. extending the raw-frame proof to validate an X/R boundary that falls inside
   one contiguous raw axis using the same range mapping as codegen.

The second part must not be implemented as generic dependency normalization or
by deleting the boundary guard. It must prove the actual codegen mapping and
retain the existing shifted/transposed fail-closed cases. This is materially
more scheduler surface than retaining the group dimension in the source graph.

Changing only the final byte store layout cannot implement AITER's order. Each
triplet packs codes `(2j, 16+2j, 2j+1, 17+2j)`, so this is a pre-pack field
permutation, not a permutation of already-packed bytes.

## Relevance of PR #193599

PR #193599 solves an output-side scale-layout problem. Its `PaddedScatter`
iterates over logical scale values and uses an arbitrary output indexer for the
primary stores; `AuxiliaryWriteRegion` fills dynamic padding holes with a
constant in a second loop inside the same Triton kernel.

That mechanism is useful for a padded or symbolic MXFP6 scale swizzle. It does
not solve the full-resolution field-order relation consumed by `(4, 3)` packing:
it does not remap forwarded register values, and auxiliary writes cannot compute
the packed data.

For the current aligned MXFP6 scale layout, existing `ir.Scatter` is already
enough and ordinary `reshape`/`permute` codegen already emits the same direct
scattered scale store. A local MXFP6 `ir.Scatter` prototype stayed at one kernel
and was bitwise identical to the view formulation:

| Mode | View/permute | Scatter |
|---|---:|---:|
| Persistent | 16.280 us | 16.485 us |
| Looped | 51.301 us | 51.346 us |

The scatter prototype was not a drop-in simplification: its mutation layout
also required narrowing the current staged planner's blanket
`has_aliasing_or_mutation()` rejection so a parent-stage scatter could be
admitted. The view/permute formulation needs no such planner relaxation.

Therefore #193599 is a useful model for future padded/dynamic scale layouts,
but it is neither necessary for the current scale-only preshuffle nor a fix for
the AITER packed-data field-order failure.

## Scratch artifacts

- `agent_space/diagnose_aiter_field_order.py`
- `agent_space/bench_aiter_field_order_variants.py`
- `agent_space/compare_mxfp6_scale_scatter.py`
- `agent_space/pr193599_plus_191775_wt`
