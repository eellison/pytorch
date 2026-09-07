# Current-main padded scale kernel anatomy

Environment: `pytorch-3.12`, PyTorch `2.15.0a0+gitcdd22ad`, git
`cdd22ade2948699188c3e2d0d80b5a396a8489ae`, CUDA 13.0, B200.

This is a static generated-code and IR investigation. Timing runs started during
this investigation overlapped another B200 user and are non-authoritative; no
timing conclusions are included here.

## Explicit `F.pad` transform

The same topology appears for NVFP4, MXFP4, and MXFP8.

| Shape | Padding | Generated kernels | Generated topology |
|---|---:|---:|---|
| `128x4096` | none | 1 | nested RMSNorm/quant reduction writes the scale directly with the final 128x4 blocked index |
| `129x4096` | rows only | 3 | nested RMSNorm/quant reduction writes valid logical scales into a row-major padded backing buffer; zero-fill remaining rows; copy/swizzle the full padded buffer |
| `129x4128` | rows and scale columns | 2 | nested RMSNorm/quant reduction materializes logical scale data; one pointwise kernel fuses masked 2-D padding with the full swizzle |

For NVFP4 at `129x4128`, the first kernel materializes BF16 group maxima and
the second performs the E4M3 scale conversion, padding, and swizzle. MXFP4 and
MXFP8 materialize their final logical uint8 E8M0 scales in the first kernel;
their second kernel performs padding and swizzle.

The `129x4096` run increments `counters["inductor"]["pad_rewritten_as_cat"]`.
Its post-fusion IR contains a `ConcatKernel`: valid scale output and the zero
region are `NonOwningLayout` slices of one contiguous padded backing buffer.
The `129x4128` run does not increment that counter and has no `ConcatKernel`.

The reason is `torch/_inductor/lowering.py`:

- `constant_pad_nd()` first calls `_pad_as_cat()` (lines 5608-5623).
- `_pad_as_cat()` accepts only a single right-padded dimension and rejects the
  second nonzero pad at lines 5572-5580.
- Row-only padding therefore reaches `cat()`. The scale value has another
  pointwise consumer in quantization, so the multiple-consumer guard at lines
  2440-2478 selects `ConcatKernel` at line 2487.
- Row-plus-column padding falls through to the generic masked pointwise pad,
  which fuses with the reshape/permute/clone swizzle chain.

As a static confirmation, `config.force_pointwise_cat=True` changes row-only
`129x4096` from 3 kernels to 2: the nested reduction plus a fused masked
pad/swizzle kernel. This is the same topology as the two-dimensional-pad case.

## `flex_gemm.to_blocked` in an ordinary compiled function

Replacing the explicit transform with clean-main
`torch._higher_order_ops.flex_gemm.to_blocked` does not expose its body to
ordinary Inductor lowering. Generated `output_code.py` contains:

```python
buf = torch.ops.flex_gemm.to_blocked.default(logical_scale)
```

Metrics report one generated nested-reduction kernel plus one `extern_call`.
That is not a one-kernel implementation: the custom op executes its eager
zero/copy/permute implementation behind the external-call boundary. A dispatch
profile of NVFP4 observed the compiled reduction followed by fill, D2D copy,
and two elementwise copy operations.

The direct blocked-output support on current main is scoped to a FlexGemm
epilogue. `normalize_gemm_epilogue_fx_node()` recognizes the custom op as
`NormalizedToBlocked` in `torch/_inductor/kernel/gemm_epilogue.py:405`, and
`match_flex_gemm_local_reduce_output_storage()` maps it to
`BLOCKED_128X4` in
`torch/_inductor/kernel/flex_gemm/fx_cutedsl_codegen.py:110`. Ordinary RMSNorm
nested reductions never enter that FlexGemm-specific planner.

## Optimization boundary and smallest shared step

The quantization math is already fused. Padding breaks propagation of the
terminal blocked output layout back to the reduction producer:

- aligned view/permute/clone is represented as a final store index and fuses;
- row-only pad becomes a materialized `ConcatKernel` backing allocation;
- two-dimensional pad remains a pointwise producer with a larger iteration
  domain, so it cannot fuse into the logical reduction domain.

The smallest shared improvement is a targeted pad/cat heuristic that keeps the
row-only pad pointwise when its sole downstream chain is the terminal blocked
layout transform. The forced-pointwise IR proves this removes the zero-fill
kernel for NVFP4, MXFP4, and MXFP8, reducing 3 kernels to 2 without
format-specific logic.

The complete optimization is a generic terminal blocked-output layout for
ordinary producer kernels: allocate padded blocked storage, map valid logical
scale stores directly to final physical indices, and initialize invalid lanes
without a full-buffer copy. That would reuse one mechanism across all three
formats; reaching one kernel also requires representing the padded lanes in the
producer launch/store domain.

## Artifacts

Harness:
`agent_space/nested_quant_comparison/inspect_padded_kernel_anatomy_agent.py`

Generated traces are under:
`agent_space/nested_quant_comparison/traces_padded_anatomy_agent/`

Each case contains `ir_pre_fusion.txt`, `ir_post_fusion.txt`, and
`output_code.py`. The key case directories are:

- `nvfp4_128x4096`, `mxfp4_128x4096`, `mxfp8_128x4096`
- `nvfp4_129x4096`, `mxfp4_129x4096`, `mxfp8_129x4096`
- `nvfp4_129x4128`, `mxfp4_129x4128`, `mxfp8_129x4128`
- `nvfp4_129x4096_force_pointwise`
- `nvfp4_129x4096_custom`, `mxfp4_129x4096_custom`,
  `mxfp8_129x4096_custom`
