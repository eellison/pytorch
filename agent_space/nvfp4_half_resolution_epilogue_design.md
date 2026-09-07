# NVFP4 Half-Resolution Epilogue Fusion Design

This change targets the standalone dim0 NVFP4 cast pattern:

```python
xg = x.view(B, D // 16, 16)
amax = xg.float().abs().amax(dim=-1)
scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(torch.float8_e4m3fn)
even = xg.view(B, D // 16, 8, 2)[..., 0].float() / scale.float().unsqueeze(-1)
odd = xg.view(B, D // 16, 8, 2)[..., 1].float() / scale.float().unsqueeze(-1)
packed = nvfp4_inline_asm(even, odd)
```

Before this change, standalone NVFP4 lowered as two kernels: one reduction for `amax/scale`, then a pointwise pack kernel that reloaded `x`. That is bad for this shape because the pack kernel should reuse the same input tile already loaded by the reduction. The intended codegen is a single persistent reduction kernel that keeps the full `[XBLOCK, 16]` tile live, computes scale, then emits the half-resolution pack epilogue over `[XBLOCK, 8]`.

The legality check is intentionally narrow. It only accepts `rnumel == 16`, persistent reductions, no aliasing/mutation, and consumers whose shape is exactly half of the full reduction tile. External half-epilogue reads must be provably constant-lane reads of the same reduction input: lane 0 or lane 1 of the final reduction dimension. If the epilogue also reads the same input buffer in any unrelated way, such as a scalar `x[0, 0]`, the path rejects fusion and falls back. The whole path is also gated by `triton.nested_reduction`, so compiled unnested baselines do not accidentally exercise the same special codegen.

Codegen uses the existing nested pointwise remap machinery with a new lazy half-resolution split. When a full tile value is materialized at half resolution, we create lane placeholders but do not immediately emit `tl.split`. The pointwise remap handler materializes the split only when an operation really needs lane values. This lets division and casts optimize at parent-full resolution first:

```python
tmp = full_tile / scale
even, odd = tl.split(tl.reshape(tmp, [XBLOCK, 8, 2]))
packed = inline_asm(even, odd)
```

That avoids the slower form:

```python
even, odd = tl.split(tl.reshape(full_tile, [XBLOCK, 8, 2]))
even = even / scale
odd = odd / scale
packed = inline_asm(even, odd)
```

This path is separate from the older RMSNorm producer case. RMSNorm plus block amax still uses the existing nested producer/reduction machinery. Standalone `amax + NVFP4 pack` uses this new persistent half-resolution reduction epilogue. A true non-reduction pack, where scale is already available and there is no `amax`, should remain ordinary pointwise fusion/codegen; it should follow the same divide-before-split principle, but it should not use reduction scheduling or persistent reduction assumptions.

Current validation covers numeric correctness against an unnested compiled baseline for standalone NVFP4 and a non-inline-asm half-resolution epilogue, plus kernel-form checks for both the SM100 NVFP4 single-kernel shape and the generic lazy cast/split path. Focused benchmarks with CUDAGraph replay at `M=16384, K=16384, G=16` show one Inductor kernel, about 153 us by default and about 138 us with coordinate descent tuning, versus about 134 us for the hand Triton kernel.
