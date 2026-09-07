# MLA / Nested Block Quant Follow-up Notes

## What falls into the current nested-reduction work

The current PR covers standalone SIMD kernels shaped like:

```python
x = norm(x)
x_groups = x.view(B, D // G, G)
amax = x_groups.abs().amax(dim=-1)
scale = f(amax)
out = quantize(x_groups, scale)
```

This includes RMSNorm/LayerNorm -> block amax -> FP8/NVFP4 quant. The new
grouped-reduction lowering (`[(XBLOCK * groups), group_size] -> [XBLOCK,
groups]`) is directly useful here.

`/home/eellison/local/pytorch/mla_fusion.py` is mostly a different family:
heterogeneous RoPE/cat/scatter fusion with mismatched pointwise iteration
spaces. That is related to loop remapping, but it is not the same as nested
block-local reductions.

## Measurement caveat

`torch._inductor.metrics.generated_kernel_count` does not count external ATen
matmul launches. A result like `generated_kernel_count == 1` for
`mm -> block quant` can mean:

1. one generated quant kernel plus an external matmul, not
2. one fused matmul+quant template kernel.

For matmul-epilogue work, inspect generated code or force Triton GEMM templates
with `max_autotune_gemm` / `max_autotune_gemm_backends="TRITON"`.

## Matmul epilogue block quant

This is the important follow-up, but it is not a one-line scheduler relaxation.
Template epilogue fusion currently assumes pointwise epilogues. The scheduler
explicitly rejects:

```python
node1.is_template() and node2.is_reduction()
```

I temporarily removed that rejection and compiled a simple:

```python
y = x @ w.t()
return y.view(M, N // G, G).abs().amax(dim=-1)
```

With Triton GEMM templates forced, codegen failed in:

```python
TritonKernel.reduction()
assert self.inside_reduction
```

So the lower-level template kernel is not set up to emit a reduction epilogue.
It runs epilogue nodes in the template output pointwise context, not inside a
separate reduction loop/range tree.

The real design likely needs a template-level "grouped reduction epilogue"
stage:

1. Keep the matmul accumulator tile in registers.
2. Apply any pointwise epilogue/prologue needed before quantization, e.g. RoPE.
3. Reshape the tile into the same compact grouped-reduction form we now use:
   `[(output_rows * groups), group_size]`.
4. Reduce over `group_size` to produce scale/amax.
5. Store the reduced output if requested.
6. Broadcast/lift the scale back to the matmul tile resolution.
7. Quantize and store the full-resolution packed/FP8 output.

That should reuse the grouped-reduction layout concepts from the current PR,
but the integration point is `TritonTemplateKernel` / template epilogue codegen,
not `codegen_nested_reduction`.

## Combo kernels

Combo kernels do not currently support `FusedNestedReductions` as subkernels.
A minimal repro with two independent RMSNorm -> FP8 quants and
`combo_kernels=True` previously failed with:

```text
NotImplementedError: unexpected group: (64, 4096) != (2048, 128)
```

Reason: combo codegen unwraps the fused node with `node.get_nodes()` and then
calls normal `generate_node_schedule()` on the raw reduction pair. That bypasses
the special `codegen_nested_reduction()` orchestration and hits the original
iteration-space mismatch.

I added a conservative guard to filter `FusedNestedReductions` out of combo
candidates, matching the existing treatment for `FusedMixOrderReductions`.
True combo support would require teaching `ComboKernel` how to create a nested
reduction subkernel directly.

This is secondary to matmul epilogue fusion. Combo packaging may reduce launch
overhead for independent kernels, but it does not keep the matmul output in
registers or avoid materializing it.

## Current status

- Standalone nested block quant: current PR.
- Matmul epilogue block quant: important follow-up, requires template reduction
  epilogue support.
- Combo + nested reductions: guarded off for correctness; real support is a
  separate lowering project.
- MLA RoPE/cat/scatter blockers: adjacent heterogeneous pointwise fusion work,
  outside this nested-reduction PR.

I also tried the existing scheduler knobs on the three isolated
`mla_fusion.py` blockers:

```python
expand_dimension_for_pointwise_nodes = True
loop_ordering_after_fusion = True
loop_reindexing_after_fusion = True
loop_index_inversion_in_fusion = True
score_fusion_memory_threshold = 1
min_overlap_ratio = 0.0
```

They did not reduce the generated-kernel counts for the Q/K RoPE, vertical
cat/scatter, or cat mismatch repros. Those cases need new heterogeneous
iteration-space fusion logic rather than just enabling an existing knob.
