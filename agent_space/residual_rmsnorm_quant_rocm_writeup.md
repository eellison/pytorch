# Residual RMSNorm Quant ROCm Failure

## Summary

The ROCm `CantSplit` failure is real, but the latest investigation found that
our FP8-to-FP16 test rewrite was not equivalent to the original test.

The original residual test kept the producer side in low precision:

```python
normed_bf16 = normed.to(torch.bfloat16) * weight
grouped = normed_bf16.view(B, D // G, G)
```

Because `weight` was also `bfloat16`, the multiply result stayed `bfloat16`.

The first FP16 rewrite did this:

```python
normed_fp16 = normed.to(torch.float16) * weight
grouped = normed_fp16.view(B, D // G, G)
```

Here `weight` is still `bfloat16`. PyTorch promotes `float16 * bfloat16` to
`float32`, so the grouped producer became `float32`. That is a real semantic
change beyond removing direct FP8 materialization.

The corrected rewrite keeps the original BF16 producer and only replaces the
final FP8 materialization boundary:

```python
normed_bf16 = normed.to(torch.bfloat16) * weight
grouped = normed_bf16.view(B, D // G, G)
...
x_quant = x_scaled.to(torch.float16).view(B, D)
```

That preserves the old producer semantics and avoids direct FP8 only at the
quant/materialization boundary that was numerically unstable.

## Evidence

I compared the traced ATen graph with `make_fx` and then reran the focused
Inductor tests locally. Triton is available, but this environment needs:

```bash
LD_PRELOAD=/home/eellison/.conda/envs/pytorch-3.12/lib/libstdc++.so.6
TORCHINDUCTOR_COMPILE_THREADS=1
```

The preload avoids loading an older system `libstdc++` before Triton's
`libtriton.so`. The compile-thread setting avoids a local compile-worker driver
discovery issue.

Original FP8 structure:

```text
view    shape=[128, 16, 128] dtype=torch.bfloat16
amax    shape=[128, 16, 1]   dtype=torch.bfloat16
to      shape=[128, 16, 1]   dtype=torch.float32
...
to      shape=[128, 16, 128] dtype=torch.float8_e4m3fn
view    shape=[128, 2048]    dtype=torch.float8_e4m3fn
to      shape=[128, 2048]    dtype=torch.float32
```

Bad FP16 rewrite:

```text
to      shape=[128, 2048]    dtype=torch.float16
mul     shape=[128, 2048]    dtype=torch.float32
view    shape=[128, 16, 128] dtype=torch.float32
amax    shape=[128, 16, 1]   dtype=torch.float32
...
to      shape=[128, 16, 128] dtype=torch.float16
```

Corrected rewrite:

```text
to      shape=[128, 2048]    dtype=torch.bfloat16
mul     shape=[128, 2048]    dtype=torch.bfloat16
view    shape=[128, 16, 128] dtype=torch.bfloat16
amax    shape=[128, 16, 1]   dtype=torch.bfloat16
to      shape=[128, 16, 1]   dtype=torch.float32
...
to      shape=[128, 16, 128] dtype=torch.float16
```

This confirms the first rewrite changed the graph before Inductor scheduling.

## Relationship To The ROCm Failure

The ROCm traceback was:

```text
torch._inductor.exc.InductorError: CantSplit: 16 not divisible by 2048
...
SIMDKernel._split_iteration_ranges
```

The suspicious shape is still:

```text
B, D, G = 128, 2048, 128
[B, D/G, G] = [128, 16, 128]
[B * D] = [262144]
```

The failing path is a full-resolution consumer remap over a nested grouped
producer domain. However, since the bad rewrite accidentally made the producer
`float32`, we should not use that failure alone to justify broadening
`_split_iteration_ranges` beyond its existing bmm-specific three-way split.

## Current Recommendation

Fix the test rewrite so it preserves the original BF16 producer and changes
only the final FP8 quant boundary:

```python
normed_bf16 = normed.to(torch.bfloat16) * weight
```

The test now parametrizes the broadcast layouts that can still represent the
same `[B, D]` logical producer:

```text
()
(1,)
(D,)
(1, 1)
(1, D)
(B, 1)
(B, D)
(1, 1, 1)
(1, 1, D)
(1, B, 1)
(1, B, D)
```

It also asserts the weighted producer result is `torch.bfloat16` and has the
same numel as `x`, which catches the accidental `float16 * bfloat16 -> float32`
promotion without adding a new cast to the weight path.

Then rerun the ROCm test family:

```bash
PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_nested_reduction.py -k residual_rmsnorm_quant
```

Keep the `is_bmm_then_pw` guard restored unless the corrected ROCm test still
fails. Nested full-resolution pointwise consumers should be selected in the
`PARENT_FULL` domain; globally allowing the local grouped domain to absorb a
flattened `[B * D]` consumer is broader than the corrected test justifies.

## Local Validation

Local validation performed:

```text
python -m py_compile torch/_inductor/codegen/simd.py test/inductor/test_nested_reduction.py
git diff --check
LD_PRELOAD=/home/eellison/.conda/envs/pytorch-3.12/lib/libstdc++.so.6 TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py -k residual_rmsnorm_quant
LD_PRELOAD=/home/eellison/.conda/envs/pytorch-3.12/lib/libstdc++.so.6 TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py -k producer_consumer_rmsnorm
LD_PRELOAD=/home/eellison/.conda/envs/pytorch-3.12/lib/libstdc++.so.6 TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py NestedReductionTest.test_combo_kernels_skip_nested_reductions NestedReductionTest.test_producer_consumer_rmsnorm_nvfp4_inline_asm_B_1 NestedReductionTest.test_producer_consumer_rmsnorm_nvfp4_inline_asm_B_128
```

Results:

```text
- residual_rmsnorm_quant: 22 tests passed
- producer_consumer_rmsnorm: 14 tests passed
- combo/NVFP4 focused top tests: 3 tests passed
```

This is CUDA/B200 validation, not ROCm validation. ROCm should still run the
expanded residual matrix before landing.
