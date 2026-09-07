# Nested Reduction FP8 CI Failure

## Reported failure

Test:

```text
NestedReductionTest::test_fullres_epilogue_with_multiple_outputs
```

CI failure:

```text
AssertionError: Tensor-likes are not close
Mismatched elements: 1018 / 262144 (0.4%)
Greatest absolute difference: 32.0
Greatest relative difference: 0.111
```

The test compares compiled nested reduction against compiled unfused output with
`emulate_precision_casts=True`.

The test function is:

```python
x = F.rms_norm(x, (D,), weight)
x_groups = x.view(B, D // G, G)
amax = x_groups.abs().amax(dim=-1)
scale = (amax / fp8_max).clamp(min=1e-12)
x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
return x_fp8.view(B, D).float(), scale
```

## Generated schedules

### Unfused

The unfused compiled schedule uses three kernels.

Kernel 0 computes the RMS reduction:

```python
tmp3 = tl.sum(x * x, axis=1)
tl.store(buf0, tmp3)
```

Kernel 1 recomputes the RMSNorm value and computes grouped scale:

```python
tmp10 = rsqrt(tl.load(buf0) / 4096 + eps)
tmp13 = x * tmp10 * weight
tmp18 = max(abs(tmp13)) / fp8_max
tl.store(scale_buf, tmp18)
```

Kernel 2 recomputes the RMSNorm value, reloads scale, and casts to FP8:

```python
tmp10 = rsqrt(tl.load(buf0) / 4096 + eps)
tmp13 = x * tmp10 * weight
tmp11 = tmp13 / tl.load(scale_buf)
tmp12 = tmp11.to(tl.float8e4nv)
tmp13 = tmp12.to(tl.float32)
tl.store(out, tmp13)
```

### Nested

The nested schedule uses one kernel:

```python
tmp3 = tl.sum(x * x, axis=1)

tmp10 = rsqrt(tmp3 / 4096 + eps)
tmp13 = x * tmp10 * weight

tmp16 = max(abs(reshape(tmp13, [X, D / G, G])), axis=2)
tmp20 = max(tmp16 / fp8_max, 1e-12)

tmp21 = broadcast(tmp20, [X, D])
tmp22 = tmp13 / tmp21
tmp23 = tmp22.to(tl.float8e4nv)
tmp24 = tmp23.to(tl.float32)

tl.store(scale, tmp20)
tl.store(out, tmp24)
```

## What was verified

The outer RMS reduction launch config is the same in both schedules:

```text
XBLOCK=1, R0_BLOCK=1024, num_warps=8
```

So this is not caused by a different RMS reduction tiling or reduction order.

The FP8 truncation is present in the nested kernel. Generated nested code has:

```python
tmp23 = tmp22.to(tl.float8e4nv)
tmp24 = tmp23.to(tl.float32)
```

So this does not look like a missing explicit FP8 cast in
`emulate_precision_casts`.

The first observed difference is before the FP8 cast, in the fp32 normalized
value:

```text
y / tmp13 max diff:  ~9.5e-07 to 1.9e-06
scale max diff:      ~2.8e-09
div max diff:        ~9e-05
```

On the local B200/CUDA 12.8 machine, the final FP8 output usually does not cross
a bucket boundary. On CUDA13 CI, it apparently does, producing a dequantized FP8
diff such as `32.0`.

## Smaller diagnostic repro

A simpler form reproduces the first fp32 drift:

```python
def f(x, weight):
    y = F.rms_norm(x, (D,), weight)
    a = y.view(B, D // G, G).abs().amax(dim=-1)
    return y, a
```

Observed locally:

```text
return y alone:
  y matches the unfused producer bitwise

unfused return y, amax(y):
  y matches return-y-alone bitwise

nested return y, amax(y):
  y differs from unfused by about 1 ulp
```

This means the larger nested Triton kernel changes the fp32 bits of the producer
expression itself, even though the algebraic expression is the same.

## Materialization boundary experiment

If the test function forces materialization after RMSNorm:

```python
x = F.rms_norm(x, (D,), weight)
x = torch.ops._inductor_test.realize(x)
```

then nested fusion still occurs for the grouped reduction, but nested and
unfused outputs become bitwise identical for the relevant values.

This shows the sensitive boundary is the materialized normalized value `x`, not
the RMS sum alone.

Realizing only the RMS sum does not remove the drift.

## Current interpretation

This does not appear to be a missing FP8 truncation in
`emulate_precision_casts`. The explicit FP8 cast is present.

It also does not appear to be different RMS reduction tiling.

The concrete difference is that the unfused schedule consumes a materialized
RMSNorm output in later kernels, while the nested schedule consumes the fp32
producer expression inside the same larger Triton kernel. That larger kernel can
produce fp32 results that differ by about 1 ulp. FP8 bucket boundaries amplify
the difference.

`emulate_precision_casts` currently preserves low-precision fp16/bf16 eager
barriers and explicit casts. It does not promise to reproduce every fp32
multi-kernel materialization boundary inside a fused kernel.

## Possible fixes

### Test-only: deterministic FP8-stable inputs

Use deterministic values away from FP8 rounding thresholds. This keeps the test
focused on nested full-resolution epilogue wiring and avoids random FP8 bucket
boundary sensitivity.

Example:

```python
x = torch.ones(B, D, device=GPU_TYPE)
w = torch.tensor([-0.5, 0.25, 1.0, -0.25], device=GPU_TYPE).repeat(D // 4)
```

This preserves nested fusion coverage locally.

### Test-only: explicit materialization

Add:

```python
x = torch.ops._inductor_test.realize(x)
```

after RMSNorm. This makes the nested test compare against the same materialized
producer boundary as the unfused schedule.

This is more semantically direct, but it also changes what the test is covering:
it no longer exercises full fusion through the RMSNorm producer value.

### Production lowering: insert a barrier

Force an fp32 materialization or optimization barrier inside nested codegen for
this producer-consumer path.

This would likely defeat part of the point of the fusion, and it is broader than
what `emulate_precision_casts` currently means. It should probably not be done
unless we decide nested fusion must preserve multi-kernel fp32 materialization
numerics exactly.

## Open question

Why does the larger nested Triton kernel produce a different fp32 `y` by about
1 ulp while using the same visible expression and same RMS reduction launch
config?

This may be due to Triton/NVIDIA codegen decisions for the larger kernel, such
as instruction selection, scheduling, register pressure, or expression
rewriting. It is not yet explained at the PT2 IR level.
