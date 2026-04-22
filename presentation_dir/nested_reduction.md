# Nested Reduction (Different-Size Dependent Reductions)

**Status: WIP — not yet landed. Currently on branch `nested_reduction_backup`.**

**Config:** `torch._inductor.config.triton.nested_reduction` (default `True`)
**Test:** `test/inductor/test_nested_reduction.py` (22 tests)
**Repro:** `agent_space/capture_nested_reduction.py` (logs in `nested_{off,on}.log`)

---

## The problem

A very common pattern in quantization pipelines: **RMS norm** (a large
reduction over D elements) followed by **block amax** (a small reduction
over G elements of the normalized output). The second reduction *depends
on* the first — it consumes the full-resolution normalized values and
further reduces them into groups.

```python
def f(x):  # x: [B, D]
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    x_normed = x / rms
    return x_normed.reshape(B, D // G, G).abs().amax(dim=-1)
```

The two reductions have *different sizes*:
- Reduction 1 (mean): `(B=128, r=8192)` — large, one result per row.
- Reduction 2 (amax): `(B*D//G=65536, r=16)` — small, one result per group.

Because the second reduction has a data dependency on the first (it reads
the normalized output), and they have completely different iteration
domains, normal fusion can't combine them.

Without nested reduction: 2 kernels, `arg0_1` loaded from HBM **twice**.

## Pre-fusion IR (identical for both configs)

```
op0: SchedulerNode  # mean(x*x, dim=-1) → buf0 (the RMS variance)
  op0.group.iteration = (128, 8192)
  op0.sizes = ([128], [8192])
  var_ranges = {p0: 128, p1: 8192}
  body: load arg0_1[8192*p0 + p1], compute x*x, reduce sum → store buf0[p0]

op1: SchedulerNode  # abs(x/rms).reshape.amax(dim=-1) → buf1
  op1.group.iteration = (65536, 16)
  op1.sizes = ([128, 512], [16])
  op1.unmet_dependencies = [MemoryDep('buf0', ...)]
  var_ranges = {p0: 128, p1: 512, p2: 16}
  body: load arg0_1[8192*p0 + 16*p1 + p2],     # re-read full x
        load buf0[p0],                           # read variance
        compute x_normed = x / sqrt(mean + eps),
        abs → reduce max → store buf1[512*p0 + p1]
```

These can't fuse because:
1. **Different iteration groups**: `(128, 8192)` vs `(65536, 16)`.
2. **Data dependency**: op1 needs `buf0` from op0.

## Post-fusion — OFF

```
op0: SchedulerNode   # variance reduction
op1: SchedulerNode   # norm + group amax
```

**2 kernels**, `arg0_1` loaded twice:

```python
triton_red_fused_mean_mul_0.run(arg0_1, buf0, 128, 8192, ...)
triton_per_fused_abs_add_amax_div_mean_mul_sqrt_view_1.run(arg0_1, buf0, buf1, 65536, 16, ...)
```

## Post-fusion — ON

```
op0_op1: FusedNestedReductions(SchedulerNode, SchedulerNode)
  writes = [buf0 (128), buf1 (128*512)]
  met_deps = [arg0_1 (two different access patterns)]
  snodes[0] = op0 (variance, iteration (128, 8192))
  snodes[1] = op1 (norm + amax, iteration (65536, 16))
```

**1 kernel:**

```python
triton_per_fused_mean_mul_0.run(arg0_1, buf1, 128, 8192, ...)
```

Note: `buf0` (the variance) doesn't appear in the call — it never
materializes to HBM.

## Output code — OFF (2 kernels)

```python
# Kernel 1: x*x → sum → buf0 (variance per row)
@triton.jit
def triton_red_fused_mean_mul_0(in_ptr0, out_ptr0, xnumel, r0_numel, ...):
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), ...)   # load arg0_1
        tmp1 = tmp0 * tmp0
        _tmp3 = _tmp3 + tmp1
    tl.store(out_ptr0 + (x0), tl.sum(_tmp3, 1), xmask)     # store variance

# Kernel 2: re-read arg0_1, re-read buf0, normalize, group amax → buf1
@triton.jit
def triton_per_fused_amax_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, ...):
    tmp0 = tl.load(in_ptr0 + (16*x1 + r0_1 + 8192*x0), ...)  # re-load arg0_1
    tmp1 = tl.load(in_ptr1 + (x0), ...)                        # load variance
    tmp10 = tl_math.abs(tmp0 / tl.sqrt(tmp1/8192 + 1e-6))
    tmp14 = triton_helpers.max2(...)                             # group amax
    tl.store(out_ptr0 + (x1 + 512*x0), tmp14, ...)
```

## Output code — ON (1 fused kernel)

```python
@triton.jit
def triton_per_fused_mean_mul_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 8192
    r0_1 = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex

    # ---- Pass 1: variance (full 8192-element reduction) ----
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp5 = tl.sum(tmp1, 1)[:, None].to(tl.float32)

    # ---- Epilogue: normalize (full resolution, no round-trip) ----
    tmp10 = tl.sqrt_rn(tmp5 / 8192.0 + 1e-6)
    tmp11 = tmp0 / tmp10                      # x_normed, still in registers
    tmp12 = tl_math.abs(tmp11)

    # ---- Pass 2: group amax (nested 16-element reduction) ----
    tmp13 = tl.reshape(tmp12, [XBLOCK, R0_BLOCK // 16, 16])    # reshape in-register
    tmp14 = triton_helpers.max2(tmp13, 2)                       # reduce over groups of 16
    pass2_r = tl.arange(0, R0_BLOCK // 16)[None, :]
    tl.store(out_ptr0 + (pass2_r + 512*pass2_x), tmp14, pass2_mask)
```

**One kernel, one load of `arg0_1`.** The flow is:

1. Load `x`, compute `x²`, reduce to variance (pass 1 — `tl.sum`).
2. In the same tile, normalize: `x_normed = x / sqrt(var + eps)`.
3. In-register `tl.reshape` to `[XBLOCK, D//G, G]`, then `max2(..., 2)` —
   a **nested** 16-element reduction inside the 8192-element tile (pass 2).
4. Store the group-level amax results.

The variance never touches HBM. The full-resolution `x_normed` lives
entirely in registers between pass 1 and pass 2.

## The two "patterns"

The test file exercises two geometric patterns:

**Pattern 1 (small dim in x):** RMS norm → weighted sum over a small
outer dim (e.g. `(w * x_normed).sum(dim=1)`). Here the nested reduction
is over the outer `K` dimension while the full-resolution epilogue lives
at `[B*K, D]`.

**Pattern 2 (small dim in r):** RMS norm → block amax/sum over groups of
the inner dim. This is the NVFP4 / MX-fp8 pattern: norm → reshape into
groups → per-group reduction. Shown above.

Both patterns produce a `FusedNestedReductions` node and generate a
single kernel with a nested `tl.reshape + reduce` after the main
reduction.

## Full NVFP4 example (sm_100+)

`test_pass3_rmsnorm_nvfp4` is the end-to-end case:

```python
x = F.rms_norm(x, (D,), weight)      # pass 1: big reduction
x = x.view(B, D//G, G)
amax = x.abs().amax(dim=-1)           # pass 2: group amax (nested)
scale = (amax / 448.0).clamp(min=1e-12).to(float8_e4m3fn)
# pass 3: even/odd split → inline_asm → NVFP4 pack
packed = inline_asm_elementwise(even, odd,
    asm_str='cvt.rn.satfinite.e2m1x2.f32 ...', ...)
```

With nested reduction **and** pass 3 enabled, the entire pipeline
(norm + block amax + scale + NVFP4 quantize) fuses into **1 kernel**.

## Narrative for slides

- **Problem:** quantization pipelines chain two (or more) reductions
  of completely different sizes — e.g. 8192-element norm then 16-element
  group max. They have a data dependency (the second consumes the first's
  full-resolution output). Normal fusion can't handle this.
- **Before:** 2 kernels. The full `x` is loaded from HBM twice (once for
  the norm reduction, once for the group amax). The variance intermediate
  (`buf0`) materializes to HBM.
- **After:** 1 kernel. `x` loaded once. Variance stays in registers.
  Full-resolution `x_normed` stays in registers. The group amax is
  computed via an **in-register reshape + reduce** (`tl.reshape` →
  `max2(dim=2)`) at a different granularity than the first reduction.
  No HBM traffic for intermediates.
- **Key insight:** a persistent-reduction kernel that holds 8192 elements
  in registers can do *any smaller reduction* on those same values
  by reshaping the register tile. The "nested" reduction is just a
  differently-shaped view of the same registers.
- **Performance (from memory):** vLLM RMS+FP8 quant: 2-3× faster than
  hand-written CUDA kernel. NVFP4 pass 3: 0.034ms vs 0.031ms hand-written,
  vs 0.094ms unfused.

## Runnable repro

```python
# Toggle NESTED between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

NESTED = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

B, D, G = 128, 8192, 16

@torch.compile(fullgraph=True)
def f(x):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    x_normed = x / rms
    return x_normed.reshape(B, D // G, G).abs().amax(dim=-1)

x = torch.randn(B, D, device="cuda")

with inductor_config.patch({
    "triton.nested_reduction": NESTED,
    "triton.unique_kernel_names": True,
}):
    out = f(x)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```

For the full NVFP4 end-to-end (requires sm_100+):
```bash
python test/inductor/test_nested_reduction.py -k test_pass3_rmsnorm_nvfp4
```
