# Index Inversion in Fusion

**Config:** `torch._inductor.config.loop_index_inversion_in_fusion` (default `True`)
**Motivating test:** `test/inductor/test_fp8.py::TestFP8Types::test_mx_fusion`
**Real-world repro:** `agent_space/test_reindex_deps.py` (MX-fp8 scaled_mm preamble)
**Minimal repro for slides:** `agent_space/capture_index_inversion.py` (logs in `index_inv_{off,on}.log`)

---

## The problem

A producer computes something in its "natural" layout. A consumer reads the
producer after a chain of `reshape → permute → reshape` (a *swizzle*). In the
consumer's iteration space the producer's read index is a nasty expression
involving `FloorDiv` / `ModularIndexing`. Scheduler sees mismatched deps,
declines to fuse.

Canonical case: MX-fp8 scale swizzle. Per-block `amax` writes a scale tensor;
the MX kernel requires that scale in a tiled `[outer, inner, 128, 4]` layout.

## Minimal example

```python
M, K, GROUP, SWIZZLE_OUTER = 256, 256, 32, 128

def f(x):
    x_blocks = x.reshape(M, K // GROUP, GROUP)
    scale = x_blocks.abs().amax(dim=-1)                    # [256, 8]
    scale_sw = (
        scale
        .reshape(M // SWIZZLE_OUTER, SWIZZLE_OUTER, (K // GROUP) // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)                                        # [2048]
    )
    return scale_sw + 1.0
```

## Pre-fusion IR (identical for both configs)

```
op0: SchedulerNode(ComputedBuffer)   # amax reduction
  op0.group.iteration = (2048, 32)
  var_ranges = {p0: 256, p1: 8, p2: 32}
  index0 = 256*p0 + 32*p1 + p2        # load arg0_1 (natural layout)
  index1 = 8*p0 + p1                  # store buf0 (scale, natural layout)

op1: SchedulerNode(ComputedBuffer)   # swizzle + add
  op1.group.iteration = (2048, 1)
  var_ranges = {p0: 2048}
  # Loads scale with the inverse-swizzle pile of FloorDiv / ModularIndexing:
  index0 = 1024*(p0//1024)
         + ModularIndexing(p0, 1, 4)
         + 8*ModularIndexing(p0, 4, 128)
         + 4*ModularIndexing(p0, 512, 2)
  index1 = p0                          # store buf1 (flat output)
```

op0 writes `buf0` at `8*d0 + d1` (2 vars), op1 reads `buf0` at that tortured
expression in 1 var. The `MemoryDep`s don't match structurally → fusion fails.

## Post-fusion — inversion OFF

```
op0: SchedulerNode   # reduction, unchanged
op1: SchedulerNode   # swizzle, unchanged
```

2 SchedulerNodes, **2 kernels**, `buf0` materialized to HBM.

## Post-fusion — inversion ON

```
op0_op1: FusedSchedulerNode(SchedulerNode, SchedulerNode)
  op0_op1.writes = [
    MemoryDep('buf0', 8*d0 + d1, {d0: 256, d1: 8}),                         # producer write
    MemoryDep('buf1', 1024*(c0//1024)                                       # ← swizzle moved to the STORE
                      + 4*ModularIndexing(c0, 8, 128)
                      + 512*ModularIndexing(ModularIndexing(c0, 1, 1024), 4, 2)
                      + ModularIndexing(ModularIndexing(ModularIndexing(c0, 1, 1024), 1, 8), 1, 4),
              {c0: 2048}),
  ]
  op0_op1.met_dependencies = [
    MemoryDep('arg0_1', 256*d0 + 32*d1 + d2, {...}),                        # reduction read
    MemoryDep('buf0', c0, {c0: 2048}),                                      # ← consumer read is now TRIVIAL
  ]

  op1 (swizzle), REWRITTEN:
    var_ranges = {p0: 2048}
    index0 = p0                                                              # ← load buf0 directly!
    index1 = 1024*(p0//1024) + 4*ModularIndexing(p0, 8, 128) + ...           # ← swizzle is now the STORE index
```

The scheduler generated the **inverse** of the reshape-permute-reshape and
used it to move the swizzle from the consumer's load side to its store side.
The consumer's dep on `buf0` is now just `c0` — identical to op0's flat
store, so the two share a loop and fuse. `buf0` never materializes.

## Output code — OFF (2 kernels)

```python
# Kernel 1: amax reduction → buf0 (natural scale layout)
@triton.jit
def triton_per_fused_abs_amax_0(in_ptr0, out_ptr0, xnumel, r0_numel, ...):
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp5 = triton_helpers.max2(...)
    tl.store(out_ptr0 + (x0), tmp5, xmask)

# Kernel 2: swizzled read → flat write
@triton.jit
def triton_poi_fused_add_view_1(in_ptr0, out_ptr0, xnumel, ...):
    x0 = xindex
    # load with the swizzle-inverse index
    tmp0 = tl.load(in_ptr0 + (1024*(x0//1024) + ((x0) % 4) + 8*(((x0//4)) % 128) + 4*(((x0//512)) % 2)), ...)
    tmp2 = tmp0 + 1.0
    tl.store(out_ptr0 + (x0), tmp2, None)
```

2 launches. `buf0` written then re-read.

## Output code — ON (1 fused kernel)

```python
@triton.jit
def triton_per_fused_abs_add_amax_clone_permute_view_0(in_ptr0, out_ptr1, xnumel, r0_numel, ...):
    # xnumel=2048, r0_numel=32 — one loop over the reduction's natural iteration
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)          # arg0_1, natural
    tmp1 = tl_math.abs(tmp0)
    tmp5 = triton_helpers.max2(...)                                      # amax in-register
    tmp7 = tmp5 + 1.0                                                    # epilogue
    # SWIZZLED STORE — the transformation ended up here
    tl.store(out_ptr1 + (4*((x0//8) % 128)
                         + 512*((((x0 % 1024))//4) % 2)
                         + 1024*(x0//1024)
                         + ((((x0 % 1024)) % 8) % 4)),
             tmp7, xmask)
```

One kernel. Scale never touches HBM. The swizzle is encoded in the store's
address calculation.

## What the transformation did

1. Scheduler wants to fuse the reduction (producer of `buf0`) with the
   consumer. Consumer's `buf0` dep has 1 var, producer's `buf0` write has 2
   vars — no structural match.
2. With `loop_index_inversion_in_fusion=True`, it tries to **invert** the
   consumer's read-index expression to get a formula for the consumer's
   iteration var in terms of the producer's write vars — using
   `generate_inverse_formula` (see `torch/_inductor/invert_expr_analysis.py`).
3. If inversion succeeds, the consumer's loop body is rewritten: the old
   load becomes a trivial `buf0[p0]`, and the inverted expression is pushed
   onto the **store** side. Now producer and consumer share an iteration
   domain → fuse.
4. In codegen, the complex index remains, but as an address-calc for the
   final store — costs a few integer ops, saves a full round trip to HBM.

## Real-world example (MX-fp8 scaled_mm preamble)

`agent_space/test_reindex_deps_toggle.py` is the full MX-fp8 preamble: the
`relu → abs → amax` reduction, the E8M0 exponent encoding (isnan / bitshift /
where), the fp8 quantize, **and** the 2-step reshape-permute-reshape swizzle
of the scale tile before it's fed to `_scaled_mm`.

- **Inversion OFF**: 2 kernels
  ```
  triton_per_fused_..._0.run(arg0_1, buf0, buf1, 32768, 32, ...)
  triton_poi_fused_..._1.run(buf0, buf2, 32768, ...)     # the scale swizzle
  extern_kernels._scaled_mm(buf1, arg1_1, buf2, arg2_1, ...)
  ```
- **Inversion ON**: 1 kernel for the whole preamble
  ```
  triton_per_fused_..._0.run(arg0_1, buf1, buf2, 32768, 32, ...)   # everything
  extern_kernels._scaled_mm(buf1, arg1_1, buf2, arg2_1, ...)
  ```

Scheduler log: `op0_op1_op2: FusedSchedulerNode(SchedulerNode, SchedulerNode, SchedulerNode)`.
Both the quantize pointwise and the scale swizzle fold into the reduction.

## Narrative for slides

- **Where this shows up in practice:** MX-fp8 scale swizzle (block amax →
  `[N/128, K/4, 128, 4]` tile layout that the scaled_mm kernel wants),
  NVFP4 scale layouts, and any reduction whose output needs to be reshaped
  into a hardware-specific tile before being consumed.
- **Before:** 2 kernels — scale materialized, re-read with gather-y indexing.
- **After:** 1 kernel — scale stays in registers, swizzle becomes store-side
  address math (address math is free; HBM round-trips aren't).
- **Limitation:** inversion must succeed symbolically. Works for chains of
  affine reshapes + permutes. Fails on data-dependent indexing.
- **Design note:** gist explaining the math —
  https://gist.github.com/eellison/6f9f4a7ec10a860150b15b719f9285a9

## Runnable repro

```python
# Toggle INVERT between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

INVERT = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

M, K = 256, 256
GROUP = 32
SWIZZLE_OUTER = 128

@torch.compile(fullgraph=True)
def f(x):
    x_blocks = x.reshape(M, K // GROUP, GROUP)
    scale = x_blocks.abs().amax(dim=-1)
    scale_sw = (
        scale
        .reshape(M // SWIZZLE_OUTER, SWIZZLE_OUTER, (K // GROUP) // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )
    return scale_sw + 1.0

x = torch.randn(M, K, dtype=torch.float32, device="cuda")

with inductor_config.patch({
    "loop_index_inversion_in_fusion": INVERT,
    "triton.unique_kernel_names": True,
}):
    y = f(x)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```

For a **full MX-fp8 scaled_mm preamble** (more realistic, closer to what the
motivating test does), run `agent_space/test_reindex_deps_toggle.py`:

```bash
TORCH_LOGS="ir_pre_fusion,ir_post_fusion,output_code" \
  INVERT=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
  python agent_space/test_reindex_deps_toggle.py 2> out.log
```

Toggle `INVERT=0` vs `INVERT=1` to see 2 kernels vs 1 kernel for the
pre-`_scaled_mm` block.
