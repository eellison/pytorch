# Loop Reindexing After Fusion

**Config:** `torch._inductor.config.loop_reindexing_after_fusion`  (defaults to `True`)
**Landing PR:** [#176927](https://github.com/pytorch/pytorch/pull/176927)
**Test:** `test/inductor/test_loop_ordering.py::LoopOrderingTest::test_reshape_reindexing_for_reduction`
**Repro:** `agent_space/capture_loop_reindexing.py` (logs in `loop_reindex_{off,on}.log`)

---

## The problem

A reshape before a reduction re-factors the iteration space. The reduction
picks up the factored loop `[M*num_heads, head_dim]`. The pointwise
epilogue (after `.reshape(M, N)`) lives in `[M, N]`. Same numel, different
iteration domain → different `MemoryDep` index expressions → no fusion.

Canonical case: RMS norm on a per-head layout.

```python
def f(x):
    head_dim = 128
    M, N = x.shape                       # [16, 8192]
    x_reshaped = x.reshape(-1, head_dim) # [1024, 128]
    x_f32 = x_reshaped.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
    return x_normed.reshape(M, N).to(x.dtype)
```

(Input is non-contiguous — a slice of a QKV projection, stride `[10240, 1]`.)

## Pre-fusion IR (identical for both configs)

```
op0: SchedulerNode(ComputedBuffer)  # pow + mean reduction
  op0.group.iteration = (1024, 128)
  op0.sizes = ([1024], [128])
  var_ranges = {p0: 1024, p1: 128}
  index0 = 10240*(p0//64) + ModularIndexing(128*p0 + p1, 1, 8192)   # x.reshape(1024,128) read
  index1 = p0                                                        # store variance

op1: SchedulerNode(ComputedBuffer)  # rsqrt + mul + reshape + cast
  op1.group.iteration = (131072, 1)                                  # ← fully flat pointwise
  op1.sizes = ([16, 8192], [])
  var_ranges = {p0: 16, p1: 8192}
  index0 = 10240*p0 + p1                                             # read arg0_1 in [M,N] layout
  index1 = 64*p0 + (p1//128)                                         # read variance (per head)
  index2 = 8192*p0 + p1                                              # store output
```

`op0` and `op1` walk the same data with **different factorizations**:
`(1024, 128)` vs `(16, 8192)`. Scheduler's fusion check compares
`MemoryDep`s structurally, sees mismatched var counts / coefficients, and
declines to fuse.

## Post-fusion — reindexing OFF

Two separate SchedulerNodes survive:

```
op0: SchedulerNode   # unchanged reduction
op1: SchedulerNode   # unchanged pointwise
```

Result: **2 kernels.** `arg0_1` is loaded from HBM twice.

## Post-fusion — reindexing ON

```
op0_op1: FusedSchedulerNode(SchedulerNode, SchedulerNode)
  op0_op1.met_dependencies = [
    MemoryDep('arg0_1', 10240*(d0//64) + ModularIndexing(128*d0 + d1, 1, 8192),
              {d0: 1024, d1: 128})    # ← now uses op0's factorization
  ]

  op1 (pointwise), REWRITTEN:
    var_ranges = {p0: 1024, p1: 128}                                 # ← was (16, 8192)
    index0 = 10240*(p0//64) + ModularIndexing(128*p0 + p1, 1, 8192)  # ← matches op0 exactly
    index1 = p0                                                       # load variance (now 1D)
    index2 = 128*p0 + p1                                              # store output in [1024,128] layout
```

The scheduler re-expressed `op1`'s iteration via `FloorDiv` / `ModularIndexing`
of a flat index, lifting `[16, 8192]` into `[1024, 128]`. The shared read
of `arg0_1` now has identical index expressions → fusion succeeds.

## Output code — OFF (2 kernels)

```python
# Kernel 1: persistent_reduction (pow + mean)
@triton.jit
def triton_per_fused_clone_mean_pow_0(in_ptr0, out_ptr0, xnumel, r0_numel, ...):
    # xnumel=1024, r0_numel=128
    tmp0 = tl.load(in_ptr0 + (10240*(x0 // 64) + ((r0_1 + 128*x0) % 8192)), ...)
    tmp2 = tmp0 * tmp0
    tmp6 = tl.sum(...)
    tl.store(out_ptr0 + (x0), tmp6, xmask)

# Kernel 2: pointwise (rsqrt + mul + cast)
@triton.jit
def triton_poi_fused_mul_rsqrt_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, ...):
    # xnumel=131072 — flat pointwise
    tmp0 = tl.load(in_ptr0 + (x0 + 10240*x1), None)           # re-reads arg0_1
    tmp2 = tl.load(in_ptr1 + (x2 // 128), None)               # reads variance
    tmp7 = libdevice.rsqrt(tmp4 + 1e-5)
    tmp8 = tmp1 * tmp7
    tl.store(out_ptr0 + (x2), tmp9, None)

def call(args):
    triton_per_fused_clone_mean_pow_0.run(arg0_1, buf0, 1024, 128, ...)
    triton_poi_fused_mul_rsqrt_view_1.run(arg0_1, buf0, buf1, 131072, ...)
```

## Output code — ON (1 fused kernel)

```python
@triton.jit
def triton_per_fused_mean_mul_pow_rsqrt_0(in_ptr0, out_ptr1, xnumel, r0_numel, ...):
    # xnumel=1024, r0_numel=128 — matches reduction's factored iteration
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10240*(x0 // 64) + ((r0_1 + 128*x0) % 8192)),
                   xmask, other=0.0).to(tl.float32)         # single read of arg0_1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp6 = tl.sum(...)                                       # variance
    tmp8 = tmp6 / 128.0
    tmp11 = libdevice.rsqrt(tmp8 + 1e-5)
    tmp12 = tmp1 * tmp11                                     # epilogue fused in-place
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp13, xmask)       # store in [1024,128]
```

One kernel. `arg0_1` read once. Variance never materializes to HBM.

## What the transformation did

1. Scheduler sees two nodes on the same `numel` but with different
   size factorizations (`[1024, 128]` vs `[16, 8192]`).
2. With `loop_reindexing_after_fusion=True`, it tries to re-express the
   pointwise's iteration vars as `FloorDiv` / `ModularIndexing` of a flat
   index so its index expressions match the reduction's.
3. It actually rewrites the pointwise `loop_body`: swaps `var_ranges`,
   rewrites every load/store index, re-validates the `MemoryDep`s.
4. If the re-expressed deps now match, fusion goes through. If not, the
   scheduler rolls back (see `test_reindex_rollback_on_no_improvement`).

## Narrative for slides

- **Why it matters:** any op that does a reshape followed by a reduction hits
  this. RMS norm on per-head layouts (qknorm), grouped quantization,
  block-wise norms — all have a reshape that splits a dim, a reduction
  over the new inner dim, and then a reshape back for the epilogue.
- **Before:** 2 kernels, input read twice (bandwidth-bound case → ~2× slowdown).
- **After:** 1 kernel, single pass over input, variance stays in registers.
- **Motivating workload:** qknorm in attention — see
  `test_qknorm_rope_fusion` / `test_qknorm_interleaved_rope_fusion`.
- **Complements loop reordering:** reordering swaps axis *order* of a
  fixed factorization; reindexing changes the *factorization* itself.
  Both run during the fusion pass; either (or both) can unlock a fusion.

## Runnable repro

```python
# Toggle REINDEX between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

REINDEX = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(x):
    head_dim = 128
    M, N = x.shape
    x_reshaped = x.reshape(-1, head_dim)
    x_f32 = x_reshaped.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
    return x_normed.reshape(M, N).to(x.dtype)

# Non-contiguous input (slice of a qkv projection)
qkv = torch.randn(16, 10240, dtype=torch.bfloat16, device="cuda")
x = qkv[:, :8192]

with inductor_config.patch({
    "loop_reindexing_after_fusion": REINDEX,
    "triton.unique_kernel_names": True,
}):
    y = f(x)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```
