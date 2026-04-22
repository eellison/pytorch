# Loop Reordering After Fusion

**Config:** `torch._inductor.config.loop_ordering_after_fusion`
**Test:** `test/inductor/test_loop_ordering.py::LoopOrderingTest::test_sum_and_t`
**Repro:** `agent_space/capture_loop_reordering.py` (full logs in `loop_reorder_{off,on}.log`)

---

## The problem

Inductor's scheduler groups nodes into kernels by matching their iteration order
(`op.group.iteration`, `op.sizes`). Two nodes that touch the same data but walk it
in different orders land in separate kernels — even when they could share loads.

```python
def f(x):
    return x.sum(dim=-1), x.t().contiguous()
```

`x.sum(dim=-1)` wants loop order `(256, 512)` (row-major).
`x.t().contiguous()` wants loop order `(512, 256)` (walks the output contiguously).
Same input, incompatible loop nests → 2 kernels.

## Pre-fusion IR (same whether reordering is on or off)

```
op0: SchedulerNode(ComputedBuffer)  # x.sum(dim=-1)
  op0.group.iteration = (256, 512)           # ← pointwise dim 256, reduction 512
  op0.sizes = ([256], [512])
  var_ranges = {p0: 256, p1: 512}
  index0 = 512*p0 + p1                        # load arg0_1 row-major
  index1 = p0                                 # store buf0[p0]

op1: SchedulerNode(ComputedBuffer)  # x.t().contiguous()
  op1.group.iteration = (131072, 1)          # ← purely pointwise, flattened
  op1.sizes = ([512, 256], [])
  var_ranges = {p0: 512, p1: 256}             # note: outer=512, inner=256
  index0 = p0 + 512*p1                        # load arg0_1 (strided along outer!)
  index1 = 256*p0 + p1                        # store buf1 row-major
```

The two nodes are incompatible: op0 iterates `(256, 512)`, op1 iterates `(512, 256)`.

## Post-fusion IR — reordering OFF (baseline)

```
op0: SchedulerNode(ComputedBuffer)   # unchanged
op1: SchedulerNode(ComputedBuffer)   # unchanged
```

Two separate `SchedulerNode`s. No `FusedSchedulerNode`. **2 kernels.**

## Post-fusion IR — reordering ON

```
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)      ← fused!
  op0_op1.writes = [buf0 (256), buf1 (d0 + 256*d1, 256x512)]
  op0_op1.met_dependencies = [MemoryDep('arg0_1', 512*d0 + d1, {d0: 256, d1: 512})]

  op0 (sum): unchanged, var_ranges={p0: 256, p1: 512}

  op1 (transpose), REWRITTEN:
    var_ranges = {p0: 256, p1: 512}           # ← swapped from (512, 256)
    index0 = 512*p0 + p1                       # ← now matches op0's load!
    index1 = p0 + 256*p1                       # transposed store (strided)
```

The scheduler rewrote `op1`'s loop body to walk `(256, 512)` instead of `(512, 256)`.
The load index becomes `512*p0 + p1`, identical to `op0`'s load — so the shared read
can be hoisted and the two nodes fuse. The transposed store absorbs the strided
access pattern that used to live on the *load* side.

## Output code — OFF (2 kernels)

```python
# Kernel 1: persistent reduction
@triton.jit
def triton_per_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    # xnumel=256, r0_numel=512
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp4 = tl.sum(...)
    tl.store(out_ptr0 + (x0), tmp4, xmask)

# Kernel 2: 2D pointwise transpose
@triton.jit
def triton_poi_fused_clone_t_1(in_ptr0, out_ptr0, ynumel, xnumel, ...):
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), ...)
    tl.store(out_ptr0 + (x1 + 256*y0), ...)

def call(args):
    triton_per_fused_sum_0.run(arg0_1, buf0, 256, 512, ...)
    triton_poi_fused_clone_t_1.run(arg0_1, buf1, 512, 256, ...)
```

Two launches. `arg0_1` is loaded from HBM twice.

## Output code — ON (1 fused kernel)

```python
@triton.jit
def triton_red_fused_clone_sum_t_0(in_ptr0, out_ptr0, out_ptr1,
                                   xnumel, r0_numel,
                                   XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    # xnumel=256, r0_numel=512  — matches op0's (256, 512) loop order
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask & xmask, other=0.0)   # shared load
        _tmp2 = _tmp2 + tmp0                                                     # accumulate sum
        tl.store(out_ptr1 + (x0 + 256*r0_1), tmp0, r0_mask & xmask)             # transposed store
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)                                       # final sum
```

One kernel. `arg0_1` loaded **once**. The transposed store is interleaved into the
reduction loop. Two writes share one read.

## What the transformation did

1. Scheduler tried to fuse `op0` and `op1` but saw their iteration orders differ.
2. With `loop_ordering_after_fusion=True`, it asks: *can I rewrite op1's loop nest
   to match op0's?* It uses the inverse of op1's index expressions to remap vars.
3. Result: op1's `var_ranges` flip from `{p0: 512, p1: 256}` to `{p0: 256, p1: 512}`,
   its load index becomes `512*p0 + p1` (equal to op0's), and its store picks up
   the strided access pattern.
4. Now both nodes share the same iteration group → they fuse into one kernel.

## Narrative for slides

- **Before (2 kernels):** arg0_1 read twice, 2 kernel launches.
- **After (1 kernel):** arg0_1 read once, transposed store absorbed into the
  reduction loop. Roughly 2× HBM traffic saved for this pattern.
- **Generalizes to:** any pair of consumers of the same input where one wants the
  "natural" layout and the other wants a transposed/permuted layout — e.g.
  `q.sum(-1), q.transpose(...).contiguous()` patterns that show up in attention
  pre-processing.

## Runnable repro

```python
# Toggle LOOP_REORDER between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

LOOP_REORDER = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(x):
    return x.sum(dim=-1), x.t().contiguous()

x = torch.randn(256, 512, device="cuda")

with inductor_config.patch({
    "loop_ordering_after_fusion": LOOP_REORDER,
    "triton.unique_kernel_names": True,
}):
    outs = f(x)
    torch.cuda.synchronize()
```

Run with caches disabled so you see fresh compilation output:

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```

