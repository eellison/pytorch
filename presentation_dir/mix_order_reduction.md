# Mix-Order Reduction

**Config:** `torch._inductor.config.triton.mix_order_reduction` (default `True` in OSS)
**Test:** `test/inductor/test_mix_order_reduction.py::MixOrderReductionTest`
**Repro:** `agent_space/capture_mix_order_reduction.py` (logs in `mix_order_{off,on}.log`)

---

## The problem

Two reductions over the **same tensor** along **different** dimensions cannot
fuse with normal loop-based scheduling: they want opposite iteration orders.

```python
def f(x):
    return x.sum(dim=0), x.sum(dim=1)   # x: [32768, 768]
```

- `x.sum(dim=0)` → iteration `(768, 32768)` — outer is cols, inner reduce is rows.
- `x.sum(dim=1)` → iteration `(32768, 768)` — outer is rows, inner reduce is cols.

Every normal fusion trick fails: the loops are genuinely opposing. Default
behavior is 2 kernels, each doing a full pass over `x`.

## Pre-fusion IR (identical for both configs)

```
op0: SchedulerNode(ComputedBuffer)            # x.sum(dim=0)
  op0.group.iteration = (768, 32768)          # outer=cols, reduce=rows
  var_ranges = {p0: 768, p1: 32768}
  index0 = p0 + 768*p1                         # col-walk input
  index1 = p0

op1: SchedulerNode(ComputedBuffer)            # x.sum(dim=1)
  op1.group.iteration = (32768, 768)          # outer=rows, reduce=cols
  var_ranges = {p0: 32768, p1: 768}
  index0 = 768*p0 + p1                         # row-walk input
  index1 = p0
```

## Post-fusion — OFF

```
op0: SchedulerNode   # unchanged (768, 32768)
op1: SchedulerNode   # unchanged (32768, 768)
```

**2 kernels**, each with a full pass over `x`:

```python
triton_red_fused_sum_0.run(arg0_1, buf0, 768, 32768, ...)   # sum(dim=0)
triton_per_fused_sum_1.run(arg0_1, buf1, 32768, 768, ...)   # sum(dim=1)
```

## Post-fusion — ON

```
op1_op0: FusedMixOrderReductions(SchedulerNode, SchedulerNode)   ← new node type
  writes = [buf0 (768), buf1 (32768)]
  met_deps = [
    MemoryDep('arg0_1', 768*d0 + d1, {d0: 32768, d1: 768}),   # row-walk
    MemoryDep('arg0_1', d0 + 768*d1, {d0: 768, d1: 32768}),   # col-walk
  ]
  snodes[0] = op1  (inner reduction, iter = (32768, 768))
  snodes[1] = op0  (outer reduction, iter = (768, 32768))
```

The scheduler does NOT rewrite either node's loop body — it creates a
`FusedMixOrderReductions` that bundles both and defers fusion to codegen.

## Output code — OFF (2 kernels, 2 passes over `x`)

```python
# Kernel 1: outer reduction sum(dim=0) → buf0[768]
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK, R0_BLOCK):
    # xnumel=768 (cols), r0_numel=32768 (rows)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), ...)
        _tmp2 = _tmp2 + tmp0
    tl.store(out_ptr0 + (x0), tl.sum(_tmp2, 1), xmask)

# Kernel 2: inner reduction sum(dim=1) → buf1[32768]
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK):
    # xnumel=32768 (rows), r0_numel=768 (cols)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tl.store(out_ptr0 + (x0), tl.sum(tmp3, 1), None)
```

## Output code — ON (1 pass over `x` + small finalization)

```python
@triton.jit
def triton_per_fused_sum_0(in_ptr0, out_ptr0, ws_ptr, xnumel, r0_numel,
                           XBLOCK: tl.constexpr, RSPLIT_SIZE: tl.constexpr,
                           NUM_STAGES: tl.constexpr):
    # xnumel=32768 (rows), r0_numel=768 (cols)
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]           # accumulator for sum(dim=0)
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
        # --- inner reduction (sum(dim=1)) in the fast direction ---
        tmp4 = tl.sum(tmp0, 1)[:, None].to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp4, None)                      # buf1[x0]
        # --- outer reduction (sum(dim=0)): accumulate per-block ---
        tmp5 = tl.sum(tmp0, 0)
        accum0 = accum0 + tmp5
    # Flush per-block partials for sum(dim=0) to workspace
    tl.store(ws_ptr + (pid * r0_numel + r0_index), accum0, r0_mask)

# Wrapper then finalizes sum(dim=0) across blocks:
def call(args):
    workspace_0 = empty_strided_cuda((1572864,), (1,), torch.float32)
    triton_per_fused_sum_0.run(arg0_1, buf1, workspace_0, 32768, 768, ...)
    buf0 = workspace_0.view(2048, 768).sum(dim=0, keepdim=False)    # small finalize
```

The main kernel streams over `x` **once**, producing `buf1` directly and
writing per-block partials for the outer reduction into a small workspace.
A second (tiny) reduction over the `(num_blocks, 768)` workspace produces
`buf0`. Net effect: one pass over the big tensor instead of two.

## What the transformation did

1. Scheduler recognizes two reductions with "opposite" loop orders over the
   same data. It fuses them into a `FusedMixOrderReductions` node.
2. At codegen, the kernel picks the "fast" reduction (inner) as its main
   iteration space and lays out a block-split outer loop over the
   non-reduced dim of the other reduction.
3. Each iteration of the outer loop:
   - computes one row of `buf1` (the inner reduction) and stores it,
   - contributes partial sums to an in-register accumulator for `buf0`.
4. After the kernel, per-block partials for `buf0` are finalized with a
   small reduction over the workspace.

## Narrative for slides

- **Where this shows up in practice:** layer-norm / RMS-norm backward
  computes both `sum(dim=-1)` (for per-row stats) **and** `sum(dim=0..-2)`
  (for weight grads). Attention backward, various loss backwards. Any
  "reduce along different axes of the same activation" pattern.
- **Before:** 2 full passes over the activation. Memory bandwidth dominated.
- **After:** 1 full pass + 1 small pass over block partials. ~2× speedup
  on bandwidth-bound cases.
- **Key idea:** at codegen time, we don't need identical loop nests — as
  long as each tile of the input can be consumed by both reductions in
  parallel, we can stream it once.
- **Different from other fusions:** doesn't try to make iteration domains
  match (impossible here); generates a special kernel that drives both
  reductions from one input walk with a workspace for the "slow axis" one.

## Runnable repro

```python
# Toggle MIX_ORDER between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

MIX_ORDER = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(x):
    return x.sum(dim=0), x.sum(dim=1)

# Needs a "transformer-ish" shape — small shapes don't qualify.
x = torch.randn(32768, 768, device="cuda")

with inductor_config.patch({
    "triton.mix_order_reduction": MIX_ORDER,
    "triton.unique_kernel_names": True,
    "split_reductions": False,
}):
    out = f(x)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```

For a more realistic demonstration, swap in an RMSNorm/LayerNorm backward —
see `test_rms_norm_bwd` and `test_layer_norm_bwd_*` for templates.
