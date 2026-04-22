# Combo Kernels (Horizontal Fusion)

**Config:** `torch._inductor.config.combo_kernels` + `combo_kernel_per_subkernel_blocks`
**Test:** `test/inductor/test_combo_kernels.py`
**Repro:** `agent_space/capture_combo_kernels.py` (logs in `combo_{off,on}.log`)

---

## The problem

Two *independent* kernels that touch different tensors and have no data
dependency still cost 2 kernel launches and hide any potential overlap of
their reads behind the launch overhead. For small/medium-sized kernels,
launch overhead alone can dominate.

```python
def f(a, b):
    return a.sum(dim=-1), b.sum(dim=-1)   # a: [1024, 64], b: [1024, 512]
```

These two reductions share nothing. Normal fusion doesn't apply.

## Pre-fusion IR (identical for both configs)

```
op0: SchedulerNode  # a.sum(-1)
  op0.group.iteration = (1024, 64)
  reads arg0_1, writes buf0 (1024)

op1: SchedulerNode  # b.sum(-1)
  op1.group.iteration = (1024, 512)
  reads arg1_1, writes buf1 (1024)
```

Different reduction dims (`64` vs `512`), different inputs — nothing for
loop-based fusion to do.

## Post-fusion — OFF

```
op0: SchedulerNode   # a.sum(-1)
op1: SchedulerNode   # b.sum(-1)
```

2 SchedulerNodes → 2 kernels:

```python
triton_per_fused_sum_0.run(arg0_1, buf0, 1024, 64, ...)     # a.sum(-1)
triton_per_fused_sum_1.run(arg1_1, buf1, 1024, 512, ...)    # b.sum(-1)
```

## Post-fusion — ON

```
op0_op1: ForeachKernelSchedulerNode(SchedulerNode, SchedulerNode)   ← new node type
```

A `ForeachKernelSchedulerNode` bundles the independent nodes. Each
subkernel keeps its own loop structure (no attempt to align iteration
spaces — that's what `FusedSchedulerNode` would do). Codegen emits
**one kernel** whose `pid` dispatches into one of the sub-blocks.

## Output code — OFF (2 launches)

```python
triton_per_fused_sum_0.run(arg0_1, buf0, 1024, 64, stream=stream0)
triton_per_fused_sum_1.run(arg1_1, buf1, 1024, 512, stream=stream0)
```

## Output code — ON (1 launch, internal dispatch)

```python
@triton.jit
def triton_per_fused_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
                       XBLOCK_0: tl.constexpr, XBLOCK_1: tl.constexpr):
    pid = tl.program_id(0)
    x_blocks_0 = tl.cdiv(1024, XBLOCK_0)
    num_blocks_0 = x_blocks_0
    x_blocks_1 = tl.cdiv(1024, XBLOCK_1)
    num_blocks_1 = num_blocks_0 + x_blocks_1

    if pid < num_blocks_0:
        # -------- Subkernel 0: a.sum(dim=-1) --------
        local_pid = pid
        R0_BLOCK_0: tl.constexpr = 64             # autotuned per subkernel
        xoffset = local_pid * XBLOCK_0
        xindex = xoffset + tl.arange(0, XBLOCK_0)[:, None]
        r0_1 = tl.arange(0, R0_BLOCK_0)[None, :]
        tmp0 = tl.load(in_ptr0 + (r0_1 + 64*xindex), ...)
        tmp4 = tl.sum(tmp0, 1)[:, None].to(tl.float32)
        tl.store(out_ptr0 + (xindex), tmp4, xmask)

    elif pid < num_blocks_1:
        # -------- Subkernel 1: b.sum(dim=-1) --------
        local_pid = pid - num_blocks_0
        R0_BLOCK_1: tl.constexpr = 512            # different block size!
        xoffset = local_pid * XBLOCK_1
        xindex = xoffset + tl.arange(0, XBLOCK_1)[:, None]
        r0_3 = tl.arange(0, R0_BLOCK_1)[None, :]
        tmp5 = tl.load(in_ptr1 + (r0_3 + 512*xindex), ...)
        tmp9 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
        tl.store(out_ptr1 + (xindex), tmp9, xmask)

    else:
        pass

# One launch:
triton_per_fused_0.run(arg0_1, arg1_1, buf0, buf1, stream=stream0)
```

Key detail: `combo_kernel_per_subkernel_blocks=True` autotunes **per-subkernel**
block sizes (`R0_BLOCK_0=64`, `R0_BLOCK_1=512`). Without that flag the combo
kernel would have to pick one block size for all subkernels, often forcing
a compromise.

The grid metadata carries both subkernels' size hints:

```python
'grid_type': 'SequentialFlattenComboKernelGrid',
'combo_grid_meta': {
    'num_kernels': 2,
    'heuristic_0': 'persistent_reduction',
    'size_hints_0': {'x': 1024, 'r0_': 64},       # subkernel 0
    'heuristic_1': 'persistent_reduction',
    'size_hints_1': {'x': 1024, 'r0_': 512},      # subkernel 1
    ...
}
```

## What the transformation did

1. Scheduler finds independent nodes with compatible launch characteristics
   (both pointwise-ish, or both reductions with sane ranges). It wraps them
   in a `ForeachKernelSchedulerNode` rather than a `FusedSchedulerNode`.
2. Codegen emits a single Triton kernel whose program dispatches on `pid`.
   Each subkernel keeps its own loop body verbatim — no rewriting.
3. Each subkernel gets its own `XBLOCK_i` / `R0_BLOCK_i` autotune params so
   shapes with very different reduction sizes don't fight each other.
4. Launch grid is `sum(blocks_per_subkernel)` — one dispatch covers both.

## Narrative for slides

- **Where this helps:** optimizers applying per-parameter updates to many
  tensors (the historical motivator — foreach-style ops), any
  `for t in tensors: ...` pattern, graph-level horizontal fusion of
  independent small kernels.
- **Launch overhead matters:** a kernel launch is ~5µs; a small reduction
  may take only ~10µs. Packing 10 small kernels into one saves ~45µs of
  overhead.
- **Different from loop fusion:** no attempt to share loads or align
  iteration spaces. Pure "one launch, one PC, dispatch on pid" style.
  Think SIMT-within-CUDA-grid rather than SIMD-within-a-loop.
- **`combo_kernel_per_subkernel_blocks`:** lets each subkernel have its
  own autotuned block size. Without it, a `sum([1024, 64])` and a
  `sum([1024, 512])` would have to share the same R0_BLOCK — bad for one
  of them.
- **Scope:** optional; off by default in some settings. Works best when
  you have many small independent kernels that would otherwise be
  launch-bound.

## Runnable repro

```python
# Toggle COMBO between False and True to see both cases.
import torch
from torch._inductor import config as inductor_config

COMBO = True

torch._logging.set_logs(
    ir_pre_fusion=True,
    ir_post_fusion=True,
    output_code=True,
)

@torch.compile(fullgraph=True)
def f(a, b):
    return a.sum(dim=-1), b.sum(dim=-1)

# Two independent reductions, different shapes.
a = torch.randn(1024, 64, device="cuda")
b = torch.randn(1024, 512, device="cuda")

with inductor_config.patch({
    "combo_kernels": COMBO,
    "benchmark_combo_kernel": False,
    "combo_kernel_per_subkernel_blocks": COMBO,
    "triton.unique_kernel_names": True,
}):
    out = f(a, b)
    torch.cuda.synchronize()
```

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python repro.py 2> out.log
```
